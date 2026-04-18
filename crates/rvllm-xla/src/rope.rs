//! RoPE table precomputation for Gemma 4 on TPU.
//!
//! Gemma 4 uses dual RoPE tables:
//! - Sliding attention: theta=10000, rot_dim=256 -> cos/sin [max_pos, 128] f32
//! - Global attention:  theta=1000000, rot_dim=128 -> cos/sin [max_pos, 64] f32

#[cfg(feature = "tpu")]
use crate::client::{PjrtBufferHandle, PjrtClientHandle};
#[cfg(feature = "tpu")]
use crate::ffi::PjrtElementType;
#[cfg(feature = "tpu")]
use crate::Result;

/// Precompute cos/sin RoPE tables on the CPU.
///
/// Returns `(cos_table, sin_table)` each flattened as `[max_pos * half_dim]`
/// where `half_dim = rot_dim / 2`.
///
/// freq[i] = 1.0 / theta^(2*i / rot_dim)  for i in 0..half_dim
/// cos[pos, i] = cos(pos * freq[i])
/// sin[pos, i] = sin(pos * freq[i])
pub fn precompute_rope(theta: f32, rot_dim: usize, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half_dim = rot_dim / 2;
    let inv_rot_dim = 1.0 / rot_dim as f32;

    // Precompute inverse frequencies
    let freqs: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 * inv_rot_dim))
        .collect();

    let total = max_pos * half_dim;
    let mut cos_table = Vec::with_capacity(total);
    let mut sin_table = Vec::with_capacity(total);

    for pos in 0..max_pos {
        let pos_f = pos as f32;
        for &freq in &freqs {
            let angle = pos_f * freq;
            let (s, c) = angle.sin_cos();
            cos_table.push(c);
            sin_table.push(s);
        }
    }

    (cos_table, sin_table)
}

/// Precomputed RoPE cos/sin tables for Gemma 4 on TPU device memory.
///
/// Holds four device buffers:
/// - `cos_sliding` / `sin_sliding`: shape `[max_pos, 128]` f32 (theta=10000, rot_dim=256)
/// - `cos_global`  / `sin_global`:  shape `[max_pos, 64]`  f32 (theta=1000000, rot_dim=128)
#[cfg(feature = "tpu")]
pub struct Gemma4RopeTables {
    pub cos_sliding: PjrtBufferHandle,
    pub sin_sliding: PjrtBufferHandle,
    pub cos_global: PjrtBufferHandle,
    pub sin_global: PjrtBufferHandle,
    pub max_pos: usize,
}

#[cfg(feature = "tpu")]
impl Gemma4RopeTables {
    const SLIDING_THETA: f32 = 10_000.0;
    const SLIDING_ROT_DIM: usize = 256;
    const GLOBAL_THETA: f32 = 1_000_000.0;
    const GLOBAL_ROT_DIM: usize = 128;

    /// Precompute RoPE tables and upload to device 0.
    ///
    /// Tables are replicated -- call once and share across all layers.
    pub fn new(client: &PjrtClientHandle, max_pos: usize) -> Result<Self> {
        let (cos_s, sin_s) =
            precompute_rope(Self::SLIDING_THETA, Self::SLIDING_ROT_DIM, max_pos);
        let (cos_g, sin_g) =
            precompute_rope(Self::GLOBAL_THETA, Self::GLOBAL_ROT_DIM, max_pos);

        let half_sliding = (Self::SLIDING_ROT_DIM / 2) as i64; // 128
        let half_global = (Self::GLOBAL_ROT_DIM / 2) as i64; // 64
        let max_pos_i64 = max_pos as i64;

        let sliding_shape = [max_pos_i64, half_sliding];
        let global_shape = [max_pos_i64, half_global];

        let upload = |data: &[f32], shape: &[i64]| -> Result<PjrtBufferHandle> {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            };
            client.buffer_from_host(bytes, shape, PjrtElementType::F32, 0)
        };

        Ok(Self {
            cos_sliding: upload(&cos_s, &sliding_shape)?,
            sin_sliding: upload(&sin_s, &sliding_shape)?,
            cos_global: upload(&cos_g, &global_shape)?,
            sin_global: upload(&sin_g, &global_shape)?,
            max_pos,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_table_shape() {
        let max_pos = 512;
        let (cos, sin) = precompute_rope(10_000.0, 256, max_pos);
        assert_eq!(cos.len(), max_pos * 128);
        assert_eq!(sin.len(), max_pos * 128);
    }

    #[test]
    fn global_table_shape() {
        let max_pos = 512;
        let (cos, sin) = precompute_rope(1_000_000.0, 128, max_pos);
        assert_eq!(cos.len(), max_pos * 64);
        assert_eq!(sin.len(), max_pos * 64);
    }

    #[test]
    fn position_zero_is_identity() {
        let (cos, sin) = precompute_rope(10_000.0, 256, 4);
        // At pos=0, angle=0 for all freqs -> cos=1, sin=0
        for i in 0..128 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0,{i}] = {}", cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[0,{i}] = {}", sin[i]);
        }
    }

    #[test]
    fn freq_decreases_with_index() {
        // Higher indices should have lower frequencies (longer wavelengths)
        let (cos, _) = precompute_rope(10_000.0, 256, 64);
        // At pos=1, cos(freq[0]) should differ more from 1.0 than cos(freq[127])
        // because freq[0] > freq[127]
        let cos_pos1_first = cos[128]; // pos=1, i=0
        let cos_pos1_last = cos[128 + 127]; // pos=1, i=127
        let delta_first = (1.0 - cos_pos1_first).abs();
        let delta_last = (1.0 - cos_pos1_last).abs();
        assert!(
            delta_first > delta_last,
            "freq[0] should cause larger deviation: delta_first={delta_first}, delta_last={delta_last}"
        );
    }

    #[test]
    fn sin_cos_unit_circle() {
        let (cos, sin) = precompute_rope(10_000.0, 256, 128);
        // sin^2 + cos^2 = 1 for every entry
        for idx in 0..cos.len() {
            let sum = cos[idx] * cos[idx] + sin[idx] * sin[idx];
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "sin^2 + cos^2 = {sum} at idx {idx}"
            );
        }
    }
}
