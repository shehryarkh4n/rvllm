#![cfg(feature = "tpu")]

use tracing::info;

use crate::client::{PjrtBufferHandle, PjrtClientHandle};
use crate::ffi::PjrtElementType;
use crate::Result;

const NUM_LAYERS: usize = 60;
const KV_DIM: usize = 4096;
#[cfg(test)]
const NUM_DEVICES_DEFAULT: usize = 4;
#[cfg(test)]
const SHARD_DIM_DEFAULT: usize = KV_DIM / NUM_DEVICES_DEFAULT; // 1024

/// KV cache buffers for Gemma 4 on TPU v4/v5e (4-chip pod slice).
///
/// Full shape: [60, max_ctx, 4096] bf16
/// Sharded along dim 2 across 4 devices => each device holds [60, max_ctx, 1024] bf16.
///
/// Lifecycle per decode step:
///   1. donate() consumes self, yields raw handles for execute inputs
///   2. PJRT execute with donation flag -- runtime reuses the memory
///   3. from_outputs() wraps the returned buffers into a new KvCacheSet
pub struct KvCacheSet {
    pub k_cache: Vec<PjrtBufferHandle>,
    pub v_cache: Vec<PjrtBufferHandle>,
    max_ctx: usize,
    num_devices: usize,
}

impl KvCacheSet {
    /// Allocate zero-filled KV caches across all devices.
    ///
    /// Each device gets a shard of shape [NUM_LAYERS, max_ctx, SHARD_DIM] bf16.
    /// For a single-device setup (e.g. 1-chip TPU), the full [60, max_ctx, 4096]
    /// lives on device 0.
    pub fn new(client: &PjrtClientHandle, max_ctx: usize) -> Result<Self> {
        let nd = client.num_devices();
        assert!(nd > 0, "no TPU devices");
        let shard_dim = KV_DIM / nd;
        assert_eq!(
            shard_dim * nd,
            KV_DIM,
            "KV_DIM ({KV_DIM}) not evenly divisible by {nd} devices"
        );

        let shard_shape: Vec<i64> = vec![NUM_LAYERS as i64, max_ctx as i64, shard_dim as i64];
        let shard_bytes = NUM_LAYERS * max_ctx * shard_dim * 2; // bf16 = 2 bytes
        let zeros = vec![0u8; shard_bytes];

        let mut k_cache = Vec::with_capacity(nd);
        let mut v_cache = Vec::with_capacity(nd);

        for dev in 0..nd {
            let k = client.buffer_from_host(&zeros, &shard_shape, PjrtElementType::BF16, dev)?;
            let v = client.buffer_from_host(&zeros, &shard_shape, PjrtElementType::BF16, dev)?;
            k_cache.push(k);
            v_cache.push(v);
        }

        let total_mb = (shard_bytes * 2 * nd) / (1024 * 1024);
        info!(
            num_devices = nd,
            max_ctx = max_ctx,
            shard_dim = shard_dim,
            total_mb = total_mb,
            "KV cache allocated"
        );

        Ok(Self {
            k_cache,
            v_cache,
            max_ctx,
            num_devices: nd,
        })
    }

    /// Allocate zero-filled KV caches on a single device (no sharding).
    ///
    /// Shape: [NUM_LAYERS, max_ctx, KV_DIM] bf16 on device 0.
    pub fn new_single(client: &PjrtClientHandle, max_ctx: usize) -> Result<Self> {
        let shape: Vec<i64> = vec![NUM_LAYERS as i64, max_ctx as i64, KV_DIM as i64];
        let nbytes = NUM_LAYERS * max_ctx * KV_DIM * 2;
        let zeros = vec![0u8; nbytes];

        let k = client.buffer_from_host(&zeros, &shape, PjrtElementType::BF16, 0)?;
        let v = client.buffer_from_host(&zeros, &shape, PjrtElementType::BF16, 0)?;

        let total_mb = (nbytes * 2) / (1024 * 1024);
        info!(max_ctx = max_ctx, total_mb = total_mb, "KV cache allocated (single device)");

        Ok(Self {
            k_cache: vec![k],
            v_cache: vec![v],
            max_ctx,
            num_devices: 1,
        })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Per-shard shape: [NUM_LAYERS, max_ctx, shard_dim].
    pub fn shard_shape(&self) -> [i64; 3] {
        let shard_dim = KV_DIM / self.num_devices;
        [NUM_LAYERS as i64, self.max_ctx as i64, shard_dim as i64]
    }

    /// Total bytes across all shards (K + V).
    pub fn total_bytes(&self) -> usize {
        let shard_dim = KV_DIM / self.num_devices;
        NUM_LAYERS * self.max_ctx * shard_dim * 2 * 2 * self.num_devices
    }

    /// Consume self and return the raw buffer handles for donation to PJRT execute.
    ///
    /// Returns (k_handles, v_handles) where each vec has one handle per device.
    /// After this call, `self` is gone -- the PJRT runtime takes ownership of
    /// the underlying device memory and reuses it for the output buffers.
    pub fn donate(self) -> (Vec<PjrtBufferHandle>, Vec<PjrtBufferHandle>) {
        (self.k_cache, self.v_cache)
    }

    /// Reconstruct a KvCacheSet from execute outputs (the updated caches).
    ///
    /// `k_outputs` and `v_outputs` must each have one buffer per device,
    /// matching the shard layout of the donated inputs.
    pub fn from_outputs(
        k_outputs: Vec<PjrtBufferHandle>,
        v_outputs: Vec<PjrtBufferHandle>,
        max_ctx: usize,
    ) -> Self {
        let nd = k_outputs.len();
        assert_eq!(nd, v_outputs.len(), "k/v output count mismatch");
        assert!(nd > 0, "empty output buffers");
        Self {
            k_cache: k_outputs,
            v_cache: v_outputs,
            max_ctx,
            num_devices: nd,
        }
    }

    /// Collect references to all cache buffers in the order expected by the
    /// MLIR execute call: [k_dev0, k_dev1, ..., v_dev0, v_dev1, ...].
    ///
    /// Use this when building the input list for non-donating execute calls
    /// (e.g. speculative / debug runs).
    pub fn as_input_refs(&self) -> Vec<&PjrtBufferHandle> {
        let mut refs = Vec::with_capacity(self.k_cache.len() + self.v_cache.len());
        for k in &self.k_cache {
            refs.push(k);
        }
        for v in &self.v_cache {
            refs.push(v);
        }
        refs
    }

    /// Collect references ordered as [k_dev0, v_dev0, k_dev1, v_dev1, ...].
    ///
    /// Some MLIR programs expect interleaved k/v per device rather than
    /// all-k-then-all-v. Use whichever matches the compiled program.
    pub fn as_interleaved_refs(&self) -> Vec<&PjrtBufferHandle> {
        let mut refs = Vec::with_capacity(self.k_cache.len() + self.v_cache.len());
        for i in 0..self.num_devices {
            refs.push(&self.k_cache[i]);
            refs.push(&self.v_cache[i]);
        }
        refs
    }
}

/// Build the non-donatable input indices array for PJRT_ExecuteOptions.
///
/// When using buffer donation, all inputs are donatable by default.
/// Call this to mark specific input indices as non-donatable (e.g. weights,
/// RoPE tables -- anything that must survive across steps).
///
/// `total_inputs` is the total number of inputs to the execute call.
/// `kv_start_idx` is the index of the first KV cache input.
/// `num_kv_inputs` is the number of KV cache inputs (k + v across devices).
///
/// Returns indices of all inputs that are NOT KV caches (i.e. everything
/// except the KV buffers should be marked non-donatable).
pub fn non_donatable_indices(
    total_inputs: usize,
    kv_start_idx: usize,
    num_kv_inputs: usize,
) -> Vec<i64> {
    let kv_end = kv_start_idx + num_kv_inputs;
    (0..total_inputs)
        .filter(|&i| i < kv_start_idx || i >= kv_end)
        .map(|i| i as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_math() {
        assert_eq!(KV_DIM, 4096);
        assert_eq!(NUM_DEVICES_DEFAULT, 4);
        assert_eq!(SHARD_DIM_DEFAULT, 1024);
    }

    #[test]
    fn non_donatable_indices_basic() {
        // 18 inputs total, KV caches at indices 16,17
        let ndi = non_donatable_indices(18, 16, 2);
        assert_eq!(ndi.len(), 16);
        assert_eq!(ndi, (0..16).map(|i| i as i64).collect::<Vec<_>>());
    }

    #[test]
    fn non_donatable_indices_sharded() {
        // 24 inputs, KV at 16..24 (4 devices, k+v each = 8 buffers)
        let ndi = non_donatable_indices(24, 16, 8);
        assert_eq!(ndi.len(), 16);
        for i in 0..16 {
            assert_eq!(ndi[i], i as i64);
        }
    }

    #[test]
    fn from_outputs_roundtrip_shape() {
        // Cannot test with real buffers without a TPU, but verify the struct logic.
        let max_ctx = 2048;
        let cache = KvCacheSet {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            max_ctx,
            num_devices: 4,
        };
        assert_eq!(cache.shard_shape(), [60, 2048, 1024]);
        assert_eq!(cache.max_ctx(), 2048);
        assert_eq!(cache.num_devices(), 4);
    }

    #[test]
    fn total_bytes_calc() {
        let cache = KvCacheSet {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            max_ctx: 8192,
            num_devices: 4,
        };
        // 60 * 8192 * 1024 * 2 bytes * 2 (k+v) * 4 devices
        let expected = 60 * 8192 * 1024 * 2 * 2 * 4;
        assert_eq!(cache.total_bytes(), expected);
    }
}
