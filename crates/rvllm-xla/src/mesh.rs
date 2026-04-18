#![cfg(feature = "tpu")]

use tracing::info;

use crate::buffer::XlaDtype;
use crate::client::{PjrtBufferHandle, PjrtClientHandle};
use crate::device::XlaDeviceId;
use crate::{LLMError, Result};

/// 1D device mesh for tensor parallelism across TPU cores.
pub struct TpuMesh {
    client: PjrtClientHandle,
    /// Physical device indices in the mesh (ordered).
    devices: Vec<usize>,
    /// Tensor parallel degree (== devices.len()).
    tp_size: usize,
}

impl TpuMesh {
    /// Create a mesh using the first `tp_size` devices from the PJRT client.
    pub fn new(client: PjrtClientHandle, tp_size: usize) -> Result<Self> {
        let num_devices = client.num_devices();
        if tp_size == 0 || tp_size > num_devices {
            return Err(LLMError::GpuError(format!(
                "tp_size={tp_size} invalid (have {num_devices} devices)"
            )));
        }
        let devices: Vec<usize> = (0..tp_size).collect();
        info!(tp_size, num_devices, "TpuMesh created");
        Ok(Self { client, devices, tp_size })
    }

    /// Create a mesh from explicit device indices.
    pub fn from_devices(client: PjrtClientHandle, device_indices: Vec<usize>) -> Result<Self> {
        let num_devices = client.num_devices();
        for &idx in &device_indices {
            if idx >= num_devices {
                return Err(LLMError::GpuError(format!(
                    "device index {idx} out of range (have {num_devices})"
                )));
            }
        }
        if device_indices.is_empty() {
            return Err(LLMError::GpuError("empty device list".into()));
        }
        let tp_size = device_indices.len();
        info!(tp_size, ?device_indices, "TpuMesh created from explicit devices");
        Ok(Self { client, devices: device_indices, tp_size })
    }

    pub fn tp_size(&self) -> usize {
        self.tp_size
    }

    pub fn devices(&self) -> &[usize] {
        &self.devices
    }

    pub fn client(&self) -> &PjrtClientHandle {
        &self.client
    }

    /// Map logical shard index to physical device index.
    pub fn shard_to_device(&self, shard_idx: usize) -> usize {
        self.devices[shard_idx % self.tp_size]
    }

    /// Map logical shard index to XlaDeviceId.
    pub fn shard_to_device_id(&self, shard_idx: usize) -> XlaDeviceId {
        XlaDeviceId(self.shard_to_device(shard_idx))
    }
}

/// How a tensor is distributed across devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShardingSpec {
    /// Full copy on every device.
    Replicated,
    /// Split evenly along one axis.
    Sharded { axis: usize, num_shards: usize },
}

impl ShardingSpec {
    pub fn sharded(axis: usize, num_shards: usize) -> Self {
        Self::Sharded { axis, num_shards }
    }
}

/// Shard a host buffer across the mesh according to `spec`.
///
/// `data` is a flat byte slice in row-major order.
/// `shape` is the full (unsharded) tensor shape.
/// `dtype` determines element size.
///
/// Returns one `PjrtBufferHandle` per device in the mesh.
pub fn shard_buffer(
    mesh: &TpuMesh,
    data: &[u8],
    shape: &[i64],
    dtype: XlaDtype,
    spec: &ShardingSpec,
) -> Result<Vec<PjrtBufferHandle>> {
    let elem_size = dtype.size_bytes();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    let expected_bytes = total_elems * elem_size;
    if data.len() != expected_bytes {
        return Err(LLMError::GpuError(format!(
            "shard_buffer: data length {} != expected {} (shape {:?}, dtype {:?})",
            data.len(), expected_bytes, shape, dtype
        )));
    }

    let pjrt_dtype = dtype.to_pjrt();

    match spec {
        ShardingSpec::Replicated => {
            let shape_i64: Vec<i64> = shape.to_vec();
            let mut buffers = Vec::with_capacity(mesh.tp_size());
            for shard_idx in 0..mesh.tp_size() {
                let dev = mesh.shard_to_device(shard_idx);
                let handle = mesh.client().buffer_from_host(data, &shape_i64, pjrt_dtype, dev)?;
                buffers.push(handle);
            }
            Ok(buffers)
        }
        ShardingSpec::Sharded { axis, num_shards } => {
            let axis = *axis;
            let num_shards = *num_shards;

            if num_shards != mesh.tp_size() {
                return Err(LLMError::GpuError(format!(
                    "num_shards={num_shards} != tp_size={}",
                    mesh.tp_size()
                )));
            }
            if axis >= shape.len() {
                return Err(LLMError::GpuError(format!(
                    "shard axis {axis} >= ndim {}",
                    shape.len()
                )));
            }
            let dim = shape[axis] as usize;
            if dim % num_shards != 0 {
                return Err(LLMError::GpuError(format!(
                    "dim[{axis}]={dim} not divisible by num_shards={num_shards}"
                )));
            }
            let shard_dim = dim / num_shards;

            // Compute strides for slicing along `axis`.
            // outer_count = product of dims before axis
            // inner_count = product of dims after axis (in bytes)
            let outer_count: usize = shape[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);
            let inner_bytes: usize = shape[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1) * elem_size;
            let full_axis_stride = dim * inner_bytes;
            let shard_bytes_per_outer = shard_dim * inner_bytes;

            let mut shard_shape: Vec<i64> = shape.to_vec();
            shard_shape[axis] = shard_dim as i64;
            let shard_total_bytes = shard_bytes_per_outer * outer_count;

            let mut buffers = Vec::with_capacity(num_shards);
            let mut shard_buf = vec![0u8; shard_total_bytes];

            for shard_idx in 0..num_shards {
                let shard_offset = shard_idx * shard_bytes_per_outer;

                // Copy shard slice from the full buffer.
                for outer in 0..outer_count {
                    let src_start = outer * full_axis_stride + shard_offset;
                    let dst_start = outer * shard_bytes_per_outer;
                    shard_buf[dst_start..dst_start + shard_bytes_per_outer]
                        .copy_from_slice(&data[src_start..src_start + shard_bytes_per_outer]);
                }

                let dev = mesh.shard_to_device(shard_idx);
                let handle = mesh.client().buffer_from_host(
                    &shard_buf,
                    &shard_shape,
                    pjrt_dtype,
                    dev,
                )?;
                buffers.push(handle);
            }
            Ok(buffers)
        }
    }
}

/// Gather sharded buffers from all devices into a single host byte vector.
///
/// `shape` is the full (unsharded) tensor shape.
pub fn gather_buffer(
    mesh: &TpuMesh,
    buffers: &[PjrtBufferHandle],
    shape: &[i64],
    dtype: XlaDtype,
    spec: &ShardingSpec,
) -> Result<Vec<u8>> {
    let elem_size = dtype.size_bytes();
    let total_elems: usize = shape.iter().map(|&d| d as usize).product();
    let total_bytes = total_elems * elem_size;

    match spec {
        ShardingSpec::Replicated => {
            if buffers.is_empty() {
                return Err(LLMError::GpuError("no buffers to gather".into()));
            }
            // Just copy from device 0.
            let mut out = vec![0u8; total_bytes];
            mesh.client().buffer_to_host(&buffers[0], &mut out)?;
            Ok(out)
        }
        ShardingSpec::Sharded { axis, num_shards } => {
            let axis = *axis;
            let num_shards = *num_shards;

            if buffers.len() != num_shards {
                return Err(LLMError::GpuError(format!(
                    "gather: got {} buffers, expected {num_shards}",
                    buffers.len()
                )));
            }
            if axis >= shape.len() {
                return Err(LLMError::GpuError(format!(
                    "gather axis {axis} >= ndim {}",
                    shape.len()
                )));
            }

            let dim = shape[axis] as usize;
            let shard_dim = dim / num_shards;
            let outer_count: usize = shape[..axis].iter().map(|&d| d as usize).product::<usize>().max(1);
            let inner_bytes: usize = shape[axis + 1..].iter().map(|&d| d as usize).product::<usize>().max(1) * elem_size;
            let full_axis_stride = dim * inner_bytes;
            let shard_bytes_per_outer = shard_dim * inner_bytes;
            let shard_total_bytes = shard_bytes_per_outer * outer_count;

            let mut out = vec![0u8; total_bytes];
            let mut shard_buf = vec![0u8; shard_total_bytes];

            for (shard_idx, buf) in buffers.iter().enumerate() {
                mesh.client().buffer_to_host(buf, &mut shard_buf)?;

                let shard_offset = shard_idx * shard_bytes_per_outer;
                for outer in 0..outer_count {
                    let dst_start = outer * full_axis_stride + shard_offset;
                    let src_start = outer * shard_bytes_per_outer;
                    out[dst_start..dst_start + shard_bytes_per_outer]
                        .copy_from_slice(&shard_buf[src_start..src_start + shard_bytes_per_outer]);
                }
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharding_spec_basics() {
        let r = ShardingSpec::Replicated;
        let s = ShardingSpec::sharded(0, 4);
        assert_eq!(r, ShardingSpec::Replicated);
        assert_eq!(s, ShardingSpec::Sharded { axis: 0, num_shards: 4 });
    }
}
