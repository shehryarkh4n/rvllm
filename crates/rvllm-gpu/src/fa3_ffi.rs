//! FFI bindings to FlashAttention-3 SM90 shared library (libfa3_kernels.so).
//!
//! Provides WGMMA/TMA-accelerated paged-KV decode attention for SM90 (H100).
//! The .so is built from Dao-AILab/flash-attention hopper kernels with a thin
//! extern "C" wrapper (fa3_sm90_wrapper.cu).

use std::ffi::c_void;
use std::path::Path;

/// Function pointer types matching extern "C" signatures in fa3_sm90_wrapper.cu.
type WorkspaceSizeFn = unsafe extern "C" fn(
    batch_size: i32,
    num_heads: i32,
    max_num_splits: i32,
) -> i32;

type PagedDecodeFn = unsafe extern "C" fn(
    q_ptr: *mut c_void,
    k_cache_ptr: *mut c_void,
    v_cache_ptr: *mut c_void,
    o_ptr: *mut c_void,
    block_tables_ptr: *mut c_void,
    context_lens_ptr: *mut c_void,
    workspace_ptr: *mut c_void,
    scale: f32,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    max_blocks_per_seq: i32,
    num_blocks_total: i32,
    stream: *mut c_void,
) -> i32;

/// Handle to the loaded FA3 shared library.
pub struct Fa3Kernels {
    _lib: libloading::Library,
    fn_workspace_size: WorkspaceSizeFn,
    fn_paged_decode: PagedDecodeFn,
}

unsafe impl Send for Fa3Kernels {}
unsafe impl Sync for Fa3Kernels {}

impl Fa3Kernels {
    /// Load the FA3 shared library and resolve function pointers.
    pub fn load(lib_path: &Path) -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new(lib_path) }
            .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?;

        unsafe {
            let fn_workspace_size: WorkspaceSizeFn = *lib
                .get(b"fa3_sm90_workspace_size\0")
                .map_err(|e| format!("fa3_sm90_workspace_size: {e}"))?;
            let fn_paged_decode: PagedDecodeFn = *lib
                .get(b"fa3_sm90_paged_decode\0")
                .map_err(|e| format!("fa3_sm90_paged_decode: {e}"))?;

            Ok(Self {
                _lib: lib,
                fn_workspace_size,
                fn_paged_decode,
            })
        }
    }

    /// Returns minimum workspace size in bytes.
    pub fn workspace_size(&self, batch_size: i32, num_heads: i32) -> usize {
        let sz = unsafe { (self.fn_workspace_size)(batch_size, num_heads, 128) };
        sz as usize
    }

    /// Run paged-KV decode attention.
    /// All pointer args are raw device pointers (u64 from cudarc DevicePtr).
    pub fn paged_decode(
        &self,
        q: u64,
        k_cache: u64,
        v_cache: u64,
        output: u64,
        block_tables: u64,
        context_lens: u64,
        workspace: u64,
        scale: f32,
        batch_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks_per_seq: i32,
        num_blocks_total: i32,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_paged_decode)(
                q as *mut c_void,
                k_cache as *mut c_void,
                v_cache as *mut c_void,
                output as *mut c_void,
                block_tables as *mut c_void,
                context_lens as *mut c_void,
                workspace as *mut c_void,
                scale,
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_blocks_per_seq,
                num_blocks_total,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("fa3_sm90_paged_decode failed: {status}"));
        }
        Ok(())
    }
}
