//! Vendored cublasLt FFI re-exports with type fixes.
//!
//! cudarc 0.19's `cublaslt::sys` doesn't export `cublasOperation_t` (it lives
//! in `cublas::sys`), and `cublasLtMatmul` expects `cudaStream_t` (runtime API)
//! while cudarc uses `CUstream` (driver API). They're the same `*mut CUstream_st`
//! pointer but Rust sees them as different types because each sys module defines
//! its own opaque struct. This shim re-exports everything we need from one place.

// All cublasLt types and functions we use:
pub use cudarc::cublaslt::sys::{
    cublasComputeType_t,
    cublasLtHandle_t,
    cublasLtMatmulAlgo_t,
    cublasLtMatmulAlgoGetHeuristic,
    cublasLtMatmulDesc_t,
    cublasLtMatmulDescAttributes_t,
    cublasLtMatmulDescCreate,
    cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    cublasLtMatmulHeuristicResult_t,
    cublasLtMatmulPreferenceAttributes_t,
    cublasLtMatmulPreference_t,
    cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatrixLayoutCreate,
    cublasLtMatrixLayoutDestroy,
    cublasLtMatrixLayout_t,
    cublasLtEpilogue_t,
    cublasStatus_t,
    cudaDataType_t,
    cublasLtMatmul,
};

// Fix 1: cublasOperation_t lives in cublas::sys, not cublaslt::sys
pub use cudarc::cublas::sys::cublasOperation_t;

// Fix 2: cublasLtMatmul expects cudaStream_t (cudarc::cublaslt::sys::cudaStream_t)
// but CudaStream::cu_stream() returns CUstream (cudarc::driver::sys::CUstream).
// Both are *mut <opaque>, same at the ABI level, different in Rust's type system.
pub use cudarc::cublaslt::sys::cudaStream_t;

/// Convert a driver-API CUstream to the runtime-API cudaStream_t that
/// cublasLtMatmul expects. Both are `*mut CUstream_st` at the C level.
#[inline(always)]
pub fn cu_stream_to_cuda_stream(s: cudarc::driver::sys::CUstream) -> cudaStream_t {
    s as *mut _ as cudaStream_t
}
