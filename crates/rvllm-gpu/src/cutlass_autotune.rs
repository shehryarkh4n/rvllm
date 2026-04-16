//! CUTLASS GEMM autotuning: benchmark all tile/schedule variants for a given
//! (M,N,K) shape, pick the fastest, cache results to JSON on disk.
//!
//! The autotune pass runs each compiled CUTLASS variant, times them with CUDA
//! events, and stores the best variant per shape. When CUTLASS is loaded at
//! runtime, every shape MUST have a cache entry -- missing entries panic.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use crate::cutlass_ffi::{FP8_GEMM_VARIANTS, FP8_GEMM_RESIDUAL_VARIANTS, GATEUP_SILU_VARIANTS, HGEMM_VARIANTS, OPROJ_RESIDUAL_VARIANTS};

/// Persistent cache mapping (M,N,K) shapes to the best CUTLASS variant index.
/// Every shape used at runtime must have an entry -- missing entries panic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CutlassAutotuneCache {
    #[serde(default)]
    pub hgemm: HashMap<String, usize>,
    #[serde(default)]
    pub oproj_residual: HashMap<String, usize>,
    #[serde(default)]
    pub gateup_silu: HashMap<String, usize>,
    #[serde(default)]
    pub fp8_gemm: HashMap<String, usize>,
    #[serde(default)]
    pub fp8_gemm_residual: HashMap<String, usize>,
}

impl Default for CutlassAutotuneCache {
    fn default() -> Self {
        Self {
            hgemm: HashMap::new(),
            oproj_residual: HashMap::new(),
            gateup_silu: HashMap::new(),
            fp8_gemm: HashMap::new(),
            fp8_gemm_residual: HashMap::new(),
        }
    }
}

fn shape_key(m: usize, n: usize, k: usize) -> String {
    format!("{m}_{n}_{k}")
}

/// Find the best variant for a given (M, N, K) from the cache.
/// Exact M match first; if no exact match, uses the nearest M with the same N,K
/// (handles arbitrary prefill lengths not in the autotune set).
fn lookup_nearest(map: &HashMap<String, usize>, m: usize, n: usize, k: usize) -> Option<usize> {
    if let Some(&v) = map.get(&shape_key(m, n, k)) {
        return Some(v);
    }
    let suffix = format!("_{n}_{k}");
    let mut best: Option<(usize, usize)> = None;
    for (key, &variant) in map {
        if !key.ends_with(&suffix) { continue; }
        if let Ok(cached_m) = key[..key.len() - suffix.len()].parse::<usize>() {
            let dist = cached_m.abs_diff(m);
            match best {
                None => best = Some((dist, variant)),
                Some((bd, _)) if dist < bd => best = Some((dist, variant)),
                _ => {}
            }
        }
    }
    best.map(|(_, v)| v)
}

impl CutlassAutotuneCache {
    pub fn best_hgemm(&self, m: usize, n: usize, k: usize) -> Option<usize> {
        lookup_nearest(&self.hgemm, m, n, k)
    }

    pub fn best_oproj_residual(&self, m: usize, n: usize, k: usize) -> Option<usize> {
        lookup_nearest(&self.oproj_residual, m, n, k)
    }

    pub fn best_gateup_silu(&self, m: usize, n: usize, k: usize) -> Option<usize> {
        lookup_nearest(&self.gateup_silu, m, n, k)
    }

    pub fn insert_hgemm(&mut self, m: usize, n: usize, k: usize, variant: usize) {
        self.hgemm.insert(shape_key(m, n, k), variant);
    }

    pub fn insert_oproj_residual(&mut self, m: usize, n: usize, k: usize, variant: usize) {
        self.oproj_residual.insert(shape_key(m, n, k), variant);
    }

    pub fn insert_gateup_silu(&mut self, m: usize, n: usize, k: usize, variant: usize) {
        self.gateup_silu.insert(shape_key(m, n, k), variant);
    }

    pub fn best_fp8_gemm(&self, m: usize, n: usize, k: usize) -> Option<usize> {
        lookup_nearest(&self.fp8_gemm, m, n, k)
    }

    pub fn insert_fp8_gemm(&mut self, m: usize, n: usize, k: usize, variant: usize) {
        self.fp8_gemm.insert(shape_key(m, n, k), variant);
    }

    pub fn best_fp8_gemm_residual(&self, m: usize, n: usize, k: usize) -> Option<usize> {
        lookup_nearest(&self.fp8_gemm_residual, m, n, k)
    }

    pub fn insert_fp8_gemm_residual(&mut self, m: usize, n: usize, k: usize, variant: usize) {
        self.fp8_gemm_residual.insert(shape_key(m, n, k), variant);
    }

    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(data) => match serde_json::from_str::<CutlassAutotuneCache>(&data) {
                Ok(cache) => {
                    let total =
                        cache.hgemm.len() + cache.oproj_residual.len() + cache.gateup_silu.len() + cache.fp8_gemm.len() + cache.fp8_gemm_residual.len();
                    tracing::info!(
                        path = %path.display(),
                        entries = total,
                        "loaded CUTLASS autotune cache"
                    );
                    cache
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        %e,
                        "corrupt CUTLASS autotune cache, starting fresh"
                    );
                    Self::default()
                }
            },
            Err(_) => {
                tracing::info!(
                    path = %path.display(),
                    "no CUTLASS autotune cache found"
                );
                Self::default()
            }
        }
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn cache_path() -> PathBuf {
        if let Ok(p) = std::env::var("RVLLM_CUTLASS_AUTOTUNE_CACHE") {
            return PathBuf::from(p);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home).join(".cache/rvllm/cutlass_autotune.json")
    }

    pub fn is_empty(&self) -> bool {
        self.hgemm.is_empty() && self.oproj_residual.is_empty() && self.gateup_silu.is_empty() && self.fp8_gemm.is_empty() && self.fp8_gemm_residual.is_empty()
    }
}

// ---------------------------------------------------------------------------
// CUDA event-based benchmarking
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
use crate::cutlass_ffi::CutlassKernels;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

/// Time GPU work launched by `f` using CUDA events.
/// Runs `warmup` iterations first (untimed), then `iters` timed iterations.
/// Returns average microseconds per iteration.
#[cfg(feature = "cuda")]
unsafe fn time_gpu_us(
    stream: &Arc<CudaStream>,
    warmup: usize,
    iters: usize,
    mut f: impl FnMut(),
) -> Result<f32, String> {
    use cudarc::driver::result::event as cu_event;
    use cudarc::driver::sys::CUevent_flags;

    for _ in 0..warmup {
        f();
    }

    let ev_start = cu_event::create(CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(|e| format!("event create: {e}"))?;
    let ev_stop = cu_event::create(CUevent_flags::CU_EVENT_DEFAULT)
        .map_err(|e| format!("event create: {e}"))?;
    let raw_stream = stream.cu_stream();

    cu_event::record(ev_start, raw_stream).map_err(|e| format!("event record: {e}"))?;
    for _ in 0..iters {
        f();
    }
    cu_event::record(ev_stop, raw_stream).map_err(|e| format!("event record: {e}"))?;
    cu_event::synchronize(ev_stop).map_err(|e| format!("event sync: {e}"))?;

    let ms = cu_event::elapsed(ev_start, ev_stop).unwrap_or(f32::MAX);
    cu_event::destroy(ev_start).ok();
    cu_event::destroy(ev_stop).ok();

    Ok((ms * 1000.0) / iters as f32)
}

/// Benchmark all HGEMM variants for a given shape.
/// Returns (best_variant_index, best_time_us) or None if no variant was loaded.
#[cfg(feature = "cuda")]
pub fn autotune_hgemm_shape(
    cutlass: &CutlassKernels,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<(usize, f32)> {
    let a_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * k).ok()?;
    let b_buf: CudaSlice<half::f16> = stream.alloc_zeros(n * k).ok()?;
    let mut c_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;

    let mut max_ws: usize = 0;
    for v in 0..HGEMM_VARIANTS {
        if let Some(ws) = cutlass.hgemm_variant_workspace_size(v, m as i32, n as i32, k as i32) {
            max_ws = max_ws.max(ws);
        }
    }
    max_ws = max_ws.max(4 * 1024 * 1024);
    let ws_buf: CudaSlice<u8> = stream.alloc_zeros(max_ws).ok()?;

    let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, stream);
    let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, stream);
    let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, stream);
    let (ws_ptr, _wsg) = DevicePtr::device_ptr(&ws_buf, stream);

    let a_raw = a_ptr as u64;
    let b_raw = b_ptr as u64;
    let c_raw = c_ptr as u64;
    let ws_raw = ws_ptr as u64;
    let stream_raw = stream.cu_stream() as u64;
    let mi = m as i32;
    let ni = n as i32;
    let ki = k as i32;

    let mut best: Option<(usize, f32)> = None;

    for v in 0..HGEMM_VARIANTS {
        // Probe: skip variants that aren't loaded or fail on this shape
        if cutlass
            .hgemm_variant(v, c_raw, a_raw, b_raw, mi, ni, ki, ws_raw, max_ws, stream_raw)
            .is_err()
        {
            continue;
        }
        // Sync after probe to catch async errors
        if stream.synchronize().is_err() {
            continue;
        }

        let time = unsafe {
            time_gpu_us(stream, warmup_iters, bench_iters, || {
                let _ = cutlass.hgemm_variant(
                    v, c_raw, a_raw, b_raw, mi, ni, ki, ws_raw, max_ws, stream_raw,
                );
            })
        };

        if let Ok(t) = time {
            tracing::debug!(variant = v, time_us = t, m, n, k, "hgemm variant timed");
            match &best {
                None => best = Some((v, t)),
                Some((_, bt)) if t < *bt => best = Some((v, t)),
                _ => {}
            }
        }
    }

    best
}

/// Benchmark all oproj+residual variants for a given shape.
#[cfg(feature = "cuda")]
pub fn autotune_oproj_residual_shape(
    cutlass: &CutlassKernels,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<(usize, f32)> {
    let a_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * k).ok()?;
    let b_buf: CudaSlice<half::f16> = stream.alloc_zeros(n * k).ok()?;
    let res_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;
    let mut c_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;

    let mut max_ws: usize = 0;
    for v in 0..OPROJ_RESIDUAL_VARIANTS {
        if let Some(ws) =
            cutlass.oproj_residual_variant_workspace_size(v, m as i32, n as i32, k as i32)
        {
            max_ws = max_ws.max(ws);
        }
    }
    max_ws = max_ws.max(4 * 1024 * 1024);
    let ws_buf: CudaSlice<u8> = stream.alloc_zeros(max_ws).ok()?;

    let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, stream);
    let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, stream);
    let (r_ptr, _rg) = DevicePtr::device_ptr(&res_buf, stream);
    let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, stream);
    let (ws_ptr, _wsg) = DevicePtr::device_ptr(&ws_buf, stream);

    let a_raw = a_ptr as u64;
    let b_raw = b_ptr as u64;
    let r_raw = r_ptr as u64;
    let c_raw = c_ptr as u64;
    let ws_raw = ws_ptr as u64;
    let stream_raw = stream.cu_stream() as u64;
    let mi = m as i32;
    let ni = n as i32;
    let ki = k as i32;

    let mut best: Option<(usize, f32)> = None;

    for v in 0..OPROJ_RESIDUAL_VARIANTS {
        if cutlass
            .oproj_residual_variant(
                v, c_raw, a_raw, b_raw, r_raw, mi, ni, ki, ws_raw, max_ws, stream_raw,
            )
            .is_err()
        {
            continue;
        }
        if stream.synchronize().is_err() {
            continue;
        }

        let time = unsafe {
            time_gpu_us(stream, warmup_iters, bench_iters, || {
                let _ = cutlass.oproj_residual_variant(
                    v, c_raw, a_raw, b_raw, r_raw, mi, ni, ki, ws_raw, max_ws, stream_raw,
                );
            })
        };

        if let Ok(t) = time {
            tracing::debug!(
                variant = v,
                time_us = t,
                m,
                n,
                k,
                "oproj_residual variant timed"
            );
            match &best {
                None => best = Some((v, t)),
                Some((_, bt)) if t < *bt => best = Some((v, t)),
                _ => {}
            }
        }
    }

    best
}

/// Benchmark all gateup+silu variants for a given shape.
#[cfg(feature = "cuda")]
pub fn autotune_gateup_silu_shape(
    cutlass: &CutlassKernels,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<(usize, f32)> {
    let a_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * k).ok()?;
    let b_buf: CudaSlice<half::f16> = stream.alloc_zeros(n * k).ok()?;
    // Output is [M, N/2] after SiLU activation
    let out_elems = m * (n / 2);
    let mut c_buf: CudaSlice<half::f16> = stream.alloc_zeros(out_elems).ok()?;

    let mut max_ws: usize = 0;
    for v in 0..GATEUP_SILU_VARIANTS {
        if let Some(ws) =
            cutlass.gateup_silu_variant_workspace_size(v, m as i32, n as i32, k as i32)
        {
            max_ws = max_ws.max(ws);
        }
    }
    max_ws = max_ws.max(4 * 1024 * 1024);
    let ws_buf: CudaSlice<u8> = stream.alloc_zeros(max_ws).ok()?;

    let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, stream);
    let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, stream);
    let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, stream);
    let (ws_ptr, _wsg) = DevicePtr::device_ptr(&ws_buf, stream);

    let a_raw = a_ptr as u64;
    let b_raw = b_ptr as u64;
    let c_raw = c_ptr as u64;
    let ws_raw = ws_ptr as u64;
    let stream_raw = stream.cu_stream() as u64;
    let mi = m as i32;
    let ni = n as i32;
    let ki = k as i32;

    let mut best: Option<(usize, f32)> = None;

    for v in 0..GATEUP_SILU_VARIANTS {
        if cutlass
            .gateup_silu_variant(v, c_raw, a_raw, b_raw, mi, ni, ki, ws_raw, max_ws, stream_raw)
            .is_err()
        {
            continue;
        }
        if stream.synchronize().is_err() {
            continue;
        }

        let time = unsafe {
            time_gpu_us(stream, warmup_iters, bench_iters, || {
                let _ = cutlass.gateup_silu_variant(
                    v, c_raw, a_raw, b_raw, mi, ni, ki, ws_raw, max_ws, stream_raw,
                );
            })
        };

        if let Ok(t) = time {
            tracing::debug!(
                variant = v,
                time_us = t,
                m,
                n,
                k,
                "gateup_silu variant timed"
            );
            match &best {
                None => best = Some((v, t)),
                Some((_, bt)) if t < *bt => best = Some((v, t)),
                _ => {}
            }
        }
    }

    best
}

/// Benchmark all FP8 GEMM variants for a given shape.
/// Returns (best_variant_index, best_time_us) or None if no variant was loaded.
#[cfg(feature = "cuda")]
pub fn autotune_fp8_gemm_shape(
    cutlass: &CutlassKernels,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<(usize, f32)> {
    // A and B are FP8 (u8-sized elements)
    let a_buf: CudaSlice<u8> = stream.alloc_zeros(m * k).ok()?;
    let b_buf: CudaSlice<u8> = stream.alloc_zeros(n * k).ok()?;
    // Per-row A scales (f32)
    let a_scales: CudaSlice<f32> = stream.alloc_zeros(m).ok()?;
    // Per-tensor B scale (f32, 1 element)
    let b_scale: CudaSlice<f32> = stream.alloc_zeros(1).ok()?;
    // Output is [M, N] f16
    let mut c_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;

    let mut max_ws: usize = 0;
    for v in 0..FP8_GEMM_VARIANTS {
        if let Some(ws) = cutlass.fp8_gemm_variant_workspace_size(v, m as i32, n as i32, k as i32) {
            max_ws = max_ws.max(ws);
        }
    }
    max_ws = max_ws.max(4 * 1024 * 1024);
    let ws_buf: CudaSlice<u8> = stream.alloc_zeros(max_ws).ok()?;

    let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, stream);
    let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, stream);
    let (as_ptr, _asg) = DevicePtr::device_ptr(&a_scales, stream);
    let (bs_ptr, _bsg) = DevicePtr::device_ptr(&b_scale, stream);
    let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, stream);
    let (ws_ptr, _wsg) = DevicePtr::device_ptr(&ws_buf, stream);

    let a_raw = a_ptr as u64;
    let b_raw = b_ptr as u64;
    let as_raw = as_ptr as u64;
    let bs_raw = bs_ptr as u64;
    let c_raw = c_ptr as u64;
    let ws_raw = ws_ptr as u64;
    let stream_raw = stream.cu_stream() as u64;
    let mi = m as i32;
    let ni = n as i32;
    let ki = k as i32;

    let mut best: Option<(usize, f32)> = None;

    for v in 0..FP8_GEMM_VARIANTS {
        // Probe: skip variants that aren't loaded or fail on this shape
        if cutlass
            .fp8_gemm_variant(v, c_raw, a_raw, b_raw, as_raw, bs_raw, mi, ni, ki, ws_raw, max_ws, stream_raw)
            .is_err()
        {
            continue;
        }
        // Sync after probe to catch async errors
        if stream.synchronize().is_err() {
            continue;
        }

        let time = unsafe {
            time_gpu_us(stream, warmup_iters, bench_iters, || {
                let _ = cutlass.fp8_gemm_variant(
                    v, c_raw, a_raw, b_raw, as_raw, bs_raw, mi, ni, ki, ws_raw, max_ws, stream_raw,
                );
            })
        };

        if let Ok(t) = time {
            tracing::debug!(variant = v, time_us = t, m, n, k, "fp8_gemm variant timed");
            match &best {
                None => best = Some((v, t)),
                Some((_, bt)) if t < *bt => best = Some((v, t)),
                _ => {}
            }
        }
    }

    best
}

/// Benchmark all FP8 GEMM + residual variants for a given shape.
/// Returns (best_variant_index, best_time_us) or None if no variant was loaded.
#[cfg(feature = "cuda")]
pub fn autotune_fp8_gemm_residual_shape(
    cutlass: &CutlassKernels,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<(usize, f32)> {
    let a_buf: CudaSlice<u8> = stream.alloc_zeros(m * k).ok()?;
    let b_buf: CudaSlice<u8> = stream.alloc_zeros(n * k).ok()?;
    let a_scales: CudaSlice<f32> = stream.alloc_zeros(m).ok()?;
    let b_scale: CudaSlice<f32> = stream.alloc_zeros(1).ok()?;
    let res_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;
    let mut c_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;

    let mut max_ws: usize = 0;
    for v in 0..FP8_GEMM_RESIDUAL_VARIANTS {
        if let Some(ws) = cutlass.fp8_gemm_residual_workspace_size(v, m as i32, n as i32, k as i32) {
            max_ws = max_ws.max(ws);
        }
    }
    max_ws = max_ws.max(4 * 1024 * 1024);
    let ws_buf: CudaSlice<u8> = stream.alloc_zeros(max_ws).ok()?;

    let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, stream);
    let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, stream);
    let (as_ptr, _asg) = DevicePtr::device_ptr(&a_scales, stream);
    let (bs_ptr, _bsg) = DevicePtr::device_ptr(&b_scale, stream);
    let (r_ptr, _rg) = DevicePtr::device_ptr(&res_buf, stream);
    let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, stream);
    let (ws_ptr, _wsg) = DevicePtr::device_ptr(&ws_buf, stream);

    let a_raw = a_ptr as u64;
    let b_raw = b_ptr as u64;
    let as_raw = as_ptr as u64;
    let bs_raw = bs_ptr as u64;
    let r_raw = r_ptr as u64;
    let c_raw = c_ptr as u64;
    let ws_raw = ws_ptr as u64;
    let stream_raw = stream.cu_stream() as u64;
    let mi = m as i32;
    let ni = n as i32;
    let ki = k as i32;

    let mut best: Option<(usize, f32)> = None;

    for v in 0..FP8_GEMM_RESIDUAL_VARIANTS {
        if cutlass
            .fp8_gemm_residual(v, c_raw, a_raw, b_raw, as_raw, bs_raw, r_raw, mi, ni, ki, ws_raw, max_ws, stream_raw)
            .is_err()
        {
            continue;
        }
        if stream.synchronize().is_err() {
            continue;
        }

        let time = unsafe {
            time_gpu_us(stream, warmup_iters, bench_iters, || {
                let _ = cutlass.fp8_gemm_residual(
                    v, c_raw, a_raw, b_raw, as_raw, bs_raw, r_raw, mi, ni, ki, ws_raw, max_ws, stream_raw,
                );
            })
        };

        if let Ok(t) = time {
            tracing::debug!(variant = v, time_us = t, m, n, k, "fp8_gemm_residual variant timed");
            match &best {
                None => best = Some((v, t)),
                Some((_, bt)) if t < *bt => best = Some((v, t)),
                _ => {}
            }
        }
    }

    best
}

/// Benchmark cuBLAS HGEMM as a baseline for comparison.
/// Returns time in microseconds per iteration.
#[cfg(feature = "cuda")]
pub fn bench_cublas_hgemm(
    cublas: &crate::cublas::CublasHandle,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<f32> {
    let a_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * k).ok()?;
    let b_buf: CudaSlice<half::f16> = stream.alloc_zeros(n * k).ok()?;
    let mut c_buf: CudaSlice<half::f16> = stream.alloc_zeros(m * n).ok()?;

    let alpha = half::f16::from_f32(1.0);
    let beta = half::f16::from_f32(0.0);

    let time = unsafe {
        time_gpu_us(stream, warmup_iters, bench_iters, || {
            let _ = cublas.hgemm(m, n, k, alpha, &a_buf, &b_buf, beta, &mut c_buf);
        })
    };

    time.ok()
}

/// Autotune HGEMM: benchmark all CUTLASS SM90 WGMMA variants vs cuBLAS.
/// Returns the best CUTLASS variant only if it beats cuBLAS.
/// Small-M bandwidth-bound GEMMs (N=3584) are genuinely faster on cuBLAS
/// SM80 tiles due to less overhead and zero M-padding waste.
#[cfg(feature = "cuda")]
pub fn autotune_hgemm_vs_cublas(
    cutlass: &CutlassKernels,
    cublas: &crate::cublas::CublasHandle,
    stream: &Arc<CudaStream>,
    m: usize,
    n: usize,
    k: usize,
    warmup_iters: usize,
    bench_iters: usize,
) -> Option<(usize, f32)> {
    let cublas_time = bench_cublas_hgemm(cublas, stream, m, n, k, warmup_iters, bench_iters)?;
    let cutlass_best =
        autotune_hgemm_shape(cutlass, stream, m, n, k, warmup_iters, bench_iters)?;

    if cutlass_best.1 < cublas_time {
        let speedup_pct = (1.0 - cutlass_best.1 / cublas_time) * 100.0;
        tracing::info!(
            m, n, k,
            variant = cutlass_best.0,
            cutlass_us = format!("{:.1}", cutlass_best.1),
            cublas_us = format!("{:.1}", cublas_time),
            speedup_pct = format!("{:.1}", speedup_pct),
            "CUTLASS hgemm v{} wins", cutlass_best.0,
        );
        Some(cutlass_best)
    } else {
        tracing::info!(
            m, n, k,
            cublas_us = format!("{:.1}", cublas_time),
            best_cutlass_us = format!("{:.1}", cutlass_best.1),
            "cuBLAS wins for hgemm M={m} N={n} K={k}",
        );
        None
    }
}

/// Run full autotune pass for all kernel types across a set of shapes.
/// Populates and returns a CutlassAutotuneCache.
#[cfg(feature = "cuda")]
pub fn run_full_autotune(
    cutlass: &CutlassKernels,
    cublas: &crate::cublas::CublasHandle,
    stream: &Arc<CudaStream>,
    hgemm_shapes: &[(usize, usize, usize)],
    oproj_shapes: &[(usize, usize, usize)],
    gateup_shapes: &[(usize, usize, usize)],
    fp8_gemm_shapes: &[(usize, usize, usize)],
    fp8_gemm_residual_shapes: &[(usize, usize, usize)],
    warmup_iters: usize,
    bench_iters: usize,
) -> CutlassAutotuneCache {
    let mut cache = CutlassAutotuneCache::default();

    tracing::info!(
        hgemm = hgemm_shapes.len(),
        oproj = oproj_shapes.len(),
        gateup = gateup_shapes.len(),
        fp8_gemm = fp8_gemm_shapes.len(),
        fp8_gemm_residual = fp8_gemm_residual_shapes.len(),
        "starting CUTLASS autotune pass"
    );

    for &(m, n, k) in hgemm_shapes {
        if let Some((variant, _)) =
            autotune_hgemm_vs_cublas(cutlass, cublas, stream, m, n, k, warmup_iters, bench_iters)
        {
            cache.hgemm.insert(shape_key(m, n, k), variant);
        }
    }

    for &(m, n, k) in oproj_shapes {
        if let Some((variant, time_us)) =
            autotune_oproj_residual_shape(cutlass, stream, m, n, k, warmup_iters, bench_iters)
        {
            tracing::info!(m, n, k, variant, time_us = format!("{:.1}", time_us), "oproj_residual best");
            cache.oproj_residual.insert(shape_key(m, n, k), variant);
        }
    }

    for &(m, n, k) in gateup_shapes {
        if let Some((variant, time_us)) =
            autotune_gateup_silu_shape(cutlass, stream, m, n, k, warmup_iters, bench_iters)
        {
            tracing::info!(m, n, k, variant, time_us = format!("{:.1}", time_us), "gateup_silu best");
            cache.gateup_silu.insert(shape_key(m, n, k), variant);
        }
    }

    for &(m, n, k) in fp8_gemm_shapes {
        if let Some((variant, time_us)) =
            autotune_fp8_gemm_shape(cutlass, stream, m, n, k, warmup_iters, bench_iters)
        {
            tracing::info!(m, n, k, variant, time_us = format!("{:.1}", time_us), "fp8_gemm best");
            cache.fp8_gemm.insert(shape_key(m, n, k), variant);
        }
    }

    for &(m, n, k) in fp8_gemm_residual_shapes {
        if let Some((variant, time_us)) =
            autotune_fp8_gemm_residual_shape(cutlass, stream, m, n, k, warmup_iters, bench_iters)
        {
            tracing::info!(m, n, k, variant, time_us = format!("{:.1}", time_us), "fp8_gemm_residual best");
            cache.fp8_gemm_residual.insert(shape_key(m, n, k), variant);
        }
    }

    tracing::info!(
        hgemm = cache.hgemm.len(),
        oproj = cache.oproj_residual.len(),
        gateup = cache.gateup_silu.len(),
        fp8_gemm = cache.fp8_gemm.len(),
        fp8_gemm_residual = cache.fp8_gemm_residual.len(),
        "autotune pass complete"
    );

    cache
}

/// Query the maximum workspace size across all loaded variants for all given shapes.
/// Use this at init to pre-allocate a single workspace buffer large enough for any call.
#[cfg(feature = "cuda")]
pub fn max_workspace_size(
    cutlass: &CutlassKernels,
    hgemm_shapes: &[(usize, usize, usize)],
    oproj_shapes: &[(usize, usize, usize)],
    gateup_shapes: &[(usize, usize, usize)],
    fp8_gemm_shapes: &[(usize, usize, usize)],
    fp8_gemm_residual_shapes: &[(usize, usize, usize)],
) -> usize {
    let mut max_ws: usize = 0;

    for &(m, n, k) in hgemm_shapes {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        // Default (non-variant) hgemm fallback
        max_ws = max_ws.max(cutlass.hgemm_workspace_size(mi, ni, ki));
        for v in 0..HGEMM_VARIANTS {
            if let Some(ws) = cutlass.hgemm_variant_workspace_size(v, mi, ni, ki) {
                max_ws = max_ws.max(ws);
            }
        }
    }

    for &(m, n, k) in oproj_shapes {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        for v in 0..OPROJ_RESIDUAL_VARIANTS {
            if let Some(ws) = cutlass.oproj_residual_variant_workspace_size(v, mi, ni, ki) {
                max_ws = max_ws.max(ws);
            }
        }
    }

    for &(m, n, k) in gateup_shapes {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        for v in 0..GATEUP_SILU_VARIANTS {
            if let Some(ws) = cutlass.gateup_silu_variant_workspace_size(v, mi, ni, ki) {
                max_ws = max_ws.max(ws);
            }
        }
    }

    for &(m, n, k) in fp8_gemm_shapes {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        // Default (non-variant) fp8_gemm and small-tile fallbacks
        max_ws = max_ws.max(cutlass.fp8_gemm_workspace_size(mi, ni, ki));
        if let Some(ws) = cutlass.fp8_gemm_small_workspace_size(mi, ni, ki) {
            max_ws = max_ws.max(ws);
        }
        for v in 0..FP8_GEMM_VARIANTS {
            if let Some(ws) = cutlass.fp8_gemm_variant_workspace_size(v, mi, ni, ki) {
                max_ws = max_ws.max(ws);
            }
        }
    }

    for &(m, n, k) in fp8_gemm_residual_shapes {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        for v in 0..FP8_GEMM_RESIDUAL_VARIANTS {
            if let Some(ws) = cutlass.fp8_gemm_residual_workspace_size(v, mi, ni, ki) {
                max_ws = max_ws.max(ws);
            }
        }
    }

    // Minimum 4MB
    max_ws.max(4 * 1024 * 1024)
}
