//! FFI bindings to CUTLASS shared library (libcutlass_kernels.so).
//!
//! CUTLASS kernels are self-launching -- they compute their own grid dimensions
//! internally via the CUTLASS device adapter. You cannot launch them as raw PTX
//! via cuLaunchKernel. Instead, compile to .so and call the C wrapper functions
//! from host code (same pattern as vLLM).

use std::ffi::c_void;
use std::path::Path;

pub const HGEMM_VARIANTS: usize = 67;
pub const OPROJ_RESIDUAL_VARIANTS: usize = 31;
pub const GATEUP_SILU_VARIANTS: usize = 32;
pub const FP8_GEMM_VARIANTS: usize = 40;
pub const FP8_GEMM_RESIDUAL_VARIANTS: usize = 10;

/// Handle to the loaded CUTLASS shared library.
/// Holds function pointers resolved at load time for zero-cost dispatch.
pub struct CutlassKernels {
    _lib: libloading::Library,
    // Resolved function pointers (cached at load time, not per-call dlsym)
    fn_qkv_bias: QkvBiasFn,
    fn_qkv_bias_ws: WorkspaceSizeFn,
    fn_oproj_residual: OprojResidualFn,
    fn_oproj_residual_ws: WorkspaceSizeFn,
    fn_gateup_silu: GateUpSiluFn,
    fn_gateup_silu_ws: WorkspaceSizeFn,
    fn_gate_silu_mul: GateSiluMulFn,
    fn_gate_silu_mul_ws: WorkspaceSizeFn,
    fn_hgemm: HgemmFn,
    fn_hgemm_ws: WorkspaceSizeFn,
    fn_fp8_gemm: Fp8GemmFn,
    fn_fp8_gemm_ws: WorkspaceSizeFn,
    fn_fp8_gemm_small: Option<Fp8GemmFn>,
    fn_fp8_gemm_small_wksz: Option<WorkspaceSizeFn>,
    // Autotuned HGEMM variants
    fn_hgemm_variants: [Option<HgemmFn>; HGEMM_VARIANTS],
    fn_hgemm_ws_variants: [Option<WorkspaceSizeFn>; HGEMM_VARIANTS],
    // Autotuned fused oproj+residual variants
    fn_oproj_res_variants: [Option<OprojResidualFn>; OPROJ_RESIDUAL_VARIANTS],
    fn_oproj_res_ws_variants: [Option<WorkspaceSizeFn>; OPROJ_RESIDUAL_VARIANTS],
    // Autotuned fused gateup+silu variants
    fn_gateup_silu_variants: [Option<GateUpSiluFn>; GATEUP_SILU_VARIANTS],
    fn_gateup_silu_ws_variants: [Option<WorkspaceSizeFn>; GATEUP_SILU_VARIANTS],
    // Autotuned FP8 GEMM variants
    fn_fp8_gemm_variants: [Option<Fp8GemmFn>; FP8_GEMM_VARIANTS],
    fn_fp8_gemm_ws_variants: [Option<WorkspaceSizeFn>; FP8_GEMM_VARIANTS],
    // FP8 GEMM + residual add variants (fused epilogue)
    fn_fp8_gemm_res_variants: [Option<Fp8GemmResidualFn>; FP8_GEMM_RESIDUAL_VARIANTS],
    fn_fp8_gemm_res_ws_variants: [Option<WorkspaceSizeFn>; FP8_GEMM_RESIDUAL_VARIANTS],
}

// Function pointer types matching the extern "C" signatures in the .cu files.
type QkvBiasFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    bias: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void, // cudaStream_t
) -> i32;

type OprojResidualFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    residual: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type GateUpSiluFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type GateSiluMulFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    gate_weight: *const c_void,
    aux_up: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type HgemmFn = unsafe extern "C" fn(
    output: *mut c_void,
    input: *const c_void,
    weight: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type Fp8GemmFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scales: *const c_void,
    b_scale: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type Fp8GemmResidualFn = unsafe extern "C" fn(
    output: *mut c_void,
    a: *const c_void,
    b: *const c_void,
    a_scales: *const c_void,
    b_scale: *const c_void,
    residual: *const c_void,
    m: i32,
    n: i32,
    k: i32,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: *mut c_void,
) -> i32;

type WorkspaceSizeFn = unsafe extern "C" fn(m: i32, n: i32, k: i32) -> usize;

unsafe impl Send for CutlassKernels {}
unsafe impl Sync for CutlassKernels {}

impl CutlassKernels {
    /// Load the CUTLASS shared library and resolve all function pointers.
    pub fn load(lib_path: &Path) -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new(lib_path) }
            .map_err(|e| format!("dlopen {}: {e}", lib_path.display()))?;

        unsafe {
            let fn_qkv_bias: QkvBiasFn = *lib
                .get(b"cutlass_qkv_bias_gemm\0")
                .map_err(|e| format!("cutlass_qkv_bias_gemm: {e}"))?;
            let fn_qkv_bias_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_qkv_bias_workspace_size\0")
                .map_err(|e| format!("cutlass_qkv_bias_workspace_size: {e}"))?;
            let fn_oproj_residual: OprojResidualFn = *lib
                .get(b"cutlass_oproj_residual_gemm\0")
                .map_err(|e| format!("cutlass_oproj_residual_gemm: {e}"))?;
            let fn_oproj_residual_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_oproj_residual_workspace_size\0")
                .map_err(|e| format!("cutlass_oproj_residual_workspace_size: {e}"))?;
            let fn_gateup_silu: GateUpSiluFn = *lib
                .get(b"cutlass_gateup_silu\0")
                .map_err(|e| format!("cutlass_gateup_silu: {e}"))?;
            let fn_gateup_silu_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_gateup_silu_workspace_size\0")
                .map_err(|e| format!("cutlass_gateup_silu_workspace_size: {e}"))?;
            let fn_gate_silu_mul: GateSiluMulFn = *lib
                .get(b"cutlass_gate_silu_mul\0")
                .map_err(|e| format!("cutlass_gate_silu_mul: {e}"))?;
            let fn_gate_silu_mul_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_gate_silu_mul_workspace_size\0")
                .map_err(|e| format!("cutlass_gate_silu_mul_workspace_size: {e}"))?;
            let fn_hgemm: HgemmFn = *lib
                .get(b"cutlass_hgemm\0")
                .map_err(|e| format!("cutlass_hgemm: {e}"))?;
            let fn_hgemm_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_hgemm_workspace_size\0")
                .map_err(|e| format!("cutlass_hgemm_workspace_size: {e}"))?;
            let fn_fp8_gemm: Fp8GemmFn = *lib
                .get(b"cutlass_fp8_gemm\0")
                .map_err(|e| format!("cutlass_fp8_gemm: {e}"))?;
            let fn_fp8_gemm_ws: WorkspaceSizeFn = *lib
                .get(b"cutlass_fp8_gemm_workspace_size\0")
                .map_err(|e| format!("cutlass_fp8_gemm_workspace_size: {e}"))?;

            // Small-tile FP8 GEMM for decode (M<=64, tile 64x128x256)
            let fn_fp8_gemm_small: Option<Fp8GemmFn> =
                lib.get(b"cutlass_fp8_gemm_small\0").ok().map(|f| *f);
            let fn_fp8_gemm_small_wksz: Option<WorkspaceSizeFn> =
                lib.get(b"cutlass_fp8_gemm_small_workspace_size\0").ok().map(|f| *f);

            // Load autotuned HGEMM variants (optional -- don't fail if missing)
            let mut fn_hgemm_variants: [Option<HgemmFn>; HGEMM_VARIANTS] = [None; HGEMM_VARIANTS];
            let mut fn_hgemm_ws_variants: [Option<WorkspaceSizeFn>; HGEMM_VARIANTS] = [None; HGEMM_VARIANTS];
            for i in 0..HGEMM_VARIANTS {
                let name = format!("cutlass_hgemm_v{i}\0");
                let ws_name = format!("cutlass_hgemm_v{i}_workspace_size\0");
                if let Ok(f) = lib.get::<HgemmFn>(name.as_bytes()) {
                    fn_hgemm_variants[i] = Some(*f);
                }
                if let Ok(f) = lib.get::<WorkspaceSizeFn>(ws_name.as_bytes()) {
                    fn_hgemm_ws_variants[i] = Some(*f);
                }
            }

            // Load autotuned oproj+residual variants (optional)
            let mut fn_oproj_res_variants: [Option<OprojResidualFn>; OPROJ_RESIDUAL_VARIANTS] = [None; OPROJ_RESIDUAL_VARIANTS];
            let mut fn_oproj_res_ws_variants: [Option<WorkspaceSizeFn>; OPROJ_RESIDUAL_VARIANTS] = [None; OPROJ_RESIDUAL_VARIANTS];
            for i in 0..OPROJ_RESIDUAL_VARIANTS {
                let name = format!("cutlass_oproj_residual_v{i}\0");
                let ws_name = format!("cutlass_oproj_residual_v{i}_workspace_size\0");
                if let Ok(f) = lib.get::<OprojResidualFn>(name.as_bytes()) {
                    fn_oproj_res_variants[i] = Some(*f);
                }
                if let Ok(f) = lib.get::<WorkspaceSizeFn>(ws_name.as_bytes()) {
                    fn_oproj_res_ws_variants[i] = Some(*f);
                }
            }

            // Load autotuned gateup+silu variants (optional)
            let mut fn_gateup_silu_variants: [Option<GateUpSiluFn>; GATEUP_SILU_VARIANTS] = [None; GATEUP_SILU_VARIANTS];
            let mut fn_gateup_silu_ws_variants: [Option<WorkspaceSizeFn>; GATEUP_SILU_VARIANTS] = [None; GATEUP_SILU_VARIANTS];
            for i in 0..GATEUP_SILU_VARIANTS {
                let name = format!("cutlass_gateup_silu_v{i}\0");
                let ws_name = format!("cutlass_gateup_silu_v{i}_workspace_size\0");
                if let Ok(f) = lib.get::<GateUpSiluFn>(name.as_bytes()) {
                    fn_gateup_silu_variants[i] = Some(*f);
                }
                if let Ok(f) = lib.get::<WorkspaceSizeFn>(ws_name.as_bytes()) {
                    fn_gateup_silu_ws_variants[i] = Some(*f);
                }
            }

            // Load autotuned FP8 GEMM variants (optional)
            let mut fn_fp8_gemm_variants: [Option<Fp8GemmFn>; FP8_GEMM_VARIANTS] = [None; FP8_GEMM_VARIANTS];
            let mut fn_fp8_gemm_ws_variants: [Option<WorkspaceSizeFn>; FP8_GEMM_VARIANTS] = [None; FP8_GEMM_VARIANTS];
            for i in 0..FP8_GEMM_VARIANTS {
                let name = format!("cutlass_fp8_gemm_v{i}\0");
                let ws_name = format!("cutlass_fp8_gemm_v{i}_workspace_size\0");
                if let Ok(f) = lib.get::<Fp8GemmFn>(name.as_bytes()) {
                    fn_fp8_gemm_variants[i] = Some(*f);
                }
                if let Ok(f) = lib.get::<WorkspaceSizeFn>(ws_name.as_bytes()) {
                    fn_fp8_gemm_ws_variants[i] = Some(*f);
                }
            }

            // Load FP8 GEMM + residual variants (optional)
            let mut fn_fp8_gemm_res_variants: [Option<Fp8GemmResidualFn>; FP8_GEMM_RESIDUAL_VARIANTS] = [None; FP8_GEMM_RESIDUAL_VARIANTS];
            let mut fn_fp8_gemm_res_ws_variants: [Option<WorkspaceSizeFn>; FP8_GEMM_RESIDUAL_VARIANTS] = [None; FP8_GEMM_RESIDUAL_VARIANTS];
            for i in 0..FP8_GEMM_RESIDUAL_VARIANTS {
                let name = format!("cutlass_fp8_gemm_residual_v{i}\0");
                let ws_name = format!("cutlass_fp8_gemm_residual_v{i}_workspace_size\0");
                if let Ok(f) = lib.get::<Fp8GemmResidualFn>(name.as_bytes()) {
                    fn_fp8_gemm_res_variants[i] = Some(*f);
                }
                if let Ok(f) = lib.get::<WorkspaceSizeFn>(ws_name.as_bytes()) {
                    fn_fp8_gemm_res_ws_variants[i] = Some(*f);
                }
            }

            Ok(Self {
                _lib: lib,
                fn_qkv_bias,
                fn_qkv_bias_ws,
                fn_oproj_residual,
                fn_oproj_residual_ws,
                fn_gateup_silu,
                fn_gateup_silu_ws,
                fn_gate_silu_mul,
                fn_gate_silu_mul_ws,
                fn_hgemm,
                fn_hgemm_ws,
                fn_fp8_gemm,
                fn_fp8_gemm_ws,
                fn_fp8_gemm_small,
                fn_fp8_gemm_small_wksz,
                fn_hgemm_variants,
                fn_hgemm_ws_variants,
                fn_oproj_res_variants,
                fn_oproj_res_ws_variants,
                fn_gateup_silu_variants,
                fn_gateup_silu_ws_variants,
                fn_fp8_gemm_variants,
                fn_fp8_gemm_ws_variants,
                fn_fp8_gemm_res_variants,
                fn_fp8_gemm_res_ws_variants,
            })
        }
    }

    /// QKV projection with fused bias add.
    /// All pointers are raw device pointers (u64 from cudarc DevicePtr).
    pub fn qkv_bias_gemm(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        bias: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_qkv_bias)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                bias as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_qkv_bias_gemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for QKV+bias GEMM.
    pub fn qkv_bias_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_qkv_bias_ws)(m, n, k) }
    }

    /// O-projection GEMM with fused residual add.
    pub fn oproj_residual_gemm(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        residual: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_oproj_residual)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                residual as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_oproj_residual_gemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for O-proj+residual GEMM.
    pub fn oproj_residual_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_oproj_residual_ws)(m, n, k) }
    }

    /// Gate+Up projection GEMM with fused SiLU*Mul activation.
    /// N is the full gate+up width (2 * intermediate_size).
    /// Output is [M, N/2] after SiLU activation.
    pub fn gateup_silu(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_gateup_silu)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_gateup_silu failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for GateUp+SiLU.
    pub fn gateup_silu_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_gateup_silu_ws)(m, n, k) }
    }

    /// Gate-only GEMM with Sm90 epilogue: SiLU(gate) * aux_up.
    pub fn gate_silu_mul(
        &self,
        output: u64,
        input: u64,
        gate_weight: u64,
        aux_up: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_gate_silu_mul)(
                output as *mut c_void,
                input as *const c_void,
                gate_weight as *const c_void,
                aux_up as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_gate_silu_mul failed: {status}"));
        }
        Ok(())
    }

    pub fn gate_silu_mul_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_gate_silu_mul_ws)(m, n, k) }
    }

    /// Plain half-precision GEMM (no epilogue fusion).
    pub fn hgemm(
        &self,
        output: u64,
        input: u64,
        weight: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_hgemm)(
                output as *mut c_void,
                input as *const c_void,
                weight as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_hgemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for plain HGEMM.
    pub fn hgemm_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_hgemm_ws)(m, n, k) }
    }

    /// FP8 GEMM with per-row A scaling and per-tensor B scaling.
    /// D[m,n] = cast_to_f16(a_scales[m] * b_scale[0] * sum_k(A_fp8[m,k] * B_fp8[k,n]))
    pub fn fp8_gemm(
        &self,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let status = unsafe {
            (self.fn_fp8_gemm)(
                output as *mut c_void,
                a as *const c_void,
                b as *const c_void,
                a_scales as *const c_void,
                b_scale as *const c_void,
                m,
                n,
                k,
                workspace as *mut c_void,
                workspace_size,
                stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_fp8_gemm failed: {status}"));
        }
        Ok(())
    }

    /// Query workspace size for FP8 GEMM.
    pub fn fp8_gemm_workspace_size(&self, m: i32, n: i32, k: i32) -> usize {
        unsafe { (self.fn_fp8_gemm_ws)(m, n, k) }
    }

    /// FP8 GEMM with small tile (64x128x256) for decode (M <= 64).
    pub fn fp8_gemm_small(
        &self,
        output: u64, a: u64, b: u64,
        a_scales: u64, b_scale: u64,
        m: i32, n: i32, k: i32,
        workspace: u64, workspace_size: usize, stream: u64,
    ) -> Result<(), String> {
        let f = self.fn_fp8_gemm_small
            .ok_or("cutlass_fp8_gemm_small not loaded")?;
        let status = unsafe {
            f(
                output as *mut c_void, a as *const c_void, b as *const c_void,
                a_scales as *const c_void, b_scale as *const c_void,
                m, n, k,
                workspace as *mut c_void, workspace_size, stream as *mut c_void,
            )
        };
        if status != 0 {
            return Err(format!("cutlass_fp8_gemm_small failed: {status}"));
        }
        Ok(())
    }

    pub fn has_fp8_gemm_small(&self) -> bool {
        self.fn_fp8_gemm_small.is_some()
    }

    pub fn fp8_gemm_small_workspace_size(&self, m: i32, n: i32, k: i32) -> Option<usize> {
        self.fn_fp8_gemm_small_wksz.map(|f| unsafe { f(m, n, k) })
    }

    // -- Autotuned variant dispatch --

    /// Run HGEMM variant by index. Returns Err if variant not loaded.
    pub fn hgemm_variant(
        &self,
        variant: usize,
        output: u64,
        input: u64,
        weight: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let f = self.fn_hgemm_variants.get(variant)
            .and_then(|o| *o)
            .ok_or_else(|| format!("HGEMM variant {variant} not loaded"))?;
        let status = unsafe {
            f(output as *mut c_void, input as *const c_void, weight as *const c_void,
              m, n, k, workspace as *mut c_void, workspace_size, stream as *mut c_void)
        };
        if status != 0 {
            return Err(format!("cutlass_hgemm_v{variant} failed: {status}"));
        }
        Ok(())
    }

    pub fn hgemm_variant_workspace_size(&self, variant: usize, m: i32, n: i32, k: i32) -> Option<usize> {
        self.fn_hgemm_ws_variants.get(variant).and_then(|o| *o).map(|f| unsafe { f(m, n, k) })
    }

    /// Count how many HGEMM variants are loaded.
    pub fn hgemm_variant_count(&self) -> usize {
        self.fn_hgemm_variants.iter().filter(|o| o.is_some()).count()
    }

    /// Run oproj+residual variant by index. Returns Err if variant not loaded.
    pub fn oproj_residual_variant(
        &self,
        variant: usize,
        output: u64,
        input: u64,
        weight: u64,
        residual: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let f = self.fn_oproj_res_variants.get(variant)
            .and_then(|o| *o)
            .ok_or_else(|| format!("oproj_residual variant {variant} not loaded"))?;
        let status = unsafe {
            f(output as *mut c_void, input as *const c_void, weight as *const c_void,
              residual as *const c_void, m, n, k,
              workspace as *mut c_void, workspace_size, stream as *mut c_void)
        };
        if status != 0 {
            return Err(format!("cutlass_oproj_residual_v{variant} failed: {status}"));
        }
        Ok(())
    }

    pub fn oproj_residual_variant_workspace_size(&self, variant: usize, m: i32, n: i32, k: i32) -> Option<usize> {
        self.fn_oproj_res_ws_variants.get(variant).and_then(|o| *o).map(|f| unsafe { f(m, n, k) })
    }

    /// Count how many oproj+residual variants are loaded.
    pub fn oproj_residual_variant_count(&self) -> usize {
        self.fn_oproj_res_variants.iter().filter(|o| o.is_some()).count()
    }

    /// Run gateup+silu variant by index. Returns Err if variant not loaded.
    pub fn gateup_silu_variant(
        &self,
        variant: usize,
        output: u64,
        input: u64,
        weight: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let f = self.fn_gateup_silu_variants.get(variant)
            .and_then(|o| *o)
            .ok_or_else(|| format!("gateup_silu variant {variant} not loaded"))?;
        let status = unsafe {
            f(output as *mut c_void, input as *const c_void, weight as *const c_void,
              m, n, k, workspace as *mut c_void, workspace_size, stream as *mut c_void)
        };
        if status != 0 {
            return Err(format!("cutlass_gateup_silu_v{variant} failed: {status}"));
        }
        Ok(())
    }

    pub fn gateup_silu_variant_workspace_size(&self, variant: usize, m: i32, n: i32, k: i32) -> Option<usize> {
        self.fn_gateup_silu_ws_variants.get(variant).and_then(|o| *o).map(|f| unsafe { f(m, n, k) })
    }

    /// Count how many gateup+silu variants are loaded.
    pub fn gateup_silu_variant_count(&self) -> usize {
        self.fn_gateup_silu_variants.iter().filter(|o| o.is_some()).count()
    }

    /// Run FP8 GEMM variant by index. Returns Err if variant not loaded.
    pub fn fp8_gemm_variant(
        &self,
        variant: usize,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let f = self.fn_fp8_gemm_variants.get(variant)
            .and_then(|o| *o)
            .ok_or_else(|| format!("FP8 GEMM variant {variant} not loaded"))?;
        let status = unsafe {
            f(output as *mut c_void, a as *const c_void, b as *const c_void,
              a_scales as *const c_void, b_scale as *const c_void,
              m, n, k, workspace as *mut c_void, workspace_size, stream as *mut c_void)
        };
        if status != 0 {
            return Err(format!("cutlass_fp8_gemm_v{variant} failed: {status}"));
        }
        Ok(())
    }

    pub fn fp8_gemm_variant_workspace_size(&self, variant: usize, m: i32, n: i32, k: i32) -> Option<usize> {
        self.fn_fp8_gemm_ws_variants.get(variant).and_then(|o| *o).map(|f| unsafe { f(m, n, k) })
    }

    /// Count how many FP8 GEMM variants are loaded.
    pub fn fp8_gemm_variant_count(&self) -> usize {
        self.fn_fp8_gemm_variants.iter().filter(|o| o.is_some()).count()
    }

    /// FP8 GEMM + residual add (fused in epilogue).
    /// D[m,n] = cast<f16>(a_scales[m] * b_scale * GEMM(a,b) + residual[m,n])
    pub fn fp8_gemm_residual(
        &self,
        variant: usize,
        output: u64,
        a: u64,
        b: u64,
        a_scales: u64,
        b_scale: u64,
        residual: u64,
        m: i32,
        n: i32,
        k: i32,
        workspace: u64,
        workspace_size: usize,
        stream: u64,
    ) -> Result<(), String> {
        let f = self.fn_fp8_gemm_res_variants.get(variant)
            .and_then(|o| *o)
            .ok_or_else(|| format!("FP8 GEMM residual variant {variant} not loaded"))?;
        let status = unsafe {
            f(output as *mut c_void, a as *const c_void, b as *const c_void,
              a_scales as *const c_void, b_scale as *const c_void,
              residual as *const c_void,
              m, n, k, workspace as *mut c_void, workspace_size, stream as *mut c_void)
        };
        if status != 0 {
            return Err(format!("cutlass_fp8_gemm_residual_v{variant} failed: {status}"));
        }
        Ok(())
    }

    pub fn fp8_gemm_residual_workspace_size(&self, variant: usize, m: i32, n: i32, k: i32) -> Option<usize> {
        self.fn_fp8_gemm_res_ws_variants.get(variant).and_then(|o| *o).map(|f| unsafe { f(m, n, k) })
    }

    pub fn fp8_gemm_residual_variant_count(&self) -> usize {
        self.fn_fp8_gemm_res_variants.iter().filter(|o| o.is_some()).count()
    }
}
