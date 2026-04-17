//! Engine bring-up: assemble every subsystem from paths on disk.
//!
//! This module exists so `main.rs` in the bench + serve binaries can
//! reach for one `bring_up::Engine::load(paths)` call and get a fully
//! wired runtime back. No graph capture here (that's a separate step
//! after weights are loaded and bucket shapes are known).

use std::path::PathBuf;
use std::sync::Arc;

use rvllm_attention::Fa3Kernels;
use rvllm_core::{ConfigError, Result, RvllmError};
use rvllm_cutlass::{CutlassLib, Fp8GemmPlan, Policy};
use rvllm_kernels::{manifest::KernelManifest, KernelFn, KernelLoader, LoadedModule};
use rvllm_loader::{load_model, LoadedModel, ModelArch};
use rvllm_mem::{context::CudaContextHandle, stream::Stream, HbmArena};

/// Paths (and only paths) the engine needs at init. All other config
/// is read from `model_dir/config.json` and `kernels_dir/manifest.json`.
#[derive(Clone, Debug)]
pub struct EnginePaths {
    pub model_dir: PathBuf,
    pub kernels_dir: PathBuf,
    pub cutlass_so: PathBuf,
    pub fa3_so: PathBuf,
    pub policy_json: PathBuf,
}

/// Assembled subsystems.
pub struct Bringup {
    pub ctx: Arc<CudaContextHandle>,
    pub arena: HbmArena<'static>,
    pub stream: Stream,
    pub arch: ModelArch,
    pub model: LoadedModel,
    pub kernels: Arc<KernelLoader>,
    pub cutlass: CutlassLib,
    pub fa3: Fa3Kernels,
    pub policy: Policy,
    pub fused_modules: FusedModules,
}

/// Loaded CUDA modules + resolved kernel handles for the nine fused
/// kernels a layer uses.
pub struct FusedModules {
    pub add_rmsnorm: LoadedModule,
    pub rope_kv_write: LoadedModule,
    pub silu_mul: LoadedModule,
    pub quantize: LoadedModule,
    pub argmax: LoadedModule,
    pub fn_add_rmsnorm: KernelFn,
    pub fn_rope_kv_write: KernelFn,
    pub fn_silu_mul: KernelFn,
    pub fn_quantize: KernelFn,
    pub fn_argmax: KernelFn,
}

impl Bringup {
    pub fn load(paths: EnginePaths, arena_bytes: usize) -> Result<Self> {
        // 1. CUDA context + stream.
        let ctx = Arc::new(CudaContextHandle::init(0)?);

        // SAFETY: arena lifetime 'static via leak — engine owns it for program lifetime.
        let arena = HbmArena::new(&ctx, arena_bytes)?;
        // The 'static lifetime gymnastics: HbmArena<'ctx> borrows from ctx.
        // The Arc keeps ctx alive. We transmute the lifetime to 'static
        // because Bringup owns both. This is sound as long as `ctx`
        // outlives `arena` — which it does (they live in the same
        // struct and ctx is the last dropped).
        let arena: HbmArena<'static> = unsafe { std::mem::transmute(arena) };

        let stream = Stream::new(&ctx)?;

        // 2. Arch + model.
        let arch = ModelArch::from_dir(&paths.model_dir)?;
        let model = load_model(&paths.model_dir, &arena, &arch)?;

        // 3. Kernel manifest -> loader -> modules.
        let manifest_path = paths.kernels_dir.join("manifest.json");
        let manifest = KernelManifest::load_and_verify(&manifest_path)?;
        let kernels = Arc::new(KernelLoader::new(manifest));
        let fused_modules = load_fused(&kernels)?;

        // 4. FA3 .so.
        let fa3 = Fa3Kernels::load(paths.fa3_so.clone(), arch.head_dim as u32)?;

        // 5. Policy + CUTLASS .so (resolve every variant referenced in
        //    the policy).
        let policy_bytes = std::fs::read(&paths.policy_json).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: paths.policy_json.clone(),
            source,
        })?;
        let policy: Policy = serde_json::from_slice(&policy_bytes).map_err(|e| {
            RvllmError::config(
                ConfigError::Inconsistent {
                    reasons: vec![format!("policy.json parse: {e}")],
                },
                "policy.json",
            )
        })?;
        let variants: Vec<_> = policy
            .entries
            .values()
            .map(|e| e.variant)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let cutlass = CutlassLib::load(paths.cutlass_so.clone(), &variants)?;

        Ok(Self {
            ctx,
            arena,
            stream,
            arch,
            model,
            kernels,
            cutlass,
            fa3,
            policy,
            fused_modules,
        })
    }

    /// Resolve a GEMM plan for a (M, N, K, dtype) shape. Missing plan
    /// in the policy is a typed AutotuneCacheMiss; the engine refuses
    /// to run that shape.
    pub fn plan(&self, m: u32, n: u32, k: u32) -> Result<Fp8GemmPlan> {
        Fp8GemmPlan::from_policy(&self.policy, m, n, k, rvllm_core::DType::Fp8E4M3)
    }

    /// Run `iters` decode steps against a batch of `num_seqs` fake seqs
    /// (one token each). Returns elapsed nanoseconds.
    ///
    /// The path here is eager (no graph capture): reach for a bench
    /// number, then add the graph capture optimization.
    ///
    /// # Safety
    /// Uses raw device pointers from the arena. The `Bringup` owns the
    /// arena + stream for this function's duration so pointers stay
    /// valid.
    #[cfg(feature = "cuda")]
    pub unsafe fn run_bench(&self, num_seqs: u32, iters: u32, warmup: u32) -> Result<BenchResult> {
        use crate::layer_exec;
        use rvllm_cutlass::Fp8GemmPlan;
        use rvllm_fused::require_multiple;

        let arch = &self.arch;
        let hidden = arch.hidden_size as u32;
        let head_dim = arch.head_dim as u32;
        let nh = arch.num_attention_heads as u32;
        let nkvh = arch.num_key_value_heads as u32;
        let inter = arch.intermediate_size as u32;
        let q_dim = nh * head_dim;
        let kv_dim = nkvh * head_dim;
        let qkv_rows = (nh + 2 * nkvh) * head_dim;
        require_multiple(hidden as usize, 8, "hidden")?;

        // --- scratch allocations -----------------------------------------
        let arena = &self.arena;
        let hidden_fp8 = arena.region("hidden_fp8", (num_seqs * hidden) as usize, 16)?;
        let hidden_scale = arena.region("hidden_scale", (num_seqs * 4) as usize, 16)?;
        let qkv_out = arena.region("qkv_out", (num_seqs * qkv_rows * 2) as usize, 16)?;
        let attn_out = arena.region("attn_out", (num_seqs * q_dim * 2) as usize, 16)?;
        let attn_out_fp8 = arena.region("attn_out_fp8", (num_seqs * q_dim) as usize, 16)?;
        let attn_out_scale = arena.region("attn_out_scale", (num_seqs * 4) as usize, 16)?;
        let gate_up_out =
            arena.region("gate_up_out", (num_seqs * 2 * inter * 2) as usize, 16)?;
        let gate_up_fp8 = arena.region("gate_up_fp8", (num_seqs * 2 * inter) as usize, 16)?;
        let gate_up_scale = arena.region("gate_up_scale", (num_seqs * 4) as usize, 16)?;
        let mlp_out_fp8 = arena.region("mlp_out_fp8", (num_seqs * inter) as usize, 16)?;
        let mlp_out_scale = arena.region("mlp_out_scale", (num_seqs * 4) as usize, 16)?;

        let num_blocks_total: u32 = 1024;
        let block_size: u32 = 64;
        let max_blocks_per_seq: u32 = 128;
        let kv_per_layer = 2 * num_blocks_total * block_size * nkvh * head_dim * 2;
        let kv_cache = arena.region(
            "kv_cache",
            (arch.num_hidden_layers as u64 * kv_per_layer as u64) as usize,
            256,
        )?;

        let cutlass_ws_bytes: usize = 16 * 1024 * 1024;
        let cutlass_ws = arena.region("cutlass_ws", cutlass_ws_bytes, 256)?;
        let fa3_ws = arena.region("fa3_ws", 64 * 1024 * 1024, 256)?;

        let residual = arena.region("residual", (num_seqs * hidden * 2) as usize, 16)?;

        // Metadata (zero-filled; decode step with all ctxlen=1 works for
        // timing — FA3 will still read stale but valid KV pages).
        let positions = arena.region("positions", (num_seqs * 4) as usize, 16)?;
        let slot_mapping = arena.region("slot_mapping", (num_seqs * 4) as usize, 16)?;
        let context_lens = arena.region("context_lens", (num_seqs * 4) as usize, 16)?;
        let block_tables = arena.region(
            "block_tables",
            (num_seqs * max_blocks_per_seq * 4) as usize,
            16,
        )?;

        // Plans (from policy) for this specific bucket.
        let plan_qkv = Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            qkv_rows,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let plan_o = Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            hidden,
            q_dim,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let plan_gate_up = Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            2 * inter,
            hidden,
            rvllm_core::DType::Fp8E4M3,
        )?;
        let plan_down = Fp8GemmPlan::from_policy(
            &self.policy,
            num_seqs,
            hidden,
            inter,
            rvllm_core::DType::Fp8E4M3,
        )?;

        let dims = layer_exec::LayerDims {
            num_tokens: num_seqs,
            hidden,
            num_heads: nh,
            num_kv_heads: nkvh,
            head_dim,
            intermediate: inter,
            block_size,
            max_blocks_per_seq,
            num_blocks_total,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            rms_eps: 1e-6,
        };
        let kernels = layer_exec::LayerKernels {
            fused_add_rmsnorm: self.fused_modules.fn_add_rmsnorm,
            fused_rope_kv_write: self.fused_modules.fn_rope_kv_write,
            fused_silu_mul: self.fused_modules.fn_silu_mul,
            quantize_fp8_per_token: self.fused_modules.fn_quantize,
        };
        let plans = layer_exec::LayerGemmPlans {
            qkv: plan_qkv,
            o: plan_o,
            gate_up: plan_gate_up,
            down: plan_down,
        };

        let stream = self.stream.raw();
        let residual_ptr = residual.device_ptr();

        let one_step = || -> Result<()> {
            for layer in &self.model.layers {
                let w = layer_exec::LayerWeights {
                    attn_norm_gamma: layer.input_layernorm.offset_bytes,
                    qkv_fp8: layer.qkv.offset_bytes,
                    qkv_scale: 0,
                    o_fp8: layer.o_proj.offset_bytes,
                    o_scale: 0,
                    mlp_norm_gamma: layer.post_attention_layernorm.offset_bytes,
                    gate_up_fp8: layer.gate_up.offset_bytes,
                    gate_up_scale: 0,
                    down_fp8: layer.down_proj.offset_bytes,
                    down_scale: 0,
                };
                let scratch = layer_exec::LayerScratch {
                    hidden_fp8: hidden_fp8.device_ptr(),
                    hidden_scale: hidden_scale.device_ptr(),
                    qkv_out: qkv_out.device_ptr(),
                    k_cache: kv_cache.device_ptr(),
                    v_cache: kv_cache.device_ptr() + (kv_per_layer / 2) as u64,
                    attn_out: attn_out.device_ptr(),
                    attn_out_fp8: attn_out_fp8.device_ptr(),
                    attn_out_scale: attn_out_scale.device_ptr(),
                    gate_up_out: gate_up_out.device_ptr(),
                    gate_up_fp8: gate_up_fp8.device_ptr(),
                    gate_up_scale: gate_up_scale.device_ptr(),
                    mlp_out_fp8: mlp_out_fp8.device_ptr(),
                    mlp_out_scale: mlp_out_scale.device_ptr(),
                    cutlass_workspace: cutlass_ws.device_ptr(),
                    cutlass_workspace_bytes: cutlass_ws_bytes,
                    fa3_workspace: fa3_ws.device_ptr(),
                };
                let meta = layer_exec::MetadataPtrs {
                    positions: positions.device_ptr(),
                    slot_mapping: slot_mapping.device_ptr(),
                    cos: self.model.rope_cos.offset_bytes,
                    sin: self.model.rope_sin.offset_bytes,
                    block_tables: block_tables.device_ptr(),
                    context_lens: context_lens.device_ptr(),
                };
                layer_exec::forward(
                    dims,
                    &kernels,
                    &w,
                    &scratch,
                    &meta,
                    &plans,
                    &self.cutlass,
                    &self.fa3,
                    residual_ptr,
                    stream,
                )?;
            }
            Ok(())
        };

        // Warmup.
        for _ in 0..warmup {
            one_step()?;
        }
        self.stream.fence()?;

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            one_step()?;
        }
        self.stream.fence()?;
        let elapsed = t0.elapsed();
        Ok(BenchResult {
            ns_per_step: elapsed.as_nanos() / iters.max(1) as u128,
            total_ns: elapsed.as_nanos(),
            iters,
            num_seqs,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub unsafe fn run_bench(&self, num_seqs: u32, iters: u32, _warmup: u32) -> Result<BenchResult> {
        Ok(BenchResult {
            ns_per_step: 0,
            total_ns: 0,
            iters,
            num_seqs,
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BenchResult {
    pub ns_per_step: u128,
    pub total_ns: u128,
    pub iters: u32,
    pub num_seqs: u32,
}

fn load_fused(loader: &KernelLoader) -> Result<FusedModules> {
    let add_rmsnorm = loader.load_ptx("fused_add_rmsnorm_fp8_quant")?;
    let rope_kv_write = loader.load_ptx("fused_rope_kv_write")?;
    let silu_mul = loader.load_ptx("fused_silu_mul_fp8_quant")?;
    let quantize = loader.load_ptx("quantize_fp8_per_token")?;
    let argmax = loader.load_ptx("argmax")?;

    let fn_add_rmsnorm = add_rmsnorm.get_function("fused_add_rmsnorm_fp8_quant_kernel")?;
    let fn_rope_kv_write = rope_kv_write.get_function("fused_rope_kv_write_kernel")?;
    let fn_silu_mul = silu_mul.get_function("fused_silu_mul_fp8_quant_kernel")?;
    let fn_quantize = quantize.get_function("quantize_fp8_per_token_kernel")?;
    let fn_argmax = argmax.get_function("argmax_kernel")?;

    Ok(FusedModules {
        add_rmsnorm,
        rope_kv_write,
        silu_mul,
        quantize,
        argmax,
        fn_add_rmsnorm,
        fn_rope_kv_write,
        fn_silu_mul,
        fn_quantize,
        fn_argmax,
    })
}
