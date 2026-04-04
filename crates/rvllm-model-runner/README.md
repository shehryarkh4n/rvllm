# rvllm-model-runner: Decode Paths and Kernel Tuning

The model runner orchestrates the GPU forward pass for all supported architectures. Its core design principle is **no fallbacks** -- every decode path either runs its intended kernels or fails loud. Silent degradation from missing or misconfigured kernels is treated as a bug.

## Decode Path Architecture

The runner selects a `ForwardPath` at model load time based on batch size, available kernels, weight format, and environment variables. The path is fixed for the lifetime of a CUDA graph capture.

```
Request arrives
    |
    v
T == 1?  ----yes----> Check decode path env vars
    |                      |
    no                     v
    |              FP8 weights? --yes--> Fp8Decode (cublasLt FP8 GEMMs)
    |                      |
    v                      no
Batched path               |
(cuBLAS/CUTLASS)     CUBLAS_DECODE=1? --yes--> CublasGemvDecode
                           |
                           no
                           |
                           v
                     FusedDecode (default)
                     Fused f16 GEMV kernels
```

### FusedDecode (default, T=1)

The production N=1 decode path. Each layer runs fused GEMV kernels that combine normalization with matrix-vector products, eliminating intermediate memory traffic.

Per-layer operation sequence:
```
residual + RMSNorm + QKV GEMV  (1 kernel: add_rmsnorm_qkv_gemv)
RoPE + KV cache write          (1 kernel: rope_cache_write)
GQA Flash Attention Decode      (1 kernel: gqa_attention_decode)
O-proj GEMV                     (1 kernel: o_proj_gemv)
residual + RMSNorm + GateUp GEMV (1 kernel: add_rmsnorm_gateup_gemv)
SiLU * Mul + Down GEMV         (1 kernel: silu_mul_down_gemv)
```

6 kernel launches per layer vs 10+ for unfused paths. The JIT compiler (`rvllm-fusion`) generates these kernels with shape-specialized tiling at model load time.

**121 tok/s** on H100 with Qwen2.5-7B f16, 55% HBM bandwidth utilization.

### CublasGemvDecode (T=1, `RVLLM_CUBLAS_DECODE=1`)

Separate RMSNorm kernels followed by cuBLAS HGEMM calls. No fusion, but cuBLAS can sometimes pick better algorithms for specific shapes.

Per-layer operation sequence:
```
RMSNorm                         (1 kernel)
cuBLAS HGEMM: QKV projection    (1 cuBLAS call)
RoPE + KV cache write           (1 kernel)
Flash Attention Decode           (1 kernel)
cuBLAS HGEMM: O-proj            (1 cuBLAS call)
RMSNorm                         (1 kernel)
cuBLAS HGEMM: gate_up           (1 cuBLAS call)
SiLU * Mul                      (1 kernel)
cuBLAS HGEMM: down              (1 cuBLAS call)
```

**118 tok/s** on H100, 84% HBM bandwidth utilization in standalone cuBLAS calls but slightly slower end-to-end due to extra kernel launch overhead.

### MegakernelDecode (T=1, experimental)

All 28 transformer layers and the LM head packed into a single CUDA kernel launch, driven by an instruction tape.

The instruction tape is a linear sequence of operations:
```
[op: RMSNORM, layer: 0, src: hidden, dst: scratch0]
[op: GEMV,    layer: 0, src: scratch0, dst: qkv, weight: q_proj]
[op: ROPE,    layer: 0, src: qkv, dst: qkv]
[op: ATTN,    layer: 0, src: qkv, dst: attn_out]
...
[op: GEMV,    layer: 27, src: scratch1, dst: hidden, weight: down_proj]
[op: GEMV,    layer: -1, src: hidden, dst: logits, weight: lm_head]
```

Key implementation details:
- Double-buffered residuals to avoid read-after-write hazards across layers
- Per-layer KV cache pointers embedded in the tape metadata
- Incrementing sync counters (not barriers) for inter-warp coordination
- Chunked gate/up splits to fit shared memory budgets
- RoPE bias race avoidance via warp-level sequencing

**~50 tok/s** on H100. The single-launch approach eliminates all inter-kernel launch overhead but is limited by the inability to overlap computation with memory transfers across layers.

### PersistentDecode (T=1, experimental)

SM-DAG (Streaming Multiprocessor Directed Acyclic Graph) cooperative kernel. Each SM is assigned a portion of the layer computation, and SMs coordinate via global memory flags rather than kernel launch boundaries.

Uses `cudaLaunchCooperativeKernel` for grid-level synchronization across all SMs.

**~51 tok/s** on H100. Similar throughput to megakernel but with cleaner inter-SM communication patterns.

### Fp8Decode (T=1, `RVLLM_FP8_WEIGHTS=1`)

All projection GEMMs use FP8 E4M3 weights with cublasLt. Weight quantization happens once at startup. Halves weight memory bandwidth for M=1 GEMV, which is the bottleneck for single-sequence decode.

Does NOT improve batched throughput -- at M>=8, f16 tensor cores already saturate compute and the f16->fp8 cast adds overhead.

### Batched (T>=2)

Standard cuBLAS/CUTLASS path for prefill and batched decode. Uses autotuned cublasLt algorithm selection with 32 candidates benchmarked per GEMM shape at startup.

GEMM strategy dispatch:
- **CUTLASS**: Fused epilogues (QKV+bias, O-proj+residual, GateUp+SiLU) when CUTLASS .so is available
- **cuBLAS**: Separate kernel launches when CUTLASS is not available
- **cublasLt for M<=32, cuBLAS for M>32**: Routing based on batch size

## Kernel Loader

The `KernelLoader` (`rvllm-gpu/src/kernel_loader.rs`) manages ~60 CUDA kernel modules with convention-based function name resolution:

```
activation/         -- silu_mul, gelu, swiglu
attention/          -- flash_attention_v3, gqa_attention, split_kv_attention
cache/              -- reshape_and_cache, copy_blocks
decode/             -- fused decode kernels (add_rmsnorm_qkv_gemv, etc.)
embedding/          -- token_embedding_lookup
fp8/                -- fp8_quantize, fp8_dequantize
gemv/               -- hgemv_f16, gemv_int4
norm/               -- rmsnorm_f16, layernorm_f16
rope/               -- rotary_embedding, rope_cache_write
sampling/           -- top_k_sampling, argmax
```

`validate_required_kernels()` checks at startup that all kernels needed by the selected decode path are present. Missing kernels cause a hard error, not a silent fallback.

## CUDA Graph Capture

The `GraphRunner` (`rvllm-worker/src/graph_runner.rs`) pre-captures CUDA graphs for 35 common batch sizes to eliminate kernel launch overhead in the decode loop:

```
Batch sizes: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, ...
```

For batch sizes without an exact graph, the runner pads up to the next captured size and unpads the logits afterward. Graph capture includes the full layer stack: all GEMM calls, attention, sampling prep.

## Architecture Registration

New model architectures are registered in `architectures/mod.rs` via a match arm mapping the HuggingFace `architectures` field from `config.json` to a Rust struct implementing the `Architecture` trait:

```rust
pub trait Architecture: Send + Sync {
    fn forward(
        &self,
        input: &ModelInput,
        cache: &mut [KVCache],
        attention: &dyn AttentionBackend,
    ) -> Result<GpuBuffer<f32>>;
}
```

Each architecture struct owns its weights and implements the full forward pass: embedding lookup, per-layer QKV projections, rotary embeddings, attention, MLP (dense or MoE), residual connections, final norm, and LM head projection.
