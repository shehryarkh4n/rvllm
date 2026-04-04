# rTriton: Triton-style GPU Kernel Compiler in Rust

A standalone Rust reimplementation of OpenAI's Triton GPU kernel compiler, combined with cuBLAS integration for a unified decode layer execution plan. One crate, one CUDA graph, zero Python.

## What This Is

rTriton is a research crate that explores compiling Triton-style GPU kernels entirely in Rust. It provides:

1. **A builder DSL** for constructing GPU kernels as SSA IR graphs
2. **An optimization pass pipeline** (7 passes) targeting GPU execution
3. **PTX code generation** for sm_80+ (Ampere, Hopper)
4. **cuBLAS integration** with autotuned algorithm selection and FP8 support
5. **A mixed execution graph** that interleaves JIT kernels and cuBLAS GEMMs in a single CUDA graph

The crate compiles on Mac without CUDA (all GPU code behind `cfg(feature = "cuda")`). 50 tests passing.

## Builder DSL

Kernels are constructed programmatically using an SSA IR builder:

```rust
use rtriton::builder::KernelBuilder;
use rtriton::ir::{ScalarType, BinaryOp};

let mut builder = KernelBuilder::new("rmsnorm");

// Define parameters
let input = builder.add_param("input", ScalarType::F16);
let weight = builder.add_param("weight", ScalarType::F16);
let output = builder.add_param("output", ScalarType::F16);
let eps = builder.const_f32(1e-6);

// Load input tile
let x = builder.load(input, /* offset */ tid);

// Compute variance: mean(x^2)
let x_sq = builder.binary(BinaryOp::Mul, x, x);
let var = builder.reduce_sum(x_sq);
let var_mean = builder.binary(BinaryOp::Div, var, n_elements);

// Normalize: x * rsqrt(var + eps)
let var_eps = builder.binary(BinaryOp::Add, var_mean, eps);
let rstd = builder.rsqrt(var_eps);
let normed = builder.binary(BinaryOp::Mul, x, rstd);

// Scale by weight
let scaled = builder.binary(BinaryOp::Mul, normed, weight_val);

// Store result
builder.store(output, scaled);
```

## IR and Optimization Passes

The IR uses SSA form with 30+ operations covering:
- Arithmetic (add, mul, div, fma, rsqrt, exp, tanh)
- Memory (load, store, atomic)
- Control (branch, select)
- Reductions (sum, max, min)
- Tensor operations (dot, reshape, broadcast)

### Optimization passes (in order):

1. **Dead Code Elimination (DCE)**: Remove unused values
2. **Constant Folding**: Evaluate constant expressions at compile time
3. **Operator Fusion**: Merge compatible adjacent operations
4. **Memory Coalescing**: Reorder loads/stores for coalesced global memory access
5. **Shared Memory Planning**: Allocate shared memory for reused data (tiling)
6. **Software Pipelining**: Overlap memory loads with computation via double buffering
7. **Register Pressure Management**: Spill to shared memory when register usage exceeds SM limits

## PTX Code Generation

The codegen backend (`codegen.rs`) targets PTX ISA 7.0+ (sm_80+). Key features:

- 128-bit vectorized global loads (`ld.global.v4.b32`)
- Warp shuffle reductions (`shfl.sync.bfly`)
- Shared memory with configurable bank conflict avoidance
- cp.async for bulk global-to-shared transfers (sm_80+)
- Predicated execution for divergent paths

## Pre-Built LLM Kernels

8 kernels ready for LLM decode:

| Kernel | Module | Description |
|---|---|---|
| RMSNorm | `kernels/rmsnorm.rs` | Root mean square layer normalization |
| Fused Residual+RMSNorm | `kernels/rmsnorm.rs` | Add residual then normalize (saves 1 GMEM round-trip) |
| RoPE | `kernels/rope.rs` | Rotary position embeddings with cache write |
| SiLU*Mul | `kernels/silu_mul.rs` | Fused gated activation: SiLU(gate) * up |
| Tiled GEMM | `kernels/gemm.rs` | General matrix multiply with shared memory tiling |
| GEMV | `kernels/gemm.rs` | Matrix-vector product (M=1 specialization) |
| Persistent GEMM | `kernels/gemm.rs` | Stream-K work distribution across SMs |
| Flash Attention Decode | `kernels/fused_attention.rs` | Online softmax, paged KV cache, grouped-query attention |

### GEMM Dispatch

`kernels/gemm_dispatch.rs` routes GEMM operations based on problem dimensions:

- **M=1**: GEMV kernel (memory-bandwidth bound, one row)
- **M<=32**: cublasLt with autotuned algorithm (small batch decode)
- **M>32**: cuBLAS with heuristic algorithm (large batch / prefill)

## cuBLAS Integration

`cublas_gemm.rs` provides:

- **FP8 cublasLt plan caching**: Pre-computed execution plans for FP8 E4M3 GEMMs, cached by (M,N,K) shape
- **Autotuned algorithm selection**: Benchmarks 32 candidate algorithms per shape at startup, caches the winner
- **Graph workspace pre-allocation**: 4 MiB workspace allocated before CUDA graph capture to avoid graph-capture-time allocations
- **M-threshold routing**: Automatic routing between cublasLt (small M, better for decode) and cuBLAS (large M, better for prefill)

## Mixed Execution Graph

The decode layer plan (`graph.rs`) interleaves rTriton JIT kernels and cuBLAS GEMMs in a single CUDA graph:

```
Per-layer decode plan (9 operations):
  1. [rTriton] fused_residual_rmsnorm
  2. [cuBLAS]  QKV GEMM
  3. [rTriton] RoPE + KV cache write
  4. [rTriton] Flash Attention Decode
  5. [cuBLAS]  O-proj GEMM
  6. [rTriton] fused_residual_rmsnorm
  7. [cuBLAS]  gate_up GEMM
  8. [rTriton] SiLU * mul
  9. [cuBLAS]  down GEMM
```

Buffer allocation uses liveness-based interval coloring: temporary buffers are reused across operations when their live ranges don't overlap, minimizing peak memory usage.

The entire 28-layer decode step (252 operations) is captured in a single CUDA graph. Replay cost: one `cudaGraphLaunch` call, zero per-kernel launch overhead.

## Autotune

`autotune.rs` benchmarks kernel configurations at startup:

- Thread block dimensions (128, 256, 512)
- Shared memory tile sizes
- Vectorization widths
- cuBLAS algorithm IDs (32 candidates)

Results are cached per (GPU model, kernel name, problem shape) tuple.

## Module Structure

| File | Purpose |
|---|---|
| `builder.rs` | Kernel builder DSL (SSA IR construction) |
| `ir.rs` | Intermediate representation types |
| `passes.rs` | 7 optimization passes |
| `codegen.rs` | PTX code generation |
| `runtime.rs` | Kernel loading and launching |
| `graph.rs` | Mixed execution graph (JIT + cuBLAS) |
| `cublas_gemm.rs` | cuBLAS/cublasLt integration with autotuning |
| `autotune.rs` | Runtime autotuning framework |
| `kernels/` | Pre-built LLM kernel implementations |

## Status

Research crate. The kernels are tested and the compiler pipeline works end-to-end, but the integration with the main `rvllm-model-runner` forward pass is still in progress. Currently, the production decode path uses `rvllm-fusion` (Tier 1 JIT) and hand-written CUDA (Tier 2) while rTriton matures.
