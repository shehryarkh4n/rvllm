# rvllm-fusion: JIT PTX Kernel Compiler

A Rust-native JIT compiler that generates shape-specialized fused GPU kernels at model load time. No nvcc, no Python, no Triton dependency -- pure Rust string-based PTX generation that produces kernels 2-7.5x faster than our hand-written nvcc-compiled CUDA.

## Why

The N=1 decode path for a 7B parameter model is entirely memory-bandwidth bound. Each token generation reads ~14 GB of weights from HBM. The dominant cost is not computation but memory access.

Fusing multiple operations into a single kernel eliminates intermediate reads and writes between operations. For example, fusing RMSNorm + GEMV saves a full hidden_size read-write pair (2 x 3584 x 2 bytes = 14 KB per layer, 28 layers = 392 KB per step). At H100 HBM bandwidth (~3.3 TB/s), this is negligible per-operation, but the real win is reducing kernel launch overhead and enabling the compiler to keep intermediate values in registers.

## Architecture

```
Model config (hidden_size, num_heads, ...)
    |
    v
Pattern Matcher (matcher.rs)
    |
    v
Fusion IR (ir.rs)
    |
    v
PTX Emitter (ptx_emit.rs)        LLVM Backend (llvm_backend.rs, experimental)
    |                                    |
    v                                    v
PTX string                          LLVM IR -> NVPTX -> PTX
    |                                    |
    v                                    v
cuModuleLoadDataEx               cuModuleLoadDataEx
    |                                    |
    v                                    v
CUfunction handle              CUfunction handle
```

## Fusion Patterns

The matcher identifies sequences of operations that can be fused:

### RMSNorm + GEMV
```
Input: hidden_states [1, hidden_size]
  -> RMSNorm(hidden_states, weight, eps)
  -> GEMV(normed, projection_weight) [1, out_features]
Output: projected [1, out_features]
```

Fused kernel: One pass reads hidden_states, computes RMSNorm in shared memory, immediately multiplies by projection weight. The normed intermediate never touches global memory.

### Add + RMSNorm + GEMV
```
Input: residual [1, hidden_size], hidden_states [1, hidden_size]
  -> Add(residual, hidden_states)
  -> RMSNorm(sum, weight, eps)
  -> GEMV(normed, projection_weight) [1, out_features]
Output: projected [1, out_features], updated residual [1, hidden_size]
```

Fused kernel: Residual addition, normalization, and projection in one pass. The updated residual is written back as a side output.

### SiLU * Mul + GEMV
```
Input: gate [1, intermediate_size], up [1, intermediate_size]
  -> SiLU(gate) * up
  -> GEMV(activated, down_weight) [1, hidden_size]
Output: projected [1, hidden_size]
```

Fused kernel: Activation and down projection in one pass. The gate and up values are read once, activated in registers, and immediately used for the GEMV.

## PTX Generation Details

The PTX emitter (`ptx_emit.rs`) generates PTX assembly targeting sm_80+. Key optimizations:

### Vectorized Loads
All weight loads use 128-bit (`ld.global.v4.b32`) vector loads, reading 4 f16x2 pairs (8 f16 values) per instruction. This is critical for GEMV bandwidth:

```ptx
ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [weight_ptr];
// Unpacks to 8 x f16 values via mov.b32 + cvt.f32.f16
```

### Warp Shuffle Reductions
GEMV partial sums are reduced across warps using `shfl.sync.bfly`:

```ptx
shfl.sync.bfly.b32 %f_sum, %f_acc, 16, 31, 0xffffffff;
add.f32 %f_acc, %f_acc, %f_sum;
shfl.sync.bfly.b32 %f_sum, %f_acc, 8, 31, 0xffffffff;
add.f32 %f_acc, %f_acc, %f_sum;
// ... down to stride 1
```

### Shared Memory Tiling
For the RMSNorm component, the input vector is loaded into shared memory once and reused across all output rows:

```ptx
// Phase 1: Load input to shared memory, compute RMS
st.shared.f32 [smem + tid*4], %f_input;
bar.sync 0;
// Phase 2: Each warp reads from shared memory for its output row
ld.shared.f32 %f_val, [smem + k*4];
```

### Shape Specialization
Every kernel is generated with compile-time constants for the specific model dimensions:

```
hidden_size = 3584      (Qwen2.5-7B)
intermediate_size = 18944
num_heads = 28
head_dim = 128
```

This eliminates bounds checks, enables exact block/thread configuration, and allows the emitter to unroll inner loops to match the exact dimensions.

## Benchmark Results

H100 SXM 80GB, Qwen2.5-7B (hidden_size=3584, intermediate_size=18944):

| Fused Kernel | JIT (us) | Hand-written nvcc (us) | Speedup |
|---|---:|---:|---|
| Add+RMSNorm+QKV GEMV [1,4608,3584] | 5.5 | 10.6 | **1.92x** |
| Add+RMSNorm+GateUp GEMV [1,37888,3584] | 19.3 | 98.6 | **5.12x** |
| SiLU*Mul+Down GEMV [1,3584,18944] | 9.5 | 70.7 | **7.48x** |
| RMSNorm+QKV GEMV [1,4608,3584] | 5.3 | 10.8 | **2.03x** |

Per-step savings at N=1 (28 layers): **4.2ms** = estimated **1.8x** single-sequence speedup.

The hand-written kernels used for comparison are our own nvcc-compiled CUDA implementations. The JIT wins primarily because:
1. Shape specialization eliminates runtime bounds checks and branching
2. Vectorized loads are tuned to the exact row width
3. Block/thread configuration is computed for the specific matrix dimensions rather than using generic launch parameters

## Verification

The `verify.rs` module runs numerical comparison between fused and unfused execution paths at model load time, checking that outputs match within f16 precision tolerance. This catches PTX generation bugs before they cause silent accuracy degradation.

## LLVM NVPTX Backend (experimental)

`llvm_backend.rs` provides an alternative code generation path through LLVM's NVPTX target (the same backend Triton uses). Gated behind `--features llvm` because it requires LLVM 20.1 headers and libraries at build time.

Pipeline: Fusion IR -> inkwell LLVM IR -> NVPTX -> PTX -> cuModuleLoadDataEx

This path exists for future investigation of LLVM's optimization passes (instruction scheduling, register allocation) on the generated kernels, but the string-based PTX emitter is currently faster to iterate on and produces competitive code.

## Module Structure

| File | Purpose |
|---|---|
| `ir.rs` | Fusion intermediate representation (operation graph) |
| `matcher.rs` | Pattern detection: identifies fusable operation sequences |
| `ptx_emit.rs` | PTX string generation with vectorized loads and warp shuffles |
| `codegen.rs` | Orchestrates IR -> PTX pipeline |
| `compiler.rs` | Top-level compile API (model config -> loaded CUfunctions) |
| `jit.rs` | Runtime JIT entry point (called at model load) |
| `dispatch.rs` | Kernel launch dispatch (selects fused vs unfused at runtime) |
| `cache.rs` | On-disk PTX caching (keyed by model dimensions + version) |
| `verify.rs` | Numerical verification of fused vs unfused outputs |
| `llvm_backend.rs` | Experimental LLVM NVPTX code generation |

## Usage

The JIT compiler runs automatically at model load time. No user configuration is needed. Generated PTX is cached to `$RVLLM_PTX_DIR` (or a temp directory) keyed by model dimensions, so subsequent loads of the same model skip compilation.

To benchmark JIT kernels:
```bash
bench/bench_jit.sh
```

To verify fused kernel correctness:
```bash
bench/verify_fusion.sh
```
