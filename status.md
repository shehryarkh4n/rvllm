# rvLLM GPU Memory & Data Path Audit

Model: Qwen2.5-32B-Instruct, FP8 CUTLASS path, H100 80GB
Date: 2025-04-15

## Model Dimensions (Qwen2.5-32B)

```
hidden_size      = 3584
num_heads        = 28   (query heads)
num_kv_heads     = 4    (GQA, 7:1 ratio)
head_dim         = 128
q_dim            = 28 * 128  = 3584
kv_dim           = 4  * 128  = 512
qkv_dim          = 3584 + 2*512 = 4608
intermediate     = 18944
gate_up_dim      = 37888  (2 * intermediate)
num_layers       = 64
vocab_size       = 152064
max_batch_tokens = 8192  (default)
block_size       = 16    (KV cache)
```

---

## 1. GPU Weight Memory

### F16 Weights (loaded at model init, NEVER FREED after FP8 quant)

Per-layer f16 weights stored in `ModelWeightsStore` (runner.rs:35-65):

| Weight | Shape (out x in) | Elements | F16 Bytes | Status |
|--------|-----------------|----------|-----------|--------|
| fused_qkv | 4608 x 3584 | 16,515,072 | 31.5 MB | DEAD after FP8 |
| o_proj | 3584 x 3584 | 12,845,056 | 24.5 MB | DEAD after FP8 |
| fused_gate_up | 37888 x 3584 | 135,794,432 | 259.1 MB | DEAD after FP8 |
| down_proj | 3584 x 18944 | 67,897,216 | 129.5 MB | DEAD after FP8 |
| input_layernorm | 3584 | 3,584 | 7 KB | still used |
| post_attn_layernorm | 3584 | 3,584 | 7 KB | still used |
| **Per-layer f16 total** | | | **444.6 MB** | |
| **64 layers f16** | | | **27.8 GB** | |

Non-layer f16 weights:

| Weight | Shape | F16 Bytes | Status |
|--------|-------|-----------|--------|
| embed_tokens | 152064 x 3584 | 1.04 GB | still used (embedding lookup) |
| lm_head_weight | 152064 x 3584 | 1.04 GB | DEAD after FP8 quant |
| final_norm_weight | 3584 | 7 KB | still used |

### FP8 Weights (created by enable_fp8_weights(), runner.rs:269-375)

Per-layer FP8 weights:

| Weight | Shape | FP8 Bytes | Scale Bytes |
|--------|-------|-----------|-------------|
| fp8_qkv | 4608 x 3584 | 15.7 MB | 4 B (per-tensor f32) |
| fp8_o_proj | 3584 x 3584 | 12.3 MB | 4 B |
| fp8_gate_up | 37888 x 3584 | 129.5 MB | 4 B |
| fp8_down | 3584 x 18944 | 64.8 MB | 4 B |
| gemv_qkv_scale | 4608 | 9 KB | (per-row f16 for GEMV) |
| gemv_o_proj_scale | 3584 | 7 KB | |
| gemv_gate_up_scale | 37888 | 74 KB | |
| gemv_down_scale | 3584 | 7 KB | |
| **Per-layer FP8 total** | | **222.3 MB** | |
| **64 layers FP8** | | **13.9 GB** | |

Non-layer FP8:

| Weight | FP8 Bytes |
|--------|-----------|
| fp8_lm_head | 0.52 GB |
| fp8_lm_head_scale | 4 B |

### Weight Memory Summary

```
F16 layer weights (DEAD, never freed):   27.8 GB
F16 lm_head (DEAD, never freed):          1.0 GB
F16 embed_tokens (active):                1.0 GB
F16 norms + final_norm (active):          0.9 MB
FP8 layer weights (active):              13.9 GB
FP8 lm_head (active):                     0.5 GB
FP8 scales + GEMV scales (active):        0.6 MB
----------------------------------------------------
TOTAL WEIGHT MEMORY:                      44.2 GB
WASTED (dead f16 after FP8 quant):        28.8 GB
SHOULD BE:                                15.4 GB
```

**BUG: 28.8 GB of f16 weights are never freed after FP8 quantization.**
The `#[allow(dead_code)]` annotations on `fused_qkv`, `fused_gate_up`,
`o_proj`, `down_proj` in `ModelWeightsStore` (runner.rs:36-43) confirm
the compiler knows they're unused. They remain allocated on GPU.

Fix: After `enable_fp8_weights()` completes, clear the f16 weight vecs
and drop lm_head_weight. This reclaims 28.8 GB for KV cache blocks.

---

## 2. KV Cache Memory

Allocated in kv_cache.rs:60-66, per-layer:

```
elements_per_block = block_size * num_kv_heads * head_dim
                   = 16 * 4 * 128 = 8192 elements
bytes_per_block    = 8192 * 2 = 16,384 bytes (16 KB, f16)
per_layer          = 2 * num_gpu_blocks * 16 KB  (key + value)
64 layers          = 128 * num_gpu_blocks * 16 KB = 2 MB per block
```

**CORRECT**: CacheConfig.num_heads is set from `num_kv_heads` (4), not
`num_heads` (28). Verified at integration.rs:155. No 7x overallocation.

With current 44.2 GB weights + 13.1 GB buffers = 57.3 GB used:
- Remaining for KV: ~22.7 GB on H100 80GB
- Blocks available: 22.7 GB / 2 MB = ~11,350 blocks
- Tokens cacheable: 11,350 * 16 = ~181,600 tokens

After freeing dead f16 weights (28.8 GB reclaimed):
- Remaining for KV: ~51.5 GB
- Blocks available: 51.5 GB / 2 MB = ~25,750 blocks
- Tokens cacheable: 25,750 * 16 = ~412,000 tokens
- **2.3x more concurrent sequences**

---

## 3. Scratch Buffers (F16LayerScratch, layer.rs:500-571)

Allocated once, reused across all 64 layers per step.
Sized for max_batch_tokens (t = 8192 default):

| Buffer | Formula | Bytes | Purpose |
|--------|---------|-------|---------|
| normed | t * hidden * 2 | 56 MB | RMSNorm output |
| qkv_buf | t * qkv_dim * 2 | 72 MB | Fused QKV GEMM output |
| attn_out | t * q_dim * 2 | 56 MB | Attention output (q_dim = hidden for Qwen) |
| o_proj_out | t * hidden * 2 | 56 MB | O-proj GEMM output |
| gate_up_out | t * gate_up_dim * 2 | 593 MB | GateUp GEMM output |
| silu_out | t * intermediate * 2 | 296 MB | SiLU activation output |
| gateup_workspace | t * gate_up_dim * 2 * 4 | **2,374 MB** | Fused gateup+silu CUTLASS workspace |
| attn_split_out | 16 * t * num_heads * head_dim * 4 | **1,792 MB** | Split-K attention partial results |
| attn_split_max | 16 * t * num_heads * 4 | 14 MB | Split-K row max |
| attn_split_sum | 16 * t * num_heads * 4 | 14 MB | Split-K row sum |
| residual_tmp | t * hidden * 2 | 56 MB | Layer-internal residual |
| fp8_act_scratch | t * intermediate * 2 | 296 MB | FP8 quantized activations |
| fp8_act_scale | t * 4 | 32 KB | Per-token FP8 scales |
| fp8_absmax | 1 * 4 | 4 B | FP8 absmax reduction |
| cutlass_workspace | max_workspace_size() | ~20 MB | CUTLASS GEMM workspace |
| **Scratch total** | | **~5,665 MB** | |

### Issues

**gateup_workspace (2.37 GB)**: Only used in the F16 fused gateup+silu
path (layer.rs:1059-1066). The FP8 path never touches this buffer --
it uses separate gate_up_out + fused_silu_mul_fp8_quant. When FP8 is
enabled, this 2.37 GB is completely wasted.

**attn_split_out (1.79 GB)**: 16-way split-K attention workspace. Only
used when context length exceeds split threshold. Allocated at full
max_batch_tokens * 16 splits unconditionally. At N=128 decode, actual
usage is 128 * 28 * 128 * 4 * (actual_splits) which is a fraction of
the allocation.

**silu_out (296 MB)**: Only used in F16 path (separate SiLU kernel).
FP8 path uses fused_silu_mul_fp8_quant which writes to fp8_act_scratch.
Dead allocation when FP8 is enabled.

---

## 4. Runner-Level Buffers (runner.rs:160-204)

| Buffer | Formula | Bytes | Notes |
|--------|---------|-------|-------|
| meta_packed | 256 * (max_seq_len + max_blocks + 10) * 4 | ~275 KB | Packed attention metadata |
| residual_a | t * hidden * 2 | 56 MB | Double-buffer residual (even layers) |
| residual_b | t * hidden * 2 | 56 MB | Double-buffer residual (odd layers) |
| down_a | t * hidden * 2 | 56 MB | Double-buffer down-proj output (even) |
| down_b | t * hidden * 2 | 56 MB | Double-buffer down-proj output (odd) |
| final_normed | t * hidden * 2 | 56 MB | Final RMSNorm before LM head |
| residual_tmp | t * hidden * 2 | 56 MB | Runner-level residual temp |
| logits_gpu | t * vocab * 4 | **4,736 MB** | Full-vocab f32 logits |
| lm_head_out_f16 | t * vocab * 2 | **2,368 MB** | DEAD -- `#[allow(dead_code)]` |
| embed_output | t * hidden * 2 | 56 MB | Embedding lookup output |
| argmax_output | t * 4 | 32 KB | GPU argmax results |
| pinned_argmax | t * 4 | 32 KB | Pinned host for async DtoH |
| pinned_meta | same as meta_packed | ~275 KB | Pinned host for async HtoD |
| **Runner total** | | **~7,496 MB** | |

### Issues

**lm_head_out_f16 (2.37 GB)**: Marked `#[allow(dead_code)]`, allocated
but never used in the current FP8 LM head path. The FP8 path writes
directly to logits_gpu (f32). Pure waste.

**logits_gpu (4.74 GB)**: f32 at full vocab (152064) for max_batch_tokens
(8192). At decode with N=128, actual usage is 128 * 152064 * 4 = 74 MB.
The buffer is 64x oversized for typical decode workloads.

**Duplicate residual_tmp**: One in F16LayerScratch (56 MB, layer.rs:565)
and one in GpuModelRunner (56 MB, runner.rs:195). Both are
t * hidden * 2. Need to verify they serve different purposes or if one
can be eliminated.

---

## 5. Data Flow Per Decode Step (FP8 CUTLASS, N=128)

All reads/writes are to HBM unless noted. "Reuse" means the buffer
is read by the next kernel without an intervening write by another
kernel to a different buffer.

```
Layer i (of 64):

  Input: residual_X[t * hidden, f16]  (X = a or b, alternating)
  Prev MLP: down_Y[t * hidden, f16]   (Y = a or b, previous layer's down-proj)

  1. FUSED_ADD_RMSNORM_FP8_QUANT
     Read:  residual_X (56 MB), down_Y (56 MB), input_layernorm_weight (7 KB)
     Write: fp8_act_scratch (4.5 MB fp8), fp8_act_scale (512 B), residual_tmp (56 MB)
     Note:  residual_tmp = residual_X + down_Y (saved for later residual add)

  2. QKV GEMM (CUTLASS FP8, autotuned variant)
     Read:  fp8_act_scratch (4.5 MB), fp8_act_scale (512 B),
            fp8_qkv_weight (15.7 MB), fp8_qkv_scale (4 B)
     Write: qkv_buf (72 MB at t=8192, but 1.125 MB at N=128)

  3. ROPE + KV CACHE WRITE (fused kernel)
     Read:  qkv_buf (Q+K portions), rope_cos, rope_sin
     Write: kv_cache key/val slots (via slot_mapping)
     Note:  Q stays in qkv_buf, K/V written directly to cache blocks

  4. FLASH ATTENTION 3 (FA3 or split-K fallback)
     Read:  Q from qkv_buf, K/V from kv_cache (random access by block table)
     Write: attn_out (56 MB at max, 0.875 MB at N=128)

  5. FP8 QUANT (per-token quantization of attn_out)
     Read:  attn_out (0.875 MB)
     Write: fp8_act_scratch (0.4375 MB), fp8_act_scale (512 B)

  6a. O-PROJ GEMM + RESIDUAL (if autotuned fp8_gemm_residual exists)
     Read:  fp8_act_scratch, fp8_act_scale,
            fp8_o_proj_weight (12.3 MB), fp8_o_proj_scale (4 B),
            residual_tmp (from step 1)
     Write: residual_Y (output = GEMM + residual, directly to next layer's residual)
     Saves: one fused_add_rmsnorm read of o_proj_out
  6b. O-PROJ GEMM (plain, if no residual autotune entry)
     Read:  same minus residual_tmp
     Write: o_proj_out (56 MB max)

  7. FUSED_ADD_RMSNORM_FP8_QUANT (or just RMSNORM_FP8_QUANT if 6a used)
     If 6a: Read residual_Y + post_attn_layernorm_weight
            Write fp8_act_scratch + fp8_act_scale  (no residual add needed)
     If 6b: Read residual_tmp + o_proj_out + post_attn_layernorm_weight
            Write fp8_act_scratch + fp8_act_scale + residual_Y

  8. GATEUP GEMM (CUTLASS FP8, autotuned)
     Read:  fp8_act_scratch, fp8_act_scale,
            fp8_gate_up_weight (129.5 MB), fp8_gate_up_scale (4 B)
     Write: gate_up_out (593 MB max, 9.25 MB at N=128)

  9. FUSED_SILU_MUL_FP8_QUANT
     Read:  gate_up_out (9.25 MB)
     Write: fp8_act_scratch (4.625 MB), fp8_act_scale (512 B)
     Note:  SiLU(gate) * up, then quantize to FP8, single kernel

  10a. DOWN GEMM + RESIDUAL (if autotuned fp8_gemm_residual exists -- currently NO for K=18944)
     Would read: fp8_act_scratch + residual_Y -> write down_X
  10b. DOWN GEMM (plain FP8, current path)
     Read:  fp8_act_scratch (4.625 MB), fp8_act_scale (512 B),
            fp8_down_weight (64.8 MB), fp8_down_scale (4 B)
     Write: down_X (56 MB max, 0.875 MB at N=128)

  Output: residual_Y, down_X carry forward to layer i+1
```

### HBM Traffic Per Layer at N=128 (decode)

Activations (read + write, both directions):

| Step | Read | Write | Total |
|------|------|-------|-------|
| 1. add+norm+fp8quant | 1.75 MB | 1.33 MB | 3.08 MB |
| 2. QKV GEMM | 20.2 MB (mostly weight) | 1.13 MB | 21.3 MB |
| 3. RoPE+KV write | ~0.5 MB | ~0.1 MB | 0.6 MB |
| 4. FA3 attention | varies by ctx_len | 0.88 MB | varies |
| 5. FP8 quant | 0.88 MB | 0.44 MB | 1.32 MB |
| 6. O-proj GEMM | 12.7 MB | 0.88 MB | 13.6 MB |
| 7. norm+fp8quant | 0.88 MB | 0.44 MB | 1.32 MB |
| 8. GateUp GEMM | 134.3 MB | 9.25 MB | 143.5 MB |
| 9. SiLU+fp8quant | 9.25 MB | 4.63 MB | 13.9 MB |
| 10. Down GEMM | 69.4 MB | 0.88 MB | 70.3 MB |
| **Total (excl attn)** | | | **~269 MB/layer** |
| **64 layers** | | | **~16.8 GB/step** |

Weight reads dominate (222 MB/layer). Activation traffic is ~47 MB/layer.
At 3.35 TB/s HBM bandwidth, pure bandwidth limit = 16.8 GB / 3.35 TB/s = 5.0 ms/step.

### Redundant HBM Round-Trips

1. **Steps 4->5->6 (attn_out -> fp8_quant -> O-proj)**:
   attn_out written by FA3, read by fp8_quant, fp8 version read by O-proj.
   3 touches of the same data. If FP8 quant were fused into FA3 epilogue
   or O-proj prologue, saves one 0.88 MB round-trip per layer.

2. **Steps 8->9->10 (gate_up_out -> silu+fp8quant -> Down)**:
   gate_up_out written by GateUp GEMM, read by fused_silu_fp8_quant,
   fp8 version read by Down GEMM. Same 3-touch pattern. The fused_silu
   kernel already combines SiLU + quant, but gate_up_out still
   materializes to HBM between steps 8 and 9.

3. At N=128 these round-trips cost ~0.27 us each (0.88 MB / 3.35 TB/s).
   64 layers * 2 extra round-trips * 0.27 us = ~35 us/step.
   Out of a ~5-7 ms step, that's <1%. Not the bottleneck.

---

## 6. Fallback Audit Results

### FP8 Hot Path (CUTLASS loaded, FP8 enabled)

| Location | Code | Status |
|----------|------|--------|
| cutlass_fp8_gemm_dispatch L150 | Was: hardcoded tile fallback when no autotune | **FIXED: now panics** |
| cuBLASLt FP8 L868, 967, 1024 | `else if let (Some(lt), Some(fp8w))` | Unreachable (CUTLASS branch taken first) |
| fp8_gemm_residual -> plain L959, 1094 | Falls back to plain FP8 GEMM | By design (down-proj residual 40% slower) |
| GEMV FP8 L822 | Gated behind `RVLLM_GEMV_FP8` env var | Correct, pending A/B profiling |

### F16 Path (only active when FP8 disabled)

| Location | Code | Status |
|----------|------|--------|
| f16_gemm_autotuned L243 | Falls to cuBLAS when no CUTLASS entry | By design (cuBLAS wins some shapes) |
| fused gateup+silu L1069 | Falls to separate GEMM+SiLU | By design (autotune only stores winners) |
| fused oproj+residual L990 | Falls to separate GEMM | By design |

No silent fallbacks in the FP8 hot path. The only hardcoded tile
fallback has been replaced with a panic.

---

## 7. Waste Summary & Priority Fixes

| Issue | Wasted GPU Memory | Difficulty | Impact |
|-------|-------------------|------------|--------|
| Dead f16 weights after FP8 quant | **28.8 GB** | Easy (clear vecs) | 2.3x more KV cache blocks |
| lm_head_out_f16 (dead_code) | **2.37 GB** | Easy (don't alloc when FP8) | Free memory |
| gateup_workspace (F16-only) | **2.37 GB** | Easy (skip alloc when FP8) | Free memory |
| silu_out (F16-only) | **296 MB** | Easy (skip alloc when FP8) | Free memory |
| logits_gpu oversized | ~4.66 GB unused at N=128 | Medium (lazy alloc) | Depends on max batch |
| attn_split_out oversized | ~1.79 GB at max | Medium | Depends on split-K usage |
| Duplicate residual_tmp | 56 MB | Easy (verify, remove one) | Minor |
| **Total recoverable** | **~33.8 GB** | | |

### After fixes, H100 80GB memory map:

```
FP8 weights + scales:           15.4 GB
Embedding (f16):                 1.0 GB
Scratch (FP8-only needed):      ~2.9 GB
Runner buffers (logits etc):    ~5.1 GB
Available for KV cache:         ~55.6 GB
KV blocks (at 2 MB/block):     ~27,800
Tokens cacheable:               ~445,000
```

vs current:

```
All weights (f16 + FP8):        44.2 GB
Scratch (all paths):            ~5.7 GB
Runner buffers:                 ~7.5 GB
Available for KV cache:         ~22.6 GB
KV blocks:                      ~11,300
Tokens cacheable:               ~181,000
```

---

## 8. Benchmark Baselines (H100 SXM5, Qwen2.5-32B FP8)

Instance: vast.ai 34803302, ssh8.vast.ai:13302
Binary: commit 3eacacd0f, fresh kernel compile (56/57 kernels)
Autotune: full cache with fp8_gemm + fp8_gemm_residual entries

```
N=1:    139.6 tok/s  (target: 160+, gap: GEMV path pending profiling)
N=32:  4,257.1 tok/s
N=64: 10,702.8 tok/s
N=128: 18,922.8 tok/s  (baseline: 19,259)
```

O-proj residual fusion active (2-6% GEMM overhead, saves one norm kernel).
Down-proj residual fusion disabled (38-46% overhead at K=18944).

---

## 9. Competitive Position vs vLLM

The dead f16 weight fix (28.8 GB) brings us to parity with vLLM, not
ahead of it. vLLM loads FP8 weights directly from the checkpoint --
it never has f16 copies on GPU. We load f16, quantize at runtime,
and forget to free. Fixing this is a correctness bug, not a
competitive advantage.

What actually differentiates rvLLM from vLLM:
- Kernel-level speed: CUTLASS FP8 GEMMs are 2-6x faster individually
- Fused kernels: add+rmsnorm+fp8quant, silu+mul+fp8quant, oproj+residual
- Lower kernel count per layer (10 vs 12-15)
- FA3 integration

The throughput gap we're chasing is CPU overhead and scheduling, not
memory layout or kernel speed. The GPU is fast; the CPU is the
bottleneck between steps.
