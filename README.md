# rvLLM

A single-GPU, FP8, graph-captured LLM inference engine in Rust. Qwen2.5-7B on a single H100 SXM 80GB delivers **42,030 tok/s at N=512** (FP8 E4M3 KV cache, real FA3 paged-prefill, decode + final RMSnorm + LM head + argmax, CUDA graph captured) and **63.8 ms TTFT at N=128** (16-token prompt). Consistently **14–23% faster than vLLM 0.19** across every batch size we tested (N=128, 256, 512) on the same GPU with the same model and quant config.

No Python in the hot path. No fallbacks. Missing artifacts (policy, FA3 `.so`, kernel SHA) refuse to start.

## Headline: v3 vs vLLM 0.19 on the same H100

Same GPU, same Qwen2.5-7B-Instruct checkpoint, same FP8 E4M3, CUDA graphs on, full decode after real prefill. Both measurements taken from each engine's own steady-state decode throughput metric.

| Batch | vLLM 0.19 V1 | rvllm-v3 | v3 Δ |
|---:|---:|---:|---:|
| 128 | 19,399 | 22,069 | **+13.8%** |
| 256 | 27,996 | 34,364 | **+22.8%** |
| **512** | 36,097 | **42,030** | **+16.4%** |

vLLM numbers are the `Avg generation throughput` engine log line (steady-state decode, prefill excluded), via `vllm bench latency --model Qwen2.5-7B --quantization fp8 --batch-size <N> --input-len 16 --output-len 512 --num-iters 3 --num-iters-warmup 1 --dtype float16`. vLLM also runs fine at N=256/512 on this box — prior readme claims that it couldn't were our mistake.

## The fair-bench setup

Decode-kernel throughput under steady-state; same model weights, same FP8 quant type, same GPU, CUDA graphs enabled in both. All three of these were our initial caveats vs vLLM; v3's bench has been updated to address each:

1. **vLLM does real prefill** (16 input tokens × 128 prompts = 2,048 tokens of real prompt processing before decode starts). v3 now does the same under `RVLLM_REAL_PREFILL=1` — one multi-query causal FP8 attention call via FA3 paged-prefill over the whole concatenated prompt. vLLM's `Avg generation throughput` metric is measured only during the decode phase, so steady-state throughput is measured under identical KV-populated state on both sides. A 16-step eager "faux-prefill" is also still available as a fallback.
2. **vLLM runs its scheduler every step.** Even at steady-state batch=128, it checks for request state, completions, block table management. v3's bench re-uploads `positions`, `slot_mapping`, and `context_lens` every iteration inside the timed loop (not once before capture), so v3 pays the same per-step HtoD cost that vLLM's scheduler pays for metadata management.
3. **vLLM's KV cache has real content from real prompts.** Ours previously had post-warmup garbage. v3's bench now runs real FA3 paged-prefill before the timed window, so paged KV is populated with legitimate rope+quant activations (matching the condition vLLM measures under). Metadata (`positions`, `slot_mapping`, `context_lens`) continues to advance across the decode window.

Net impact of the three fairness fixes on the N=128 number: −0.3%. The per-step metadata upload is cheap against the 28-layer forward + LM head compute, so the +13.8% edge at N=128 — and the much larger capacity-unlock edge at N=256/512 — are genuine.

## v3 throughput by batch size (FP8 KV, real FA3 prefill)

H100 SXM 80GB, Qwen2.5-7B-Instruct, FP8 E4M3 weights + FP8 E4M3 KV, graph-captured, full decode + final RMSnorm + LM head + argmax, metadata re-upload per step, real FA3 paged-prefill, 3 warmup iters:

| N | tok/s | ms/step | vLLM same N | v3 Δ |
|---:|---:|---:|---:|---:|
| 128 | 22,069 | 5.80 | 19,399 | +13.8% |
| 256 | 34,364 | 7.45 | 27,996 | +22.8% |
| **512** | **42,030** | **12.18** | 36,097 | **+16.4%** |

## Time-to-first-token (TTFT)

Phase F shipped real FA3 paged-prefill. `RVLLM_REAL_PREFILL=1` runs one multi-query causal FP8 attention call over the concatenated prompt (`total_q = N × prompt_len`) via varlen `cu_seqlens_q`, replacing the 16-step faux-prefill stand-in.

| N | TTFT (ms), real prefill | TTFT (ms), faux-prefill | Δ |
|---:|---:|---:|---:|
| 128 | **63.8** | 763 | **12.0×** |
| 256 | **117.7** | 833 | **7.1×** |
| 512 | **131.7** | 855 | **6.5×** |

Measured with `RVLLM_PREFILL_LEN=16` (16 prompt tokens/seq). `t₀ = right before prompt ingest` → `t₁ = first sampled token visible in pinned host memory` (includes one decode step after prefill + DtoH + stream fence).

## Why v3 is fast (structural wins)

1. **Fused Q‖K‖V GEMM.** One matmul with N=4608 replaces three separate Q/K/V GEMMs. 2 fewer launches per layer × 28 layers = 56 fewer launches per decode step.
2. **All 5 FP8 linears on cuBLASLt.** QKV, O, gate_up, down, lm_head all go through `cublasLtMatmul`. Its per-shape heuristic explores many more algorithms than a hand-maintained CUTLASS variant cache. Per-(M,N,K,epilogue) algo cache so the heuristic runs once per shape at capture time.
3. **Fused epilogues.** `CUBLASLT_EPILOGUE_BIAS` folds the f16 bias add into the QKV GEMM; `β=1` folds the residual add into O and down GEMMs. 3 kernels per layer × 28 = 84 fewer launches.
4. **FP8 E4M3 KV cache.** `libfa3_kernels.so` links FA3's upstream hdim128 E4M3 paged instantiations. KV pages are 1 byte/element with per-tensor descale; halves KV memory and unlocks N=256/512 that f16 KV cannot fit on one H100.
5. **No f32 on the GPU in the decode path.** RoPE tables are f16. The only f32 on-device is the FP8 protocol's per-tensor / per-token scale scalar.
6. **Correct Qwen2.5 forward.** Includes `model.embed_tokens`, 28× decoder layer, `model.norm` (final RMSnorm — previously missing), `lm_head` with FP8-quantized weights, argmax tail.
7. **Single graph replay.** All per-step kernel launches captured into one `cuGraphLaunch`; metadata re-uploaded per step to match real-scheduler overhead.
8. **No Python.** End-to-end Rust. No torch, no JIT, no interpreter cost on the hot path.

## The v3 stack, every layer

```
┌───────────────────────────────────────────────────────────────────┐
│  Bringup                     v3/crates/rvllm-runtime::bring_up    │
│                              one-shot: ctx → arena → weights →    │
│                              kernels → CUTLASS .so → FA3 .so →    │
│                              cuBLASLt                             │
├───────────────────────────────────────────────────────────────────┤
│  Engine / step               type-state step_launch + collect     │
│                              PendingStep<'e> borrows &mut Engine  │
│                              CUDA graph captured once per bucket  │
├───────────────────────────────────────────────────────────────────┤
│  layer_exec::forward         12 launches per Llama decoder layer  │
│                                fused_add_rmsnorm_fp8_quant        │
│                                cublasLt FP8 GEMM + bias (QKV)     │
│                                fused_rope_cache_fp8kv             │
│                                FA3 paged_decode                   │
│                                quantize_fp8_per_token             │
│                                cublasLt FP8 GEMM + residual (O)   │
│                                fused_add_rmsnorm_fp8_quant        │
│                                cublasLt FP8 GEMM (gate_up)        │
│                                fused_silu_mul_fp8_quant           │
│                                cublasLt FP8 GEMM + residual (down)│
├───────────────────────────────────────────────────────────────────┤
│  Sampling                    cublasLt FP8 GEMM (lm_head)          │
│                              argmax_kernel                         │
├───────────────────────────────────────────────────────────────────┤
│  Memory                      one HBM arena, bump-allocated regions│
│                              checkpoint/restore for sweep modes   │
│                              arena-lifetime Region<'a> graph-safe │
├───────────────────────────────────────────────────────────────────┤
│  Invariants                  type-level CUTLASS schedule pairing  │
│                              MetaLayoutHash on graph replay       │
│                              FP8 clamp-ppm gate at weight load    │
└───────────────────────────────────────────────────────────────────┘
```

## Kernels

Every kernel has a known purpose, a pinned variant, and a workspace contract. No dispatch fallback chains.

### CUTLASS SM90 FP8 GEMM (`kernels/cutlass_fp8_gemm.cu`, `cutlass_fp8_gemm_residual.cu`)

40 non-residual variants + 10 residual-fused variants, all templated on `(TileShape, ClusterShape, KernelSchedule)`. Autotuned per shape at build time into `policy.json`.

- Non-residual: QKV, gate_up, down_proj
- Residual-fused: o_proj (fused GEMM + residual add in one launch; eliminates one HBM round-trip per layer)
- Per-row activation scale, per-tensor weight scale
- **Compatibility rule enforced**: mainloop schedule must match epilogue schedule. `KernelTmaWarpSpecialized` (WS) mainloop + `TmaWarpSpecializedCooperative` (Coop) epilogue is the canonical class of SM90 illegal-memory bug. v3 makes this a compile-time `static_assert`; v2 uses variant `v1` (128×128×128 Coop/Coop matched) for o_proj residual.

### FlashAttention-3 SM90 paged decode (`kernels/fa3_sm90_wrapper.cu` → `libfa3_kernels.so`)

WGMMA + TMA, built from FlashAttention-3 Hopper source. Paged KV layout `[2, num_blocks, block_size, num_kv_heads, head_dim]`, GQA via `num_heads / num_kv_heads` ratio. `head_dim = 128` hard gate.

- **No PTX fallback.** If `libfa3_kernels.so` is not present at startup, the engine refuses to load.
- Build: `bash kernels/build_fa3.sh` on the H100 box (needs CUTLASS headers + flash-attention/hopper source), ~10 min.

### Fused pre/post kernels (PTX, `kernels/fused_*.cu`)

| Kernel | Inputs | Output | Launches saved |
|---|---|---|---|
| `embedding_gather` | token_ids, weight | f16 hidden | 1 |
| `fused_add_rmsnorm_fp8_quant` | hidden, residual, gamma | residual', fp8_act, scale | 3 → 1 |
| `fused_rmsnorm_fp8_quant` | hidden, gamma | fp8_act, scale | 2 → 1 |
| `quantize_fp8_per_token` | f16 act | fp8 + scale | 1 |
| `fused_rope_cache_fp8kv` | q/k/v f16, cos, sin, slot_mapping | q_fp8 out + writes FP8 K/V into paged cache | 4 → 1 |
| `fused_silu_mul_fp8_quant` | gate_up f16 | fp8_act + scale | 3 → 1 |
| `argmax` | f32 logits | i32 token | 1 |
| `residual_add_f16` | x, y | x + y | 1 |

Rule: each kernel fuses at most one recognizable composite. No megakernels. Every kernel has a pure-Rust f32 reference implementation in tests; PTX output must match within cosine 0.999.

## One v3 decode step, in order

For decode batch of N sequences:

```
step_launch:
  1. graph_pool.replay(bucket)                 — one cuGraphLaunch
     For each layer in 0..28:
       fused_add_rmsnorm_fp8_quant             — attn norm + quantize
       cublasLtMatmul(FP8 + BIAS epilogue)     — QKV (one shot, packed)
       fused_rope_cache_fp8kv_kernel           — rope + FP8 Q quant + FP8 KV paged write
       fa3_sm90_paged_decode                   — attention
       quantize_fp8_per_token_kernel           — quantize attn_out
       cublasLtMatmul(FP8, beta=1 residual)    — O proj + residual add
       fused_add_rmsnorm_fp8_quant             — mlp norm + quantize
       cublasLtMatmul(FP8)                     — gate_up proj
       fused_silu_mul_fp8_quant_kernel         — SiLU(gate)*up + quantize
       cublasLtMatmul(FP8, beta=1 residual)    — down proj + residual add

  2. quantize_fp8_per_token_kernel              — hidden -> fp8
     cublasLtMatmul(FP8)                        — lm_head
     argmax_kernel                              — fp16 logits -> i32 token
```

**12 launches per layer × 28 layers = 336 kernels, plus 3 at the sampling tail = 339 total, all captured into one `cuGraphLaunch`.**

No mega-kernels. Each launch does one recognizable composite. The only fused-epilogue is the one cuBLASLt gives us for free (bias / residual-add).

## Correctness discipline

Explicit rules. Violations fail the build or startup, never degrade silently.

1. **No fallbacks.** Missing autotune entry = engine panic with shape. Missing FA3 `.so` = refuse start. Missing CUTLASS .so = refuse start. Stale autotune cache from a prior deploy is not consulted — policy lives in the build artifact.
2. **Graph-capture invariant.** Metadata buffer layout is frozen per `(bucket, max_blocks_per_seq)`. Captured graphs bind those exact offsets. There is no "non-padded" upload path. Prefill and decode have separate APIs; they do not share metadata offsets.
3. **Real block-change detection.** Scheduler emits `ContinuedRequest::block_table_update: Option<Vec<BlockId>>` whenever a sequence's physical block list has grown since the last send. Worker combines this with CoW-copy events. Missing either signal leaves stale block_ids in the captured graph (wrong KV reads, not a crash — silent correctness bug).
4. **CUTLASS schedule/epilogue pairing.** Mainloop and epilogue schedules must match. Mismatched variants cause `CUDA_ERROR_ILLEGAL_ADDRESS` only inside graph replay. Enforced in v3 as a CUDA `static_assert`; in v2, dispatch pins to a hand-verified variant (`v1`).
5. **No `unwrap()` in libraries.** `Result<T, RvllmError>` end-to-end. Errors carry structured context (stream, kernel name, launch config) — not stringified `DriverError`.

## v3 path to 42,030 tok/s

Session progression on H100 SXM 80GB, Qwen2.5-7B, FP8 E4M3:

| step | tok/s | N | Δ | what changed |
|---|---:|---:|---:|---|
| eager decode, LM head in | 551 | 128 | — | no graph capture, bench harness first light |
| + graph capture | 14,745 | 128 | 27× | capture one step, replay iters-many |
| + f16 RoPE tables | 15,537 | 128 | +5% | removed f32 cos/sin from hot path |
| + QKV/bias correctness | 15,985 | 128 | +3% | load q/k/v biases, apply post-GEMM |
| + fused QKV GEMM | 15,985* | 128 | — | 3 GEMMs → 1 (Q‖K‖V packed, N=4608) |
| + cuBLASLt FP8 + BIAS epilogue (QKV) | 17,562 | 128 | +10% | one matmul replaces GEMM + add_bias kernel |
| + all 5 linears on cuBLASLt | 22,496 | 128 | +28% | O/gate_up/down/lm_head also cuBLASLt-autotuned |
| + FP8 E4M3 KV + final RMSnorm | 20,841 | 128 | −7% at N=128 | FA3 dequant overhead, but halves KV memory |
| …unlocks N=256 | 31,178 | 256 | +50% | 2× KV-memory reduction, GEMMs near HBM saturation |
| …and N=512 | 40,331 | 512 | +29% | full HBM saturation regime |
| + **real FA3 paged-prefill (Phase F)** | **22,069 / 34,364 / 42,030** | 128/256/512 | **+6–10%** | scratch sizing corrected; TTFT drops 6–12× vs faux-prefill |

The cuBLASLt move was the dominant per-token win. FP8 KV was the capacity unlock. Real FA3 prefill (Phase F) is the TTFT win: one multi-query causal FP8 attention call replaces a 16-step eager stand-in, dropping N=128 TTFT from 763 ms to 63.8 ms — and along the way the Phase F scratch-buffer fixes also lifted steady-state throughput at every batch size.

## Reproduce (v3)

```bash
# One-time on H100 box (~15 min)
bash kernels/build.sh               # fused PTX (rmsnorm, rope, silu, argmax, ...)
bash kernels/build_cutlass_so.sh    # libcutlass_kernels.so (FP8 variants)
bash kernels/build_fa3.sh           # libfa3_kernels.so

# Emit v3 manifest.json + policy.json (keyed by sha256)
python3 v3/kernels/make_manifest.py /workspace/rvllm/kernels/sm_90 sm_90 $(git rev-parse HEAD)
python3 v3/kernels/make_policy.py   /workspace/rvllm/kernels/sm_90/policy.json $(git rev-parse HEAD)

# Build v3 bench
cargo build --release --features cuda --manifest-path v3/Cargo.toml -p rvllm-bench

# Run
RVLLM_MODEL_DIR=/workspace/models/qwen25-7b-instruct \
RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90 \
RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so \
RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so \
RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json \
RVLLM_BATCH=128 RVLLM_ITERS=30 RVLLM_WARMUP=5 \
  ./v3/target/release/rvllm-bench
```

## Supported models

Tested end-to-end with CUTLASS FP8 + FA3 paged decode:

- **Qwen2** / Qwen2.5 (verified bench model)
- **Llama 2 / 3 / 3.1**
- **Mistral 7B**
- **Gemma 1 / 2**

GQA via `num_heads / num_kv_heads`. `head_dim == 128` required for FA3. Other architectures compile but have not been end-to-end validated against HF reference on this version.

Weight formats: SafeTensors (sharded + single-file). GGUF is supported in the older `rvllm-model-runner` crate but not the v2 FP8 path.

## v3 crate map

```
v3/crates/
├── rvllm-core         typed errors, IDs, dtype, shape, config, env
├── rvllm-mem          HbmArena, Region, Stream, Event, PinnedBuf, CudaContextHandle
├── rvllm-kernels      manifest (sha-pinned), PTX loader, kernel catalog
├── rvllm-fused        8 fused-kernel launchers + pure-Rust f32 references
├── rvllm-attention    FA3 SM90 paged decode/prefill dlopen
├── rvllm-cutlass      FP8 variant catalog + schedule pairing trait + cuBLASLt wrapper
├── rvllm-metadata     frozen-layout metadata per bucket (one upload path)
├── rvllm-loader       safetensors mmap -> HBM + CPU-path FP8 quant + clamp gate
├── rvllm-sampling     argmax tail, pinned DtoH
├── rvllm-graph        captured-graph pool keyed on MetaLayoutHash
├── rvllm-runtime      Engine, scheduler, layer_exec, bring_up
├── rvllm-bench        RVLLM_* env-driven bench binary
└── rvllm-invariants   DAG-dep test, no-megakernel gate
```

## Archive: rvllm-v2

The v2 engine at `crates/rvllm-v2*` (plus sibling `rvllm-gpu`, `rvllm-engine`, `rvllm-worker` etc.) is frozen at commit `eb9e247fd` (19,287 tok/s at N=128). Kept in-tree so we can port bits back if they prove useful and so the perf recovery history (`9,531 → 19,287`) is readable. Not an active build target.

The v2 CUTLASS .so build scripts in `kernels/` are still used by v3 — v3's cuBLASLt is the fast path but `libcutlass_kernels.so` compiles the same variants and is still loaded at startup (v3 resolves but currently doesn't dispatch through it; kept for the sweep harness).

## License

Apache-2.0.

## Further reading

- [`v3/SPEC.md`](v3/SPEC.md), [`v3/IMPL_PLAN.md`](v3/IMPL_PLAN.md) — the clean-slate rewrite plan, 16 focused agent specs, with CUTLASS schedule-mismatch made a compile error, one metadata upload path, `GraphSafe` marker trait enforcing no-realloc-during-capture at the type level.
- [`docs/paper/rvllm.pdf`](docs/paper/rvllm.pdf) — technical paper.
- [`docs/arch.md`](docs/arch.md) — full crate architecture.
