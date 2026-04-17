# rvLLM

A single-GPU, FP8, graph-captured LLM inference engine in Rust. Qwen2.5-7B on a single H100 SXM at **19,287 tok/s** at N=128, 512 output tokens, greedy, CUDA graphs on.

No Python in the hot path. No fallbacks. Missing artifacts (autotune policy, FA3 `.so`, kernel SHA) refuse to start.

Measured April 16, 2026 on commit `eb9e247fd`, rvllm-v2-bench direct engine, FP8 E4M3, CUDA 12.4. Reproducible from source — see *Reproduce* below.

## Current throughput

H100 SXM 80GB, Qwen2.5-7B, FP8 E4M3, 512 output tokens, greedy, graphs captured, FA3 `.so` loaded:

| N | tok/s | stdev |
|---:|---:|---:|
| 1 | 195.2 | 0.0 |
| 4 | 770.2 | 0.7 |
| 8 | 1,529.6 | 0.8 |
| 16 | 3,011.8 | 0.6 |
| 32 | 5,935.0 | 0.4 |
| 64 | 11,064.3 | 1.6 |
| **128** | **19,287.5** | **38.5** |

## The stack, every layer

```
┌───────────────────────────────────────────────────────────────────┐
│  HTTP / OpenAI API             rvllm-api (axum)                   │
├───────────────────────────────────────────────────────────────────┤
│  Engine                        rvllm-v2 Engine                    │
│                                step_pipelined() = launch + collect│
│                                double-buffered pinned argmax DtoH │
├───────────────────────────────────────────────────────────────────┤
│  Scheduler                     continuous batching                │
│                                paged KV (block_size=64)           │
│                                page-boundary growth signal        │
├───────────────────────────────────────────────────────────────────┤
│  Worker                        CUDA graph pool                    │
│                                35 pre-captured batch buckets      │
│                                1 compute stream per worker        │
├───────────────────────────────────────────────────────────────────┤
│  Runner                        1 HBM arena, pre-allocated slab    │
│                                packed metadata (1 HtoD / step)    │
├───────────────────────────────────────────────────────────────────┤
│  Layer                         11 launches per layer (decode)     │
├───────────────────────────────────────────────────────────────────┤
│  Kernels                       CUTLASS 3.x SM90 FP8 GEMM          │
│                                FlashAttention-3 SM90 paged decode │
│                                custom fused PTX (norm/silu/rope)  │
│                                cuBLAS HGEMM for LM head           │
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
| `fused_rope_kv_write` | qkv, cos, sin, slot_mapping | q_out, writes K/V into cache | 3 → 1 |
| `fused_silu_mul_fp8_quant` | gate_up f16 | fp8_act + scale | 3 → 1 |
| `argmax` | f32 logits | i32 token | 1 |
| `residual_add_f16` | x, y | x + y | 1 |

Rule: each kernel fuses at most one recognizable composite. No megakernels. Every kernel has a pure-Rust f32 reference implementation in tests; PTX output must match within cosine 0.999.

## One decode step, in order

For decode batch of N sequences:

```
step_launch(diff):
  1. schedule(diff)                          — CPU, O(N), no allocs
  2. metadata pack + 1× HtoD                 — positions, context_lens,
                                               block_tables, slot_mapping
                                               into one packed i32 buffer
  3. graph_pool.replay(bucket = pad_up(N))   — one cuGraphLaunch
       ├─ embedding_gather                    — 1 kernel
       ├─ for layer in 0..28:                 — 11 kernels × 28 layers
       │    fused_add_rmsnorm_fp8_quant
       │    CUTLASS FP8 GEMM (QKV)
       │    fused_rope_kv_write
       │    FA3 paged_decode
       │    quantize_fp8_per_token (post-attn)
       │    CUTLASS FP8 GEMM + residual (o_proj, v1)
       │    fused_rmsnorm_fp8_quant
       │    CUTLASS FP8 GEMM (gate_up)
       │    fused_silu_mul_fp8_quant
       │    CUTLASS FP8 GEMM (down_proj)
       │    residual_add_f16
       ├─ fused_residual_rmsnorm (final)      — 1 kernel
       ├─ cuBLAS HGEMM f32→lm_head            — 1 kernel
       └─ argmax                              — 1 kernel
  4. cuMemcpyDtoHAsync argmax → pinned[w_idx] — 512 B at N=128
     cuEventRecord event[w_idx]
  5. return; w_idx ^= 1                       — double buffer flip

step_collect():
  1. cuEventSynchronize event[r_idx]          — waits for step-1 DtoH
  2. read pinned[r_idx][0..N]                 — CPU reads new tokens
  3. scheduler.commit()                        — mark tokens, free finished
```

Total GPU kernels per decode step: embed (1) + layer ops (28 × 11 = 308) + final norm (1) + LM head (1) + argmax (1) = **311 launches**, all captured into **1 `cuGraphLaunch`**.

Total DtoH per step: **one** 4·N byte copy for the argmax'd token IDs. Nothing else.
Total HtoD per step: **one** packed metadata buffer (<100 KB at N=128, max_blocks=129).

## Correctness discipline

Explicit rules. Violations fail the build or startup, never degrade silently.

1. **No fallbacks.** Missing autotune entry = engine panic with shape. Missing FA3 `.so` = refuse start. Missing CUTLASS .so = refuse start. Stale autotune cache from a prior deploy is not consulted — policy lives in the build artifact.
2. **Graph-capture invariant.** Metadata buffer layout is frozen per `(bucket, max_blocks_per_seq)`. Captured graphs bind those exact offsets. There is no "non-padded" upload path. Prefill and decode have separate APIs; they do not share metadata offsets.
3. **Real block-change detection.** Scheduler emits `ContinuedRequest::block_table_update: Option<Vec<BlockId>>` whenever a sequence's physical block list has grown since the last send. Worker combines this with CoW-copy events. Missing either signal leaves stale block_ids in the captured graph (wrong KV reads, not a crash — silent correctness bug).
4. **CUTLASS schedule/epilogue pairing.** Mainloop and epilogue schedules must match. Mismatched variants cause `CUDA_ERROR_ILLEGAL_ADDRESS` only inside graph replay. Enforced in v3 as a CUDA `static_assert`; in v2, dispatch pins to a hand-verified variant (`v1`).
5. **No `unwrap()` in libraries.** `Result<T, RvllmError>` end-to-end. Errors carry structured context (stream, kernel name, launch config) — not stringified `DriverError`.

## Measured recovery

One-day root-cause pass on April 16, 2026 took v2 from 9,531 → 19,287 tok/s at N=128 (+102%). Three commits, all structural, none were stopgaps:

| commit | fix | N=128 gain |
|---|---|---|
| `6dabb76a5` | `last_padded_batch` tracking: prefill's non-padded metadata no longer overwrites decode's padded layout → captured graph reads correct offsets | crash → works |
| `31716269d` | Re-enable fused FP8 o_proj + residual with variant `v1` (Coop/Coop matched); revert the earlier "disable all variants" stopgap | +83% |
| `eb9e247fd` | Restore `patch_metadata_decode` fast path with **real** block-change detection via `ContinuedRequest::block_table_update` (catches page-boundary growth, not just CoW) | +10% |

Plus `libfa3_kernels.so` built on the box (23 MB, once) — eliminates the silent .ptx fallback.

## Reproduce

```bash
# One-shot: rent an H100 on vast.ai, build from source, run bench, land JSON locally.
VASTAI_API_KEY=<your_key> ./race.sh
```

Manual:

```bash
# Build kernels on an H100 box (first time only, ~15 min)
bash kernels/build.sh              # custom PTX
bash kernels/build_cutlass_so.sh   # CUTLASS .so
bash kernels/build_fa3.sh          # FA3 .so

# Populate autotune policy
./target/release/autotune-cutlass  # writes /root/.cache/rvllm/cutlass_autotune.json

# Run bench
cargo build --release --features cuda-graphs --bin rvllm-v2-bench
./target/release/rvllm-v2-bench \
  --model Qwen/Qwen2.5-7B \
  --fp8 \
  --n "1,4,8,16,32,64,128" \
  --output-len 512 \
  --iters 3
```

## Supported models

Tested end-to-end with CUTLASS FP8 + FA3 paged decode:

- **Qwen2** / Qwen2.5 (verified bench model)
- **Llama 2 / 3 / 3.1**
- **Mistral 7B**
- **Gemma 1 / 2**

GQA via `num_heads / num_kv_heads`. `head_dim == 128` required for FA3. Other architectures compile but have not been end-to-end validated against HF reference on this version.

Weight formats: SafeTensors (sharded + single-file). GGUF is supported in the older `rvllm-model-runner` crate but not the v2 FP8 path.

## Crate map (hot path)

```
rvllm-api         HTTP/OpenAI routes
  └── rvllm-engine / rvllm-v2::engine   step_pipelined()
       └── rvllm-worker / rvllm-v2::worker
            ├── rvllm-scheduler    request state, paged KV, preemption
            ├── rvllm-v2::runner   HBM arena, packed metadata, layer dispatch
            │    └── rvllm-v2::layer
            │         ├── rvllm-gpu::cutlass_ffi   FP8 GEMM variants + residual
            │         ├── rvllm-gpu::fa3_ffi        libfa3_kernels.so bindings
            │         └── kernel_loader             PTX/.so loader
            └── rvllm-gpu::cuda_graph  GraphPool, bucket capture
```

Full crate table for multimodal, embeddings, speculative decode, gRPC, telemetry, Python bindings: see [`docs/arch.md`](docs/arch.md).

## License

Apache-2.0.

## Further reading

- [`v3/SPEC.md`](v3/SPEC.md), [`v3/IMPL_PLAN.md`](v3/IMPL_PLAN.md) — the clean-slate rewrite plan, 16 focused agent specs, with CUTLASS schedule-mismatch made a compile error, one metadata upload path, `GraphSafe` marker trait enforcing no-realloc-during-capture at the type level.
- [`docs/paper/rvllm.pdf`](docs/paper/rvllm.pdf) — technical paper.
- [`docs/arch.md`](docs/arch.md) — full crate architecture.
