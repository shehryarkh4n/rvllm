# Profiling rvLLM with nsys

nsys (NVIDIA Nsight Systems) gives kernel-level GPU timing for the full inference pipeline. This is how we identified the exact sources of the ~10% gap vs vLLM 0.19.0.

## Quick start

```bash
# On an H100 instance with rvllm built
NSYS=/opt/nvidia/nsight-compute/2025.1.0/host/target-linux-x64/nsys

# Profile a short benchmark run
$NSYS profile --stats=true --force-overwrite=true -o /root/rvllm_prof \
  ./target/release/rvllm benchmark \
  --model /root/models/Qwen2.5-7B --dtype half --n 32 --output-len 16
```

## What to look for

The `--stats=true` flag prints three tables at the end:

### 1. CUDA kernel summary (where GPU time goes)

```
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)    Name
      6.6         27899364        980    28468.7     fa3_v3_decode_gqa_kernel
      4.3         18401252       3193     5763.0     fused_residual_rmsnorm_f16_kernel
      3.0         12942583       1596     8109.4     silu_mul_interleaved_f16_kernel
      2.6         11072837       4788     2312.6     add_bias_f16_kernel
      ...
```

Sort by `Time (%)` to find the dominant kernels. For Qwen2.5-7B at N=32:
- cuBLAS GEMMs dominate (~60%)
- Attention is ~7%
- Non-GEMM fused ops (norm, activation, bias) are ~13%
- The rest is memory ops

### 2. Memory operation summary

```
 Time (%)  Total Time (ns)  Count    Operation
     97.6       1468681249    435     [CUDA memcpy Host-to-Device]
      1.2         18692416    739     [CUDA memcpy Device-to-Device]
```

**If HtoD memcpy dominates, you're uploading metadata every step instead of using pre-packed CUDA graph inputs.** This was our biggest single overhead -- 1.47 seconds of HtoD in a 4-second benchmark.

### 3. Memory size summary

```
 Total (MB)  Count  Avg (MB)  Operation
  15237.281    435    35.028   [CUDA memcpy Host-to-Device]
```

Large per-call averages (>1 MB) indicate weight or KV cache transfers. Small frequent transfers (<1 KB) indicate per-step metadata that should be pre-packed.

## Comparing with vLLM

For vLLM, profiling is harder because it spawns a subprocess (EngineCore). Use torch profiler instead:

```python
import torch
from vllm import LLM, SamplingParams

llm = LLM(model="/root/models/Qwen2.5-7B", dtype="half")
_ = llm.generate(["warm"], SamplingParams(temperature=0, max_tokens=5))

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16))
    torch.cuda.synchronize()

# Print top kernels
for e in sorted(prof.key_averages(), key=lambda e: -e.cpu_time_total)[:20]:
    print(f"{e.key[:60]:60s} {e.count:6d} {e.cpu_time_total/1000:10.2f} ms")
```

## Known bottlenecks (April 2026)

| Source | Impact | Status |
|---|---|---|
| HtoD metadata upload per step | ~5% | CUDA graph inputs need pre-packing |
| FA3 v3 attention vs FlashInfer | ~3% | 28.5 us vs 22 us per call |
| Separate add_bias kernel launches | ~2% | Should fuse into GEMM epilogue |
| Separate deinterleave_qkv kernel | ~1% | Should fuse into QKV projection |

## Extracting detailed reports

```bash
# Kernel summary
$NSYS stats --force-export=true -r cuda_gpu_kern_sum /root/rvllm_prof.nsys-rep

# Memory operation summary
$NSYS stats --force-export=true -r cuda_gpu_mem_size_sum /root/rvllm_prof.nsys-rep

# Full trace (large output)
$NSYS stats --force-export=true -r cuda_gpu_trace /root/rvllm_prof.nsys-rep
```

## Downloading profiles for GUI analysis

```bash
# Copy .nsys-rep to local machine
scp -P PORT root@HOST:/root/rvllm_prof.nsys-rep .

# Open in NVIDIA Nsight Systems GUI (free download)
# https://developer.nvidia.com/nsight-systems
```

The GUI shows a timeline view where you can see kernel overlap, stream utilization, and identify bubbles between kernel launches.

## rvLLM vs vLLM comparison pipeline

For a publishable side-by-side profile run, use:

```bash
scripts/profile_compare.sh \
  --model /root/models/Qwen2.5-7B \
  --n 1,32,64,96,128 \
  --output-len 128 \
  --profile-ns 1,32,64,96,128 \
  --profile-output-len 16
```

This produces:

- benchmark JSON for `rvLLM` and `vLLM 0.19`
- `nsys` traces for both engines
- exported kernel and memcpy summaries
- rendered comparison artifacts:
  - `rendered/profile_compare.svg`
  - `rendered/profile_compare.html`
  - `rendered/summary.json`

The pipeline uses the direct-engine benchmark path for both systems, `temperature=0.0`, and `ignore_eos=true` so the throughput and profile captures stay aligned.
