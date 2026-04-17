#!/usr/bin/env bash
# End-to-end tok/s + TTFT bench on TPU using vLLM 0.6.6.post1 (torch-xla 2.5).
# Mirrors the H100 methodology in docs/bench.html:
#   Qwen2.5-7B-Instruct, 16 in / 512 out, N in {1,8,16,64,128,256,512},
#   two runs per N (cold then hot).
# Differences (unavoidable on v6e):
#   - bf16 weights + bf16 KV (no StableHLO FP8 op yet)
#   - vllm 0.6.6 (TPU-compatible) vs vllm 0.19 on H100
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
MODEL="Qwen/Qwen2.5-7B-Instruct"
OUT=/tmp/tpu_e2e_bench.json

echo ">> vllm:" ; pip show vllm 2>/dev/null | head -2 | tail -1
echo ">> torch-xla:" ; python3 -c 'import torch_xla; print(torch_xla.__version__)' 2>&1 | tail -1
echo ">> aiohttp:" ; pip install --quiet aiohttp 2>&1 | tail -1 || true

echo ">> launching server"
rm -f /tmp/vllm.log
nohup vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --device tpu \
    > /tmp/vllm.log 2>&1 &
SERVER_PID=$!

echo ">> waiting for server (up to 1200s)"
for i in $(seq 1 600); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo ">> server up after $((i*2))s"
    break
  fi
  sleep 2
done

if ! curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
  echo ">> server never came up. last 60 log lines:"
  tail -60 /tmp/vllm.log
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

python3 /tmp/bench_driver.py --model "$MODEL" --out "$OUT"

kill $SERVER_PID 2>/dev/null || true
wait 2>/dev/null || true
echo ">> done, results at $OUT"
