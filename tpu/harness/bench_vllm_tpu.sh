#!/usr/bin/env bash
# End-to-end tok/s + TTFT bench on TPU using vLLM's TPU backend.
# Mirrors the H100 bench methodology in docs/bench.html:
#   - Qwen2.5-7B-Instruct
#   - 16 input tokens / 512 output tokens
#   - N in {1, 8, 16, 64, 128, 256, 512}
#   - two runs per N (first = cold, second = hot), report hot
# Differences (all unavoidable on v6e):
#   - bf16 weights + bf16 KV  (TPU has no FP8 StableHLO op yet)
#   - dense (no quantization)
#
# Runs directly on the TPU VM. Emits a single JSON to /tmp/tpu_e2e_bench.json.
set -euo pipefail

MODEL="Qwen/Qwen2.5-7B-Instruct"
SEQ_IN=16
SEQ_OUT=512
OUT=/tmp/tpu_e2e_bench.json

echo ">> installing vllm[tpu]"
pip install --quiet -U \
    "vllm" --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torch-xla 2>&1 | tail -3 || true
# vLLM TPU path uses torch-xla; pinning via the tpu extras.
pip install --quiet -U "vllm" || true

echo ">> launching server"
rm -f /tmp/vllm.log
nohup vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    > /tmp/vllm.log 2>&1 &
SERVER_PID=$!

# wait for server
for i in $(seq 1 180); do
  if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo ">> server up after ${i}s"; break
  fi
  sleep 2
done

results='{"model":"'"$MODEL"'","runs":['
first=1
for N in 1 8 16 64 128 256 512; do
  for run in cold hot; do
    echo ">> bench N=$N $run"
    raw=$(vllm bench serve \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts $N \
        --max-concurrency $N \
        --random-input-len $SEQ_IN \
        --random-output-len $SEQ_OUT \
        --ignore-eos 2>&1 | tail -40)
    echo "$raw" > /tmp/bench_N${N}_${run}.log
    toks=$(echo "$raw" | grep -Eo 'Output token throughput.*[0-9]+\.?[0-9]*' | grep -Eo '[0-9]+\.?[0-9]*$' | head -1)
    ttft=$(echo "$raw" | grep -Eo 'Mean TTFT.*[0-9]+\.?[0-9]*' | grep -Eo '[0-9]+\.?[0-9]*$' | head -1)
    [ -z "$first" ] && results="$results,"
    first=
    results="$results{\"n\":$N,\"phase\":\"$run\",\"toks\":${toks:-null},\"ttft_ms\":${ttft:-null}}"
  done
done
results="$results]}"
echo "$results" > "$OUT"
echo ">> wrote $OUT"

kill $SERVER_PID 2>/dev/null || true
wait 2>/dev/null || true
