#!/usr/bin/env bash
# vLLM tok/s + TTFT bench on H100 for multiple models.
# Usage: bash bench_h100_vllm.sh [model_name]
# Models: qwen (default), gemma, opt
set -euo pipefail

case "${1:-qwen}" in
  qwen)  MODEL="/workspace/models/qwen25-7b-instruct" ;;
  gemma) MODEL="google/gemma-7b" ;;
  opt)   MODEL="facebook/opt-1.3b" ;;
  *)     MODEL="$1" ;;
esac

MODEL_SHORT=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
OUT="/tmp/bench_h100_${MODEL_SHORT}.json"
SEQ_IN=16
SEQ_OUT=512
PORT=8000

pkill -f "vllm serve" 2>/dev/null || true
sleep 2

echo ">> model: $MODEL"
echo ">> launching vLLM server (BF16 + FP8 dynamic quant)..."

# FP8 for 7B+ models, BF16-only for small models
if [[ "$MODEL_SHORT" == *"opt"* ]]; then
  # OPT-1.3B: too small for FP8 to matter, run BF16
  nohup vllm serve "$MODEL" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    > /tmp/vllm_${MODEL_SHORT}.log 2>&1 &
else
  nohup vllm serve "$MODEL" \
    --quantization fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    > /tmp/vllm_${MODEL_SHORT}.log 2>&1 &
fi
SERVER_PID=$!

echo ">> waiting for server (up to 300s)..."
for i in $(seq 1 150); do
  if curl -s http://localhost:${PORT}/v1/models >/dev/null 2>&1; then
    echo ">> server up after $((i*2))s"
    break
  fi
  sleep 2
done

if ! curl -s http://localhost:${PORT}/v1/models >/dev/null 2>&1; then
  echo ">> server failed. last 40 lines:"
  tail -40 /tmp/vllm_${MODEL_SHORT}.log
  kill $SERVER_PID 2>/dev/null || true
  exit 1
fi

# Use bench_driver.py if available, else inline bench
if [ -f /tmp/bench_driver.py ]; then
  python3 /tmp/bench_driver.py --model "$MODEL" --out "$OUT"
else
  # Inline: use vllm bench serve
  results='{"model":"'"$MODEL"'","device":"H100","runs":['
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
          --ignore-eos 2>&1 | tail -60)
      echo "$raw" > /tmp/bench_${MODEL_SHORT}_N${N}_${run}.log
      toks=$(echo "$raw" | grep -iE 'Output token throughput' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
      ttft=$(echo "$raw" | grep -iE 'Mean TTFT' | grep -Eo '[0-9]+\.[0-9]+' | head -1)
      [ -z "$first" ] && results="$results,"
      first=
      results="$results{\"n\":$N,\"phase\":\"$run\",\"toks\":${toks:-null},\"ttft_ms\":${ttft:-null}}"
      echo "   toks=${toks:-NA}  ttft=${ttft:-NA}"
    done
  done
  results="$results]}"
  echo "$results" > "$OUT"
fi

echo ">> wrote $OUT"
kill $SERVER_PID 2>/dev/null || true
wait 2>/dev/null || true
echo ">> done: $MODEL_SHORT"
