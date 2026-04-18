#!/bin/bash
set -e

run_bench() {
  local MODEL=$1 NAME=$2
  echo "=== $NAME FP8 ===" >> /tmp/fp8_all_status
  vllm serve "$MODEL" \
    --quantization fp8 --kv-cache-dtype fp8_e4m3 \
    --dtype bfloat16 --max-model-len 2048 --max-num-seqs 512 \
    --gpu-memory-utilization 0.85 \
    &> /tmp/vllm_${NAME}.log &
  local SPID=$!
  for i in $(seq 1 150); do
    if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "UP after $((i*2))s" >> /tmp/fp8_all_status
      python3 /tmp/bench_driver.py --model "$MODEL" --out "/tmp/bench_h100_${NAME}-fp8.json" 2>&1 >> /tmp/fp8_all_status
      echo "${NAME}_DONE" >> /tmp/fp8_all_status
      kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true
      sleep 3
      return 0
    fi
    sleep 2
  done
  echo "${NAME}_FAILED" >> /tmp/fp8_all_status
  tail -10 /tmp/vllm_${NAME}.log >> /tmp/fp8_all_status
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null || true
  sleep 3
  return 1
}

echo "starting" > /tmp/fp8_all_status
run_bench /workspace/models/qwen3-8b qwen3-8b
run_bench /workspace/models/opt-1.3b opt-1.3b
echo "ALL_DONE" >> /tmp/fp8_all_status
