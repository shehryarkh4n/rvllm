#!/bin/bash
set -euo pipefail

# Benchmark rvllm on a vast.ai H100 instance.
#
# Usage:
#   ./deploy/vastai-benchmark.sh
#
# Environment:
#   MODEL          model name (default: Qwen/Qwen2.5-7B)
#   NUM_PROMPTS    total prompts (default: 500)
#   CONCURRENT     concurrency level (default: 32)
#   MAX_TOKENS     max output tokens (default: 256)

MODEL=${MODEL:-"Qwen/Qwen2.5-7B"}
NUM_PROMPTS=${NUM_PROMPTS:-500}
CONCURRENT=${CONCURRENT:-32}
MAX_TOKENS=${MAX_TOKENS:-256}
WARMUP_PROMPTS=${WARMUP_PROMPTS:-100}

RVLLM_INSTANCE=${RVLLM_INSTANCE:-$(cat deploy/.instance_id_rvllm 2>/dev/null || echo "")}

if [[ -z "$RVLLM_INSTANCE" ]]; then
    echo "ERROR: No rvllm instance. Set RVLLM_INSTANCE or run vastai-provision.sh"
    exit 1
fi

echo "=========================================="
echo "BENCHMARK CONFIGURATION"
echo "=========================================="
echo "Instance:    $RVLLM_INSTANCE"
echo "Model:       $MODEL"
echo "Prompts:     $NUM_PROMPTS (warmup: $WARMUP_PROMPTS)"
echo "Concurrency: $CONCURRENT"
echo "Max tokens:  $MAX_TOKENS"
echo ""

echo "=========================================="
echo "BENCHMARK: rvllm (instance $RVLLM_INSTANCE)"
echo "=========================================="

vastai ssh $RVLLM_INSTANCE << REMOTE
set -euo pipefail
export PATH="/root/.cargo/bin:\$PATH"
cd /root/rvllm

pkill -9 -f "rvllm serve" 2>/dev/null || true
sleep 2

echo "Starting rvllm server..."
STARTUP_START=\$(date +%s%N)
target/release/rvllm serve --model $MODEL --gpu-memory-utilization 0.90 --port 8000 > /tmp/rvllm_bench.log 2>&1 &
PID=\$!

for i in \$(seq 1 120); do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then break; fi
    sleep 2
done
STARTUP_END=\$(date +%s%N)
STARTUP_MS=\$(( (STARTUP_END - STARTUP_START) / 1000000 ))

if ! curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "FAIL: rvllm did not start"
    tail -30 /tmp/rvllm_bench.log
    kill \$PID 2>/dev/null || true
    exit 1
fi
echo "rvllm ready (startup: \${STARTUP_MS}ms, PID \$PID)"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "--- Warmup ($WARMUP_PROMPTS prompts) ---"
python3 deploy/benchmark_client.py --url http://localhost:8000 --num-prompts $WARMUP_PROMPTS --concurrent 16 --max-tokens 128 --output /tmp/rvllm_warmup.json 2>&1 | grep -E "Throughput|Errors"
sleep 2

echo ""
echo "--- Full benchmark ($NUM_PROMPTS prompts, concurrency=$CONCURRENT, max_tokens=$MAX_TOKENS) ---"
python3 deploy/benchmark_client.py --url http://localhost:8000 --num-prompts $NUM_PROMPTS --concurrent $CONCURRENT --max-tokens $MAX_TOKENS --output /root/results_rvllm.json

echo ""
echo "Server errors: \$(grep -c ERROR /tmp/rvllm_bench.log 2>/dev/null || echo 0)"

kill \$PID 2>/dev/null || true
wait \$PID 2>/dev/null || true
echo "=== rvllm benchmark done ==="
REMOTE

echo "Downloading results..."
vastai scp $RVLLM_INSTANCE /root/results_rvllm.json deploy/results_rvllm.json 2>/dev/null || true

echo ""
echo "Results saved to deploy/results_rvllm.json"
