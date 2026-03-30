#!/bin/bash
set -euo pipefail

# Benchmark rvllm vs Python vLLM on SEPARATE H100 instances.
#
# CRITICAL: Each engine runs on its own GPU instance. Never run both
# on the same GPU -- shared CUDA driver state contaminates results.
#
# Usage:
#   ./deploy/vastai-benchmark.sh                    # auto-detect both instances
#   ./deploy/vastai-benchmark.sh --rvllm-only       # benchmark rvllm only
#   ./deploy/vastai-benchmark.sh --vllm-only        # benchmark vLLM only
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
VLLM_INSTANCE=${VLLM_INSTANCE:-$(cat deploy/.instance_id_vllm 2>/dev/null || echo "")}

MODE="${1:---both}"

echo "=========================================="
echo "BENCHMARK CONFIGURATION"
echo "=========================================="
echo "Model:       $MODEL"
echo "Prompts:     $NUM_PROMPTS (warmup: $WARMUP_PROMPTS)"
echo "Concurrency: $CONCURRENT"
echo "Max tokens:  $MAX_TOKENS"
echo ""

# ---------- rvllm benchmark ----------
run_rvllm_bench() {
    if [[ -z "$RVLLM_INSTANCE" ]]; then
        echo "ERROR: No rvllm instance. Set RVLLM_INSTANCE or run vastai-provision.sh --rvllm"
        return 1
    fi
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

    echo "Downloading rvllm results..."
    vastai scp $RVLLM_INSTANCE /root/results_rvllm.json deploy/results_rvllm.json 2>/dev/null || true
}

# ---------- vLLM benchmark ----------
run_vllm_bench() {
    if [[ -z "$VLLM_INSTANCE" ]]; then
        echo "ERROR: No vLLM instance. Set VLLM_INSTANCE or run vastai-provision.sh --vllm"
        return 1
    fi
    echo "=========================================="
    echo "BENCHMARK: Python vLLM (instance $VLLM_INSTANCE)"
    echo "=========================================="

    vastai ssh $VLLM_INSTANCE << REMOTE
set -euo pipefail

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "api_server" 2>/dev/null || true
sleep 3

echo "Starting Python vLLM server..."
STARTUP_START=\$(date +%s%N)
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --gpu-memory-utilization 0.90 \
    --port 8001 > /tmp/vllm_bench.log 2>&1 &
PID=\$!

for i in \$(seq 1 120); do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then break; fi
    sleep 2
done
STARTUP_END=\$(date +%s%N)
STARTUP_MS=\$(( (STARTUP_END - STARTUP_START) / 1000000 ))

if ! curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "FAIL: vLLM did not start"
    tail -30 /tmp/vllm_bench.log
    kill \$PID 2>/dev/null || true
    exit 1
fi
echo "vLLM ready (startup: \${STARTUP_MS}ms, PID \$PID)"

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "--- Warmup ($WARMUP_PROMPTS prompts) ---"
python3 /root/rvllm/deploy/benchmark_client.py --url http://localhost:8001 --num-prompts $WARMUP_PROMPTS --concurrent 16 --max-tokens 128 --output /tmp/vllm_warmup.json 2>&1 | grep -E "Throughput|Errors"
sleep 2

echo ""
echo "--- Full benchmark ($NUM_PROMPTS prompts, concurrency=$CONCURRENT, max_tokens=$MAX_TOKENS) ---"
python3 /root/rvllm/deploy/benchmark_client.py --url http://localhost:8001 --num-prompts $NUM_PROMPTS --concurrent $CONCURRENT --max-tokens $MAX_TOKENS --output /root/results_vllm.json

echo ""
echo "Server errors: \$(grep -c ERROR /tmp/vllm_bench.log 2>/dev/null || echo 0)"

kill \$PID 2>/dev/null || true
wait \$PID 2>/dev/null || true
echo "=== vLLM benchmark done ==="
REMOTE

    echo "Downloading vLLM results..."
    vastai scp $VLLM_INSTANCE /root/results_vllm.json deploy/results_vllm.json 2>/dev/null || true
}

# ---------- Run ----------
if [[ "$MODE" == "--rvllm-only" ]]; then
    run_rvllm_bench
elif [[ "$MODE" == "--vllm-only" ]]; then
    run_vllm_bench
else
    run_rvllm_bench
    echo ""
    run_vllm_bench
fi

# ---------- Compare ----------
if [[ -f deploy/results_rvllm.json && -f deploy/results_vllm.json ]]; then
    echo ""
    echo "=========================================="
    echo "COMPARISON"
    echo "=========================================="
    python3 deploy/compare_results.py \
        --rust deploy/results_rvllm.json \
        --python deploy/results_vllm.json 2>/dev/null || \
    echo "(compare_results.py not found or failed -- check results manually)"
fi

echo ""
echo "Results saved to deploy/results_rvllm.json and deploy/results_vllm.json"
