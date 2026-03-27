#!/usr/bin/env bash
# rvLLM Benchmark Script
# One-shot: builds from source, runs server, benchmarks, reports results.
#
# Usage:
#   # On a fresh vast.ai instance with CUDA:
#   curl -sSL https://raw.githubusercontent.com/m0at/rvllm/main/bench/run.sh | bash
#
#   # Or locally:
#   bash bench/run.sh
#
# Requirements: CUDA GPU, Rust toolchain, ~10GB disk, ~8GB GPU VRAM
# Tested on: A100 80GB SXM4 with Qwen2.5-1.5B

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
PORT="${PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-32}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 4}"

BASE_URL="http://localhost:${PORT}"
RESULTS_FILE="/tmp/rvllm_bench_results.txt"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[bench]${NC} $*"; }
ok()  { echo -e "${GREEN}[ok]${NC} $*"; }
err() { echo -e "${RED}[err]${NC} $*" >&2; }

# --- Step 0: Check prerequisites ---
log "Checking prerequisites..."
command -v cargo >/dev/null 2>&1 || {
    log "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
}
command -v nvcc >/dev/null 2>&1 || { err "nvcc not found. CUDA toolkit required."; exit 1; }
command -v curl >/dev/null 2>&1 || { err "curl not found."; exit 1; }
nvidia-smi >/dev/null 2>&1 || { err "nvidia-smi failed. No GPU?"; exit 1; }

# --- Step 1: Build ---
log "Building rvllm-server with CUDA support..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/.."
if [ ! -f "${REPO_DIR}/Cargo.toml" ]; then
    log "Cloning repository..."
    REPO_DIR="/tmp/rvllm"
    git clone https://github.com/m0at/rvllm.git "$REPO_DIR" 2>/dev/null || true
fi
cd "$REPO_DIR"

BUILD_START=$(date +%s%N)
cargo build --release --features cuda -p rvllm-server 2>&1 | tail -3
BUILD_END=$(date +%s%N)
BUILD_TIME_MS=$(( (BUILD_END - BUILD_START) / 1000000 ))
log "Build time: ${BUILD_TIME_MS}ms"

BINARY="./target/release/rvllm"
BINARY_SIZE=$(ls -lh "$BINARY" | awk '{print $5}')
log "Binary size: ${BINARY_SIZE}"

# --- Step 2: Start server ---
log "Starting rvllm server (model=${MODEL})..."
pkill -9 rvllm 2>/dev/null || true
sleep 1

SERVER_START=$(date +%s%N)
nohup "$BINARY" serve --model "$MODEL" --port "$PORT" > /tmp/rvllm_server.log 2>&1 &
SERVER_PID=$!

cleanup() {
    log "Cleaning up..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for server to be ready
for i in $(seq 1 60); do
    if curl -s "${BASE_URL}/health" | grep -q "ok"; then
        break
    fi
    if [ "$i" -eq 60 ]; then
        err "Server failed to start in 60s"
        cat /tmp/rvllm_server.log
        exit 1
    fi
    sleep 1
done
SERVER_END=$(date +%s%N)
STARTUP_MS=$(( (SERVER_END - SERVER_START) / 1000000 ))
ok "Server ready in ${STARTUP_MS}ms"

# --- Step 3: Collect static metrics ---
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CPU_RSS_KB=$(ps -o rss= -p "$SERVER_PID" 2>/dev/null || echo "0")
CPU_RSS_MB=$(( CPU_RSS_KB / 1024 ))

log "GPU: ${GPU_NAME}"
log "GPU memory: ${GPU_MEM} MiB"
log "CPU RSS: ${CPU_RSS_MB} MB"

# --- Step 4: Quality check (single request) ---
log "Running quality check..."
QUALITY_RESP=$(curl -s -X POST "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"The capital of France is\",\"max_tokens\":20,\"temperature\":0.0}")
QUALITY_TEXT=$(echo "$QUALITY_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "PARSE_ERROR")
ok "Quality check: ${QUALITY_TEXT}"

# --- Step 5: Latency benchmark ---
PROMPTS=(
    "The capital of France is"
    "Explain quantum computing:"
    "Write a Python sort function:"
    "The theory of relativity"
    "Artificial intelligence in 2024"
    "The speed of light is"
    "A binary search algorithm"
    "The Fibonacci sequence"
    "Machine learning models"
    "The periodic table contains"
    "Rust programming language"
    "Neural networks learn by"
    "The sun is approximately"
    "HTTP status code 404"
    "A linked list is"
    "The Pythagorean theorem"
)

for CONC in $CONCURRENCY_LEVELS; do
    log "Running benchmark: concurrency=${CONC}, prompts=${NUM_PROMPTS}, max_tokens=${MAX_TOKENS}"

    BATCH_START=$(date +%s%N)
    TOTAL_TOKENS=0
    PIDS=()
    TMPDIR_BENCH=$(mktemp -d)

    for i in $(seq 0 $((NUM_PROMPTS - 1))); do
        PROMPT="${PROMPTS[$((i % ${#PROMPTS[@]}))]}"
        (
            START=$(date +%s%N)
            RESP=$(curl -s -X POST "${BASE_URL}/v1/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"max_tokens\":${MAX_TOKENS},\"temperature\":0.7}" \
                --max-time 120)
            END=$(date +%s%N)
            ELAPSED_MS=$(( (END - START) / 1000000 ))
            TOKENS=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
            echo "${ELAPSED_MS} ${TOKENS}" > "${TMPDIR_BENCH}/result_${i}.txt"
        ) &
        PIDS+=($!)

        # Throttle concurrency
        if [ "${#PIDS[@]}" -ge "$CONC" ]; then
            wait "${PIDS[0]}"
            PIDS=("${PIDS[@]:1}")
        fi
    done
    # Wait for remaining
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    BATCH_END=$(date +%s%N)
    WALL_MS=$(( (BATCH_END - BATCH_START) / 1000000 ))

    # Collect results
    LATENCIES=()
    TOTAL_TOKENS=0
    for f in "${TMPDIR_BENCH}"/result_*.txt; do
        read -r lat tok < "$f"
        LATENCIES+=("$lat")
        TOTAL_TOKENS=$((TOTAL_TOKENS + tok))
    done
    rm -rf "$TMPDIR_BENCH"

    # Sort latencies for percentiles
    SORTED=($(printf '%s\n' "${LATENCIES[@]}" | sort -n))
    P50_IDX=$(( NUM_PROMPTS / 2 ))
    P95_IDX=$(( NUM_PROMPTS * 95 / 100 ))
    P50="${SORTED[$P50_IDX]}"
    P95="${SORTED[$P95_IDX]}"
    AVG=$(( $(printf '%s+' "${LATENCIES[@]}" | sed 's/+$//') ))
    AVG=$(( AVG / NUM_PROMPTS ))
    TOKS_PER_SEC=0
    if [ "$WALL_MS" -gt 0 ]; then
        TOKS_PER_SEC=$(( TOTAL_TOKENS * 1000 / WALL_MS ))
    fi

    echo ""
    echo "=== Results: concurrency=${CONC} ==="
    echo "  Prompts:      ${NUM_PROMPTS}"
    echo "  Max tokens:   ${MAX_TOKENS}"
    echo "  Wall time:    ${WALL_MS} ms"
    echo "  Total tokens: ${TOTAL_TOKENS}"
    echo "  Throughput:   ${TOKS_PER_SEC} tok/s"
    echo "  Avg latency:  ${AVG} ms"
    echo "  P50 latency:  ${P50} ms"
    echo "  P95 latency:  ${P95} ms"
    echo ""
done

# --- Step 6: Summary ---
echo ""
echo "========================================"
echo "  rvLLM Benchmark Summary"
echo "========================================"
echo "  Model:        ${MODEL}"
echo "  GPU:          ${GPU_NAME}"
echo "  Binary size:  ${BINARY_SIZE}"
echo "  Startup time: ${STARTUP_MS} ms"
echo "  GPU memory:   ${GPU_MEM} MiB"
echo "  CPU RSS:      ${CPU_RSS_MB} MB"
echo "  Quality:      ${QUALITY_TEXT}"
echo "========================================"

# Cleanup
kill "$SERVER_PID" 2>/dev/null || true
log "Done."
