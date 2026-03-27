#!/usr/bin/env bash
# rvLLM vs vLLM -- side-by-side streaming inference demo
#
# Records a tmux split-pane session where identical prompts are sent to
# rvLLM (port 8000) and Python vLLM (port 8001) simultaneously.
# Output: bench/video/rvllm-vs-vllm.cast  (asciinema)
#         bench/video/rvllm-vs-vllm.mp4   (if agg+ffmpeg available)
#
# Usage:
#   bash bench/video_demo.sh                   # default
#   bash bench/video_demo.sh --skip-servers    # servers already running
#   bash bench/video_demo.sh --prompts 5       # number of prompts

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_DIR="${SCRIPT_DIR}/video"
CAST_FILE="${VIDEO_DIR}/rvllm-vs-vllm.cast"
GIF_FILE="/tmp/rvllm-vs-vllm.gif"
MP4_FILE="${VIDEO_DIR}/rvllm-vs-vllm.mp4"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B}"
RVLLM_PORT="${RVLLM_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8001}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"
SKIP_SERVERS=0
TMUX_SESSION="rvllm-demo"

RVLLM_PID=""
VLLM_PID=""

# ── Parse args ────────────────────────────────────────────────────────────
shift_next=0
for arg in "$@"; do
    if [ "$shift_next" = "1" ]; then
        NUM_PROMPTS="$arg"
        shift_next=0
        continue
    fi
    case "$arg" in
        --skip-servers) SKIP_SERVERS=1 ;;
        --prompts)      shift_next=1 ;;
    esac
done

# ── Prompts ───────────────────────────────────────────────────────────────
PROMPTS=(
    "Explain quantum entanglement in simple terms."
    "Write a short Python function to compute the Fibonacci sequence."
    "What are the main differences between Rust and C++?"
    "Describe the process of photosynthesis step by step."
    "Write a haiku about machine learning."
)

# ── Cleanup on exit ──────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[cleanup] Shutting down..."
    [ -n "$RVLLM_PID" ] && kill "$RVLLM_PID" 2>/dev/null && echo "[cleanup] Killed rvLLM (PID $RVLLM_PID)"
    [ -n "$VLLM_PID" ]  && kill "$VLLM_PID"  2>/dev/null && echo "[cleanup] Killed vLLM  (PID $VLLM_PID)"
    tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
    wait 2>/dev/null
    echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

# ── Preflight ─────────────────────────────────────────────────────────────
for cmd in tmux asciinema curl python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first."
        exit 1
    fi
done

mkdir -p "$VIDEO_DIR"

# ── Helper: wait for server health ────────────────────────────────────────
wait_healthy() {
    local url="$1" name="$2" timeout="${3:-120}"
    echo -n "[wait] $name on $url "
    local elapsed=0
    while ! curl -sf "${url}/v1/models" >/dev/null 2>&1; do
        if [ "$elapsed" -ge "$timeout" ]; then
            echo " TIMEOUT after ${timeout}s"
            exit 1
        fi
        echo -n "."
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo " ready (${elapsed}s)"
}

# ── Step 1: Start servers ────────────────────────────────────────────────
if [ "$SKIP_SERVERS" = "0" ]; then
    echo "============================================"
    echo "  rvLLM vs vLLM  --  Video Demo"
    echo "  Model: ${MODEL}"
    echo "============================================"
    echo ""

    # Start rvLLM
    echo "[start] rvLLM on port ${RVLLM_PORT}..."
    "${REPO_ROOT}/target/release/rvllm" serve \
        --model "$MODEL" --port "$RVLLM_PORT" \
        > /tmp/rvllm-demo.log 2>&1 &
    RVLLM_PID=$!

    # Start vLLM
    echo "[start] vLLM on port ${VLLM_PORT}..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --port "$VLLM_PORT" \
        > /tmp/vllm-demo.log 2>&1 &
    VLLM_PID=$!

    wait_healthy "http://localhost:${RVLLM_PORT}" "rvLLM" 120
    wait_healthy "http://localhost:${VLLM_PORT}" "vLLM"  180
else
    echo "[skip] Assuming servers already running on ports ${RVLLM_PORT} and ${VLLM_PORT}"
fi

# ── Step 2: Write the SSE token extractor ─────────────────────────────────
SSE_PARSER="/tmp/rvllm-sse-parse.py"
cat > "$SSE_PARSER" << 'PYEOF'
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith("data: "):
        line = line[6:]
    if line == "[DONE]" or not line:
        continue
    try:
        d = json.loads(line)
        t = d.get("choices", [{}])[0].get("text", "")
        if t:
            print(t, end="", flush=True)
    except Exception:
        pass
print()
PYEOF

# ── Step 3: Build the curl command for a pane ────────────────────────────
build_pane_cmd() {
    local port="$1" prompt="$2" label="$3"
    # Escape single quotes in prompt for shell embedding
    local escaped="${prompt//\'/\'\\\'\'}"
    printf "echo '--- %s ---' && echo 'Prompt: %s' && echo '' && curl -sN 'http://localhost:%s/v1/completions' -H 'Content-Type: application/json' -d '{\"model\": \"%s\", \"prompt\": \"%s\", \"max_tokens\": %s, \"stream\": true}' 2>/dev/null | python3 %s && echo '' && echo '--- done ---'" \
        "$label" "$escaped" "$port" "$MODEL" "$escaped" "$MAX_TOKENS" "$SSE_PARSER"
}

# ── Step 4: Set up tmux session ──────────────────────────────────────────
echo ""
echo "[record] Setting up tmux session..."

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# Create detached session with fixed geometry
tmux new-session -d -s "$TMUX_SESSION" -x 160 -y 45
tmux split-window -h -t "$TMUX_SESSION"

# Set pane titles via border format
tmux set-option -t "$TMUX_SESSION" pane-border-status top 2>/dev/null || true
tmux select-pane -t "${TMUX_SESSION}:0.0" -T "rvLLM (Rust) :${RVLLM_PORT}" 2>/dev/null || true
tmux select-pane -t "${TMUX_SESSION}:0.1" -T "vLLM (Python) :${VLLM_PORT}" 2>/dev/null || true

# Print headers in each pane
tmux send-keys -t "${TMUX_SESSION}:0.0" "printf '\\033[1;32m[ rvLLM  --  Rust  --  port ${RVLLM_PORT} ]\\033[0m\\n'" Enter
tmux send-keys -t "${TMUX_SESSION}:0.1" "printf '\\033[1;34m[ vLLM  --  Python  --  port ${VLLM_PORT} ]\\033[0m\\n'" Enter
sleep 1

# ── Step 5: Record with asciinema ────────────────────────────────────────
# asciinema records the tmux attach. We drive keys from a background process.
# When the driver finishes, it sends 'exit' to detach, ending the recording.

echo "[record] Starting asciinema recording..."

# Write the driver that runs in background while asciinema records
DRIVER="/tmp/rvllm-demo-driver.sh"
cat > "$DRIVER" << DRIVEREOF
#!/usr/bin/env bash
# Wait for tmux attach to be established
sleep 3

SESSION="${TMUX_SESSION}"

DRIVEREOF

# Append prompt commands
for i in $(seq 0 $((NUM_PROMPTS - 1))); do
    idx=$((i % ${#PROMPTS[@]}))
    prompt="${PROMPTS[$idx]}"

    rv_cmd=$(build_pane_cmd "$RVLLM_PORT" "$prompt" "rvLLM")
    vl_cmd=$(build_pane_cmd "$VLLM_PORT" "$prompt" "vLLM (Python)")

    cat >> "$DRIVER" << DRIVEREOF

# Prompt $((i + 1)): ${prompt}
tmux send-keys -t "\${SESSION}:0.0" "$(printf '%s' "$rv_cmd")" Enter
tmux send-keys -t "\${SESSION}:0.1" "$(printf '%s' "$vl_cmd")" Enter
sleep 10

DRIVEREOF
done

# Final: wait and detach to stop recording
cat >> "$DRIVER" << 'DRIVEREOF'
sleep 3
# Detach the tmux client to end asciinema recording
tmux detach-client -s "$SESSION" 2>/dev/null || true
# If detach didn't work, just exit the session
sleep 1
DRIVEREOF

chmod +x "$DRIVER"

# Launch the driver in background
bash "$DRIVER" &
DRIVER_PID=$!

# Record: asciinema attaches to tmux (blocking until detach/exit)
asciinema rec "$CAST_FILE" \
    --overwrite \
    --cols 160 \
    --rows 45 \
    -c "tmux attach -t '${TMUX_SESSION}'" || true

# Wait for driver to finish
wait "$DRIVER_PID" 2>/dev/null || true

echo "[record] Saved: ${CAST_FILE}"

# ── Step 6: Convert to MP4 (if tools available) ──────────────────────────
if command -v agg &>/dev/null && command -v ffmpeg &>/dev/null; then
    echo "[render] Generating MP4..."

    agg "$CAST_FILE" "$GIF_FILE" \
        --theme monokai \
        --font-size 14 \
        --fps-cap 30 \
        --cols 160 \
        --rows 45 2>/dev/null || true

    if [ -f "$GIF_FILE" ]; then
        ffmpeg -y \
            -i "$GIF_FILE" \
            -movflags faststart \
            -pix_fmt yuv420p \
            -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
            -c:v libx264 \
            -preset medium \
            -crf 18 \
            "$MP4_FILE" 2>/dev/null

        echo "[render] Saved: ${MP4_FILE}"
        rm -f "$GIF_FILE"
    fi
else
    echo "[render] agg/ffmpeg not found -- skipping MP4 conversion"
    echo "[render] Install: cargo install agg && brew install ffmpeg"
fi

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Recording complete"
echo "  Cast: ${CAST_FILE}"
[ -f "$MP4_FILE" ] && echo "  MP4:  ${MP4_FILE}"
echo "============================================"
