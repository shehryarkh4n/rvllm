#!/bin/bash
# deploy_and_bench.sh -- Compile kernels, build rvllm-v2 bench binary, benchmark.
# Runs directly on the GPU box (no HTTP server, no deadlocks).
#
# Usage:
#   bash deploy/deploy_and_bench.sh
#   bash deploy/deploy_and_bench.sh --arch sm_90
#   bash deploy/deploy_and_bench.sh --model Qwen/Qwen2.5-7B --fp8
#   bash deploy/deploy_and_bench.sh --skip-build --skip-compile
#
# Flags:
#   --model <name>       model name or HF path (default: Qwen/Qwen2.5-7B)
#   --arch <sm_XX>       override GPU arch (default: auto-detect)
#   --skip-build         skip Rust binary compilation
#   --skip-compile       skip kernel PTX compilation
#   --output-len <N>     tokens per request (default: 512)
#   --with-cutlass [dir] compile CUTLASS .so (default dir: /root/cutlass)
#   --fp8                enable FP8 weights
#   --n <list>           batch sizes (default: 1,4,8,16,32,64,128)
#   --iters <N>          iterations per batch size (default: 3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Defaults ---
MODEL="Qwen/Qwen2.5-7B"
ARCH=""
SKIP_BUILD=0
SKIP_COMPILE=0
OUTPUT_LEN=512
WITH_CUTLASS=0
CUTLASS_DIR="/root/cutlass"
FP8=0
BATCH_SIZES="1,4,8,16,32,64,128"
ITERS=3

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --arch)         ARCH="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --skip-compile) SKIP_COMPILE=1; shift ;;
        --output-len)   OUTPUT_LEN="$2"; shift 2 ;;
        --fp8)          FP8=1; shift ;;
        --n)            BATCH_SIZES="$2"; shift 2 ;;
        --iters)        ITERS="$2"; shift 2 ;;
        --with-cutlass)
            WITH_CUTLASS=1
            if [[ $# -ge 2 && ! "$2" == --* ]]; then
                CUTLASS_DIR="$2"; shift
            fi
            shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
step() { echo -e "${BOLD}${BLUE}=== $* ===${NC}"; }

EXIT_CODE=0

# --- GPU arch detection ---
detect_gpu_arch() {
    if [[ -n "$ARCH" ]]; then echo "$ARCH"; return; fi
    if ! command -v nvidia-smi &>/dev/null; then echo "sm_80"; return; fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')
    if [[ -z "$cc" || "$cc" == "N/A" ]]; then echo "sm_80"; else echo "sm_${cc}"; fi
}

ARCH=$(detect_gpu_arch)

# ============================================================
# Step 0: Environment
# ============================================================
step "Step 0: Environment"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no-gpu")
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
echo "GPU:  ${GPU_NAME} (${GPU_MEM_TOTAL} MiB)"
echo "Arch: ${ARCH}"
echo "Model: ${MODEL}"
echo "FP8: ${FP8}"

if command -v nvcc &>/dev/null; then
    echo "nvcc: $(nvcc --version 2>/dev/null | tail -1)"
    pass "nvcc found"
else
    warn "nvcc not found"
fi

if command -v cargo &>/dev/null; then
    pass "cargo found ($(cargo --version 2>/dev/null | head -1))"
else
    fail "cargo not found -- install Rust first"
    exit 1
fi

# Kill rogue GPU processes
ROGUE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
if [[ -n "$ROGUE_PIDS" ]]; then
    warn "Killing rogue GPU processes: ${ROGUE_PIDS}"
    for pid in $ROGUE_PIDS; do kill -9 "$pid" 2>/dev/null || true; done
    sleep 2
fi

GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "?")
echo "GPU memory used: ${GPU_MEM_USED} MiB"
echo ""

# ============================================================
# Step 1: Compile kernels to PTX
# ============================================================
if [[ "$SKIP_COMPILE" -eq 0 ]]; then
    step "Step 1: Compile kernels (${ARCH})"

    KERNEL_DIR="$REPO_DIR/kernels"
    OUTDIR="$KERNEL_DIR/$ARCH"
    mkdir -p "$OUTDIR"

    CU_COUNT=0
    PTX_OK=0
    PTX_FAIL=0

    for cu in "$KERNEL_DIR"/*.cu; do
        [[ -f "$cu" ]] || continue
        CU_COUNT=$((CU_COUNT + 1))
        stem=$(basename "$cu" .cu)

        if [[ "$stem" == "persistent_layer_decode" ]]; then
            cubin="$OUTDIR/${stem}.cubin"
            if nvcc -cubin -arch="$ARCH" -O3 --use_fast_math -rdc=true -o "$cubin" "$cu" 2>/tmp/nvcc_${stem}.log; then
                PTX_OK=$((PTX_OK + 1))
            else
                PTX_FAIL=$((PTX_FAIL + 1))
                warn "  FAILED: ${stem}.cu (cubin)"
                tail -3 /tmp/nvcc_${stem}.log 2>/dev/null
            fi
        elif [[ "$stem" == cutlass_* ]]; then
            # CUTLASS kernels compiled in Step 1b as .so
            CU_COUNT=$((CU_COUNT - 1))
            continue
        else
            ptx="$OUTDIR/${stem}.ptx"
            if nvcc -ptx -arch="$ARCH" -O3 --use_fast_math -o "$ptx" "$cu" 2>/tmp/nvcc_${stem}.log; then
                PTX_OK=$((PTX_OK + 1))
            else
                PTX_FAIL=$((PTX_FAIL + 1))
                warn "  FAILED: ${stem}.cu"
                tail -3 /tmp/nvcc_${stem}.log 2>/dev/null
            fi
        fi
    done

    if [[ "$PTX_FAIL" -eq 0 ]]; then
        pass "Compiled ${PTX_OK}/${CU_COUNT} kernels to ${OUTDIR}/"
    else
        warn "Compiled ${PTX_OK}/${CU_COUNT} kernels (${PTX_FAIL} failed)"
    fi

    export RVLLM_PTX_DIR="${RVLLM_PTX_DIR:-$OUTDIR}"
    echo "RVLLM_PTX_DIR=${RVLLM_PTX_DIR}"
    echo ""
else
    step "Step 1: Skipping kernel compilation (--skip-compile)"
    export RVLLM_PTX_DIR="${RVLLM_PTX_DIR:-$REPO_DIR/kernels/$ARCH}"
    echo ""
fi

# ============================================================
# Step 1b: Compile CUTLASS shared library
# ============================================================
if [[ "$SKIP_COMPILE" -eq 0 && "$WITH_CUTLASS" -eq 1 ]]; then
    step "Step 1b: Compile CUTLASS shared library (${ARCH})"

    if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
        echo "CUTLASS not found at $CUTLASS_DIR, cloning..."
        git clone --depth 1 https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
    fi

    CUTLASS_SO_BUILD="$REPO_DIR/kernels/build_cutlass_so.sh"
    if [[ -f "$CUTLASS_SO_BUILD" ]]; then
        if bash "$CUTLASS_SO_BUILD" "$ARCH" "$CUTLASS_DIR"; then
            pass "CUTLASS shared library built"
        else
            warn "CUTLASS .so build failed"
        fi
    else
        warn "build_cutlass_so.sh not found"
    fi
    echo ""
fi

# ============================================================
# Step 2: Build rvllm-v2 bench binary
# ============================================================
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    step "Step 2: Build rvllm-v2"
    BUILD_START=$(date +%s)

    cargo build --release -p rvllm-v2 --features cuda-graphs --bin rvllm-v2-bench \
        --manifest-path "$REPO_DIR/Cargo.toml" 2>&1 | tail -5

    BUILD_END=$(date +%s)
    BUILD_SECS=$((BUILD_END - BUILD_START))

    BINARY="$REPO_DIR/target/release/rvllm-v2-bench"
    if [[ -x "$BINARY" ]]; then
        pass "Build complete in ${BUILD_SECS}s -> ${BINARY}"
    else
        fail "Binary not found at ${BINARY}"
        exit 1
    fi
    echo ""
else
    step "Step 2: Skipping build (--skip-build)"
    echo ""
fi

BINARY="$REPO_DIR/target/release/rvllm-v2-bench"
if [[ ! -x "$BINARY" ]]; then
    fail "rvllm-v2-bench binary not found. Run without --skip-build."
    exit 1
fi

# ============================================================
# Step 3: Download model (if needed)
# ============================================================
step "Step 3: Model"
pip3 install -q huggingface_hub 2>/dev/null || true
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')" 2>&1 | tail -2
pass "Model ready: ${MODEL}"
echo ""

# ============================================================
# Step 4: Benchmark
# ============================================================
step "Step 4: Benchmark (direct engine, no HTTP)"

FP8_FLAG=""
if [[ "$FP8" -eq 1 ]]; then FP8_FLAG="--fp8"; fi

echo ""
echo "## rvllm-v2 Benchmark Results"
echo ""
echo "- Model: ${MODEL}"
echo "- GPU: ${GPU_NAME} (${GPU_MEM_TOTAL} MiB)"
echo "- Output tokens: ${OUTPUT_LEN}"
echo "- Arch: ${ARCH}"
echo "- FP8: ${FP8}"
echo "- Batch sizes: ${BATCH_SIZES}"
echo "- Iters: ${ITERS}"
echo "- Date: $(date -u +%Y-%m-%d)"
echo ""

RVLLM_PTX_DIR="${RVLLM_PTX_DIR}" "$BINARY" \
    --model "$MODEL" \
    --n "$BATCH_SIZES" \
    --output-len "$OUTPUT_LEN" \
    --iters "$ITERS" \
    --gpu-memory-utilization 0.95 \
    $FP8_FLAG \
    --json 2>&1

BENCH_EXIT=$?

echo ""
if [[ "$BENCH_EXIT" -eq 0 ]]; then
    pass "Benchmark complete"
else
    fail "Benchmark failed (exit $BENCH_EXIT)"
    EXIT_CODE=1
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "  Deploy + Benchmark Summary"
echo "========================================"
echo "  GPU:   ${GPU_NAME}"
echo "  Arch:  ${ARCH}"
echo "  Model: ${MODEL}"
echo "  FP8:   ${FP8}"
echo ""
if [[ "$EXIT_CODE" -eq 0 ]]; then
    pass "All steps passed"
else
    fail "Some steps failed (see above)"
fi
echo "========================================"

exit "$EXIT_CODE"
