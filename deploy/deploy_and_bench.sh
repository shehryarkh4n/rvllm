#!/bin/bash
# deploy_and_bench.sh -- Pull kernels from HF, build rvllm-v2 bench binary, benchmark.
# Runs directly on the GPU box (no HTTP server, no deadlocks).
#
# Kernel flow:
#   Default:              Download pre-compiled kernels from HF for this SHA
#   --compile-kernels:    Compile fresh on box + upload to HF (when .cu files changed)
#
# Old kernels on HF are never deleted -- each SHA gets its own directory.
# Kernels on the box are always nuked and re-downloaded/re-compiled.
#
# Usage:
#   bash deploy/deploy_and_bench.sh --model Qwen/Qwen2.5-32B --fp8
#   bash deploy/deploy_and_bench.sh --compile-kernels --with-cutlass
#   bash deploy/deploy_and_bench.sh --skip-build
#
# Flags:
#   --model <name>         model name or HF path (default: Qwen/Qwen2.5-7B)
#   --arch <sm_XX>         override GPU arch (default: auto-detect)
#   --compile-kernels      compile kernels on box + upload to HF (use when .cu changed)
#   --with-cutlass [dir]   compile CUTLASS .so (only with --compile-kernels)
#   --skip-build           skip Rust binary compilation
#   --output-len <N>       tokens per request (default: 512)
#   --fp8                  enable FP8 weights
#   --n <list>             batch sizes (default: 1,4,8,16,32,64,128)
#   --iters <N>            iterations per batch size (default: 3)
#   --profile              enable CUPTI profiling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Defaults ---
MODEL="Qwen/Qwen2.5-7B"
ARCH=""
SKIP_BUILD=0
COMPILE_KERNELS=0
OUTPUT_LEN=512
WITH_CUTLASS=0
CUTLASS_DIR="/root/cutlass"
FP8=0
BATCH_SIZES="1,4,8,16,32,64,128"
ITERS=3
PROFILE=""

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2"; shift 2 ;;
        --arch)             ARCH="$2"; shift 2 ;;
        --compile-kernels)  COMPILE_KERNELS=1; shift ;;
        --skip-build)       SKIP_BUILD=1; shift ;;
        --output-len)       OUTPUT_LEN="$2"; shift 2 ;;
        --fp8)              FP8=1; shift ;;
        --n)                BATCH_SIZES="$2"; shift 2 ;;
        --iters)            ITERS="$2"; shift 2 ;;
        --profile)          PROFILE="--profile"; shift ;;
        --with-cutlass)
            WITH_CUTLASS=1
            if [[ $# -ge 2 && ! "$2" == --* ]]; then
                CUTLASS_DIR="$2"; shift
            fi
            shift ;;
        # Legacy flags -- warn and map
        --skip-compile)     echo "WARN: --skip-compile is deprecated, kernels come from HF by default"; shift ;;
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
# Step 0: Environment + SHA verification
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

# SHA verification
REVISION_FILE="${REPO_DIR}/REVISION"
if [[ -f "$REVISION_FILE" ]]; then
    EXPECTED_SHA=$(cat "$REVISION_FILE")
    echo "Expected SHA: ${EXPECTED_SHA}"
    pass "REVISION file present"
else
    fail "No REVISION file -- deploy via rsync_and_run.sh"
    exit 1
fi
echo ""

# ============================================================
# Step 1: Kernels -- download from HF or compile + upload
# ============================================================
KERNEL_DIR="$REPO_DIR/kernels"
OUTDIR="$KERNEL_DIR/$ARCH"
HF_KERNEL_REPO="and-y/rvllm-kernels"

# Always nuke what's on the box
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

if [[ "$COMPILE_KERNELS" -eq 1 ]]; then
    # --- Compile fresh + upload to HF ---
    step "Step 1: Compile kernels (${ARCH}) [--compile-kernels]"

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
        pass "Compiled ${PTX_OK}/${CU_COUNT} kernels"
    else
        warn "Compiled ${PTX_OK}/${CU_COUNT} kernels (${PTX_FAIL} failed)"
    fi

    # Compile CUTLASS .so if requested
    if [[ "$WITH_CUTLASS" -eq 1 ]]; then
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
                fail "CUTLASS .so build failed"
                exit 1
            fi
        else
            fail "build_cutlass_so.sh not found"
            exit 1
        fi
    fi

    # Upload to HF under <sha>/<arch>/
    step "Step 1c: Upload kernels to HF (${EXPECTED_SHA})"
    bash "$REPO_DIR/deploy/upload_kernels_hf.sh" "$ARCH" "$EXPECTED_SHA"
    echo ""

else
    # --- Download from HF ---
    step "Step 1: Download kernels from HF (${EXPECTED_SHA}/${ARCH})"
    bash "$REPO_DIR/deploy/download_kernels_hf.sh" "$ARCH" "$EXPECTED_SHA"
    echo ""
fi

export RVLLM_PTX_DIR="${RVLLM_PTX_DIR:-$OUTDIR}"
echo "RVLLM_PTX_DIR=${RVLLM_PTX_DIR}"
echo ""

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
        BINARY_AGE=$(( $(date +%s) - $(stat -c %Y "$BINARY" 2>/dev/null || stat -f %m "$BINARY") ))
        if [[ "$BINARY_AGE" -gt "$((BUILD_SECS + 10))" ]]; then
            fail "Binary is ${BINARY_AGE}s old but build took ${BUILD_SECS}s -- stale binary!"
            exit 1
        fi
        pass "Build complete in ${BUILD_SECS}s"
    else
        fail "Binary not found"
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
# Step 4: Verify binary SHA
# ============================================================
step "Step 4: Verify binary SHA"

BINARY_SHA=$("$BINARY" --model dummy --n 0 2>&1 | grep "rvLLM revision:" | awk '{print $NF}' || true)
if [[ -n "$BINARY_SHA" && "$BINARY_SHA" != "$EXPECTED_SHA" ]]; then
    fail "Binary SHA (${BINARY_SHA}) != expected (${EXPECTED_SHA}) -- STALE BINARY!"
    exit 1
elif [[ -n "$BINARY_SHA" ]]; then
    pass "Binary SHA verified: ${BINARY_SHA}"
else
    warn "Could not extract SHA from binary output (will verify from bench log)"
fi
echo ""

# ============================================================
# Step 5: Benchmark
# ============================================================
step "Step 5: Benchmark (direct engine, no HTTP)"

FP8_FLAG=""
if [[ "$FP8" -eq 1 ]]; then FP8_FLAG="--fp8"; fi

echo ""
echo "## rvllm-v2 Benchmark Results"
echo ""
echo "- Model: ${MODEL}"
echo "- GPU: ${GPU_NAME} (${GPU_MEM_TOTAL} MiB)"
echo "- SHA: ${EXPECTED_SHA}"
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
    $PROFILE \
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
echo "  SHA:   ${EXPECTED_SHA}"
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
