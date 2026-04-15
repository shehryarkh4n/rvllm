#!/bin/bash
# Build FA3 SM90 decode kernel as a shared library on H100.
# Requires: CUDA 12.x, CUTLASS headers at /root/cutlass, FA3 source at /root/flash-attention
set -euo pipefail

FA3_DIR="${FA3_DIR:-/root/flash-attention/hopper}"
CUTLASS_DIR="${CUTLASS_DIR:-/root/cutlass}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/sm_90"

mkdir -p "${OUT_DIR}"

NVCC_FLAGS=(
    -std=c++17
    -arch=sm_90a
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -Xcompiler -fPIC
    -shared
    -O3
    -DNDEBUG
    -I"${CUTLASS_DIR}/include"
    -I"${FA3_DIR}"
    -I"${SCRIPT_DIR}"
    -lineinfo
)

echo "=== Building FA3 SM90 decode kernel ==="
echo "FA3_DIR=${FA3_DIR}"
echo "CUTLASS_DIR=${CUTLASS_DIR}"
echo "OUT=${OUT_DIR}/libfa3_kernels.so"

# Source files:
#   1. Our wrapper with extern "C" entry points
#   2. Paged non-split instantiation (matches vLLM decode path)
#   3. Paged split instantiation (for low-batch long-seqlen)
#   4. Combine kernel (for split-KV reduction)
#   5. Prepare scheduler (varlen metadata)
SRCS=(
    "${SCRIPT_DIR}/fa3_sm90_wrapper.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_paged_sm90.cu"
    "${FA3_DIR}/instantiations/flash_fwd_hdim128_fp16_paged_split_sm90.cu"
    "${FA3_DIR}/flash_fwd_combine.cu"
    "${FA3_DIR}/flash_prepare_scheduler.cu"
)

echo "Sources:"
for s in "${SRCS[@]}"; do echo "  ${s}"; done

echo ""
echo "Compiling (this takes 5-15 minutes due to CuTe template expansion)..."
time nvcc "${NVCC_FLAGS[@]}" -o "${OUT_DIR}/libfa3_kernels.so" "${SRCS[@]}" 2>&1

SZ=$(stat -c%s "${OUT_DIR}/libfa3_kernels.so" 2>/dev/null || stat -f%z "${OUT_DIR}/libfa3_kernels.so")
echo ""
echo "=== FA3 build complete ==="
echo "  Size: ${SZ} bytes"
echo "  Path: ${OUT_DIR}/libfa3_kernels.so"

if [ "$SZ" -lt 1000000 ]; then
    echo "WARNING: .so seems too small (<1MB), may not contain GPU code"
fi

# Verify exported symbols
echo ""
echo "Exported symbols:"
nm -D "${OUT_DIR}/libfa3_kernels.so" | grep -E 'fa3_sm90' || echo "  (none found - check build)"
