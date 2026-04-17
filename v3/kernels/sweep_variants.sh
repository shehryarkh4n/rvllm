#!/bin/bash
# Sweep CUTLASS fp8 gemm variants at N=128 and report tok/s per pick.
# Usage: bash sweep_variants.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export RVLLM_MODEL_DIR=/workspace/models/qwen25-7b-instruct
export RVLLM_KERNELS_DIR=/workspace/rvllm/kernels/sm_90
export RVLLM_CUTLASS_SO=/workspace/rvllm/kernels/sm_90/libcutlass_kernels.so
export RVLLM_FA3_SO=/workspace/rvllm/kernels/sm_90/libfa3_kernels.so
export RVLLM_POLICY=/workspace/rvllm/kernels/sm_90/policy.json
export RVLLM_BATCH=128
export RVLLM_ITERS=30
export RVLLM_WARMUP=5

BIN=/workspace/rvllm/v3/target/release/rvllm-bench
POLICY_JSON="$RVLLM_POLICY"
MK=/workspace/rvllm/v3/kernels/make_policy.py

# Promising v0..v10 for non-residual plus all residual variants 100..109.
NONRES_CANDIDATES="0 2 5 8 10"
RES_CANDIDATES="100 102 105 108"

echo "=== variant sweep at N=128 ==="
for nr in $NONRES_CANDIDATES; do
    for r in $RES_CANDIDATES; do
        python3 "$MK" "$POLICY_JSON" "sweep" "$nr" "$r" > /dev/null
        OUT=$("$BIN" 2>/dev/null | tail -1 || true)
        echo "nonres=$nr res=$r -> $OUT"
    done
done
