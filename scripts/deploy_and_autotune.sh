#!/bin/bash
set -euo pipefail

SSH_HOST=${1:?Usage: $0 <ssh_host> <ssh_port> [prev_run_dir]}
SSH_PORT=${2:?Usage: $0 <ssh_host> <ssh_port> [prev_run_dir]}
PREV_RUN=${3:-}

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RUN_DIR="/workspace/runs/autotune-${SHA}"
TARBALL="/tmp/rvllm-autotune-${SHA}.tar.gz"
SSH="ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} root@${SSH_HOST}"
SCP="scp -o StrictHostKeyChecking=no -P ${SSH_PORT}"

echo "=== rvLLM CUTLASS Autotune Deploy ==="
echo "SHA: ${SHA}"
echo "Host: ${SSH_HOST}:${SSH_PORT}"
echo "Run dir: ${RUN_DIR}"

# 1. Package
echo "Packaging..."
tar czf "${TARBALL}" \
    --exclude=target --exclude=.git --exclude='.codex-tmp' \
    --exclude='v2' --exclude='libtypes.rlib' \
    --exclude='kernels/sm_*/*.so' --exclude='kernels/sm_*/*.dylib' \
    --exclude='*.pdf' \
    .
echo "  $(du -h "${TARBALL}" | cut -f1)"

# 2. Upload
echo "Uploading..."
${SCP} "${TARBALL}" "root@${SSH_HOST}:/tmp/rvllm-autotune.tar.gz"

# 3. Extract
echo "Extracting to ${RUN_DIR}..."
${SSH} "mkdir -p ${RUN_DIR}/rvllm && cd ${RUN_DIR}/rvllm && tar xzf /tmp/rvllm-autotune.tar.gz"

# 3b. Nuke any .so that leaked from local (Mac stubs etc)
${SSH} "rm -f ${RUN_DIR}/rvllm/kernels/sm_*/*.so ${RUN_DIR}/rvllm/kernels/sm_*/*.dylib"

# 4. Copy PTX from previous run
if [ -n "${PREV_RUN}" ]; then
    echo "Copying PTX from ${PREV_RUN}..."
    ${SSH} "cp ${PREV_RUN}/kernels/*.ptx ${RUN_DIR}/rvllm/kernels/ 2>/dev/null || true; \
            cp -r ${PREV_RUN}/kernels/sm_90/*.ptx ${RUN_DIR}/rvllm/kernels/sm_90/ 2>/dev/null || true"
fi

# 5. Build CUTLASS .so
echo "Building CUTLASS kernels..."
${SSH} "cd ${RUN_DIR}/rvllm && bash kernels/build_cutlass_so.sh sm_90 /root/cutlass"

# 5b. Verify .so is GPU-compiled (not a Mac stub)
${SSH} "SO=${RUN_DIR}/rvllm/kernels/sm_90/libcutlass_kernels.so; \
    if [ ! -f \$SO ]; then echo 'FATAL: .so not found after build'; exit 1; fi; \
    SZ=\$(stat -c%s \$SO 2>/dev/null || stat -f%z \$SO); \
    echo \"CUTLASS .so size: \${SZ} bytes\"; \
    if [ \$SZ -lt 1000000 ]; then echo 'FATAL: .so too small (<1MB) -- likely a Mac stub, not GPU-compiled'; exit 1; fi"

# 6. Build Rust
echo "Building rvllm-v2..."
${SSH} "source \$HOME/.cargo/env && cd ${RUN_DIR}/rvllm && cargo build --release -p rvllm-v2 --features cuda-graphs -p rvllm-gpu 2>&1 | tail -5"

# 7. Run autotune
echo "Running CUTLASS autotune..."
${SSH} "source \$HOME/.cargo/env && cd ${RUN_DIR}/rvllm && cargo run --release -p rvllm-gpu --bin autotune-cutlass --features cuda 2>&1"

# 8. Run benchmark
echo "Running benchmark..."
${SSH} "MODEL=\$(find /workspace/hf_cache /root/.cache/huggingface -path '*/Qwen2.5-7B/snapshots/*' -name 'config.json' 2>/dev/null | head -1 | xargs dirname); \
    if [ -z \"\$MODEL\" ]; then echo 'FATAL: Qwen2.5-7B model not found'; exit 1; fi; \
    echo \"Model: \$MODEL\"; \
    cd ${RUN_DIR}/rvllm && target/release/rvllm-v2-bench --fp8 --model \$MODEL --output-len 512 --n 1,32,64,128 --iters 3 2>&1"

echo "=== Done ==="
