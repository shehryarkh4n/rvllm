#!/bin/bash
# rsync_and_run.sh -- Push to GitHub, pull on GPU box, build + bench.
#
# Kernels come from HF by default. Pass --compile-kernels when .cu files changed.
#
# Usage:
#   bash deploy/rsync_and_run.sh ssh8.vast.ai 13302 --model Qwen/Qwen2.5-32B --fp8 --n 1,128 --iters 3
#   bash deploy/rsync_and_run.sh ssh8.vast.ai 13302 --compile-kernels --with-cutlass  # recompile + upload to HF
#
# All arguments after host and port are forwarded to deploy_and_bench.sh.

set -euo pipefail

HOST=${1:-ssh8.vast.ai}
PORT=${2:-20236}
shift 2 2>/dev/null || true
EXTRA_ARGS="$*"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -p ${PORT}"
REMOTE_DIR="/root/rvllm"
REPO_URL="git@github.com:m0at/rvllm.git"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOCAL_SHA=$(git -C "${REPO_DIR}" rev-parse HEAD)

# Abort if tree is dirty
if ! git -C "${REPO_DIR}" diff --quiet HEAD; then
    echo "ERROR: Working tree is dirty. Commit first."
    exit 1
fi

echo "Local SHA: ${LOCAL_SHA}"
echo "Target: root@${HOST}:${PORT}"
echo ""

# --- Push to GitHub ---
echo "Pushing to GitHub..."
git -C "${REPO_DIR}" push origin main

# --- Pull on remote, verify SHA, run ---
echo "Deploying on remote..."
ssh ${SSH_OPTS} "root@${HOST}" "
    export PATH=/root/.cargo/bin:\$PATH

    if [ -d ${REMOTE_DIR}/.git ]; then
        cd ${REMOTE_DIR}
        git fetch origin
        git reset --hard origin/main
    else
        git clone ${REPO_URL} ${REMOTE_DIR}
        cd ${REMOTE_DIR}
    fi

    # Verify SHA
    REMOTE_SHA=\$(git rev-parse HEAD)
    echo \"Remote SHA: \${REMOTE_SHA}\"
    if [ \"\${REMOTE_SHA}\" != \"${LOCAL_SHA}\" ]; then
        echo \"FATAL: SHA mismatch! Local=${LOCAL_SHA} Remote=\${REMOTE_SHA}\"
        exit 1
    fi

    # Write REVISION for binary and kernel scripts to read
    echo \"${LOCAL_SHA}\" > REVISION

    # Nuke stale binary
    rm -f target/release/rvllm-v2-bench

    echo ''
    echo '========================================'
    echo 'Running deploy_and_bench.sh...'
    echo '========================================'
    bash deploy/deploy_and_bench.sh ${EXTRA_ARGS}
"
