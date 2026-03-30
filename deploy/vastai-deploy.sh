#!/bin/bash
set -euo pipefail

# Deploy rvllm to the rvllm GPU instance.
# Pushes code via git, builds with CUDA on the instance.
#
# Usage:
#   ./deploy/vastai-deploy.sh                   # auto-detect instance from .instance_id_rvllm
#   ./deploy/vastai-deploy.sh <instance_id>     # explicit instance

INSTANCE_ID=${1:-$(cat deploy/.instance_id_rvllm 2>/dev/null || echo "")}
if [[ -z "$INSTANCE_ID" ]]; then
    echo "No instance ID. Run vastai-provision.sh --rvllm first, or pass instance ID."
    exit 1
fi

MODEL=${MODEL:-"Qwen/Qwen2.5-7B"}

SSH_INFO=$(vastai ssh-url $INSTANCE_ID)
echo "Deploying rvllm to instance $INSTANCE_ID ($SSH_INFO)"
echo "Model: $MODEL"

SSH_CMD="vastai ssh $INSTANCE_ID"

$SSH_CMD << REMOTE_SCRIPT
set -euo pipefail
export PATH="/root/.cargo/bin:$PATH"

# Ensure Rust is installed
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="/root/.cargo/bin:$PATH"
fi

# Clone or update repo
if [[ -d /root/rvllm/.git ]]; then
    cd /root/rvllm
    git config --global --add safe.directory /root/rvllm 2>/dev/null || true
    git fetch origin main
    git reset --hard origin/main
else
    cd /root
    git clone https://github.com/m0at/rvllm.git
    cd rvllm
fi

# Detect GPU arch
CC=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "\$CC" in
    7.0) ARCH=sm_70 ;; 7.5) ARCH=sm_75 ;; 8.0) ARCH=sm_80 ;;
    8.6) ARCH=sm_86 ;; 8.9) ARCH=sm_89 ;; 9.0) ARCH=sm_90 ;;
    10.0) ARCH=sm_100 ;; 12.0) ARCH=sm_120 ;; *) ARCH=sm_80 ;;
esac
echo "GPU compute capability: \$CC -> \$ARCH"

# Build with CUDA
echo "Building rvllm..."
CUDA_ARCH=\$ARCH cargo build --release --features cuda 2>&1 | tail -5

# Download model
echo "Downloading model: $MODEL"
pip3 install -q huggingface_hub 2>/dev/null || true
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')"

echo ""
echo "Deploy complete!"
echo "Binary: /root/rvllm/target/release/rvllm"
echo "Run: target/release/rvllm serve --model $MODEL --port 8000"
REMOTE_SCRIPT
