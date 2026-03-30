#!/bin/bash
set -euo pipefail

# Provision GPU instances on vast.ai for rvllm benchmarks.
# Requires: vastai CLI (pip install vastai)
#
# IMPORTANT: Always provision TWO separate instances:
#   1. rvllm instance  -- nvidia/cuda devel image (builds + runs rvllm)
#   2. vLLM instance   -- vllm/vllm-openai image  (runs Python vLLM)
#
# Never run both on the same GPU. Shared CUDA driver state, memory
# fragmentation, and context residue between runs contaminate results.
# Each engine gets a clean GPU with no prior allocations.
#
# Usage:
#   ./vastai-provision.sh              # provision rvllm instance
#   ./vastai-provision.sh --vllm       # provision vLLM instance
#   ./vastai-provision.sh --both       # provision both
#   ./vastai-provision.sh --dry-run    # search only

DISK_GB=${DISK_GB:-100}
GPU_TYPE=${GPU_TYPE:-"H100_SXM"}
GPU_RAM_MIN=${GPU_RAM_MIN:-75}

RVLLM_IMAGE="nvidia/cuda:12.6.3-devel-ubuntu22.04"
VLLM_IMAGE="vllm/vllm-openai:latest"

MODE="${1:---rvllm}"

provision_instance() {
    local label=$1
    local image=$2
    local onstart=$3
    local id_file=$4

    echo ""
    echo "=== Provisioning $label instance ==="
    echo "Image: $image"

    vastai search offers \
        "gpu_name=$GPU_TYPE gpu_ram>=${GPU_RAM_MIN} num_gpus=1 inet_down>200 disk_space>=${DISK_GB}" \
        --order "dph_total" \
        --limit 5

    echo ""
    read -p "Enter offer ID for $label: " OFFER_ID

    INSTANCE_ID=$(vastai create instance $OFFER_ID \
        --image "$image" \
        --disk $DISK_GB \
        --ssh \
        --env "RUST_LOG=info" \
        --onstart-cmd "$onstart" \
        --raw | jq -r '.new_contract')

    echo "Instance created: $INSTANCE_ID"
    echo "Waiting for instance to start..."

    for i in $(seq 1 60); do
        STATUS=$(vastai show instance $INSTANCE_ID --raw | jq -r '.actual_status')
        if [[ "$STATUS" == "running" ]]; then
            echo "Instance running!"
            break
        fi
        echo "  Status: $STATUS (attempt $i/60)"
        sleep 10
    done

    SSH_INFO=$(vastai ssh-url $INSTANCE_ID)
    echo "$label ready: $SSH_INFO"
    echo "$INSTANCE_ID" > "deploy/$id_file"
}

if [[ "$MODE" == "--dry-run" ]]; then
    echo "Searching for $GPU_TYPE instances (>=${GPU_RAM_MIN}GB VRAM)..."
    vastai search offers \
        "gpu_name=$GPU_TYPE gpu_ram>=${GPU_RAM_MIN} num_gpus=1 inet_down>200 disk_space>=${DISK_GB}" \
        --order "dph_total" \
        --limit 5
    exit 0
fi

RVLLM_ONSTART="apt-get update && apt-get install -y curl git build-essential pkg-config libssl-dev && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
VLLM_ONSTART="pip install aiohttp numpy"

if [[ "$MODE" == "--rvllm" || "$MODE" == "--both" ]]; then
    provision_instance "rvllm" "$RVLLM_IMAGE" "$RVLLM_ONSTART" ".instance_id_rvllm"
fi

if [[ "$MODE" == "--vllm" || "$MODE" == "--both" ]]; then
    provision_instance "vLLM" "$VLLM_IMAGE" "$VLLM_ONSTART" ".instance_id_vllm"
fi

echo ""
echo "=== Next steps ==="
echo "  1. Deploy rvllm:  ./deploy/vastai-deploy.sh"
echo "  2. Run benchmark: ./deploy/vastai-benchmark.sh"
echo ""
echo "REMINDER: rvllm and vLLM must run on SEPARATE instances."
echo "Never benchmark both engines on the same GPU."
