#!/bin/bash
# upload_kernels_hf.sh -- Upload compiled kernels to HuggingFace after compilation.
# Run ON the GPU box after kernels are compiled.
#
# Uploads: PTX, cubin, libcutlass_kernels.so for a given arch+SHA.
# Layout on HF: <sha>/<arch>/{*.ptx, *.cubin, libcutlass_kernels.so}
#
# Usage:
#   bash deploy/upload_kernels_hf.sh              # auto-detect arch, use HEAD SHA
#   bash deploy/upload_kernels_hf.sh sm_90 abc123 # explicit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ARCH=${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.' | tr -d ' ')}
ARCH="sm_${ARCH#sm_}"
SHA=${2:-$(cat "$REPO_DIR/REVISION" 2>/dev/null || git -C "$REPO_DIR" rev-parse HEAD)}

HF_REPO="and-y/rvllm-kernels"
KERNEL_DIR="$REPO_DIR/kernels/$ARCH"

if [ ! -d "$KERNEL_DIR" ]; then
    echo "FATAL: No compiled kernels at $KERNEL_DIR"
    exit 1
fi

# Count what we're uploading
PTX_COUNT=$(ls "$KERNEL_DIR"/*.ptx 2>/dev/null | wc -l)
CUBIN_COUNT=$(ls "$KERNEL_DIR"/*.cubin 2>/dev/null | wc -l)
SO_EXISTS=0
[ -f "$KERNEL_DIR/libcutlass_kernels.so" ] && SO_EXISTS=1

echo "Uploading kernels to HF: $HF_REPO"
echo "  SHA:    $SHA"
echo "  Arch:   $ARCH"
echo "  PTX:    $PTX_COUNT files"
echo "  Cubin:  $CUBIN_COUNT files"
echo "  SO:     $SO_EXISTS"

if [ "$PTX_COUNT" -eq 0 ] && [ "$CUBIN_COUNT" -eq 0 ] && [ "$SO_EXISTS" -eq 0 ]; then
    echo "FATAL: Nothing to upload"
    exit 1
fi

# Create HF repo if needed, upload
python3 - "$HF_REPO" "$KERNEL_DIR" "$SHA" "$ARCH" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi, create_repo

hf_repo = sys.argv[1]
kernel_dir = sys.argv[2]
sha = sys.argv[3]
arch = sys.argv[4]

api = HfApi()

# Create repo if it doesn't exist
try:
    create_repo(hf_repo, repo_type="model", private=True, exist_ok=True)
except Exception as e:
    print(f"Repo create: {e}")

# Upload the entire arch directory under <sha>/<arch>/
prefix = f"{sha}/{arch}"
api.upload_folder(
    repo_id=hf_repo,
    folder_path=kernel_dir,
    path_in_repo=prefix,
    ignore_patterns=["obj/*"],  # skip intermediate .o files
)

print(f"Uploaded to {hf_repo} at {prefix}/")

# Also upload as "latest/<arch>/" for convenience
api.upload_folder(
    repo_id=hf_repo,
    folder_path=kernel_dir,
    path_in_repo=f"latest/{arch}",
    ignore_patterns=["obj/*"],
)
print(f"Also uploaded to latest/{arch}/")
PYEOF

echo "Done."
