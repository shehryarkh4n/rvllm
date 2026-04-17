#!/usr/bin/env bash
# One-shot: tarball the tpu/ suite, scp to a TPU VM, install jax[tpu], run.
# Usage: ./deploy_to_tpu.sh <tpu-name> <zone> [--only <kernel>]
set -euo pipefail

NAME="${1:?tpu name}"
ZONE="${2:?zone}"
shift 2
PROJECT="${PROJECT:-finance-484520}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHA="$(cd "$ROOT/.." && git rev-parse --short HEAD)"

TAR="/tmp/rvllm-tpu-${SHA}.tar.gz"
echo ">> packaging $ROOT -> $TAR"
tar -czf "$TAR" -C "$(dirname "$ROOT")" \
    --exclude='tpu/.venv' --exclude='tpu/out' --exclude='tpu/__pycache__' \
    tpu

echo ">> uploading to $NAME"
gcloud compute tpus tpu-vm scp --zone="$ZONE" --project="$PROJECT" \
    "$TAR" "$NAME:/tmp/rvllm-tpu.tar.gz"

REMOTE_CMD=$(cat <<'EOF'
set -euo pipefail
cd /tmp
rm -rf ~/runs && mkdir -p ~/runs && cd ~/runs
tar -xzf /tmp/rvllm-tpu.tar.gz
cd tpu
pip install --quiet --upgrade pip
pip install --quiet -U "jax[tpu]" numpy tomli
python3 -c "import jax; print('backend:', jax.default_backend(), 'devices:', jax.devices())"
python3 -m harness.execute_all "$@"
EOF
)

echo ">> executing on TPU VM"
gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --project="$PROJECT" \
    --command="$REMOTE_CMD" -- "$@"
