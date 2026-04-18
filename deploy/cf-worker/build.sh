#!/usr/bin/env bash
# Stage the repo's docs/ folder under ./site/docs/ so wrangler picks it
# up and serves at solidsf.com/rvllm/docs/... with the URL-to-file
# mapping exactly mirroring the repo layout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SITE_DIR="${SCRIPT_DIR}/site"

rm -rf "${SITE_DIR}"
mkdir -p "${SITE_DIR}/docs"
rsync -a \
    --exclude='*.aux' --exclude='*.log' --exclude='*.out' \
    --exclude='*.bbl' --exclude='*.blg' \
    "${REPO_ROOT}/docs/" "${SITE_DIR}/docs/"

# Copy _headers into site root for Cloudflare cache control
cp "${SCRIPT_DIR}/site-headers/_headers" "${SITE_DIR}/_headers"

echo "Staged into ${SITE_DIR}:"
du -sh "${SITE_DIR}"
find "${SITE_DIR}" -maxdepth 3 -type f | sort
