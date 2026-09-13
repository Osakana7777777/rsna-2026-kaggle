#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "${1:-submission}"
config_path="${2:-configs/baseline.toml}"

require_configured_metadata "${config_path}"

require_command uv

mkdir -p "${BUILD_DIR}"
find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

cp "${METADATA_FILE}" "${BUILD_DIR}/kernel-metadata.json"
UV_CACHE_DIR="${TMPDIR:-/tmp}/rsna-2026-kaggle-uv-cache" \
  uv run --no-project python "${PROJECT_ROOT}/scripts/build_launcher.py" \
    --project-root "${PROJECT_ROOT}" \
    --mode "${RUN_MODE}" \
    --config "${config_path}" \
    --output "${BUILD_DIR}/launcher.ipynb"

echo "Prepared Kaggle upload payload at ${BUILD_DIR}"
