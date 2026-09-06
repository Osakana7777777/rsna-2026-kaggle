#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" train

require_command uv

cache_build_dir="${PROJECT_ROOT}/.kaggle-build/selection-cache"
mkdir -p "${cache_build_dir}"
find "${cache_build_dir}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

cp "${PROJECT_ROOT}/kaggle/selection-cache-dataset-metadata.json" \
  "${cache_build_dir}/dataset-metadata.json"
selection_source="${PROJECT_ROOT}/outputs/train/outputs/slice-selection.json"
selection_sha256="$(shasum -a 256 "${selection_source}" | awk '{print $1}')"
UV_CACHE_DIR="${TMPDIR:-/tmp}/rsna-2026-kaggle-uv-cache" \
  uv run --no-project python "${PROJECT_ROOT}/scripts/build_selection_cache.py" \
    --source "${selection_source}" \
    --output "${cache_build_dir}/slice-selection.json.xz" \
    --expected-sha256 "${selection_sha256}"

echo "Prepared private Kaggle dataset at ${cache_build_dir}"
