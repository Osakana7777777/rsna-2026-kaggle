#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

required_files=(
  "README.md"
  "pyproject.toml"
  "configs/baseline.toml"
  "kaggle/kernel-metadata.json"
  "kaggle/launcher.py"
  "src/rsna2026/config.py"
  "src/rsna2026/constants.py"
  "src/rsna2026/data.py"
  "src/rsna2026/dataset.py"
  "src/rsna2026/model.py"
  "src/rsna2026/public_labels.py"
  "src/rsna2026/slice_selection.py"
  "src/rsna2026/train.py"
  "scripts/prepare.sh"
  "scripts/build_launcher.py"
  "scripts/push.sh"
  "scripts/status.sh"
  "scripts/output.sh"
  "scripts/wait.sh"
  "tests/test_public_labels.py"
  "tests/test_slice_selection.py"
  "submissions/README.md"
  "submissions/TEMPLATE.md"
)

for relative_path in "${required_files[@]}"; do
  test -f "${PROJECT_ROOT}/${relative_path}" || {
    echo "Missing required file: ${relative_path}" >&2
    exit 1
  }
done

for script in "${PROJECT_ROOT}"/scripts/*.sh; do
  bash -n "${script}"
done

if command -v uv >/dev/null 2>&1; then
  UV_CACHE_DIR="${TMPDIR:-/tmp}/rsna-2026-kaggle-uv-cache"
  export UV_CACHE_DIR
  if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    uv venv "${PROJECT_ROOT}/.venv"
  fi
  "${PROJECT_ROOT}/.venv/bin/python" -m compileall -q \
    "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/kaggle/launcher.py" \
    "${PROJECT_ROOT}/scripts/build_launcher.py" "${PROJECT_ROOT}/tests"
  "${PROJECT_ROOT}/.venv/bin/python" -c \
    'import json, pathlib, tomllib; root = pathlib.Path(__import__("sys").argv[1]); json.loads((root / "kaggle/kernel-metadata.json").read_text()); tomllib.loads((root / "configs/baseline.toml").read_text())' \
    "${PROJECT_ROOT}"
else
  echo "uv not installed; skipped Python syntax check." >&2
fi

download_pattern='kaggle (competitions|datasets) down''load|competition_down''load|dataset_down''load'
if grep -R -nE "${download_pattern}" \
  "${PROJECT_ROOT}/scripts" "${PROJECT_ROOT}/src" "${PROJECT_ROOT}/kaggle"; then
  echo "A local data-download command was found." >&2
  exit 1
fi

echo "Structure, shell syntax, Python syntax, and no-local-data policy verified."
if grep -q 'YOUR_' "${METADATA_FILE}" "${PROJECT_ROOT}/configs/baseline.toml"; then
  echo "Configuration placeholders remain, as expected for the initial template."
fi
