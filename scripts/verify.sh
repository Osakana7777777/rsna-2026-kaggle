#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

required_files=(
  "README.md"
  "pyproject.toml"
  "configs/baseline.toml"
  "configs/coatnet-0-224.toml"
  "kaggle/kernel-metadata.json"
  "kaggle/train-kernel-metadata.json"
  "kaggle/eda-kernel-metadata.json"
  "kaggle/preprocess-0-kernel-metadata.json"
  "kaggle/preprocess-1-kernel-metadata.json"
  "kaggle/preprocess-2-kernel-metadata.json"
  "kaggle/preprocess-3-kernel-metadata.json"
  "kaggle/selection-cache-dataset-metadata.json"
  "kaggle/launcher.py"
  "src/rsna2026/checkpoint.py"
  "src/rsna2026/acquisition.py"
  "src/rsna2026/augmentation.py"
  "src/rsna2026/config.py"
  "src/rsna2026/constants.py"
  "src/rsna2026/cross_validation.py"
  "src/rsna2026/data.py"
  "src/rsna2026/dataset.py"
  "src/rsna2026/image_cache.py"
  "src/rsna2026/model.py"
  "src/rsna2026/ensemble.py"
  "src/rsna2026/public_labels.py"
  "src/rsna2026/slice_selection.py"
  "src/rsna2026/train.py"
  "src/rsna2026/weak_label_calibration.py"
  "src/rsna2026/weak_label_codes.py"
  "src/rsna2026/weak_label_decode.py"
  "src/rsna2026/weak_label_infer.py"
  "src/rsna2026/weak_label_postprocess.py"
  "src/rsna2026/weak_label_schema.py"
  "prompts/rsna_knee_extractor_v1.txt"
  "modal_weak_labels.py"
  "docs/weak-label-extraction.md"
  "scripts/prepare.sh"
  "scripts/build_launcher.py"
  "scripts/build_selection_cache.py"
  "scripts/run_eda.py"
  "scripts/run-eda.sh"
  "scripts/run-image-cache.sh"
  "scripts/prepare-selection-cache.sh"
  "scripts/push.sh"
  "scripts/status.sh"
  "scripts/output.sh"
  "scripts/wait.sh"
  "scripts/runpod.sh"
  "scripts/runpod-train.sh"
  "tests/test_public_labels.py"
  "tests/test_checkpoint.py"
  "tests/test_cross_validation.py"
  "tests/test_ensemble.py"
  "tests/test_acquisition.py"
  "tests/test_model.py"
  "tests/test_slice_selection.py"
  "tests/test_slice_windows.py"
  "tests/test_distributed.py"
  "tests/test_study_records_cache.py"
  "tests/test_physical_crop.py"
  "tests/test_augmentation.py"
  "tests/test_dense_windows.py"
  "tests/test_image_cache.py"
  "tests/test_build_launcher.py"
  "tests/test_weak_labels.py"
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
    "${PROJECT_ROOT}/modal_weak_labels.py" \
    "${PROJECT_ROOT}/scripts/build_launcher.py" "${PROJECT_ROOT}/tests"
  "${PROJECT_ROOT}/.venv/bin/python" -c \
    'import json, pathlib, tomllib; root = pathlib.Path(__import__("sys").argv[1]); files = ["kernel-metadata.json", "train-kernel-metadata.json", "eda-kernel-metadata.json", *[f"preprocess-{index}-kernel-metadata.json" for index in range(4)]]; [json.loads((root / "kaggle" / name).read_text()) for name in files]; [tomllib.loads(path.read_text()) for path in sorted((root / "configs").glob("*.toml"))]' \
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
if grep -q 'YOUR_' "${METADATA_FILE}" "${PROJECT_ROOT}"/configs/*.toml; then
  echo "Configuration placeholders remain, as expected for the initial template."
fi
