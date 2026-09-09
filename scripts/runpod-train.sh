#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_path="${PROJECT_ROOT}/configs/baseline.toml"
input_root="/workspace/input"
output_dir="/workspace/outputs"
gpu_count="auto"
venv_path="/workspace/.venvs/rsna-2026"
build_selection_cache=false

usage() {
  cat <<EOF
Usage: ./scripts/runpod-train.sh [options]

  --input-root PATH       Staged read-only inputs (default: ${input_root})
  --output-dir PATH       Persistent experiment outputs (default: ${output_dir})
  --gpu-count COUNT       Visible GPUs to use, or auto (default: auto)
  --config PATH           Experiment TOML (default: configs/baseline.toml)
  --venv PATH             uv virtual environment path (default: ${venv_path})
  --build-selection-cache Build slice selection metadata from staged DICOM files
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --input-root) input_root="${2:?Missing value for --input-root}"; shift 2 ;;
    --output-dir) output_dir="${2:?Missing value for --output-dir}"; shift 2 ;;
    --gpu-count) gpu_count="${2:?Missing value for --gpu-count}"; shift 2 ;;
    --config) config_path="${2:?Missing value for --config}"; shift 2 ;;
    --venv) venv_path="${2:?Missing value for --venv}"; shift 2 ;;
    --build-selection-cache) build_selection_cache=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required inside the Pod." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this does not appear to be a GPU Pod." >&2
  exit 1
fi
if [[ ! -d "${input_root}" ]]; then
  echo "Input root not found: ${input_root}" >&2
  exit 1
fi
if [[ ! -f "${config_path}" ]]; then
  echo "Configuration not found: ${config_path}" >&2
  exit 1
fi

visible_gpu_count="$(nvidia-smi --list-gpus | wc -l | tr -d '[:space:]')"
if [[ "${gpu_count}" == "auto" ]]; then
  gpu_count="${visible_gpu_count}"
fi
if ! [[ "${gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--gpu-count must be a positive integer or auto." >&2
  exit 2
fi
if (( gpu_count > visible_gpu_count )); then
  echo "Requested ${gpu_count} GPUs, but only ${visible_gpu_count} are visible." >&2
  exit 1
fi

mkdir -p "$(dirname "${venv_path}")" "${output_dir}"
if [[ ! -x "${venv_path}/bin/python" ]]; then
  uv venv --system-site-packages "${venv_path}"
fi
if ! "${venv_path}/bin/python" -c \
  'import numpy, pandas, pydicom, transformers' >/dev/null 2>&1; then
  uv pip install --python "${venv_path}/bin/python" \
    'numpy<3' 'pandas<4' 'pydicom<4' 'transformers>=4.40,<6'
fi
"${venv_path}/bin/python" -c \
  'import numpy, pandas, pydicom, torch, transformers; assert torch.cuda.is_available()'

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

echo "Starting training with ${gpu_count}/${visible_gpu_count} visible GPUs."
training_command=(
  "${venv_path}/bin/python" -m torch.distributed.run
  --standalone
  --nproc-per-node="${gpu_count}"
  -m rsna2026.train
  --config "${config_path}"
  --mode train
  --input-root "${input_root}"
  --output-dir "${output_dir}"
  --expected-gpu-count "${gpu_count}"
)
if [[ "${build_selection_cache}" == true ]]; then
  training_command+=(--build-selection-cache)
fi
exec "${training_command[@]}"
