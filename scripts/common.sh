#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_KIND="${1:-submission}"
case "${RUN_KIND}" in
  train)
    METADATA_FILE="${PROJECT_ROOT}/kaggle/train-kernel-metadata.json"
    BUILD_DIR="${PROJECT_ROOT}/.kaggle-build/train"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/train"
    RUN_MODE="train"
    ;;
  submission)
    METADATA_FILE="${PROJECT_ROOT}/kaggle/kernel-metadata.json"
    BUILD_DIR="${PROJECT_ROOT}/.kaggle-build/submission"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/submission"
    RUN_MODE="inference"
    ;;
  eda)
    METADATA_FILE="${PROJECT_ROOT}/kaggle/eda-kernel-metadata.json"
    BUILD_DIR="${PROJECT_ROOT}/.kaggle-build/eda"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/eda"
    RUN_MODE="eda"
    ;;
  preprocess-[0-3])
    shard_index="${RUN_KIND#preprocess-}"
    METADATA_FILE="${PROJECT_ROOT}/kaggle/preprocess-${shard_index}-kernel-metadata.json"
    BUILD_DIR="${PROJECT_ROOT}/.kaggle-build/preprocess-${shard_index}"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/preprocess-${shard_index}"
    RUN_MODE="preprocess-${shard_index}"
    ;;
  *)
    echo "Usage: $0 [train|submission|eda|preprocess-{0..3}]" >&2
    exit 2
    ;;
esac

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

kernel_id() {
  sed -n 's/^[[:space:]]*"id":[[:space:]]*"\([^"]*\)".*/\1/p' "${METADATA_FILE}"
}

require_configured_metadata() {
  if grep -q 'YOUR_' "${METADATA_FILE}" "${PROJECT_ROOT}/configs/baseline.toml"; then
    echo "Replace every YOUR_... placeholder in kernel metadata and config first." >&2
    exit 1
  fi
}
