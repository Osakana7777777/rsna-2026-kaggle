#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METADATA_FILE="${PROJECT_ROOT}/kaggle/kernel-metadata.json"
BUILD_DIR="${PROJECT_ROOT}/.kaggle-build"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"

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
