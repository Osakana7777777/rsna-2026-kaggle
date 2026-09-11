#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  echo "Usage: $0 deploy|smoke|run [Modal arguments]" >&2
  exit 2
fi
shift

export UV_CACHE_DIR="${TMPDIR:-/tmp}/rsna-2026-kaggle-uv-cache"
case "${command_name}" in
  deploy)
    exec uv run --group labels modal deploy \
      "${PROJECT_ROOT}/modal/report_labels.py" "$@"
    ;;
  smoke)
    exec uv run --group labels modal run \
      "${PROJECT_ROOT}/modal/report_labels.py" --smoke "$@"
    ;;
  run)
    exec uv run --group labels modal run \
      "${PROJECT_ROOT}/modal/report_labels.py" "$@"
    ;;
  *)
    echo "Usage: $0 deploy|smoke|run [Modal arguments]" >&2
    exit 2
    ;;
esac
