#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command kaggle
require_configured_metadata

interval="${KAGGLE_POLL_SECONDS:-30}"
while true; do
  status="$(kaggle kernels status "$(kernel_id)" 2>&1)"
  echo "${status}"
  status_lower="$(printf '%s' "${status}" | tr '[:upper:]' '[:lower:]')"
  case "${status_lower}" in
    *complete*) exit 0 ;;
    *failed*|*error*|*cancel*) exit 1 ;;
  esac
  sleep "${interval}"
done
