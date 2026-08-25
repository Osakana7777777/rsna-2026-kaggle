#!/usr/bin/env bash

set -euo pipefail
run_kind="${1:-submission}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "${run_kind}"

require_command kaggle
"${PROJECT_ROOT}/scripts/prepare.sh" "${run_kind}"
kaggle kernels push -p "${BUILD_DIR}"
