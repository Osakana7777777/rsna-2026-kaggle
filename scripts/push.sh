#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command kaggle
"${PROJECT_ROOT}/scripts/prepare.sh"
kaggle kernels push -p "${BUILD_DIR}"
