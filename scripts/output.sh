#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command kaggle
require_configured_metadata
mkdir -p "${OUTPUT_DIR}"
kaggle kernels output "$(kernel_id)" -p "${OUTPUT_DIR}" -o
echo "Downloaded run artifacts to ${OUTPUT_DIR}"
