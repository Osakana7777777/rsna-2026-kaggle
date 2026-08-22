#!/usr/bin/env bash

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require_command kaggle
require_configured_metadata
kaggle kernels status "$(kernel_id)"
