#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${script_dir}/push.sh" eda
"${script_dir}/wait.sh" eda
"${script_dir}/output.sh" eda
