#!/usr/bin/env bash

set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for shard_index in 0 1 2 3; do
  "${project_root}/scripts/push.sh" "preprocess-${shard_index}"
done

for shard_index in 0 1 2 3; do
  "${project_root}/scripts/wait.sh" "preprocess-${shard_index}"
done

for shard_index in 0 1 2 3; do
  "${project_root}/scripts/status.sh" "preprocess-${shard_index}"
done

echo "All four persistent image-cache shards completed."
