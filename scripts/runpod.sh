#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/runpod.sh gpus
  ./scripts/runpod.sh create --gpu-id GPU_ID [options]
  ./scripts/runpod.sh get POD_ID
  ./scripts/runpod.sh stop POD_ID
  ./scripts/runpod.sh delete POD_ID

Create options:
  --gpu-id GPU_ID             Exact ID shown by the gpus command (required)
  --gpu-count COUNT           Number of GPUs (default: 2)
  --name NAME                 Pod name (default: rsna-2026-experiment)
  --image IMAGE               Container image
  --cloud-type TYPE           SECURE or COMMUNITY (default: SECURE)
  --container-disk-gb SIZE    Ephemeral container disk (default: 50)
  --volume-gb SIZE             Persistent Pod volume (default: 200)
  --network-volume-id ID      Attach an existing network volume instead
  --data-center-ids IDS       Comma-separated preferred data centers
  --dry-run                   Print the create command without renting a Pod
EOF
}

require_runpodctl() {
  if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl is required. Install it and run: runpodctl doctor" >&2
    exit 1
  fi
}

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  usage >&2
  exit 2
fi
shift

case "${command_name}" in
  gpus)
    require_runpodctl
    exec runpodctl gpu list
    ;;
  get|stop|delete)
    require_runpodctl
    if [[ "$#" -ne 1 ]]; then
      usage >&2
      exit 2
    fi
    exec runpodctl pod "${command_name}" "$1"
    ;;
  create)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

gpu_id=""
gpu_count=2
pod_name="rsna-2026-experiment"
image_name="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
cloud_type="SECURE"
container_disk_gb=50
volume_gb=200
network_volume_id=""
data_center_ids=""
dry_run=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --gpu-id) gpu_id="${2:?Missing value for --gpu-id}"; shift 2 ;;
    --gpu-count) gpu_count="${2:?Missing value for --gpu-count}"; shift 2 ;;
    --name) pod_name="${2:?Missing value for --name}"; shift 2 ;;
    --image) image_name="${2:?Missing value for --image}"; shift 2 ;;
    --cloud-type) cloud_type="${2:?Missing value for --cloud-type}"; shift 2 ;;
    --container-disk-gb) container_disk_gb="${2:?Missing value for --container-disk-gb}"; shift 2 ;;
    --volume-gb) volume_gb="${2:?Missing value for --volume-gb}"; shift 2 ;;
    --network-volume-id) network_volume_id="${2:?Missing value for --network-volume-id}"; shift 2 ;;
    --data-center-ids) data_center_ids="${2:?Missing value for --data-center-ids}"; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${gpu_id}" ]]; then
  echo "--gpu-id is required. Run './scripts/runpod.sh gpus' first." >&2
  exit 2
fi
if ! [[ "${gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--gpu-count must be a positive integer." >&2
  exit 2
fi
if [[ "${cloud_type}" != "SECURE" && "${cloud_type}" != "COMMUNITY" ]]; then
  echo "--cloud-type must be SECURE or COMMUNITY." >&2
  exit 2
fi

create_command=(
  runpodctl pod create
  --name "${pod_name}"
  --gpu-id "${gpu_id}"
  --gpu-count "${gpu_count}"
  --image "${image_name}"
  --cloud-type "${cloud_type}"
  --container-disk-in-gb "${container_disk_gb}"
  --volume-mount-path /workspace
  --ports 8888/http,22/tcp
  --docker-args "sleep infinity"
)
if [[ -n "${network_volume_id}" ]]; then
  create_command+=(--network-volume-id "${network_volume_id}")
else
  create_command+=(--volume-in-gb "${volume_gb}")
fi
if [[ -n "${data_center_ids}" ]]; then
  create_command+=(--data-center-ids "${data_center_ids}")
fi

if [[ "${dry_run}" == true ]]; then
  printf '%q ' "${create_command[@]}"
  printf '\n'
  exit 0
fi

require_runpodctl
echo "Creating ${gpu_count} GPU Pod with ${gpu_id}. Billing starts when deployed."
"${create_command[@]}"
echo "After connecting and staging the permitted inputs, run scripts/runpod-train.sh inside the repository."
