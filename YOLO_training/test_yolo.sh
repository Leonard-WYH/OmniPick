#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERCEPTION_DIR="${SCRIPT_DIR}/my_baseline/examples/supermarket_sorting/perception"
IMAGE="${YOLO_SERVER_IMAGE:-supermarket_sorting:server}"
CONTAINER_PERCEPTION="/workspace/supermarket_sorting_task/examples/supermarket_sorting/perception"
CACHE_DIR="${SCRIPT_DIR}/.cache"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[yolo-test] Docker image not found: ${IMAGE}" >&2
  exit 1
fi

if [[ ! -f "${PERCEPTION_DIR}/test_yolo.py" ]]; then
  echo "[yolo-test] Test script not found: ${PERCEPTION_DIR}/test_yolo.py" >&2
  exit 1
fi

if [[ ! -f "${PERCEPTION_DIR}/checkpoints/products.pt" ]]; then
  echo "[yolo-test] Trained weights not found: ${PERCEPTION_DIR}/checkpoints/products.pt" >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}/xdg"

echo "[yolo-test] image=${IMAGE}"
echo "[yolo-test] weights=${PERCEPTION_DIR}/checkpoints/products.pt"

exec docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e HOME=/tmp/yolo-home \
  -e XDG_CACHE_HOME=/tmp/yolo-cache \
  -e PYTHONUNBUFFERED=1 \
  -v "${PERCEPTION_DIR}:${CONTAINER_PERCEPTION}" \
  -v "${CACHE_DIR}/xdg:/tmp/yolo-cache" \
  "${IMAGE}" \
  bash -c "cd /workspace/supermarket_sorting_task/examples/supermarket_sorting && python3 perception/test_yolo.py \"\$@\"" \
  -- "$@"
