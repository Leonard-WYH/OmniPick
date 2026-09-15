#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERCEPTION_DIR="${SCRIPT_DIR}/my_baseline/examples/supermarket_sorting/perception"
IMAGE="${YOLO_SERVER_IMAGE:-supermarket_sorting:server}"
CONTAINER_PERCEPTION="/workspace/supermarket_sorting_task/examples/supermarket_sorting/perception"
CACHE_DIR="${SCRIPT_DIR}/.cache"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[yolo-dataset] Docker image not found: ${IMAGE}" >&2
  exit 1
fi

if [[ ! -f "${PERCEPTION_DIR}/gen_dataset.py" ]]; then
  echo "[yolo-dataset] Generator not found: ${PERCEPTION_DIR}/gen_dataset.py" >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}/torch_extensions" "${CACHE_DIR}/torch_inductor" "${CACHE_DIR}/xdg"

if (( $# == 0 )); then
  set -- \
    --frames 1600 \
    --variants 1 \
    --pose-mode diverse \
    --layout-shuffle-every 8 \
    --debug-overlay \
    --debug-max 240 \
    --overwrite
fi

echo "[yolo-dataset] image=${IMAGE}"
echo "[yolo-dataset] host perception=${PERCEPTION_DIR}"

exec docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" \
  -e LOGNAME="$(id -un)" \
  -e HOME=/tmp/yolo-home \
  -e XDG_CACHE_HOME=/tmp/yolo-cache \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-yolo \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch-extensions-yolo \
  -e PYTHONUNBUFFERED=1 \
  -e MUJOCO_GL=egl \
  -v "${PERCEPTION_DIR}:${CONTAINER_PERCEPTION}" \
  -v "${CACHE_DIR}/torch_extensions:/tmp/torch-extensions-yolo" \
  -v "${CACHE_DIR}/torch_inductor:/tmp/torchinductor-yolo" \
  -v "${CACHE_DIR}/xdg:/tmp/yolo-cache" \
  "${IMAGE}" \
  bash -c "cd /workspace/supermarket_sorting_task/examples/supermarket_sorting && python3 perception/gen_dataset.py \"\$@\"" \
  -- "$@"
