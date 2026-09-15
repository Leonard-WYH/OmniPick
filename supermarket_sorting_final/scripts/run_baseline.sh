#!/usr/bin/env bash
set -eo pipefail

# 中文说明：单可乐 Baseline 启动器。后台运行专用视觉，前台运行基础抓取节点；
# 任一退出或收到终止信号时回收视觉进程。本脚本没有人为 sleep 时延。

baseline_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$baseline_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$baseline_root"

python3 -m supermarket_sorting_nav2.perception.cola_detector \
  --weights "${SUPERMARKET_BASELINE_WEIGHTS:-$baseline_root/weights/products.pt}" \
  --device "${SUPERMARKET_DETECTOR_DEVICE:-auto}" &
detector_pid=$!

cleanup() {
  kill "$detector_pid" 2>/dev/null || true
  wait "$detector_pid" 2>/dev/null || true
}
trap cleanup EXIT

python3 -m supermarket_sorting_nav2.baseline_grasp_controller
