#!/usr/bin/env bash
set -eo pipefail

# 中文说明：Client 容器内部的任务启动器。依次设置 ROS/PYTHON 环境、校验
# 模式和机械臂安全开关、调整扫描高度、启动 SLAM+Nav2、等待就绪、启动视觉，
# 最后前台运行对应任务状态机。退出时统一回收所有后台子进程。

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

# 仓库只维护一份跨显卡可移植的九类 PyTorch 权重。若需要 TensorRT，调用方可
# 通过 SUPERMARKET_PRODUCT_WEIGHTS 指向在目标电脑上临时生成的 .engine 文件。
default_product_weights="$project_root/weights/products.pt"
product_weights="${SUPERMARKET_PRODUCT_WEIGHTS:-$default_product_weights}"

run_mode="${1:-${RUN_MODE:-nav_only}}"
controller="${CONTROLLER:-dwb}"
full_arm_clearance_verified="${FULL_ARM_CLEARANCE_VERIFIED:-0}"
case "$run_mode" in
  nav_only|scan_only|full|e_kele_cycle|e_maidong_cycle|e_mixed_cycle|random_cycle) ;;
  *)
    echo "ERROR: mode must be nav_only, scan_only, full, e_kele_cycle, e_maidong_cycle, e_mixed_cycle or random_cycle (got: $run_mode)" >&2
    exit 2
    ;;
esac
case "$controller" in
  dwb|mppi) ;;
  *)
    echo "ERROR: CONTROLLER must be dwb or mppi (got: $controller)" >&2
    exit 2
    ;;
esac
case "$full_arm_clearance_verified" in
  0|1) ;;
  *)
    echo "ERROR: FULL_ARM_CLEARANCE_VERIFIED must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "$run_mode" == "full" || "$run_mode" == "e_kele_cycle" \
      || "$run_mode" == "e_maidong_cycle" \
      || "$run_mode" == "e_mixed_cycle" \
      || "$run_mode" == "random_cycle" ]] \
    && [[ "$full_arm_clearance_verified" != "1" ]]; then
  echo "ERROR: $run_mode mode is safety-locked because it commands the arms." >&2
  echo "Run with FULL_ARM_CLEARANCE_VERIFIED=1 only after acknowledging the" >&2
  echo "current Server MJCF clearance validation." >&2
  exit 2
fi
cd "$project_root"

nav2_pid=""
detector_pid=""
manual_goal_pid=""
cleanup() {
  # wait 只回收已发送终止信号的子进程，不引入额外固定时延。
  if [[ -n "$manual_goal_pid" ]]; then
    kill "$manual_goal_pid" 2>/dev/null || true
    wait "$manual_goal_pid" 2>/dev/null || true
  fi
  if [[ -n "$detector_pid" ]]; then
    kill "$detector_pid" 2>/dev/null || true
    wait "$detector_pid" 2>/dev/null || true
  fi
  if [[ -n "$nav2_pid" ]]; then
    kill "$nav2_pid" 2>/dev/null || true
    wait "$nav2_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$run_mode" == "scan_only" || "$run_mode" == "full" \
      || "$run_mode" == "e_kele_cycle" \
      || "$run_mode" == "e_maidong_cycle" \
      || "$run_mode" == "e_mixed_cycle" \
      || "$run_mode" == "random_cycle" ]]; then
  echo "===== Lowering torso into the startup scanning posture ====="
  python3 -m supermarket_sorting_nav2.navigation.scan_posture_initializer \
    --slide "${SUPERMARKET_SCAN_SLIDE:-0.15}"
fi

echo "===== Starting fresh online SLAM + Nav2: controller=$controller ====="
ros2 launch "$project_root/launch/navigation_stack.launch.py" \
  params_file:="$project_root/config/nav2_params.yaml" \
  mppi_params_file:="$project_root/config/nav2_mppi_controller.yaml" \
  slam_params_file:="$project_root/config/slam_toolbox_params.yaml" \
  controller:="$controller" &
nav2_pid=$!

echo "===== Waiting for Nav2 lifecycle readiness ====="
# 具体启动上限与每项探测时限由 check_nav2_ready.sh 统一管理。
CONTROLLER="$controller" NAV2_LAUNCH_PID="$nav2_pid" \
  "$project_root/scripts/check_nav2_ready.sh"

echo "===== Starting explicit RViz manual-goal bridge ====="
python3 -m supermarket_sorting_nav2.navigation.manual_goal_bridge &
manual_goal_pid=$!

if [[ "$run_mode" == "scan_only" || "$run_mode" == "e_kele_cycle" \
      || "$run_mode" == "e_maidong_cycle" \
      || "$run_mode" == "e_mixed_cycle" \
      || "$run_mode" == "random_cycle" ]]; then
  echo "===== Starting nine-class product scanner: weights=$product_weights ====="
  python3 -m supermarket_sorting_nav2.perception.product_detector \
    --weights "$product_weights" \
    --aruco-truth "$project_root/config/aruco_truth.json" \
    --confidence "${SUPERMARKET_PRODUCT_CONFIDENCE:-0.50}" \
    --min-votes "${SUPERMARKET_INVENTORY_MIN_VOTES:-3}" \
    --scan-stand-off "${SUPERMARKET_SCAN_STAND_OFF:-0.85}" \
    --approach-stand-off "${SUPERMARKET_APPROACH_STAND_OFF:-0.70}" \
    --device "${SUPERMARKET_DETECTOR_DEVICE:-auto}" &
  detector_pid=$!
elif [[ "$run_mode" == "full" ]]; then
  echo "===== Starting verified kele detector ====="
  python3 -m supermarket_sorting_nav2.perception.cola_detector \
    --weights "${SUPERMARKET_BASELINE_WEIGHTS:-$project_root/weights/products.pt}" \
    --device "${SUPERMARKET_DETECTOR_DEVICE:-auto}" &
  detector_pid=$!
else
  echo "===== nav_only: detector and manipulator control disabled ====="
fi

if [[ "$run_mode" == "scan_only" ]]; then
  echo "===== scan_only ready: torso lowered; no arm/gripper commands ====="
  echo "Use RViz 2D Goal Pose to visit product viewpoints."
  echo "Manual goal route: /goal_pose -> /navigate_to_pose."
  echo "The complete 45-marker /aruco/map comes from config/aruco_truth.json."
  echo "Observe /product/result_image, /aruco/map and /inventory/map."
  scan_pids=("$nav2_pid" "$detector_pid" "$manual_goal_pid")
  set +e
  wait -n "${scan_pids[@]}"  # 长驻模式：任一核心进程退出即结束，不是定时等待
  status=$?
  set -e
  echo "ERROR: scan_only component exited (status=$status)" >&2
  exit "$status"
fi

if [[ "$run_mode" == "e_kele_cycle" || "$run_mode" == "e_maidong_cycle" \
      || "$run_mode" == "e_mixed_cycle" \
      || "$run_mode" == "random_cycle" ]]; then
  case "$run_mode" in
    e_maidong_cycle)
      target_kind="maidong"
      mission_command="clear_e_maidong"
      ;;
    e_mixed_cycle)
      target_kind="mixed"
      mission_command="clear_e_mixed"
      ;;
    random_cycle)
      target_kind="random"
      mission_command="clear_random"
      ;;
    *)
      target_kind="kele"
      mission_command="clear_e_kele"
      ;;
  esac
  echo "===== Shelf $target_kind cycle ready; waiting for mission command ====="
  echo "Start publisher: ros2 topic pub -r 1 /supermarket_sorting/mission_command std_msgs/msg/String 'data: $mission_command'"
  echo "Status topic:   /supermarket_sorting/mission_status"
  python3 -m supermarket_sorting_nav2.autonomous_sorting_mission \
    --target-kind "$target_kind" \
    --target-sequence "${SUPERMARKET_E_MIXED_SEQUENCE:-maidong,kele,maidong}" \
    --scan-slide "${SUPERMARKET_SCAN_SLIDE:-0.30}" \
    --waypoints "$project_root/config/nav_waypoints.yaml"
  exit $?
fi

echo "===== Starting Nav2 mission: $run_mode ====="
python3 -m supermarket_sorting_nav2.nav2_manipulation_client \
  --mode "$run_mode" \
  --scan-slide "${SUPERMARKET_SCAN_SLIDE:-0.15}" \
  --waypoints "$project_root/config/nav_waypoints.yaml"
