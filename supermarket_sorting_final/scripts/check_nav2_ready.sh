#!/usr/bin/env bash
set -eo pipefail

# 中文说明：Nav2 启动就绪检查器。默认最多等待 45 s，每 1 s 重试一次；
# 同时核对 action、节点、话题、生命周期、控制器插件、消息流和完整 TF 链。

source /opt/ros/humble/setup.bash
set -u

timeout_sec="${NAV2_READY_TIMEOUT:-45}"  # 整体就绪检查上限（秒）
controller="${CONTROLLER:-dwb}"
if ! [[ "$timeout_sec" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NAV2_READY_TIMEOUT must be a positive integer" >&2
  exit 2
fi
case "$controller" in
  dwb)
    expected_controller_plugin="dwb_core::DWBLocalPlanner"
    ;;
  mppi)
    expected_controller_plugin="nav2_mppi_controller::MPPIController"
    ;;
  *)
    echo "ERROR: CONTROLLER must be dwb or mppi (got: $controller)" >&2
    exit 2
    ;;
esac

required_nodes=(
  /controller_server
  /planner_server
  /behavior_server
  /bt_navigator
  /waypoint_follower
)
required_graph_nodes=(
  /nav2_odom_adapter
  /slam_toolbox
)
required_graph_topics=(
  /map
)
required_topics=(
  /slamware_ros_sdk_server_node/odom
  /nav2/odom
  /slamware_ros_sdk_server_node/scan
)
if [[ "$controller" == "mppi" ]]; then
  required_graph_nodes+=(/mppi_local_plan_adapter)
  # The publisher exists before a goal; messages begin when MPPI computes.
  required_graph_topics+=(/local_plan)
fi

deadline=$((SECONDS + timeout_sec))
last_reason="waiting for ROS graph"

while (( SECONDS < deadline )); do  # 所有失败分支均在 1 s 后重新检查
  if [[ -n "${NAV2_LAUNCH_PID:-}" ]] && ! kill -0 "$NAV2_LAUNCH_PID" 2>/dev/null; then
    echo "ERROR: Nav2 launch process exited before readiness" >&2
    exit 1
  fi

  action_list="$(ros2 action list 2>/dev/null || true)"
  node_list="$(ros2 node list 2>/dev/null || true)"
  topic_list="$(ros2 topic list 2>/dev/null || true)"

  if ! grep -Fxq /navigate_to_pose <<<"$action_list"; then
    last_reason="missing /navigate_to_pose action"
    sleep 1
    continue
  fi

  graph_nodes_ready=1
  for node in "${required_graph_nodes[@]}"; do
    if ! grep -Fxq "$node" <<<"$node_list"; then
      last_reason="missing required node $node"
      graph_nodes_ready=0
      break
    fi
  done
  if (( ! graph_nodes_ready )); then
    sleep 1
    continue
  fi

  graph_topics_ready=1
  for topic in "${required_graph_topics[@]}"; do
    if ! grep -Fxq "$topic" <<<"$topic_list"; then
      last_reason="missing required graph topic $topic"
      graph_topics_ready=0
      break
    fi
  done
  if (( ! graph_topics_ready )); then
    sleep 1
    continue
  fi

  topics_ready=1
  for topic in "${required_topics[@]}"; do
    if ! grep -Fxq "$topic" <<<"$topic_list"; then
      last_reason="missing required topic $topic"
      topics_ready=0
      break
    fi
  done
  if (( ! topics_ready )); then
    sleep 1
    continue
  fi

  lifecycle_ready=1
  for node in "${required_nodes[@]}"; do
    state="$(ros2 lifecycle get "$node" 2>/dev/null || true)"
    lifecycle_name="${state%%[[:space:]]*}"
    if [[ "$lifecycle_name" != "active" ]]; then
      last_reason="$node lifecycle is not ACTIVE (${state:-unavailable})"
      lifecycle_ready=0
      break
    fi
  done

  if (( lifecycle_ready )); then
    loaded_controller="$(
      ros2 param get /controller_server FollowPath.plugin 2>/dev/null || true
    )"
    if ! grep -Fq "$expected_controller_plugin" <<<"$loaded_controller"; then
      last_reason="controller plugin mismatch: expected $expected_controller_plugin, got ${loaded_controller:-unavailable}"
      sleep 1
      continue
    fi
    if [[ "$controller" == "mppi" ]]; then
      mppi_visualize="$(
        ros2 param get /controller_server FollowPath.visualize 2>/dev/null || true
      )"
      if ! grep -Fq "True" <<<"$mppi_visualize"; then
        last_reason="MPPI visualization must be enabled for /local_plan (${mppi_visualize:-unavailable})"
        sleep 1
        continue
      fi
    fi

    messages_ready=1
    for topic in "${required_topics[@]}"; do
      # 每个关键话题最多等 3 s 收到一条消息。
      if ! timeout 3 ros2 topic echo --once "$topic" >/dev/null 2>&1; then
        last_reason="required topic has no message: $topic"
        messages_ready=0
        break
      fi
    done
    if (( ! messages_ready )); then
      sleep 1
      continue
    fi

    # 在线建图首帧通常较慢，因此 /map 单独允许 7 s。
    if ! timeout 7 ros2 topic echo --once /map \
      --qos-durability transient_local >/dev/null 2>&1; then
      last_reason="slam_toolbox /map has no message"
      sleep 1
      continue
    fi

    # TF 三段各最多查询 3 s，防止 tf2_echo 永久阻塞启动器。
    map_odom_tf="$(timeout 3 ros2 run tf2_ros tf2_echo map odom 2>&1 || true)"
    if ! grep -q "Translation:" <<<"$map_odom_tf"; then
      last_reason="missing SLAM TF map -> odom"
      sleep 1
      continue
    fi

    odom_tf="$(timeout 3 ros2 run tf2_ros tf2_echo odom base_link 2>&1 || true)"
    if ! grep -q "Translation:" <<<"$odom_tf"; then
      last_reason="missing TF odom -> base_link"
      sleep 1
      continue
    fi
    laser_tf="$(timeout 3 ros2 run tf2_ros tf2_echo base_link laser 2>&1 || true)"
    if ! grep -q "Translation:" <<<"$laser_tf"; then
      last_reason="missing TF base_link -> laser"
      sleep 1
      continue
    fi

    echo "NAV2_READY controller=$controller plugin=$expected_controller_plugin action=/navigate_to_pose lifecycle=ACTIVE slam=online map=live odom_adapter=ready local_plan=ready tf=map->odom->base_link->laser"
    exit 0
  fi
  sleep 1
done

echo "ERROR: Nav2 not ready after ${timeout_sec}s: $last_reason" >&2
echo "Actions:" >&2
ros2 action list 2>/dev/null >&2 || true
printf 'Controller plugin: ' >&2
ros2 param get /controller_server FollowPath.plugin 2>/dev/null >&2 || echo "unavailable" >&2
for node in "${required_nodes[@]}"; do
  printf '%s: ' "$node" >&2
  ros2 lifecycle get "$node" 2>/dev/null >&2 || echo "unavailable" >&2
done
exit 1
