# supermarket_sorting_final：SLAM 与 Nav2 说明

本文对应当前 `supermarket_sorting_final` 的导航实现。正式启动使用工作区根目录的
`run_final_test.sh`，不要使用历史项目的 `run_nav2_test.sh`。

## 1. 快速启动

只验证建图和导航：

```bash
cd ~/OmniPick
RUN_MODE=nav_only CONTROLLER=mppi ENABLE_RANDOM_OBSTACLES=1 ./run_final_test.sh
```

正式随机分拣：

```bash
RUN_MODE=random_cycle \
CONTROLLER=mppi \
ENABLE_RANDOM_OBSTACLES=1 \
FULL_ARM_CLEARANCE_VERIFIED=1 \
./run_final_test.sh
```

当前默认值是 `RUN_MODE=nav_only`、`CONTROLLER=mppi`、
`ENABLE_RANDOM_OBSTACLES=1`。启动器创建的容器为：

```text
supermarket_sorting_final_server
supermarket_sorting_final_client
```

## 2. 导航数据链路

```text
仿真 LaserScan ───────────────┐
                              ├─ slam_toolbox ── /map、map→odom
原始 Odom ── odometry_adapter ┘
      │
      ├─ odom→base_link
      └─ /nav2/odom ── Nav2 Controller Server

/map + LaserScan + TF
      └─ global/local costmap
              ├─ NavFn/Dijkstra ── /plan
              └─ MPPI ── /cmd_vel、/trajectories
```

TF 主链为：

```text
map -> odom -> base_link -> laser
```

- 仿真器发布 `/slamware_ros_sdk_server_node/scan` 和原始里程计；
- `odometry_adapter.py` 把世界坐标语义的平面速度旋转到 `base_link`，发布
  `/nav2/odom`；
- `slam_toolbox` 运行异步 mapping 模式，每次启动从空地图开始；
- 本项目不使用 AMCL、预存地图或 map saver；
- 任务航点使用 `odom` 坐标，发送 Nav2 时使用零时间戳，以便每次重规划使用最新
  `map→odom` 变换。

## 3. 在线 SLAM

配置文件：`config/slam_toolbox_params.yaml`。

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `scan_topic` | `/slamware_ros_sdk_server_node/scan` | 激光输入 |
| `resolution` | 0.05 m | 占据栅格分辨率 |
| `transform_publish_period` | 0.02 s | 以50 Hz发布 `map→odom` |
| `map_update_interval` | 5.0 s | 占据栅格更新周期 |
| `minimum_time_interval` | 0.5 s | 纳入扫描匹配的最小时间间隔 |
| `transform_timeout` | 0.2 s | 查询 TF 的容忍上限 |
| `tf_buffer_duration` | 30.0 s | TF 历史缓存长度，不是停车等待 |

动态障碍首先进入 LaserScan 障碍层；SLAM 地图按 `map_update_interval` 更新，因此
局部避障不需要等整张 `/map` 再发布一次。

## 4. 全局规划器

配置文件：`config/nav2_params.yaml`。

当前真正被行为树指定的是：

```yaml
planner_id: GridBased
GridBased:
  plugin: nav2_navfn_planner/NavfnPlanner
  use_astar: false
  allow_unknown: true
```

所以正式全局路径使用 NavFn 的 Dijkstra 搜索，不是 A*。`SmacGrid` 插件仍保留在
`planner_plugins` 中用于对照，但当前两个行为树都没有选择它。

全局代价地图参数：

| 项目 | 当前值 |
| --- | ---: |
| 坐标系 | `map` |
| 窗口 | 20 m × 20 m，rolling window |
| 分辨率 | 0.05 m |
| 更新/发布频率 | 5 Hz / 2 Hz |
| 图层 | StaticLayer、ObstacleLayer、InflationLayer |
| 膨胀半径 | 0.30 m |
| 膨胀代价系数 | 10.0 |

Planner Server 的 `expected_planner_frequency=5 Hz` 是性能期望值，不等于任务中的
全局重规划频率。实际重规划周期由行为树 `RateController` 决定。

## 5. 局部控制器与局部代价地图

正式默认控制器是 MPPI，配置文件为 `config/nav2_mppi_controller.yaml`。

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| Controller Server | 10 Hz | 每0.1秒计算一次速度 |
| `time_steps` | 30 | 每条候选轨迹30步 |
| `model_dt` | 0.1 s | 每步0.1秒 |
| 预测窗 | 3.0 s | `30 × 0.1`，不是固定等待 |
| `batch_size` | 1000 | 每轮采样轨迹数量 |
| `vx_max` | 0.75 m/s | MPPI 常规最大前向速度 |
| `wz_max` | 1.8 rad/s | MPPI 常规最大角速度 |
| `reset_period` | 1.0 s | 优化器内部复位周期 |
| `transform_tolerance` | 0.30 s | TF 时间容差 |

局部代价地图当前为 `odom` 坐标系、6 m × 6 m rolling window、0.05 m 分辨率，
8 Hz 更新、4 Hz 发布，使用 ObstacleLayer 和 InflationLayer。它不加载 StaticLayer。

双臂夹持负载导航时，状态机会动态把速度限制为 `0.045 m/s` 和 `0.35 rad/s`；
完成该商品放置后恢复常规 MPPI 角速度上限 `1.8 rad/s`。这组动态限制不会修改
YAML 中的常规上限。

需要对照时可以指定 `CONTROLLER=dwb`。DWB 与 MPPI 共用同一个 NavFn 全局规划器、
代价地图和 `/nav2/odom`；切换局部控制器不会改变全局搜索算法。

## 6. 行为树和重规划

普通导航行为树：`config/navigate_to_pose_w_replanning_and_recovery.xml`。

- `RateController hz="0.25"`：每4秒重算一次全局路径；
- 规划失败先清全局代价地图；
- 跟踪失败先清局部代价地图；
- 恢复轮询为：清局部/全局代价地图 → 原地旋转1.57 rad → 后退0.30 m；
- `RecoveryNode number_of_retries="6"`；
- 普通行为树没有固定 `Wait`。

双臂夹持负载行为树：`config/navigate_to_pose_tissue_safe.xml`。

- 同样以0.25 Hz重算全局路径；
- 仍然允许清理局部/全局代价地图；
- 禁用 Spin 和 BackUp，避免恢复动作绕过负载速度限制；
- 恢复轮次使用3秒 `Wait`。

要修改任务执行时的全局重规划频率，需要同时修改两个 XML 的
`RateController hz`，只改 `expected_planner_frequency` 不会改变实际周期。

## 7. `/cmd_vel` 控制权

底盘并非全程只由 Nav2 控制：

| 阶段 | 控制者 |
| --- | --- |
| 开放区域长距离导航 | Nav2 Controller Server |
| 第一次驶向 E 柜的高速段 | 任务状态机直接速度控制 |
| 货柜前抓取接近 | 两阶段视觉伺服直接速度控制 |
| 抓取后退出货柜 | 任务状态机直接速度控制 |
| 放置点0.40 m内航向辅助 | Nav2保留线速度，状态机覆盖角速度 |
| 放置完成后退出桌边 | 任务状态机直接速度控制 |

第一次驶向 E 柜的直接巡航配置为 `2.0 m/s`，这是发布目标值；实际速度仍受底盘
加速度、动力学、剩余距离、转向误差和安全切换门槛限制。放置点0.40 m内的固定辅助
角速度为 `0.35 rad/s`，航向误差进入0.10 rad后可以尽快切换到放置动作。

不要同时运行基线控制器和正式任务节点，否则会产生多个 `/cmd_vel` 发布者。

## 8. RViz 与常用检查

启动器加载 `rviz/supermarket_nav2.rviz`。主要显示：

- `/map`、TF 和 LaserScan；
- `/global_costmap/costmap`、`/local_costmap/costmap`；
- 全局路径 `/plan`；
- MPPI 轨迹以及由 `mppi_local_plan_adapter.py` 转换的 `/local_plan`；
- 机器人 footprint、任务目标和库存航点。

进入当前 Client 容器检查：

```bash
docker exec -it supermarket_sorting_final_client bash
source /opt/ros/humble/setup.bash

ros2 lifecycle get /controller_server
ros2 lifecycle get /planner_server
ros2 lifecycle get /behavior_server
ros2 lifecycle get /bt_navigator
ros2 action list | grep /navigate_to_pose

ros2 topic echo /map --once --qos-durability transient_local --field info
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_link
ros2 run tf2_ros tf2_echo base_link laser

ros2 param get /planner_server GridBased.plugin
ros2 param get /planner_server GridBased.use_astar
ros2 param get /global_costmap/global_costmap width
ros2 param get /local_costmap/local_costmap width
ros2 param get /controller_server FollowPath.plugin
```

没有活动导航目标时，`ros2 topic echo /plan --once` 或 `/local_plan --once` 会等待
下一条路径消息，属于正常现象。

## 9. 常见故障

### 有全局路径但机器人不动

依次检查 Controller Server 生命周期、`/nav2/odom`、局部代价地图、底盘是否被任务
状态机接管，以及是否存在多个 `/cmd_vel` 发布者。重点搜索 Client 日志中的
`No valid trajectories`、`collision`、`Failed to make progress` 和 Action 失败原因。

### 进入膨胀区后停顿

检查局部代价地图实际障碍、机器人 footprint、InflationLayer 和 MPPI critic，而不是
仅增大速度。当前局部地图已经是6 m窗口；更大的地图不能替代合理的制动距离和轨迹
代价参数。

### 全局路径贴边

NavFn 会综合栅格路径长度与代价。可检查 Static/Obstacle/Inflation 图层是否正常、
膨胀是否连续，以及目标点本身是否靠近高代价区。当前正式规划器是 Dijkstra；若改成
其他插件，必须同步修改行为树中的 `planner_id` 并重新启动。

### 修改配置后没有生效

ROS 节点不会热加载 YAML 或 XML。执行 `./run_final_test.sh --stop` 后重新启动，确认
使用的是 `supermarket_sorting_final_client`，并排除历史项目容器和旧脚本。
