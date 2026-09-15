# supermarket_sorting_final 完整运行说明

本文只描述当前正式项目 `supermarket_sorting_final`。工作区统一启动入口是根目录的
`run_final_test.sh`；旧目录和旧启动脚本仅用于历史对照，不应启动当前任务。

## 1. 系统能力

当前项目把以下模块组合成一套完整任务：

- `slam_toolbox` 根据 LaserScan、里程计和 TF 在线建图；
- Nav2 使用 NavFn/Dijkstra 全局规划和 MPPI 局部控制；
- YOLO 检测九类商品，统一加载仓库中的 PyTorch 权重；
- ArUco 固定真值、多帧投票和 RGB-D 世界坐标共同建立 A～E 柜库存；
- 调度器维护跨货柜记忆，抓取后从库存中剔除对应实体；
- 视觉伺服、商品抓取配置和状态机完成接近、抓取、运输、放置和退出；
- 每轮可以随机商品、随机货架布局、随机障碍物，并保存完整复现信息。

## 2. 正式随机任务规则

`RUN_MODE=random_cycle` 的任务契约如下：

1. 从45件商品中随机选择5个不同的物理实体；
2. 九类商品都可以出现；不同实体可以属于同一类别，所以品类允许重复；
3. 若5件中含纸巾，把本单中的第一个纸巾提前到首位，优先抓取并放到3号放置点；
4. 含纸巾时，其余4件保持原相对顺序并依次放到1、2、4、5号放置点；
5. 不含纸巾时，5件依次放到1、2、3、4、5号放置点；
6. 若同一订单含多个纸巾，只有第一个纸巾占用3号位，后续纸巾按剩余订单处理；
7. 调度依据视觉库存、机器人与货柜距离、放置代价和货柜策略选择实际抓取货柜，
   订单本身不指定货柜。

随机选择的是实体 ID，而状态机按服务器发布的商品类别计数完成任务。手工指定任务时
也必须提供5个互不重复的 `product_NNN`，不能直接填写类别名。

## 3. 启动前检查

在工作区根目录执行：

```bash
cd ~/market_game
./run_final_test.sh --check
```

检查内容包括 Docker 镜像、正式项目文件、任务参数、随机种子、显示环境和服务器镜像
所需接口。当前使用的资源为：

| 项目 | 当前值 |
| --- | --- |
| Server 镜像 | `supermarket_sorting:server` |
| Client 镜像 | `supermarket_sorting:client-nav2` |
| Server 容器 | `supermarket_sorting_final_server` |
| Client 容器 | `supermarket_sorting_final_client` |
| 项目挂载点 | `supermarket_sorting_final` → `/workspace/baseline:rw` |
| ROS 域 | `ROS_DOMAIN_ID=99` |
| DDS | `rmw_cyclonedds_cpp` |

启动器复用现有镜像，不会自动构建或拉取新镜像。

## 4. 启动正式任务

```bash
cd ~/market_game
RUN_MODE=random_cycle \
CONTROLLER=mppi \
ENABLE_RANDOM_OBSTACLES=1 \
FULL_ARM_CLEARANCE_VERIFIED=1 \
./run_final_test.sh
```

启动器默认值是 `RUN_MODE=nav_only`、`CONTROLLER=mppi`、
`ENABLE_RANDOM_OBSTACLES=1`。因此执行完整分拣时必须显式指定
`RUN_MODE=random_cycle` 和机械臂安全确认。

Client 出现 `RANDOM_CYCLE_READY` 后，在另一个终端发布：

```bash
docker exec supermarket_sorting_final_client bash -lc '
source /opt/ros/humble/setup.bash
ros2 topic pub -r 1 \
  /supermarket_sorting/mission_command \
  std_msgs/msg/String "data: clear_random"
'
```

状态机只接受一次有效启动；任务开始后可以用 `Ctrl+C` 停止重复发布器。查看状态：

```bash
docker exec supermarket_sorting_final_client bash -lc '
source /opt/ros/humble/setup.bash
ros2 topic echo /supermarket_sorting/mission_status
'
```

停止本项目创建的容器：

```bash
cd ~/market_game
./run_final_test.sh --stop
```

## 5. 随机种子与复现

任务选择、商品布局和障碍物布局使用彼此独立的种子：

```bash
RUN_MODE=random_cycle \
CONTROLLER=mppi \
ENABLE_RANDOM_OBSTACLES=1 \
FULL_ARM_CLEARANCE_VERIFIED=1 \
TASK_SEED=20260901 \
PRODUCT_SEED=20260902 \
OBSTACLE_SEED=20260903 \
./run_final_test.sh
```

| 变量 | 作用 | 未指定时 |
| --- | --- | --- |
| `TASK_SEED` | 从45件实体中抽取5件并确定初始顺序 | 纳秒级当前时间 |
| `PRODUCT_SEED` | 服务器随机分配商品货位 | 当前 Unix 秒 |
| `OBSTACLE_SEED` | 随机生成场内障碍物 | 纳秒级当前时间 |
| `TASKS` | 手工指定5个不同实体 ID | `random_cycle` 下自动生成 |

手工任务示例：

```bash
RUN_MODE=random_cycle \
FULL_ARM_CLEARANCE_VERIFIED=1 \
TASKS=product_004,product_013,product_001,product_015,product_024 \
PRODUCT_SEED=20260902 \
OBSTACLE_SEED=20260903 \
./run_final_test.sh
```

上述 ID 可以对应重复品类，但实体 ID 本身不能重复。每轮实际参数都会写进运行日志。

## 6. 运行模式

| `RUN_MODE` | 用途 |
| --- | --- |
| `nav_only` | 在线建图与 Nav2 导航，不启动商品抓取 |
| `scan_only` | 启动九类检测和库存构建，不执行机械臂抓取 |
| `full` | 兼容早期完整流程的回归模式 |
| `e_kele_cycle` | E 柜可乐专项回归 |
| `e_maidong_cycle` | E 柜脉动专项回归 |
| `e_mixed_cycle` | E 柜混合商品专项回归 |
| `random_cycle` | A～E 柜自主观察、调度、抓取和放置的正式模式 |

所有会驱动机械臂的模式都需要 `FULL_ARM_CLEARANCE_VERIFIED=1`。正式任务使用
`CONTROLLER=mppi`；`CONTROLLER=dwb` 仅用于局部控制器对照。

## 7. 感知与库存

仓库正式检测权重只有 `weights/products.pt`，可直接跨兼容的 NVIDIA 显卡运行。

主要话题：

```text
/product/detections                九类商品三维检测，frame_id=odom
/product/result_image              带检测框和关联信息的结果图
/aruco/detections                  当前画面实际识别到的 ArUco
/aruco/map                         45个 ArUco 固定真值地图
/inventory/observations            当前帧商品与货位关联
/inventory/map                     多帧累计库存
/inventory/waypoints               扫描和接近航点
/inventory/remove                  抓取完成后的库存剔除
/supermarket_sorting/mission_command
/supermarket_sorting/mission_status
```

库存不是单帧检测列表。机器人在行驶和观察期间持续累积 A～E 柜信息；已确认商品保留
在任务期记忆中，成功抓取后删除对应实体，并阻止旧观测把它重新写回。调度器优先使用
记忆中的真实候选，只有库存证据不足时才进行补扫。

## 8. 导航、抓取与放置链路

1. `slam_toolbox` 持续发布 `/map` 和 `map→odom`；
2. NavFn 的 `GridBased` 插件在全局代价地图上用 Dijkstra 生成路径；
3. MPPI 在局部代价地图上预测3秒并输出速度；
4. 接近货柜后，状态机从 Nav2 切换到视觉伺服；
5. 视觉伺服用商品中心与夹持中心的像素差动态纠偏，并保留货位记忆作为遮挡后备；
6. 抓取完成后直接速度控制退出货柜，再把控制权交回 Nav2；
7. 到放置点末端区域时，Nav2 保留线速度，状态机辅助角速度并尽快进入放置；
8. 松开、抬起并直接速度后退完成后，非最后一件立即开始下一段导航；
9. 最后一件完成放置、抬起、退出到位并停车后结束任务计时，机器人停在原地。

Nav2 的详细参数见 [README_NAV2.md](README_NAV2.md)，真实动作时延和超时见
[中文配置与时延说明.md](中文配置与时延说明.md)。

## 9. 代码与配置入口

| 文件 | 职责 |
| --- | --- |
| `../run_final_test.sh` | 宿主机参数校验、随机任务、容器、RViz 和日志 |
| `scripts/run_nav2_mission.sh` | Client 内启动 SLAM、Nav2、感知和任务节点 |
| `launch/navigation_stack.launch.py` | ROS 2 导航栈编排 |
| `config/nav2_params.yaml` | NavFn、DWB、代价地图、BT 和公共 Nav2 参数 |
| `config/nav2_mppi_controller.yaml` | MPPI 局部控制参数 |
| `config/slam_toolbox_params.yaml` | 在线 SLAM 参数 |
| `config/nav_waypoints.yaml` | 固定导航与观察航点 |
| `config/aruco_truth.json` | 45个 ArUco 固定世界坐标 |
| `src/supermarket_sorting_nav2/autonomous_sorting_mission.py` | 正式任务节点入口 |
| `src/supermarket_sorting_nav2/sorting_state_machine.py` | 抓取、运输和放置主状态机 |
| `src/supermarket_sorting_nav2/sorting_config.py` | 商品参数、速度、阈值和时延 |
| `src/supermarket_sorting_nav2/product_scheduler.py` | 跨货柜候选选择和放置点分配 |
| `src/supermarket_sorting_nav2/inventory_tracking.py` | 订单解析、库存记忆和实体剔除 |
| `src/supermarket_sorting_nav2/visual_servo.py` | 两阶段视觉伺服 |
| `src/supermarket_sorting_nav2/motion_control.py` | Nav2 交接和直接底盘控制 |
| `src/supermarket_sorting_nav2/tissue_handling.py` | 双臂夹持商品的动作实现 |
| `src/supermarket_sorting_nav2/perception/product_detector.py` | YOLO、RGB-D、ArUco 和库存发布 |
| `src/supermarket_sorting_nav2/perception/yolo_backend.py` | YOLO 权重加载与推理设备选择 |

## 10. 日志与总时长

每轮运行创建：

```text
supermarket_sorting_final/logs/<RUN_ID>/
├── metadata.env
├── launch_config.txt
├── task_message.txt
├── server.log
├── client.log
├── rviz.log
└── timeline.log
```

总计时从机器人第一次产生有效移动开始，到最后一件商品完成放置、抬起、退出并停车
结束。最近一轮时长写入 `mission_total_time.txt`，不依赖终端日志保存结果。

## 11. 回归测试

在具备 ROS/Python 依赖的环境中执行：

```bash
cd ~/market_game
PYTHONPATH="supermarket_sorting_final/src:${PYTHONPATH:-}" \
python3 -m unittest discover \
  -s supermarket_sorting_final/tests \
  -p 'test_*.py' \
  -v
```

测试覆盖随机货柜调度、重复品类计数、Nav2 目标抢占、导航重试、橙子放置、双臂
放置以及最后停车完成条件。宿主机缺少 ROS 依赖时，应在 Client 镜像或已启动的
Client 容器内执行。

## 12. 常见问题

### 机器人不开始任务

确认使用 `RUN_MODE=random_cycle` 启动、Client 已输出 `RANDOM_CYCLE_READY`，并且
任务命令发布到 `supermarket_sorting_final_client`。随后检查：

```bash
docker logs supermarket_sorting_final_client 2>&1 | tail -n 200
```

### 障碍物没有随机

设置 `ENABLE_RANDOM_OBSTACLES=1`，并且不要重复使用相同 `OBSTACLE_SEED`。当前默认
障碍种子来自纳秒级当前时间，不再由 `PRODUCT_SEED` 推导。

### 换电脑后推理变慢

检查 Client 日志中的推理设备和单帧耗时，并确认容器能访问 CUDA。仓库统一使用
`products.pt`，不会加载其他权重文件。

### 修改源码后行为没有变化

Python 节点和 YAML 不会热重载。使用 `./run_final_test.sh --stop` 停止旧容器后重新
启动，不要同时运行历史项目容器或旧任务脚本。
