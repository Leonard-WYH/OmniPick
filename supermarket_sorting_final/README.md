# supermarket_sorting_final

这是当前正式的 ROS 2 超市自主分拣项目，包含在线 SLAM、Nav2、九类 YOLO 商品
检测、ArUco 货位关联、A～E 柜库存记忆、视觉伺服、机械臂抓放和随机任务调度。

- 完整启动与任务规则：[README_FULL.md](README_FULL.md)
- Nav2、SLAM、代价地图和行为树：[README_NAV2.md](README_NAV2.md)
- 动作停稳、超时和刷新周期：[中文配置与时延说明.md](中文配置与时延说明.md)
- 工作区总说明：[../README.md](../README.md)

正式启动统一从工作区根目录执行 `./run_final_test.sh`。项目使用已有的
`supermarket_sorting:server` 和 `supermarket_sorting:client-nav2` 镜像，不会在启动时
新建镜像。Client 容器名为 `supermarket_sorting_final_client`，项目被挂载到容器内
`/workspace/baseline`。

不要同时运行 `baseline_grasp_controller.py`、`scripts/run_baseline.sh` 与正式 Nav2
任务节点，否则多个进程会同时争用 `/cmd_vel`、机械臂和夹爪话题。
