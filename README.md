# OmniPick 智慧零售机器人

基于 ROS 2 Humble 的商超移动操作机器人项目。系统在随机商品与随机障碍场景中，
完成在线建图、商品识别、库存记忆、自主调度、避障导航、视觉伺服、抓取和定点放置。

## 功能概览

- `slam_toolbox` 在线构建二维占据栅格地图；
- Nav2 使用 NavFn 全局规划和 MPPI 局部控制；
- YOLOv8 + RGB-D 完成九类商品检测与三维定位；
- ArUco 货位关联与多帧投票构建 A～E 柜库存；
- 任务状态机根据库存、距离和失败记录动态选择目标；
- 双臂/单臂视觉伺服抓取，并将商品放到指定位置。

```text
RGB-D / LiDAR / Odom
        ↓
YOLO + ArUco + 在线 SLAM
        ↓
库存记忆与任务调度
        ↓
Nav2 导航 → 视觉伺服 → 抓取 → 放置
```

## 仓库结构

```text
market_game/
├── run_final_test.sh             # 宿主机统一启动器
├── supermarket_sorting_final/    # 正式 ROS 2 导航、感知与抓放项目
│   ├── config/ launch/ scripts/
│   ├── src/supermarket_sorting_nav2/
│   ├── tests/
│   └── weights/products.pt       # 仓库唯一保留的九类权重
├── YOLO_training/                # 合成数据生成、训练与评测工具
└── images/                       # Docker 镜像下载说明；tar 文件不进入 Git
```

## 运行要求

- Ubuntu 22.04 桌面环境；
- NVIDIA GPU、可用驱动与 `nvidia-smi`；
- Docker Engine 与 NVIDIA Container Toolkit；
- X11、`xhost` 和 `gnome-terminal`；
- 建议至少预留 50 GB 磁盘空间。

将仓库克隆或解压为当前用户主目录下的 `market_game`，后续命令统一从这里执行：

```bash
cd ~/market_game
```

## Docker 镜像

运行需要 `supermarket_sorting:server` 和 `supermarket_sorting:client-nav2`。镜像归档托管在
[Hugging Face：OmniPick-Docker-Image](https://huggingface.co/datasets/Leonard-WYH/OmniPick-Docker-Image/tree/main)，
不提交到 GitHub。

安装 Hugging Face CLI 并下载：

```bash
python3 -m pip install -U "huggingface_hub[cli]"
cd ~/market_game
hf download Leonard-WYH/OmniPick-Docker-Image \
  supermarket_sorting_server.tar \
  supermarket_sorting_client-nav2.tar \
  --repo-type dataset \
  --local-dir images
```

加载镜像：

```bash
docker load -i images/supermarket_sorting_server.tar
docker load -i images/supermarket_sorting_client-nav2.tar
docker image ls | grep supermarket_sorting
```

更完整的镜像说明见 [images/README.md](images/README.md)。

## 快速启动

先做只读检查：

```bash
cd ~/market_game
chmod +x run_final_test.sh
./run_final_test.sh --check
```

启动正式随机任务：

```bash
RUN_MODE=random_cycle \
CONTROLLER=mppi \
ENABLE_RANDOM_OBSTACLES=1 \
FULL_ARM_CLEARANCE_VERIFIED=1 \
./run_final_test.sh
```

Client 显示 `RANDOM_CYCLE_READY` 后，在另一终端发布任务：

```bash
docker exec supermarket_sorting_final_client bash -lc '
source /opt/ros/humble/setup.bash
ros2 topic pub -r 1 \
  /supermarket_sorting/mission_command \
  std_msgs/msg/String "data: clear_random"
'
```

每轮随机抽取 5 个不同商品实体，商品类别允许重复，也允许包含纸巾。包含纸巾时优先
抓取第一件纸巾并放到 3 号点，其余商品依次使用 1、2、4、5 号点；不含纸巾时依次
使用 1～5 号点。

停止本项目容器：

```bash
./run_final_test.sh --stop
```

固定 `TASK_SEED`、`PRODUCT_SEED` 和 `OBSTACLE_SEED` 可复现实验。运行日志与计时文件
会在本地生成，但已被 `.gitignore` 排除。

## YOLO 数据

生成合成数据集：

```bash
cd ~/market_game/YOLO_training
chmod +x generate_dataset.sh test_yolo.sh
./generate_dataset.sh
```

数据集、训练输出和评测图均为可再生文件，不进入 Git。训练完成后如需更新正式模型，
只替换 `supermarket_sorting_final/weights/products.pt`。详见
[YOLO_training/README_DATASET.md](YOLO_training/README_DATASET.md)。

## 测试

在已启动的 Client 容器中运行回归测试：

```bash
docker exec supermarket_sorting_final_client bash -lc '
source /opt/ros/humble/setup.bash
cd /workspace/baseline
PYTHONPATH="/workspace/baseline/src:${PYTHONPATH:-}" \
python3 -m unittest discover -s tests -p "test_*.py" -v
'
```

## 文档

- [正式任务与运行模式](supermarket_sorting_final/README_FULL.md)
- [SLAM、Nav2 与控制链路](supermarket_sorting_final/README_NAV2.md)
- [配置、动作与时延说明](supermarket_sorting_final/中文配置与时延说明.md)
- [YOLO 数据集生成与评测](YOLO_training/README_DATASET.md)

## License

正式子项目采用 [MIT License](supermarket_sorting_final/LICENSE)。Docker 镜像、基础仿真
工程及第三方模型仍受各自许可证约束。
