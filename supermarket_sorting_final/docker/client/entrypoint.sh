#!/usr/bin/env bash
set -e

# 中文说明：Client 镜像入口，只加载 ROS 2 Humble 环境后原样执行传入命令；
# 本脚本没有 sleep、轮询或额外启动时延。

source /opt/ros/humble/setup.bash

exec "$@"
