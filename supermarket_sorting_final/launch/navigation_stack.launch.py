#!/usr/bin/env python3
"""Launch fresh online SLAM plus Nav2 while retaining simulator odometry.

中文说明：启动在线 SLAM、里程计适配器、可选 MPPI 路径显示适配器和 Nav2
五个生命周期节点。关键适配器退出会关闭整组进程；本文件本身不设置等待时延。
"""

from pathlib import Path
import sys

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    project_root = Path(__file__).resolve().parents[1]
    default_params = str(project_root / "config" / "nav2_params.yaml")
    default_mppi_params = str(project_root / "config" / "nav2_mppi_controller.yaml")
    default_slam_params = str(project_root / "config" / "slam_toolbox_params.yaml")
    package_root = project_root / "src" / "supermarket_sorting_nav2"
    odom_adapter = str(package_root / "navigation" / "odometry_adapter.py")
    mppi_local_plan_adapter = str(
        package_root / "navigation" / "mppi_local_plan_adapter.py"
    )
    params_file = LaunchConfiguration("params_file")
    mppi_params_file = LaunchConfiguration("mppi_params_file")
    slam_params_file = LaunchConfiguration("slam_params_file")
    controller = LaunchConfiguration("controller")
    autostart = LaunchConfiguration("autostart")

    odom_adapter_process = ExecuteProcess(
        cmd=[sys.executable, odom_adapter],
        output="screen",
    )
    odom_adapter_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=odom_adapter_process,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="required Nav2 odom adapter exited")
                )
            ],
        )
    )

    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[slam_params_file],
    )
    slam_toolbox_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=slam_toolbox_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="required slam_toolbox process exited")
                )
            ],
        )
    )

    lifecycle_nodes = [
        "controller_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Full path to the Nav2 parameter file",
            ),
            DeclareLaunchArgument(
                "autostart",
                default_value="true",
                description="Automatically configure and activate Nav2 nodes",
            ),
            DeclareLaunchArgument(
                "controller",
                default_value="dwb",
                description="Local controller selection: dwb or mppi",
            ),
            DeclareLaunchArgument(
                "mppi_params_file",
                default_value=default_mppi_params,
                description="MPPI-only parameter overlay",
            ),
            DeclareLaunchArgument(
                "slam_params_file",
                default_value=default_slam_params,
                description="Fresh online-mapping slam_toolbox parameter file",
            ),
            # Register first so immediate required-process failures shut Nav2 down.
            odom_adapter_exit_handler,
            slam_toolbox_exit_handler,
            odom_adapter_process,
            slam_toolbox_node,
            ExecuteProcess(
                cmd=[sys.executable, mppi_local_plan_adapter],
                output="screen",
                condition=IfCondition(
                    PythonExpression(["'", controller, "' == 'mppi'"])
                ),
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[params_file],
                remappings=[("cmd_vel", "/cmd_vel_nav2_raw")],
                condition=IfCondition(
                    PythonExpression(["'", controller, "' == 'dwb'"])
                ),
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[params_file, mppi_params_file],
                remappings=[("cmd_vel", "/cmd_vel_nav2_raw")],
                condition=IfCondition(
                    PythonExpression(["'", controller, "' == 'mppi'"])
                ),
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[params_file],
                remappings=[("cmd_vel", "/cmd_vel_nav2_raw")],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"use_sim_time": False},
                    {"autostart": autostart},
                    {"node_names": lifecycle_nodes},
                ],
            ),
        ]
    )
