"""Thin NavigateToPose action wrapper for the online-SLAM Nav2 stack.

中文说明：对 NavigateToPose Action 做单目标异步封装。任务循环默认采用零超时
轮询；只有启动就绪和主动取消时才使用有限时等待，避免堵住主状态机。
"""

import math
import time
from enum import Enum, auto
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .mission_manager import Waypoint


class NavResult(Enum):
    IDLE = auto()
    STARTING = auto()
    ACTIVE = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()


class Nav2Manager:
    """Manage one Nav2 goal at a time without owning the ROS executor."""

    def __init__(self, node, cmd_vel_publisher=None):
        self._node = node
        self._client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
        self._cmd_vel_pub = cmd_vel_publisher or node.create_publisher(Twist, "/cmd_vel", 5)
        goal_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._goal_debug_pub = node.create_publisher(
            PoseStamped, "/nav2_mission/goal_pose", goal_qos
        )
        self._goal_handle = None
        self._send_future = None
        self._result_future = None
        self._cancel_future = None
        self._cancel_requested = False
        self._result = NavResult.IDLE
        self._goal_name = ""
        # Nav2 can preempt an active NavigateToPose goal.  Keep a generation
        # number so the previous goal's late result cannot overwrite the
        # replacement goal's state.
        self._goal_generation = 0

    @property
    def result(self) -> NavResult:
        return self._result

    @property
    def is_active(self) -> bool:
        return self._result in (NavResult.STARTING, NavResult.ACTIVE)

    def wait_until_ready(self, timeout_sec: float = 60.0) -> bool:
        """最多等待 Action Server 60 s；就绪后立即返回。"""
        ready = self._client.wait_for_server(timeout_sec=timeout_sec)
        if not ready:
            self._node.get_logger().error(
                f"NAV_FAILED NavigateToPose server unavailable after {timeout_sec:.1f}s"
            )
        return ready

    def _make_pose(self, waypoint: Waypoint) -> PoseStamped:
        if waypoint.frame_id != "odom":
            raise ValueError("all first-version Nav2 goals must use frame_id=odom")
        pose = PoseStamped()
        pose.header.frame_id = "odom"
        # Keep odom-frame goals valid across long-running map->odom updates.
        # A zero stamp tells tf2 to use the latest transform on every replan;
        # a fixed send-time stamp eventually falls out of the TF cache.
        pose.header.stamp.sec = 0
        pose.header.stamp.nanosec = 0
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        pose.pose.orientation.z = math.sin(waypoint.yaw * 0.5)
        pose.pose.orientation.w = math.cos(waypoint.yaw * 0.5)
        return pose

    def go_to_pose(
        self,
        x,
        y=None,
        yaw=None,
        name="goal",
        frame_id="odom",
        behavior_tree="",
        replace_active=False,
    ) -> bool:
        """Send an odom-frame goal.

        Accepts either a :class:`Waypoint` or ``x, y, yaw`` numeric values.
        """

        if isinstance(x, Waypoint):
            waypoint = x
        else:
            if y is None or yaw is None:
                raise ValueError("go_to_pose requires x, y and yaw")
            waypoint = Waypoint(name, frame_id, float(x), float(y), float(yaw))
        if waypoint.needs_tuning:
            self._node.get_logger().error(f"NAV_FAILED {waypoint.name} NEEDS_TUNING")
            self._result = NavResult.FAILED
            return False
        replacing = bool(self.is_active and replace_active)
        if self.is_active and not replacing:
            self._node.get_logger().error("NAV_FAILED refusing a second active goal")
            return False
        if not self._client.server_is_ready():
            self._node.get_logger().error("NAV_FAILED NavigateToPose server is not ready")
            self._result = NavResult.FAILED
            return False

        pose = self._make_pose(waypoint)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = str(behavior_tree)
        previous_name = self._goal_name
        self._goal_generation += 1
        generation = self._goal_generation
        self._goal_name = waypoint.name
        self._cancel_requested = False
        self._result = NavResult.STARTING
        self._goal_debug_pub.publish(pose)
        self._node.get_logger().info(
            f"{'NAV_REPLACE' if replacing else 'NAV_START'} "
            + (f"previous={previous_name} " if replacing else "")
            + f"name={waypoint.name} frame=odom "
            f"x={waypoint.x:.3f} y={waypoint.y:.3f} yaw={waypoint.yaw:.3f} "
            f"behavior_tree={goal.behavior_tree or 'default'}"
        )
        self._send_future = self._client.send_goal_async(goal)
        self._send_future.add_done_callback(
            lambda future, goal_generation=generation,
            goal_name=waypoint.name: self._goal_response_cb(
                future, goal_generation, goal_name
            )
        )
        return True

    def _goal_response_cb(self, future, generation, goal_name) -> None:
        if generation != self._goal_generation:
            self._node.get_logger().info(
                f"NAV_STALE_GOAL_RESPONSE ignored name={goal_name}"
            )
            return
        try:
            goal_handle = future.result()
        except Exception as exc:  # rclpy future propagates transport failures here
            self._result = NavResult.FAILED
            self._node.get_logger().error(f"NAV_FAILED send error: {exc}")
            return
        if not goal_handle.accepted:
            self._result = NavResult.FAILED
            self._node.get_logger().error(
                f"NAV_FAILED name={goal_name} goal rejected"
            )
            return
        self._goal_handle = goal_handle
        self._result = NavResult.ACTIVE
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(
            lambda result_future, goal_generation=generation,
            accepted_name=goal_name: self._result_cb(
                result_future, goal_generation, accepted_name
            )
        )
        if self._cancel_requested:
            self._cancel_future = goal_handle.cancel_goal_async()

    def _result_cb(self, future, generation, goal_name) -> None:
        if generation != self._goal_generation:
            self._node.get_logger().info(
                f"NAV_STALE_RESULT ignored name={goal_name}"
            )
            return
        try:
            status = future.result().status
        except Exception as exc:
            self._result = NavResult.FAILED
            self._node.get_logger().error(f"NAV_FAILED result error: {exc}")
            self._goal_handle = None
            return
        self._goal_handle = None
        self._cancel_requested = False
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._result = NavResult.SUCCEEDED
            self._node.get_logger().info(f"NAV_SUCCESS name={goal_name}")
        elif status == GoalStatus.STATUS_CANCELED:
            self._result = NavResult.CANCELLED
            self._node.get_logger().info(f"NAV_CANCELLED name={goal_name}")
        else:
            self._result = NavResult.FAILED
            self._node.get_logger().error(
                f"NAV_FAILED name={goal_name} action_status={status}"
            )

    def cancel(self) -> bool:
        """Request cancellation of the current accepted goal, if one exists."""

        if self._goal_handle is None and self._result == NavResult.STARTING:
            self._cancel_requested = True
            return True
        if self._goal_handle is None:
            return not self.is_active
        self._cancel_requested = True
        self._cancel_future = self._goal_handle.cancel_goal_async()
        return True

    def cancel_and_wait(self, timeout_sec: float = 3.0) -> bool:
        """取消当前目标并最多等 3 s 进入终态，不是固定停顿。"""

        self.cancel()
        result = self.wait_result(timeout_sec=timeout_sec)
        terminal = result in (
            NavResult.IDLE,
            NavResult.SUCCEEDED,
            NavResult.FAILED,
            NavResult.CANCELLED,
        )
        if not terminal:
            self._node.get_logger().error(
                f"NAV_FAILED cancellation did not finish within {timeout_sec:.1f}s"
            )
        return terminal

    def wait_result(self, timeout_sec: float = 0.0) -> Optional[NavResult]:
        """Return a completed result, optionally spinning while waiting.

        Mission timers call this with the default zero timeout (non-blocking).
        The bounded blocking form is provided for standalone callers.

        中文：默认 ``0`` 表示只查询一次；正值才允许在此函数内有限等待。
        """

        if not self.is_active:
            return self._result
        if timeout_sec <= 0.0:
            return None
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self.is_active and time.monotonic() < deadline:
            # 每次最多阻塞 0.1 s，以便取消/结果回调及时得到处理。
            rclpy.spin_once(self._node, timeout_sec=min(0.1, deadline - time.monotonic()))
        return None if self.is_active else self._result

    def stop_robot(self) -> None:
        self._cmd_vel_pub.publish(Twist())
        self._node.get_logger().info("STOP_ROBOT")
