#!/usr/bin/env python3
"""Forward RViz ``/goal_pose`` messages to Nav2 NavigateToPose.

The stock Humble Nav2 RViz goal tool talks to the action server internally,
which makes a missed click difficult to diagnose.  This small bridge gives the
manual scan workflow an explicit ROS topic and logs every stage of the action.

中文说明：RViz 目标到 Nav2 Action 的异步桥接器。回调本身不等待；反馈日志
每 2 s 最多打印一次，该节流不会改变导航控制频率。
"""

from __future__ import annotations

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


class ManualGoalBridge(Node):
    """Turn a standard RViz PoseStamped goal into a Nav2 action goal."""

    def __init__(self) -> None:
        super().__init__("manual_goal_bridge")
        self._client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self._goal_generation = 0
        self._last_feedback_log_ns = 0
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, 10)
        self.get_logger().info(
            "MANUAL_GOAL_READY topic=/goal_pose action=/navigate_to_pose"
        )

    def _goal_cb(self, pose: PoseStamped) -> None:
        if not pose.header.frame_id:
            self.get_logger().error("MANUAL_GOAL_REJECTED empty frame_id")
            return
        if not self._client.server_is_ready():
            self.get_logger().error(
                "MANUAL_GOAL_REJECTED /navigate_to_pose server is unavailable"
            )
            return

        self._goal_generation += 1
        generation = self._goal_generation
        self._last_feedback_log_ns = 0
        self.get_logger().info(
            "MANUAL_GOAL_RX "
            f"frame={pose.header.frame_id} "
            f"x={pose.pose.position.x:.3f} y={pose.pose.position.y:.3f}"
        )

        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self._client.send_goal_async(
            goal,
            feedback_callback=lambda feedback: self._feedback_cb(
                generation, feedback
            ),
        )
        future.add_done_callback(
            lambda response: self._goal_response_cb(generation, response)
        )

    def _goal_response_cb(self, generation, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # action transport failure
            self.get_logger().error(f"MANUAL_GOAL_SEND_FAILED error={exc}")
            return
        if not goal_handle.accepted:
            self.get_logger().error("MANUAL_GOAL_REJECTED by Nav2")
            return

        self.get_logger().info("MANUAL_GOAL_ACCEPTED")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result: self._result_cb(generation, result)
        )

    def _feedback_cb(self, generation, feedback_msg) -> None:
        if generation != self._goal_generation:
            return
        now_ns = self.get_clock().now().nanoseconds
        # 2 s 仅为日志节流，目标反馈和 Nav2 控制仍按原频率运行。
        if now_ns - self._last_feedback_log_ns < 2_000_000_000:
            return
        self._last_feedback_log_ns = now_ns
        feedback = feedback_msg.feedback
        self.get_logger().info(
            "MANUAL_GOAL_ACTIVE "
            f"distance_remaining={feedback.distance_remaining:.3f}"
        )

    def _result_cb(self, generation, future) -> None:
        try:
            wrapped_result = future.result()
        except Exception as exc:  # action transport failure
            self.get_logger().error(f"MANUAL_GOAL_RESULT_FAILED error={exc}")
            return
        status = STATUS_NAMES.get(wrapped_result.status, str(wrapped_result.status))
        suffix = "" if generation == self._goal_generation else " stale=true"
        self.get_logger().info(f"MANUAL_GOAL_RESULT status={status}{suffix}")


def main() -> None:
    rclpy.init()
    node = ManualGoalBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
