#!/usr/bin/env python3
"""Expose Humble MPPI's optimal marker trajectory as ``/local_plan``.

Nav2 Humble 1.1.20 does not publish its optimized trajectory as a
``nav_msgs/Path``.  With ``visualize`` enabled it puts both candidate and
optimal trajectory points in ``/trajectories`` as a ``MarkerArray``.  RViz and
the existing project configuration expect a conventional Path on
``/local_plan``, so this adapter filters only the ``Optimal Trajectory``
markers and converts them without changing the controller output.

中文说明：仅把 MPPI 可视化 Marker 转成 RViz Path，不参与规划与速度控制，
也没有定时延迟；输出节奏完全跟随 ``/trajectories`` 输入消息。
"""

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


MPPI_TRAJECTORIES_TOPIC = "/trajectories"
LOCAL_PLAN_TOPIC = "/local_plan"
OPTIMAL_NAMESPACE = "Optimal Trajectory"


def optimal_markers_to_path(message: MarkerArray) -> Path | None:
    """Convert one Humble MPPI optimal-marker set into an ordered Path."""

    markers = sorted(
        (
            marker
            for marker in message.markers
            if marker.ns == OPTIMAL_NAMESPACE and marker.action == Marker.ADD
        ),
        key=lambda marker: marker.id,
    )
    if not markers:
        return None

    path = Path()
    path.header.frame_id = markers[0].header.frame_id
    path.header.stamp = markers[0].header.stamp

    for index, marker in enumerate(markers):
        pose = PoseStamped()
        pose.header.frame_id = marker.header.frame_id
        pose.header.stamp = marker.header.stamp
        pose.pose.position.x = marker.pose.position.x
        pose.pose.position.y = marker.pose.position.y
        pose.pose.position.z = marker.pose.position.z

        # Humble's MPPI markers contain positions only.  Supply a useful path
        # orientation from the adjacent segment for RViz pose-style displays.
        if len(markers) == 1:
            yaw = 0.0
        elif index + 1 < len(markers):
            next_position = markers[index + 1].pose.position
            yaw = math.atan2(
                next_position.y - marker.pose.position.y,
                next_position.x - marker.pose.position.x,
            )
        else:
            previous_position = markers[index - 1].pose.position
            yaw = math.atan2(
                marker.pose.position.y - previous_position.y,
                marker.pose.position.x - previous_position.x,
            )
        pose.pose.orientation.z = math.sin(0.5 * yaw)
        pose.pose.orientation.w = math.cos(0.5 * yaw)
        path.poses.append(pose)

    return path


class MPPILocalPlanAdapter(Node):
    """Publish MPPI's optimal trajectory in the standard RViz Path format."""

    def __init__(self) -> None:
        super().__init__("mppi_local_plan_adapter")
        self._publisher = self.create_publisher(Path, LOCAL_PLAN_TOPIC, 1)
        self._subscription = self.create_subscription(
            MarkerArray,
            MPPI_TRAJECTORIES_TOPIC,
            self._trajectory_callback,
            1,
        )
        self._first_path_reported = False
        self.get_logger().info(
            f"converting MPPI optimal markers: "
            f"{MPPI_TRAJECTORIES_TOPIC} -> {LOCAL_PLAN_TOPIC}"
        )

    def _trajectory_callback(self, message: MarkerArray) -> None:
        path = optimal_markers_to_path(message)
        if path is None:
            return
        self._publisher.publish(path)
        if not self._first_path_reported:
            self.get_logger().info(
                f"publishing MPPI local plan with {len(path.poses)} poses "
                f"in frame {path.header.frame_id}"
            )
            self._first_path_reported = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MPPILocalPlanAdapter()
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
