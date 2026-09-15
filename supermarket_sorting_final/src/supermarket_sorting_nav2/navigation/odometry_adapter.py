#!/usr/bin/env python3
"""Provide Nav2 with ROS-compliant, base-frame odometry twist.

The simulator's raw odometry pose is already expressed in ``odom``, but its
linear velocity comes directly from a MuJoCo ``framelinvel`` sensor and is in
world coordinates.  ROS Odometry defines twist in ``child_frame_id``.  This
node leaves the raw topic untouched and republishes only the corrected view
used by Nav2.

中文说明：仅转换里程计线速度的坐标系并立即转发，不缓存、不限频，也没有
人为等待；它不会改变 Nav2 的速度或规划频率。
"""

import math
from typing import Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


RAW_ODOM_TOPIC = "/slamware_ros_sdk_server_node/odom"
NAV2_ODOM_TOPIC = "/nav2/odom"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return yaw from a finite, non-zero quaternion without changing pose."""

    components = (float(x), float(y), float(z), float(w))
    if not all(math.isfinite(value) for value in components):
        raise ValueError("odometry orientation contains a non-finite value")

    norm_sq = sum(value * value for value in components)
    if norm_sq <= 1.0e-24:
        raise ValueError("odometry orientation quaternion has zero norm")

    inv_norm = 1.0 / math.sqrt(norm_sq)
    qx, qy, qz, qw = (value * inv_norm for value in components)
    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_yaw, cos_yaw)


def world_linear_to_base(
    vx_world: float,
    vy_world: float,
    yaw: float,
) -> Tuple[float, float]:
    """Rotate planar world/odom linear velocity into ``base_link``."""

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_base = cos_yaw * vx_world + sin_yaw * vy_world
    vy_base = -sin_yaw * vx_world + cos_yaw * vy_world
    return vx_base, vy_base


def adapt_odometry(raw: Odometry) -> Odometry:
    """Build a corrected message without mutating or aliasing the raw message."""

    orientation = raw.pose.pose.orientation
    yaw = quaternion_to_yaw(
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )
    raw_linear = raw.twist.twist.linear
    vx_base, vy_base = world_linear_to_base(raw_linear.x, raw_linear.y, yaw)

    corrected = Odometry()
    corrected.header.stamp.sec = raw.header.stamp.sec
    corrected.header.stamp.nanosec = raw.header.stamp.nanosec
    corrected.header.frame_id = ODOM_FRAME
    corrected.child_frame_id = BASE_FRAME

    raw_position = raw.pose.pose.position
    corrected.pose.pose.position.x = raw_position.x
    corrected.pose.pose.position.y = raw_position.y
    corrected.pose.pose.position.z = raw_position.z
    corrected.pose.pose.orientation.x = orientation.x
    corrected.pose.pose.orientation.y = orientation.y
    corrected.pose.pose.orientation.z = orientation.z
    corrected.pose.pose.orientation.w = orientation.w
    corrected.pose.covariance = list(raw.pose.covariance)

    corrected.twist.twist.linear.x = vx_base
    corrected.twist.twist.linear.y = vy_base
    corrected.twist.twist.linear.z = raw_linear.z
    raw_angular = raw.twist.twist.angular
    corrected.twist.twist.angular.x = raw_angular.x
    corrected.twist.twist.angular.y = raw_angular.y
    corrected.twist.twist.angular.z = raw_angular.z
    corrected.twist.covariance = list(raw.twist.covariance)
    return corrected


class OdomAdapter(Node):
    """Republish the Server's world-linear-velocity odometry for Nav2."""

    def __init__(self) -> None:
        super().__init__("nav2_odom_adapter")
        self._publisher = self.create_publisher(Odometry, NAV2_ODOM_TOPIC, 10)
        self._subscription = self.create_subscription(
            Odometry,
            RAW_ODOM_TOPIC,
            self._odom_callback,
            10,
        )
        self._invalid_orientation_reported = False
        self.get_logger().info(
            f"correcting linear twist frame: {RAW_ODOM_TOPIC} -> {NAV2_ODOM_TOPIC}"
        )

    def _odom_callback(self, raw: Odometry) -> None:
        try:
            corrected = adapt_odometry(raw)
        except ValueError as exc:
            if not self._invalid_orientation_reported:
                self.get_logger().error(f"dropping invalid raw odometry: {exc}")
                self._invalid_orientation_reported = True
            return

        self._invalid_orientation_reported = False
        self._publisher.publish(corrected)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomAdapter()
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
