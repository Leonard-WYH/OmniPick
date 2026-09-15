#!/usr/bin/env python3
"""Move the MMK2 torso smoothly to its competition scanning height once.

中文说明：启动时一次性把升降轴移动到观察高度。它按反馈渐变指令，连续稳定
0.4 s 后退出；命令行 ``--timeout`` 是总故障上限，不是固定等待。
"""

import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


SLIDE_MIN = -0.04
SLIDE_MAX = 0.87


class ScanPostureInitializer(Node):
    """One-shot slide controller that never commands the base, head or arms."""

    def __init__(self) -> None:
        super().__init__("scan_posture_initializer")
        self.slide_position: float | None = None
        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/spine_forward_position_controller/commands",
            5,
        )
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10)

    def joint_cb(self, msg: JointState) -> None:
        try:
            index = msg.name.index("slide_joint")
        except ValueError:
            return
        if index < len(msg.position):
            self.slide_position = float(msg.position[index])

    def publish_slide(self, position: float) -> None:
        self.publisher.publish(Float64MultiArray(data=[float(position)]))

    def move_to(
        self,
        target: float,
        slew_rate: float,
        timeout: float,
        tolerance: float,
    ) -> None:
        # 关节反馈最多等 min(总超时, 5 s)，收到后立即进入移动阶段。
        feedback_deadline = time.monotonic() + min(timeout, 5.0)
        while rclpy.ok() and self.slide_position is None:
            if time.monotonic() >= feedback_deadline:
                raise RuntimeError("SCAN_POSTURE failed: no slide_joint feedback")
            rclpy.spin_once(self, timeout_sec=0.05)  # 50 ms 回调轮询上限

        assert self.slide_position is not None
        start = self.slide_position
        command = start
        last_update = time.monotonic()
        deadline = last_update + timeout
        ready_since: float | None = None
        self.get_logger().info(
            f"SCAN_POSTURE_START slide={start:.3f} target={target:.3f} "
            f"slew={slew_rate:.3f}m/s"
        )

        while rclpy.ok():
            now = time.monotonic()
            if now >= deadline:
                measured = self.slide_position
                raise RuntimeError(
                    "SCAN_POSTURE timeout: "
                    f"target={target:.3f} measured={measured:.3f}"
                )
            dt = max(0.0, min(0.10, now - last_update))
            last_update = now
            error = target - command
            step = slew_rate * dt
            if abs(error) <= step:
                command = target
            else:
                command += step if error > 0.0 else -step
            self.publish_slide(command)
            rclpy.spin_once(self, timeout_sec=0.02)  # 约 50 Hz 更新升降指令

            measured = self.slide_position
            if command == target and abs(measured - target) <= tolerance:
                if ready_since is None:
                    ready_since = now
                elif now - ready_since >= 0.40:  # 反馈连续到位 0.4 s 才确认完成
                    break
            else:
                ready_since = None

        # Repeat the final setpoint so the position controller reliably retains
        # it after this one-shot node exits.
        for _ in range(5):  # 退出前重发 5 次，防止末值被 DDS 启动瞬态漏掉
            self.publish_slide(target)
            rclpy.spin_once(self, timeout_sec=0.03)  # 每次最多处理回调 30 ms
        self.get_logger().info(
            f"SCAN_POSTURE_READY slide={self.slide_position:.3f} "
            f"lowered_by={target - start:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slide", type=float, default=0.15)
    parser.add_argument("--slew-rate", type=float, default=0.10)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    if not SLIDE_MIN <= args.slide <= SLIDE_MAX:
        parser.error(f"--slide must be within [{SLIDE_MIN}, {SLIDE_MAX}]")
    if args.slew_rate <= 0.0 or args.timeout <= 0.0 or args.tolerance <= 0.0:
        parser.error("slew rate, timeout and tolerance must be positive")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ScanPostureInitializer()
    try:
        node.move_to(args.slide, args.slew_rate, args.timeout, args.tolerance)
    except RuntimeError as exc:
        node.get_logger().error(str(exc))
        raise SystemExit(1) from exc
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
