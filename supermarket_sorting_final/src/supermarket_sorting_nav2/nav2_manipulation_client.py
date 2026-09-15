#!/usr/bin/env python3
"""Nav2 mission wrapper around the verified task-one grasp controller.

Nav2 exclusively owns large-scale base motion.  Its raw Twist is relayed by
this client so a mission-specific terminal angular assist can preserve Nav2's
linear command while replacing only angular.z.  The copied Baseline controller
is allowed to originate base motion only for the verified shelf creep and short
post-grasp retreat; perception, IK and manipulator commands remain unchanged.

中文说明：该模块给单瓶基线增加 Nav2 长距离导航与桌面放置状态。Nav2 负责
开放区域，基线控制器只在货架近距和短退段接管。所有 ``TIMEOUT`` 常量都是
失败上限，``SETTLE``/``DWELL`` 常量才是反馈到位后的真实停稳时间。
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

from .baseline_grasp_controller import (
    BASE_GRASP_CLOSE_DWELL_SEC,
    CREEP_SPEED,
    CREEP_YAW_KP,
    DETECT_DWELL,
    GRASP_YAW,
    GRIP_CLOSE,
    GRIP_OPEN,
    HEAD_PITCH,
    LIFT_AMOUNT,
    REACH_FWD_MAX,
    REACH_FWD_MIN,
    REACH_LATERAL_MAX,
    REACH_Z_MAX,
    REACH_Z_MIN,
    RETREAT_SPEED,
    SLIDE_GRASP,
    TaskOneClient,
    YELLOW_MID_Y,
    wrap_to_pi,
)
from .navigation.mission_manager import MissionState, MissionStateMachine, load_waypoints
from .navigation.nav2_manager import Nav2Manager, NavResult


STOP_SETTLE_SEC = 0.75  # Nav2 取消后且里程计静止，继续保持 0.75 s 再操作
BASE_STOP_EPS = 1.0e-3
ODOM_STOP_LINEAR_EPS = 0.015
ODOM_STOP_ANGULAR_EPS = 0.03

# The verified shelf pose faces 11 degrees east of the shelf normal. Holding
# that yaw during the whole 25 cm creep moves the gripper several centimetres
# to the right of a visually locked bottle. Keep the original pose, arm IK and
# creep speed, but close the gripper's world-X error while moving. The angular
# command accounts for the long base-to-gripper lever arm, avoiding a large
# sideways wrist swing from even a small chassis rotation.
FINE_APPROACH_LATERAL_KP = 0.60
FINE_APPROACH_MAX_ANGULAR = 0.12
FINE_APPROACH_MIN_LEVER_Y = 0.20

# With the right arm retaining the bottle, fold the unused left arm down beside
# the chassis before Nav2 starts the long transport leg.  A direct joint-space
# interpolation from the initial left-arm pose to LEFT_ARM_TRANSPORT sweeps the
# left wrist through the right-arm/bottle envelope.  These waypoints first move
# the left arm outward, then lower it on the robot's left side, and only then
# bring it into the compact transport pose.  They were sampled against the
# Server image's Humble-era MMK2 MJCF at the post-grasp slide height (0.06 m).
LEFT_ARM_TRANSPORT = np.array(
    [math.pi / 2.0, -2.7548, 2.7548, 0.0, 0.0, 0.0], dtype=float
)
LEFT_ARM_STOW_WAYPOINTS = np.array(
    [
        # Move the wrist outward before lowering it past the carried bottle.
        [0.673051, -0.928052, 1.143096, -0.106383, 0.883911, 1.035591],
        # Lower the arm while keeping the wrist on the robot's left side.
        [1.564500, -2.221909, 2.402510, -0.287849, -0.219311, 0.362219],
        LEFT_ARM_TRANSPORT,
    ],
    dtype=float,
)
LEFT_ARM_TRANSPORT_TOL = 0.05
LEFT_ARM_STOW_STAGE_DWELL_SEC = 0.25  # 左臂各收拢路点到位后的保持时间

# Fixed-scene kele placement starts from the 2026-08-17 grasp handoff.  Before
# lowering, move the closed gripper farther over the table.  After releasing,
# reverse exactly that short joint-space segment so the open fingers leave the
# bottle without sweeping sideways through it.  Raising the torso then gives
# enough vertical clearance to mirror the already-validated left-arm stow path
# on the robot's right side.
PLACE_FORWARD_DISTANCE = 0.06
PLACE_FORWARD_FALLBACKS = (0.05, 0.04, 0.03)
PLACE_SLIDE = 0.17
PLACE_SLIDE_TOL = 0.02
PLACE_GRIP_OPEN_MIN = 0.85
PLACE_ARM_TOL = 0.05
# The normal grasp sequence slews joints at 1.2 rad/s.  Placement needs to be
# gentler because the released bottle is tall and has little damping on the
# tabletop.  Slow all manipulator targets only after arriving at the table.
PLACE_JOINT_SLEW = 0.45
# These delays start after feedback first confirms the action is complete.  In
# particular, lowering time is not incorrectly counted as tabletop settle time.
PLACE_ACTION_SETTLE_SEC = 0.10  # 放置伸手/撤手反馈到位后的通用保持
PLACE_LOWER_SETTLE_SEC = 0.10  # 身体下降到放置高度后的停稳时间
PLACE_RELEASE_SETTLE_SEC = 0.30  # 松爪后等待商品落稳的时间
PLACE_LIFT_SETTLE_SEC = 0.10  # 松爪后身体抬起到位的保持时间
PLACE_STAGE_TIMEOUT_SEC = 20.0  # 任一放置阶段的故障等待上限
PLACE_CLEAR_SLIDE = SLIDE_GRASP - LIFT_AMOUNT
PLACE_CLEAR_MIN_EE_DISTANCE = 0.09
RIGHT_ARM_RETURN_JOINT_SLEW = 1.20
RIGHT_ARM_RETURN_SETTLE_SEC = 0.25  # 右臂回到运输/初始姿态后的保持时间

# Raised table-clear posture: end effector is approximately
# [0.44, -0.20, 0.98] in base_footprint.  It simultaneously moves the open
# gripper backward/right and rotates the wrist out of its grasp posture.  The
# actual Server collision model reports only the small pre-existing grasp-pose
# self contact at the first sample and no table/chassis/wall contact afterward.
RIGHT_ARM_TABLE_CLEAR = np.array(
    [-0.124320, -0.866041, 0.709903, -0.077347, -1.438229, -0.637119],
    dtype=float,
)


class Nav2TaskClient(TaskOneClient):
    def __init__(self, run_mode: str, waypoint_path: str, scan_slide: float = 0.15):
        self.run_mode = run_mode
        super().__init__(node_name="supermarket_sorting_task_nav2_client")

        # scan_posture_initializer.py has already moved the torso before Nav2 starts. Keep
        # the same setpoint when full mode begins publishing joint commands so
        # the camera is not raised back to the initial 0.0 m position.
        self.scan_slide = float(scan_slide)
        self.tc[2] = self.scan_slide
        self.action[2] = self.scan_slide

        self.waypoints = load_waypoints(waypoint_path, require_tuned=False)
        self.nav2 = Nav2Manager(self, cmd_vel_publisher=self.cmd_vel_pub)
        # controller_server is remapped to /cmd_vel_nav2_raw by the Nav2
        # launch file.  Normal navigation is relayed unchanged.  A derived
        # mission may replace only angular.z near a terminal pose while Nav2
        # remains the sole source of linear.x.
        self._nav_angular_override_radps: float | None = None
        self._nav_raw_cmd_vel_sub = self.create_subscription(
            Twist,
            "/cmd_vel_nav2_raw",
            self._nav_cmd_vel_cb,
            10,
        )
        initial = MissionState.NAV_TO_PICK if run_mode == "nav_only" else MissionState.WAIT_TASK
        self.mission = MissionStateMachine(self, initial)

        self.base_control_enabled = False
        self.manipulation_enabled = False
        self.task_received = False
        self.task_payload = None
        self._last_state_log = 0.0
        self._last_stow_wait_log = 0.0
        self._left_arm_stow_stage = 0
        self._left_arm_stow_stage_started_at = 0.0
        self._place_retract_arm = None
        self._place_advanced_arm = None
        self._place_forward_distance = 0.0
        self._place_clear_arm = None
        self._placed_ee_footprint = None
        self._place_ready_since = None
        self._grasp_heading = GRASP_YAW
        self._shutdown_started = False

        # The Server publishes the task once with TRANSIENT_LOCAL durability.
        # Matching that QoS is required because this client starts after Nav2.
        task_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self.task_cb, task_qos
        )
        self.get_logger().info(
            f"Nav2 mission client ready mode={run_mode}; "
            "large-scale /cmd_vel owner=Nav2"
        )

    def odom_cb(self, msg) -> None:
        super().odom_cb(msg)
        linear = msg.twist.twist.linear
        self.odom_linear_speed = math.hypot(linear.x, linear.y)
        self.odom_angular_speed = abs(msg.twist.twist.angular.z)
        self.odom_received_at = self.now()

    def _set_nav_angular_override(self, angular_radps: float | None) -> None:
        """Replace only Nav2 angular.z while retaining its linear command."""

        self._nav_angular_override_radps = (
            None if angular_radps is None else float(angular_radps)
        )

    def _nav_cmd_vel_cb(self, msg: Twist) -> None:
        """Relay Nav2's raw Twist, optionally mixing in terminal yaw assist."""

        # A local fine/retreat state owns /cmd_vel exclusively.  Ignore a late
        # Nav2 sample during that ownership handoff rather than mixing owners.
        if self.base_control_enabled:
            return

        twist = Twist()
        twist.linear.x = float(msg.linear.x)
        twist.linear.y = float(msg.linear.y)
        twist.linear.z = float(msg.linear.z)
        twist.angular.x = float(msg.angular.x)
        twist.angular.y = float(msg.angular.y)
        angular_override = self._nav_angular_override_radps
        twist.angular.z = float(
            msg.angular.z
            if angular_override is None
            else angular_override
        )
        self.cmd_vel_pub.publish(twist)

    def _odom_stopped(self) -> bool:
        odom_fresh = (
            self.now() - getattr(self, "odom_received_at", -math.inf) <= 0.5
        )
        twist_stopped = (
            getattr(self, "odom_linear_speed", math.inf) <= ODOM_STOP_LINEAR_EPS
            and getattr(self, "odom_angular_speed", math.inf)
            <= ODOM_STOP_ANGULAR_EPS
        )
        return odom_fresh and twist_stopped

    @property
    def larm_meas(self) -> np.ndarray:
        return np.array(
            [
                self.jpos.get(f"left_arm_joint{i + 1}", self.tc[5 + i])
                for i in range(6)
            ]
        )

    # ---- task and perception callbacks ----
    def task_cb(self, msg: String) -> None:
        if self.run_mode != "full" or self.task_received:
            return
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().error(f"WAIT_TASK invalid JSON: {exc}")
            return
        targets = payload.get("targets") if isinstance(payload, dict) else None
        if not isinstance(targets, list) or not targets:
            self.get_logger().warn("WAIT_TASK ignored task without targets")
            return
        if not any(isinstance(target, dict) and target.get("kind") == "kele" for target in targets):
            self.get_logger().warn("WAIT_TASK ignored task without a kele target")
            return
        self.task_payload = payload
        self.task_received = True
        self.manipulation_enabled = True
        self.get_logger().info(
            f"TASK_ACCEPTED run_prefix={payload.get('run_prefix', 'unknown')} "
            f"targets={len(targets)}"
        )

    def det_cb(self, msg) -> None:
        """Accumulate the best reachable kele while parked at the shelf."""

        if (
            not hasattr(self, "mission")
            or self.mission.state != MissionState.PERCEPTION
            or self.target_locked
            or self.deploy_set
            or self.base_xy is None
        ):
            return
        best, best_lat = None, REACH_LATERAL_MAX
        for det in msg.detections:
            if not det.results:
                continue
            pos = det.results[0].pose.pose.position
            point_world = np.array([pos.x, pos.y, pos.z])
            point_base = self.world_to_footprint(point_world)
            forward, lateral = point_base[0], abs(point_base[1])
            if forward < REACH_FWD_MIN or forward > REACH_FWD_MAX:
                continue
            if point_world[2] < REACH_Z_MIN or point_world[2] > REACH_Z_MAX:
                continue
            if lateral < best_lat:
                best_lat, best = lateral, point_world
        if best is not None:
            self.det_buf.append(best)

    # ---- strict base command ownership ----
    def _enable_baseline_base(self) -> None:
        if self.nav2.is_active:
            raise RuntimeError("cannot hand base to Baseline while a Nav2 goal is active")
        self._set_nav_angular_override(None)
        self.nav2.cancel()
        self.nav2.stop_robot()
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0] = self.tc[1] = 0.0
        self.base_control_enabled = True
        self.get_logger().info("CMD_VEL_OWNER baseline_fine_control")

    def _release_baseline_base(self) -> None:
        if abs(self.cur_lin) > BASE_STOP_EPS or abs(self.cur_ang) > BASE_STOP_EPS:
            raise RuntimeError("refusing /cmd_vel handoff before Baseline velocity reaches zero")
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0] = self.tc[1] = 0.0
        self.base_control_enabled = False
        self._set_nav_angular_override(None)
        self.nav2.stop_robot()
        self.get_logger().info("CMD_VEL_OWNER nav2")

    def _publish_outputs(self) -> None:
        # A Nav2-owned cycle publishes no Twist at all from this task node.
        if self.base_control_enabled:
            twist = Twist()
            twist.linear.x = float(self.tc[0])
            twist.angular.z = float(self.tc[1])
            self.cmd_vel_pub.publish(twist)

        # nav_only must not move head, spine, arms, or grippers.
        if not self.manipulation_enabled:
            return
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(
            Float64MultiArray(data=[float(self.action[3]), float(self.action[4])])
        )
        self.larm_pub.publish(
            Float64MultiArray(
                data=[float(value) for value in self.action[5:11]]
                + [float(self.action[11])]
            )
        )
        self.rarm_pub.publish(
            Float64MultiArray(
                data=[float(value) for value in self.action[12:18]]
                + [float(self.action[18])]
            )
        )

    # ---- mission helpers ----
    def _navigation_failed(self, leg: str, result: NavResult) -> None:
        self.base_control_enabled = False
        self.nav2.cancel()
        self.nav2.stop_robot()
        self.mission.transition(MissionState.FAILED, f"{leg}:{result.name}")

    def _reset_perception(self) -> None:
        self.deploy_set = False
        self.det_buf.clear()
        self.target_locked = False
        self.OBJECT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self.arm_target_set = False
        self.tc[4] = HEAD_PITCH
        self.tc[2] = SLIDE_GRASP
        self.tc[18] = GRIP_OPEN
        self.state_t0 = self.now()

    def _right_arm_at_target(self, target: np.ndarray) -> bool:
        return float(np.max(np.abs(self.rarm_meas - target))) < PLACE_ARM_TOL

    @property
    def _right_gripper_is_open(self) -> bool:
        measured = self.jpos.get("right_arm_eef_gripper_joint", self.tc[18])
        return float(measured) >= PLACE_GRIP_OPEN_MIN

    def _solve_right_ee_offset(
        self,
        start_pose: np.ndarray,
        reference_arm: np.ndarray,
        dx: float,
        dy: float = 0.0,
        dz: float = 0.0,
        slide: float = None,
    ):
        """Solve a small footprint-frame offset while retaining wrist attitude."""

        target = np.array(start_pose, dtype=float, copy=True)
        target[:3, 3] += np.array([dx, dy, dz], dtype=float)
        target_slide = float(self.tc[2] if slide is None else slide)
        reference = np.concatenate([[target_slide], reference_arm])
        solutions = self.kdl.inverse_kinematics(
            T_left=None,
            T_right=target,
            ref_pos=reference,
            target_height=target_slide,
        )
        if not solutions:
            return None
        return np.asarray(solutions[0], dtype=float)[1:7]

    def _right_ee_path(
        self,
        start_arm: np.ndarray,
        end_arm: np.ndarray,
        slide: float,
        samples: int = 31,
    ) -> np.ndarray:
        points = []
        for fraction in np.linspace(0.0, 1.0, samples):
            joints = start_arm + fraction * (end_arm - start_arm)
            _, pose = self.kdl.forward_kinematics(
                np.concatenate([[slide], joints]), index="right"
            )
            points.append(pose[:3, 3])
        return np.asarray(points)

    def _release_retract_is_safe(
        self, advanced_arm: np.ndarray, retract_arm: np.ndarray
    ) -> bool:
        """Require the release retreat to move backward with negligible side/drop."""

        points = self._right_ee_path(advanced_arm, retract_arm, PLACE_SLIDE)
        backward = bool(np.all(np.diff(points[:, 0]) <= 0.002))
        lateral_span = float(np.ptp(points[:, 1]))
        vertical_span = float(np.ptp(points[:, 2]))
        return backward and lateral_span < 0.025 and vertical_span < 0.025

    def _command_place_advance(
        self, distances: tuple[float, ...] | None = None
    ) -> bool:
        self._place_retract_arm = self.rarm_meas.copy()
        start_pose = self.ee_footprint_pose()
        candidates = (
            (PLACE_FORWARD_DISTANCE,) + PLACE_FORWARD_FALLBACKS
            if distances is None
            else tuple(float(distance) for distance in distances)
        )
        if not candidates or any(distance <= 0.0 for distance in candidates):
            raise ValueError("place advance distances must be positive")
        for distance in candidates:
            target_arm = self._solve_right_ee_offset(
                start_pose,
                self._place_retract_arm,
                dx=distance,
                slide=float(self.tc[2]),
            )
            if target_arm is None:
                continue
            if not self._release_retract_is_safe(
                target_arm, self._place_retract_arm
            ):
                self.get_logger().warn(
                    f"PLACE_ADVANCE rejected unsafe retract path distance="
                    f"{distance:.3f}"
                )
                continue
            self._place_advanced_arm = target_arm
            self._place_forward_distance = distance
            self.tc[12:18] = target_arm
            self.arm_target_set = True
            self.get_logger().info(
                f"PLACE_ADVANCE commanded forward={distance:.3f} m; "
                "wrist attitude retained"
            )
            return True
        self.get_logger().error(
            "PLACE_ADVANCE failed: no reachable forward pose with a safe "
            "straight retract path"
        )
        return False

    def _command_place_clearance(self) -> bool:
        """Move the raised open gripper to the MJCF-checked table-clear pose."""

        start_arm = self.rarm_meas.copy()
        target_arm = RIGHT_ARM_TABLE_CLEAR.copy()
        if self._placed_ee_footprint is not None:
            paths = [
                self._right_ee_path(
                    start_arm, target_arm, PLACE_CLEAR_SLIDE
                ),
                self._right_ee_path(
                    target_arm,
                    self._place_retract_arm,
                    PLACE_CLEAR_SLIDE,
                ),
            ]
            clearance = float(
                min(
                    np.min(
                        np.linalg.norm(
                            path - self._placed_ee_footprint, axis=1
                        )
                    )
                    for path in paths
                )
            )
            if clearance < PLACE_CLEAR_MIN_EE_DISTANCE:
                self.get_logger().error(
                    f"PLACE_MOVE_ASIDE rejected clearance={clearance:.3f} m"
                )
                return False
        else:
            clearance = math.nan
        self._place_clear_arm = target_arm
        self.tc[12:18] = target_arm
        self.get_logger().info(
            "PLACE_MOVE_ASIDE commanded table-clear posture "
            f"predicted_bottle_clearance={clearance:.3f} m"
        )
        return True

    def _place_stage_timed_out(self, label: str) -> bool:
        if self.mission.elapsed < PLACE_STAGE_TIMEOUT_SEC:
            return False
        self.get_logger().error(
            f"{label} timed out after {PLACE_STAGE_TIMEOUT_SEC:.1f} s"
        )
        self.mission.transition(MissionState.FAILED, f"{label.lower()}_timeout")
        return True

    def _reset_place_settle(self) -> None:
        self._place_ready_since = None

    def _place_target_settled(
        self, ready: bool, settle_sec: float, label: str
    ) -> bool:
        """Wait from measured convergence, rather than from command start."""

        if not ready:
            self._place_ready_since = None
            return False
        if self._place_ready_since is None:
            self._place_ready_since = self.now()
            self.get_logger().info(
                f"{label} reached target; settling for {settle_sec:.2f} s"
            )
            return settle_sec <= 0.0
        return self.now() - self._place_ready_since >= settle_sec

    def _fine_approach_steering(self, end_effector: np.ndarray):
        """Close gripper world-X error without swinging the extended arm."""

        lateral_remaining = float(self.OBJECT_WORLD[0] - end_effector[0])
        forward_remaining = float(self.CREEP_STOP_Y - end_effector[1])

        # For a rigid point fixed to the chassis:
        #   ee_x_dot = v*cos(yaw) - omega*(ee_y - base_y)
        # Select omega so ee_x_dot follows a proportional command toward the
        # detected bottle center.  This compensates both the 11-degree chassis
        # bias and the end-effector motion caused by chassis rotation.
        lever_y = float(end_effector[1] - self.base_xy[1])
        desired_lateral_velocity = (
            FINE_APPROACH_LATERAL_KP * lateral_remaining
        )
        if abs(lever_y) < FINE_APPROACH_MIN_LEVER_Y:
            angular_raw = CREEP_YAW_KP * wrap_to_pi(
                GRASP_YAW - self.base_yaw
            )
        else:
            angular_raw = (
                CREEP_SPEED * math.cos(self.base_yaw)
                - desired_lateral_velocity
            ) / lever_y
        angular = float(
            np.clip(
                angular_raw,
                -FINE_APPROACH_MAX_ANGULAR,
                FINE_APPROACH_MAX_ANGULAR,
            )
        )
        return angular, lateral_remaining, forward_remaining

    # ---- high-level state machine ----
    def tick(self) -> None:
        if self._shutdown_started:
            return
        # nav_only needs odometry only; full mode also needs measured joints.
        if self.base_xy is None or (self.run_mode == "full" and self.jpos is None):
            return

        state = self.mission.state

        if state == MissionState.WAIT_TASK:
            self.mission.consume_entry()
            if self.task_received:
                self.mission.transition(MissionState.NAV_TO_PICK)

        elif state == MissionState.NAV_TO_PICK:
            if self.mission.consume_entry():
                self.base_control_enabled = False
                if self.nav2.go_to_pose(self.waypoints["pick_approach"]):
                    self.mission.transition(MissionState.WAIT_PICK_NAV_DONE)
                else:
                    self._navigation_failed("pick_approach", self.nav2.result)

        elif state == MissionState.WAIT_PICK_NAV_DONE:
            result = self.nav2.wait_result()
            if result == NavResult.SUCCEEDED:
                self.mission.transition(MissionState.STOP_NAV)
            elif result in (NavResult.FAILED, NavResult.CANCELLED):
                self._navigation_failed("pick_approach", result)

        elif state == MissionState.STOP_NAV:
            if self.mission.consume_entry():
                self.nav2.cancel()
                self.nav2.stop_robot()
            if (
                self.mission.elapsed >= STOP_SETTLE_SEC
                and not self.nav2.is_active
                and self._odom_stopped()
            ):
                next_state = (
                    MissionState.NAV_TO_LEFT
                    if self.run_mode == "nav_only"
                    else MissionState.PERCEPTION
                )
                self.mission.transition(next_state)

        elif state == MissionState.PERCEPTION:
            if self.mission.consume_entry():
                self._reset_perception()
            self.set_twist(0.0, 0.0)
            if not self.deploy_set:
                if (
                    not self.target_locked
                    and self.now() - self.state_t0 < DETECT_DWELL
                ):
                    pass
                elif self._lock_target():
                    if self.arm_to(self.DEPLOY_WORLD):
                        self.deploy_set = True
                        self.state_t0 = self.now()
                    else:
                        self.target_locked = False
                        self.det_buf.clear()
            if self.deploy_set and self.deploy_done():
                self.mission.transition(MissionState.FINE_APPROACH)

        elif state == MissionState.FINE_APPROACH:
            entered = self.mission.consume_entry()
            if entered:
                self._enable_baseline_base()
            end_effector = self.ee_world()
            if end_effector[1] < self.CREEP_STOP_Y:
                angular, lateral_remaining, forward_remaining = (
                    self._fine_approach_steering(end_effector)
                )
                self.set_twist(
                    CREEP_SPEED,
                    angular,
                )
                if entered:
                    self.get_logger().info(
                        f"FINE_APPROACH_AIM lateral_remaining="
                        f"{lateral_remaining:.3f} "
                        f"forward_remaining={forward_remaining:.3f}"
                    )
            else:
                self.set_twist(0.0, 0.0)
                self._grasp_heading = self.base_yaw
                lateral_error = float(end_effector[0] - self.OBJECT_WORLD[0])
                self.get_logger().info(
                    f"FINE_APPROACH_COMPLETE lateral_error={lateral_error:.3f} "
                    f"grasp_heading={self._grasp_heading:.3f}"
                )
                self.mission.transition(MissionState.GRASP)

        elif state == MissionState.GRASP:
            if self.mission.consume_entry():
                self.state_t0 = self.now()
            self.set_twist(0.0, 0.0)
            self.tc[18] = GRIP_CLOSE
            if self.now() - self.state_t0 > BASE_GRASP_CLOSE_DWELL_SEC:
                self.mission.transition(MissionState.CHECK_GRASP)

        elif state == MissionState.CHECK_GRASP:
            if self.mission.consume_entry():
                self.state_t0 = self.now()
            self.set_twist(0.0, 0.0)
            self.tc[2] = SLIDE_GRASP - LIFT_AMOUNT
            # This intentionally matches the Baseline's sequence-complete test;
            # the simulator exposes no physical grasp/contact success signal.
            if abs(self.slide_meas - self.tc[2]) < 0.02:
                self.get_logger().info(
                    "GRASP_SEQUENCE_COMPLETE "
                    f"close_dwell={BASE_GRASP_CLOSE_DWELL_SEC:.1f} slide_ok=true"
                )
                self.mission.transition(MissionState.FINE_RETREAT)

        elif state == MissionState.FINE_RETREAT:
            self.mission.consume_entry()
            # Back away along the same heading used to enter the shelf.  Turning
            # back to GRASP_YAW next to the shelf can disturb a fresh grasp.
            yaw_error = wrap_to_pi(self._grasp_heading - self.base_yaw)
            if self.base_xy[1] > YELLOW_MID_Y + self.pos_tol:
                self.set_twist(-RETREAT_SPEED, yaw_error)
            else:
                self.set_twist(0.0, 0.0)
                self.mission.transition(MissionState.STOP_BASE_DRIVE)

        elif state == MissionState.STOP_BASE_DRIVE:
            self.mission.consume_entry()
            self.set_twist(0.0, 0.0)
            if (
                abs(self.cur_lin) <= BASE_STOP_EPS
                and abs(self.cur_ang) <= BASE_STOP_EPS
                and self._odom_stopped()
            ):
                self.mission.transition(MissionState.STOW_LEFT_ARM)

        elif state == MissionState.STOW_LEFT_ARM:
            if self.mission.consume_entry():
                self._left_arm_stow_stage = 0
                self._left_arm_stow_stage_started_at = self.now()
                self.tc[5:11] = LEFT_ARM_STOW_WAYPOINTS[0]
                self.get_logger().info(
                    f"STOW_LEFT_ARM stage=1/{len(LEFT_ARM_STOW_WAYPOINTS)} "
                    "commanded; moving outward for bottle clearance"
                )
            self.set_twist(0.0, 0.0)
            stage_target = LEFT_ARM_STOW_WAYPOINTS[self._left_arm_stow_stage]
            arm_error = float(np.max(np.abs(self.larm_meas - stage_target)))
            if (
                self.now() - self._left_arm_stow_stage_started_at
                >= LEFT_ARM_STOW_STAGE_DWELL_SEC
                and arm_error < LEFT_ARM_TRANSPORT_TOL
                and self._odom_stopped()
            ):
                next_stage = self._left_arm_stow_stage + 1
                if next_stage < len(LEFT_ARM_STOW_WAYPOINTS):
                    self._left_arm_stow_stage = next_stage
                    self._left_arm_stow_stage_started_at = self.now()
                    self.tc[5:11] = LEFT_ARM_STOW_WAYPOINTS[next_stage]
                    self.get_logger().info(
                        f"STOW_LEFT_ARM stage={next_stage + 1}/"
                        f"{len(LEFT_ARM_STOW_WAYPOINTS)} commanded"
                    )
                else:
                    self.get_logger().info(
                        f"STOW_LEFT_ARM complete max_joint_error={arm_error:.3f}"
                    )
                    self._release_baseline_base()
                    self.mission.transition(MissionState.NAV_TO_LEFT)
            elif self.now() - self._last_stow_wait_log > 1.0:  # 收臂等待日志每秒一次
                self.get_logger().info(
                    f"STOW_LEFT_ARM waiting stage={self._left_arm_stow_stage + 1}/"
                    f"{len(LEFT_ARM_STOW_WAYPOINTS)} "
                    f"max_joint_error={arm_error:.3f}"
                )
                self._last_stow_wait_log = self.now()

        elif state == MissionState.NAV_TO_LEFT:
            if self.mission.consume_entry():
                self.base_control_enabled = False
                if self.nav2.go_to_pose(self.waypoints["left_goal"]):
                    self.mission.transition(MissionState.WAIT_LEFT_NAV_DONE)
                else:
                    self._navigation_failed("left_goal", self.nav2.result)

        elif state == MissionState.WAIT_LEFT_NAV_DONE:
            result = self.nav2.wait_result()
            if result == NavResult.SUCCEEDED:
                self.mission.transition(MissionState.ARRIVED)
            elif result in (NavResult.FAILED, NavResult.CANCELLED):
                self._navigation_failed("left_goal", result)

        elif state == MissionState.ARRIVED:
            if self.mission.consume_entry():
                self.nav2.cancel()
                self.nav2.stop_robot()
            if self.mission.elapsed >= STOP_SETTLE_SEC and self._odom_stopped():
                next_state = (
                    MissionState.DONE
                    if self.run_mode == "nav_only"
                    else MissionState.PLACE_ADVANCE
                )
                self.mission.transition(next_state)

        elif state == MissionState.PLACE_ADVANCE:
            if self.mission.consume_entry():
                self.nav2.cancel()
                self.nav2.stop_robot()
                self.joint_slew = PLACE_JOINT_SLEW
                self._reset_place_settle()
                self.tc[18] = GRIP_CLOSE
                if not self._command_place_advance():
                    self.mission.transition(
                        MissionState.FAILED, "place_advance_unreachable"
                    )
            if self.mission.state == MissionState.PLACE_ADVANCE:
                self.tc[18] = GRIP_CLOSE
                ready = (
                    self._odom_stopped()
                    and self._right_arm_at_target(self._place_advanced_arm)
                )
                if self._place_target_settled(
                    ready, PLACE_ACTION_SETTLE_SEC, "PLACE_ADVANCE"
                ):
                    self.mission.transition(MissionState.PLACE_LOWER)
                else:
                    self._place_stage_timed_out("PLACE_ADVANCE")

        elif state == MissionState.PLACE_LOWER:
            if self.mission.consume_entry():
                self.nav2.cancel()
                self.nav2.stop_robot()
                self._reset_place_settle()
                self.tc[2] = PLACE_SLIDE
                self.get_logger().info(
                    f"PLACE_LOWER commanded slide={PLACE_SLIDE:.3f} "
                    f"slew={PLACE_JOINT_SLEW:.2f}; "
                    "gripper remains closed"
                )
            # Never release while the bottle is still above the table.
            self.tc[18] = GRIP_CLOSE
            ready = (
                self._odom_stopped()
                and abs(self.slide_meas - PLACE_SLIDE) < PLACE_SLIDE_TOL
                and self._right_arm_at_target(self._place_advanced_arm)
            )
            if self._place_target_settled(
                ready, PLACE_LOWER_SETTLE_SEC, "PLACE_LOWER"
            ):
                self.mission.transition(MissionState.PLACE_RELEASE)
            else:
                self._place_stage_timed_out("PLACE_LOWER")

        elif state == MissionState.PLACE_RELEASE:
            if self.mission.consume_entry():
                self._reset_place_settle()
                self._placed_ee_footprint = self.ee_footprint_pose()[:3, 3].copy()
                self.tc[18] = GRIP_OPEN
                self.get_logger().info(
                    f"PLACE_RELEASE opening right gripper slowly "
                    f"slew={PLACE_JOINT_SLEW:.2f}"
                )
            self.tc[18] = GRIP_OPEN
            if self._place_target_settled(
                self._right_gripper_is_open,
                PLACE_RELEASE_SETTLE_SEC,
                "PLACE_RELEASE",
            ):
                self.get_logger().info(
                    "PLACE_RELEASE settled; retracting open gripper before stow"
                )
                self.mission.transition(MissionState.PLACE_RETRACT)
            else:
                self._place_stage_timed_out("PLACE_RELEASE")

        elif state == MissionState.PLACE_RETRACT:
            if self.mission.consume_entry():
                self._reset_place_settle()
                self.tc[18] = GRIP_OPEN
                self.tc[12:18] = self._place_retract_arm
                self.get_logger().info(
                    f"PLACE_RETRACT commanded backward="
                    f"{self._place_forward_distance:.3f} m along release path"
                )
            self.tc[18] = GRIP_OPEN
            if self._place_target_settled(
                self._right_arm_at_target(self._place_retract_arm),
                PLACE_ACTION_SETTLE_SEC,
                "PLACE_RETRACT",
            ):
                self.mission.transition(MissionState.PLACE_LIFT_CLEAR)
            else:
                self._place_stage_timed_out("PLACE_RETRACT")

        elif state == MissionState.PLACE_LIFT_CLEAR:
            if self.mission.consume_entry():
                self._reset_place_settle()
                self.tc[18] = GRIP_OPEN
                self.tc[2] = PLACE_CLEAR_SLIDE
                self.get_logger().info(
                    f"PLACE_LIFT_CLEAR commanded slide={PLACE_CLEAR_SLIDE:.3f} "
                    f"slew={PLACE_JOINT_SLEW:.2f}; "
                    "raising open gripper above bottle"
                )
            self.tc[18] = GRIP_OPEN
            ready = (
                abs(self.slide_meas - PLACE_CLEAR_SLIDE) < PLACE_SLIDE_TOL
                and self._right_arm_at_target(self._place_retract_arm)
            )
            if self._place_target_settled(
                ready, PLACE_LIFT_SETTLE_SEC, "PLACE_LIFT_CLEAR"
            ):
                self.mission.transition(MissionState.PLACE_MOVE_ASIDE)
            else:
                self._place_stage_timed_out("PLACE_LIFT_CLEAR")

        elif state == MissionState.PLACE_MOVE_ASIDE:
            if self.mission.consume_entry():
                self._reset_place_settle()
                self.tc[18] = GRIP_OPEN
                if not self._command_place_clearance():
                    self.mission.transition(
                        MissionState.FAILED, "place_clearance_unreachable"
                    )
            if self.mission.state == MissionState.PLACE_MOVE_ASIDE:
                self.tc[18] = GRIP_OPEN
                if self._place_target_settled(
                    self._right_arm_at_target(self._place_clear_arm),
                    PLACE_ACTION_SETTLE_SEC,
                    "PLACE_MOVE_ASIDE",
                ):
                    self.mission.transition(
                        MissionState.RESTORE_RIGHT_ARM_DEPLOYED
                    )
                else:
                    self._place_stage_timed_out("PLACE_MOVE_ASIDE")

        elif state == MissionState.RESTORE_RIGHT_ARM_DEPLOYED:
            if self.mission.consume_entry():
                self._reset_place_settle()
                self.joint_slew = RIGHT_ARM_RETURN_JOINT_SLEW
                self.tc[18] = GRIP_OPEN
                self.tc[12:18] = self._place_retract_arm
                self.get_logger().info(
                    "RESTORE_RIGHT_ARM_DEPLOYED commanded saved shelf-deploy pose "
                    f"slew={RIGHT_ARM_RETURN_JOINT_SLEW:.2f}"
                )
            self.tc[18] = GRIP_OPEN
            if self._place_target_settled(
                self._right_arm_at_target(self._place_retract_arm)
                and self._right_gripper_is_open,
                RIGHT_ARM_RETURN_SETTLE_SEC,
                "RESTORE_RIGHT_ARM_DEPLOYED",
            ):
                self.get_logger().info(
                    "PLACE_COMPLETE bottle released; gripper clear; "
                    "right arm restored to shelf-deploy pose"
                )
                self.mission.transition(MissionState.DONE)
            else:
                self._place_stage_timed_out("RESTORE_RIGHT_ARM_DEPLOYED")

        elif state in (MissionState.DONE, MissionState.FAILED):
            if self.mission.consume_entry():
                self.base_control_enabled = False
                self.nav2.cancel()
                self.nav2.stop_robot()
                outcome = "DONE" if state == MissionState.DONE else "FAILED"
                self.get_logger().info(f"MISSION_{outcome}")

        if self.base_control_enabled:
            self.ramp_twist()
        if self.manipulation_enabled:
            self.smooth_step()
        self._publish_outputs()

        if self.now() - self._last_state_log > 1.0:  # 只把状态日志限制为每秒一次
            self.get_logger().info(
                f"phase={self.mission.state.name.lower()} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={self.base_yaw:.2f} cmd_vel_owner="
                f"{'baseline' if self.base_control_enabled else 'nav2'}"
            )
            self._last_state_log = self.now()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.base_control_enabled = False
        self.nav2.cancel_and_wait(timeout_sec=3.0)  # 取消 Action 最多等待 3 s
        # Give DDS several final zero samples before the node is destroyed.  The
        # launcher then terminates the Nav2 stack as a second safety boundary.
        for _ in range(3):  # 共发送 3 个零速度样本
            self.nav2.stop_robot()
            time.sleep(0.05)  # 样本间隔 50 ms，仅发生在节点退出阶段


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("nav_only", "full"), default="nav_only")
    parser.add_argument("--scan-slide", type=float, default=0.15)
    parser.add_argument(
        "--waypoints",
        default=str(project_root / "config" / "nav_waypoints.yaml"),
    )
    args = parser.parse_args()
    if not -0.04 <= args.scan_slide <= 0.40:
        parser.error("--scan-slide must be within -0.04..0.40 m")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "full" and os.environ.get("FULL_ARM_CLEARANCE_VERIFIED", "0") != "1":
        raise SystemExit(
            "full mode is safety-locked: validate the carried-arm collision "
            "envelope, then set FULL_ARM_CLEARANCE_VERIFIED=1"
        )
    # Validate the complete schema before ROS initialization.  A waypoint still
    # marked for tuning is then refused immediately before its individual leg.
    load_waypoints(args.waypoints, require_tuned=False)
    rclpy.init()
    node = Nav2TaskClient(args.mode, args.waypoints, scan_slide=args.scan_slide)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
