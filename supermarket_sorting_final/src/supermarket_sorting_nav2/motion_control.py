"""Concurrent arm motion, Nav2 handoff and footprint helpers."""

from __future__ import annotations

import math
import numpy as np

from .baseline_grasp_controller import (
    DEPLOY_CART_TOL,
    DEPLOY_JOINT_TOL,
    DEPLOY_ROT_TOL,
    GRASP_ROT,
    GRIP_CLOSE,
    GRIP_OPEN,
    INIT_ARM_L,
    INIT_ARM_R,
    wrap_to_pi,
)
from .nav2_manipulation_client import (
    LEFT_ARM_STOW_STAGE_DWELL_SEC,
    LEFT_ARM_STOW_WAYPOINTS,
    LEFT_ARM_TRANSPORT,
    LEFT_ARM_TRANSPORT_TOL,
    RIGHT_ARM_RETURN_JOINT_SLEW,
    STOP_SETTLE_SEC,
)
from .navigation.sorting_geometry import (
    CARRY_FOOTPRINT,
    CARRY_HAND_WORKPOINT_XY_M,
    COMPACT_CARRY_ARM,
    FIXED_GRASP_ARM,
    HAND_WORKPOINT_XY_M,
    HAND_Z_PLUS_SLIDE_M,
    NORMAL_FOOTPRINT,
    TABLE_DROP_X_OFFSETS_M,
    TABLE_Y_MIN_M,
    TISSUE_HALF_WIDTH_M,
    table_approach_pose,
    table_place_forward_distances,
)
from .navigation.nav2_manager import NavResult
from nav2_msgs.msg import SpeedLimit
from rcl_interfaces.msg import (
    Parameter,
    ParameterType,
    ParameterValue,
)
from rcl_interfaces.srv import SetParameters

from .sorting_config import (
    ARM_SETTLE_SEC,
    BROAD_RELEASE_COMMAND_OPEN_MIN,
    CHENGZI_CARRY_STOW_JOINT_SLEW,
    CHENGZI_DEPLOY_SLIDE_TOL_M,
    COMPACT_CARRY_SAFE_SETTLE_SEC,
    FAST_ARM_STAGE_DWELL_SEC,
    LEFT_ARM_DOWN_WAYPOINTS,
    LEFT_ARM_SCAN_WAYPOINTS,
    MANIPULATION_TIMEOUT_SEC,
    NAV_NEAR_GOAL_SETTLE_SEC,
    NAV_NEAR_GOAL_YAW_TOL_RAD,
    NORMAL_MPPI_MAX_ANGULAR_RADPS,
    PICK_DEPLOY_BODY_LOWER_JOINT_SLEW,
    PICK_DEPLOY_FAST_JOINT_SLEW,
    PICK_FINE_HANDOFF_MAX_LINEAR_MPS,
    PICK_TRANSPORT_TOGETHER_TIMEOUT_SEC,
    RETURN_STOW_CONCURRENT_TIMEOUT_SEC,
    RETURN_STOW_FAST_JOINT_SLEW,
    RIGHT_ARM_DOWN_WAYPOINTS,
    SEARCH_ARMS_RESTORE_WAYPOINTS,
    SEARCH_RESTORE_FAST_JOINT_SLEW,
    SLIDE_TOL_M,
    SINGLE_HAND_TABLE_EDGE_MARGIN_M,
    TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD,
    TABLE_NEAR_GOAL_POSITION_TOL_M,
    TISSUE_DEPLOY_HALF_SEPARATION_M,
    TISSUE_HAND_CENTER_X_M,
    TISSUE_NAV_BEHAVIOR_TREE,
    TISSUE_TABLE_BASE_INSET_M,
    TISSUE_TABLE_BASE_Y_OFFSET_M,
    TISSUE_TABLE_EDGE_MARGIN_M,
    TISSUE_TABLE_INSET_M,
    TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M,
    TRANSPORT_SLIDE_M,
)
from .grasp_profiles import (
    _grip_close_command,
    _table_nav_angular_override,
    _table_nav_pose_fast_acceptable,
)
from .sorting_state import CycleState


class EProductCycleMotionMixin:
    @property
    def _broad_release_gripper_is_open(self) -> bool:
        """Require both feedback and the smoothed command to be open."""

        return bool(
            self._right_gripper_is_open
            and float(self.action[18]) >= BROAD_RELEASE_COMMAND_OPEN_MIN
        )

    def _pick_transport_is_ready(self) -> bool:
        """Return whether the loaded body/arms are safe for table handoff."""

        if self.jpos is None:
            return False
        if self._uses_tissue_two_hand_grasp:
            return self._tissue_transport_is_ready()
        compact = self._compact_carry_metrics()
        return bool(
            self._pick_left_stow_complete
            and self._pick_right_carry_ready
            and float(
                np.max(np.abs(self.larm_meas - LEFT_ARM_TRANSPORT))
            )
            < LEFT_ARM_TRANSPORT_TOL
            and bool(compact["strict"] or compact["safe"])
            and abs(self.slide_meas - TRANSPORT_SLIDE_M) <= SLIDE_TOL_M
        )

    def _start_concurrent_pick_stow(self) -> None:
        """Start loaded torso/arm stow without taking ownership of the base."""

        self._pick_deploy_left_restore_active = False
        self._pick_deploy_left_restore_ready_since = None
        self._pick_deploy_held_slide_m = None
        self._left_arm_stow_stage = 0
        self._left_arm_stow_stage_started_at = self.now()
        self._pick_left_stow_complete = False
        self._pick_right_carry_ready = False
        self._compact_carry_safe_since = None
        self._pick_stow_active = True
        self._pick_stow_started_at = self.now()
        self.tc[2] = TRANSPORT_SLIDE_M
        self.tc[5:11] = LEFT_ARM_STOW_WAYPOINTS[0]
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = COMPACT_CARRY_ARM
        self.tc[18] = _grip_close_command(self.target_kind)
        self.joint_slew = (
            CHENGZI_CARRY_STOW_JOINT_SLEW
            if self.target_kind == "chengzi"
            else RIGHT_ARM_RETURN_JOINT_SLEW
        )
        self.get_logger().info(
            "PICK_STOW_CONCURRENT_START "
            f"left_stage=1/{len(LEFT_ARM_STOW_WAYPOINTS)} "
            "right=compact_carry torso=transport nav2=true "
            f"slew={self.joint_slew:.2f}"
        )

    def _update_concurrent_pick_stow(self) -> None:
        """Advance the loaded two-arm/body motion while Nav2 owns the base."""

        if not self._pick_stow_active:
            return

        # Keep the carried product clamped throughout every intermediate pose.
        self.tc[18] = _grip_close_command(self.target_kind)
        if not self._pick_left_stow_complete:
            left_target = LEFT_ARM_STOW_WAYPOINTS[
                self._left_arm_stow_stage
            ]
            left_error = float(
                np.max(np.abs(self.larm_meas - left_target))
            )
            if (
                left_error < LEFT_ARM_TRANSPORT_TOL
                and self.now() - self._left_arm_stow_stage_started_at
                >= LEFT_ARM_STOW_STAGE_DWELL_SEC
            ):
                next_stage = self._left_arm_stow_stage + 1
                if next_stage < len(LEFT_ARM_STOW_WAYPOINTS):
                    self._left_arm_stow_stage = next_stage
                    self._left_arm_stow_stage_started_at = self.now()
                    self.tc[5:11] = LEFT_ARM_STOW_WAYPOINTS[next_stage]
                    self.get_logger().info(
                        "PICK_STOW_CONCURRENT "
                        f"left_stage={next_stage + 1}/"
                        f"{len(LEFT_ARM_STOW_WAYPOINTS)}"
                    )
                else:
                    self._pick_left_stow_complete = True
                    self.get_logger().info(
                        "PICK_STOW_CONCURRENT left_complete=true"
                    )

        compact = self._compact_carry_metrics()
        stow_elapsed = (
            0.0
            if self._pick_stow_started_at is None
            else self.now() - self._pick_stow_started_at
        )
        strict_ready = bool(
            compact["strict"] and stow_elapsed >= ARM_SETTLE_SEC
        )
        if strict_ready:
            self._pick_right_carry_ready = True
        elif not self._pick_right_carry_ready and compact["safe"]:
            if self._compact_carry_safe_since is None:
                self._compact_carry_safe_since = self.now()
                self.get_logger().info(
                    "COMPACT_CARRY_SAFE_DWELL_START "
                    f"non_wrist_error={compact['non_wrist_max_error']:.3f}rad "
                    f"wrist_error={compact['wrist_error']:.3f}rad"
                )
            elif (
                self.now() - self._compact_carry_safe_since
                >= COMPACT_CARRY_SAFE_SETTLE_SEC
            ):
                self.get_logger().warning(
                    "COMPACT_CARRY_SAFE_ACCEPT after 3s stable dwell "
                    f"non_wrist_error={compact['non_wrist_max_error']:.3f}rad "
                    f"wrist_error={compact['wrist_error']:.3f}rad "
                    f"ee={np.round(compact['ee'], 3).tolist()} "
                    f"tilt={math.degrees(compact['tilt']):.1f}deg"
                )
                self._pick_right_carry_ready = True
        elif not self._pick_right_carry_ready:
            self._compact_carry_safe_since = None

        if self._pick_transport_is_ready():
            self._pick_stow_active = False
            self._pick_stow_started_at = None
            self.get_logger().info(
                "PICK_STOW_CONCURRENT_COMPLETE "
                "left=true right=true torso=true nav2_continues=true"
            )
        elif stow_elapsed >= PICK_TRANSPORT_TOGETHER_TIMEOUT_SEC:
            self._pick_stow_active = False
            self._fail("pick_stow_concurrent_timeout")

    def _start_concurrent_return_stow(
        self,
        *,
        navigation_active: bool = True,
        both_arms_from_waiting: bool = False,
    ) -> None:
        """Raise the torso and fold arms inside the navigation envelope."""

        self._pick_deploy_left_restore_active = False
        self._pick_deploy_left_restore_ready_since = None
        self._pick_deploy_held_slide_m = None
        self._return_arm_stage = 0
        self._return_arm_stage_ready_since = None
        self._return_arms_stowed = False
        self._return_stow_active = True
        self._return_stow_both_arms = both_arms_from_waiting
        self._return_stow_navigation_active = navigation_active
        self._return_stow_started_at = self.now()
        self.tc[2] = TRANSPORT_SLIDE_M
        self.tc[5:11] = (
            LEFT_ARM_DOWN_WAYPOINTS[0]
            if both_arms_from_waiting
            else LEFT_ARM_TRANSPORT
        )
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = RIGHT_ARM_DOWN_WAYPOINTS[0]
        self.tc[18] = GRIP_OPEN
        self.joint_slew = RETURN_STOW_FAST_JOINT_SLEW
        self.get_logger().info(
            "RETURN_STOW_CONCURRENT_START "
            f"right_stage=1/{len(RIGHT_ARM_DOWN_WAYPOINTS)}; "
            f"base_mode={'nav2_return' if navigation_active else 'stationary'} "
            f"both_arms={both_arms_from_waiting} torso=raised"
        )

    def _update_concurrent_return_stow(self) -> None:
        """Advance the collision-checked arm fold without owning base motion."""

        if not self._return_stow_active:
            return
        target = RIGHT_ARM_DOWN_WAYPOINTS[self._return_arm_stage]
        left_target = (
            LEFT_ARM_DOWN_WAYPOINTS[self._return_arm_stage]
            if self._return_stow_both_arms
            else LEFT_ARM_TRANSPORT
        )
        ready = (
            self._right_arm_at_target(target)
            and float(np.max(np.abs(self.larm_meas - left_target)))
            < LEFT_ARM_TRANSPORT_TOL
            and abs(self.slide_meas - TRANSPORT_SLIDE_M) <= SLIDE_TOL_M
        )
        if ready:
            if self._return_arm_stage_ready_since is None:
                self._return_arm_stage_ready_since = self.now()
            elif (
                self.now() - self._return_arm_stage_ready_since
                >= FAST_ARM_STAGE_DWELL_SEC
            ):
                next_stage = self._return_arm_stage + 1
                if next_stage < len(RIGHT_ARM_DOWN_WAYPOINTS):
                    self._return_arm_stage = next_stage
                    self._return_arm_stage_ready_since = None
                    self.tc[12:18] = RIGHT_ARM_DOWN_WAYPOINTS[next_stage]
                    if self._return_stow_both_arms:
                        self.tc[5:11] = LEFT_ARM_DOWN_WAYPOINTS[next_stage]
                    label = (
                        "transport"
                        if next_stage == len(RIGHT_ARM_DOWN_WAYPOINTS) - 1
                        else f"fold_{next_stage}"
                    )
                    self.get_logger().info(
                        "RETURN_STOW_CONCURRENT "
                        f"right_stage={next_stage + 1}/"
                        f"{len(RIGHT_ARM_DOWN_WAYPOINTS)} target={label}"
                    )
                else:
                    navigation_active = self._return_stow_navigation_active
                    self._return_arms_stowed = True
                    self._return_stow_active = False
                    self._return_stow_navigation_active = False
                    self._return_stow_started_at = None
                    self.get_logger().info(
                        "RETURN_STOW_CONCURRENT_COMPLETE "
                        "arms_compact=true torso_raised=true lidar_clear=true"
                    )
                    # 退桌时必须用 carry 轮廓覆盖尚未收回的手臂；一旦双臂
                    # 已完全收拢，就在 Nav2 继续行驶期间异步恢复正常轮廓。
                    # 旧逻辑一直带着较大的 carry 轮廓跑到下一柜，靠墙时会
                    # 让 MPPI 的全部采样轨迹同时落入碰撞，形成原地卡死。
                    if navigation_active and self._footprint_mode == "carry":
                        if self._begin_footprint_change("normal"):
                            self._return_nav_footprint_restore_active = True
                        else:
                            self._fail(
                                "return_navigation_footprint_service_unavailable"
                            )
        else:
            self._return_arm_stage_ready_since = None

        if (
            self._return_stow_active
            and self._return_stow_started_at is not None
            and self.now() - self._return_stow_started_at
            >= RETURN_STOW_CONCURRENT_TIMEOUT_SEC
        ):
            self._return_stow_active = False
            self._return_stow_navigation_active = False
            self._fail("return_stow_concurrent_timeout")

    def _update_return_navigation_footprint(self) -> None:
        """Finish the asynchronous carry-to-normal footprint handoff."""

        if not self._return_nav_footprint_restore_active:
            return
        result = self._poll_footprint_change()
        if result is True:
            self._return_nav_footprint_restore_active = False
            self.get_logger().info(
                "RETURN_NAV_FOOTPRINT_NORMAL navigation_continues=true"
            )
        elif result is False:
            self._return_nav_footprint_restore_active = False
            self._fail("return_navigation_footprint_rejected")

    def _start_concurrent_search_restore(self) -> None:
        """Restore both waiting arms while the fixed E observation is running."""

        self.tc[11] = GRIP_OPEN
        self.tc[18] = GRIP_OPEN
        already_ready = (
            float(
                np.max(
                    np.abs(
                        self.larm_meas - np.asarray(INIT_ARM_L, dtype=float)
                    )
                )
            )
            < LEFT_ARM_TRANSPORT_TOL
            and self._right_arm_at_target(np.asarray(INIT_ARM_R, dtype=float))
        )
        if already_ready:
            self.tc[5:11] = INIT_ARM_L
            self.tc[12:18] = INIT_ARM_R
            self._search_arms_restore_active = False
            self._search_arms_ready = True
            self._search_arms_restore_started_at = None
            self._return_arms_stowed = False
            self.get_logger().info(
                "SEARCH_ARMS_ALREADY_READY observation_concurrent=true"
            )
            return

        self._return_arm_stage = 0
        self._return_arm_stage_ready_since = None
        self._search_arms_restore_active = True
        self._search_arms_ready = False
        self._search_arms_restore_started_at = self.now()
        left_target, right_target = SEARCH_ARMS_RESTORE_WAYPOINTS[0]
        self.tc[5:11] = left_target
        self.tc[12:18] = right_target
        self.joint_slew = SEARCH_RESTORE_FAST_JOINT_SLEW
        self.get_logger().info(
            "SEARCH_AND_ARMS_TOGETHER_START "
            f"arm_stage=1/{len(SEARCH_ARMS_RESTORE_WAYPOINTS)} "
            "observation=true"
        )

    def _update_concurrent_search_restore(self) -> None:
        """Advance both arms toward their waiting pose without pausing vision."""

        if not self._search_arms_restore_active:
            return
        left_target, right_target = SEARCH_ARMS_RESTORE_WAYPOINTS[
            self._return_arm_stage
        ]
        ready = (
            self._right_arm_at_target(right_target)
            and float(np.max(np.abs(self.larm_meas - left_target)))
            < LEFT_ARM_TRANSPORT_TOL
        )
        if ready:
            if self._return_arm_stage_ready_since is None:
                self._return_arm_stage_ready_since = self.now()
            elif (
                self.now() - self._return_arm_stage_ready_since
                >= FAST_ARM_STAGE_DWELL_SEC
            ):
                next_stage = self._return_arm_stage + 1
                if next_stage < len(SEARCH_ARMS_RESTORE_WAYPOINTS):
                    self._return_arm_stage = next_stage
                    self._return_arm_stage_ready_since = None
                    next_left, next_right = SEARCH_ARMS_RESTORE_WAYPOINTS[
                        next_stage
                    ]
                    self.tc[5:11] = next_left
                    self.tc[12:18] = next_right
                    self.get_logger().info(
                        "SEARCH_AND_ARMS_TOGETHER "
                        f"arm_stage={next_stage + 1}/"
                        f"{len(SEARCH_ARMS_RESTORE_WAYPOINTS)}"
                    )
                else:
                    self._search_arms_restore_active = False
                    self._search_arms_ready = True
                    self._search_arms_restore_started_at = None
                    self._return_arms_stowed = False
                    self.get_logger().info(
                        "SEARCH_AND_ARMS_TOGETHER_COMPLETE "
                        "left_wait=true right_wait=true observation=true"
                    )
        else:
            self._return_arm_stage_ready_since = None

        if (
            self._search_arms_restore_active
            and self._search_arms_restore_started_at is not None
            and self.now() - self._search_arms_restore_started_at
            >= RETURN_STOW_CONCURRENT_TIMEOUT_SEC
        ):
            self._search_arms_restore_active = False
            self._fail("search_arms_restore_concurrent_timeout")

    # ---- common mission helpers ----
    def _set_navigation_speed_limit(self, speed_mps: float) -> None:
        """Set or clear the controller-server speed limit for loaded tissue."""

        speed_mps = max(0.0, float(speed_mps))
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = False
        # Nav2 defines zero as no limit.  This message limits translation;
        # FollowPath.wz_max is managed separately for loaded-clamp steering.
        msg.speed_limit = speed_mps
        self._speed_limit_pub.publish(msg)
        self._tissue_nav_speed_limit_active = speed_mps > 0.0
        self.get_logger().info(
            "NAV_SPEED_LIMIT "
            f"speed={'unlimited' if speed_mps == 0.0 else f'{speed_mps:.2f}m/s'} "
            f"reason={'normal_navigation' if speed_mps == 0.0 else 'tissue_side_friction_carry'}"
        )

    def _stop_motion(self) -> None:
        self._first_e_direct_active = False
        self._set_nav_angular_override(None)
        self._table_nav_yaw_assist_active = False
        self._table_nav_yaw_assist_command_radps = None
        self._pick_deploy_left_restore_active = False
        self._pick_deploy_left_restore_ready_since = None
        self._pick_deploy_held_slide_m = None
        self.set_twist(0.0, 0.0)
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0] = self.tc[1] = 0.0
        self.base_control_enabled = False
        self.nav2.cancel()
        self.nav2.stop_robot()
        if self._tissue_nav_speed_limit_active:
            self._set_navigation_speed_limit(0.0)
        if (
            self._controller_angular_limit_future is not None
            and self._controller_angular_limit_future.done()
        ):
            self._poll_controller_angular_limit()
        if (
            self._controller_angular_limit_future is None
            and abs(
                self._controller_angular_limit_current
                - NORMAL_MPPI_MAX_ANGULAR_RADPS
            )
            > 1e-9
        ):
            self._begin_controller_angular_limit(
                NORMAL_MPPI_MAX_ANGULAR_RADPS
            )

    def _fail(self, reason: str) -> None:
        self._last_failure_reason = reason
        self._stop_motion()
        self.get_logger().error(
            f"{self._event_prefix}_CYCLE_FAILED reason={reason}"
        )
        self.mission.transition(CycleState.FAILED, reason)

    def _send_goal(
        self,
        pose: tuple[float, float, float],
        name: str,
        wait_state: CycleState,
        *,
        replace_active: bool = False,
    ) -> None:
        self._set_nav_angular_override(None)
        self._table_nav_yaw_assist_active = False
        self._table_nav_yaw_assist_error_rad = None
        self._table_nav_yaw_assist_command_radps = None
        self.base_control_enabled = False
        self._nav_goal_distance_m = None
        self._nav_goal_yaw_error_rad = None
        self._nav_near_goal_since = None
        self._nav_near_goal_leg = name
        self._nav_terminal_hold_commanded = False
        behavior_tree = (
            TISSUE_NAV_BEHAVIOR_TREE
            if self._uses_tissue_two_hand_grasp
            and name == "delivery_table_drop"
            else ""
        )
        if self.nav2.go_to_pose(
            *pose,
            name=name,
            behavior_tree=behavior_tree,
            replace_active=replace_active,
        ):
            self.mission.transition(wait_state)
        else:
            self._fail(f"{name}:{self.nav2.result.name}")

    def _navigation_pose_error(
        self, pose: tuple[float, float, float]
    ) -> tuple[float, float]:
        distance = float(
            np.linalg.norm(self.base_xy - np.asarray(pose[:2], dtype=float))
        )
        yaw_error = abs(
            (float(pose[2]) - self.base_yaw + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        self._nav_goal_distance_m = distance
        self._nav_goal_yaw_error_rad = yaw_error
        return distance, yaw_error

    def _update_table_nav_yaw_assist(
        self, pose: tuple[float, float, float]
    ) -> bool:
        """Mix fixed terminal yaw into Nav2 linear motion at the table.

        Return True after position and yaw are already usable and the state
        has been advanced to STOP_TABLE.  Until that instant Nav2 continues to
        own path following and linear.x; the raw-Twist relay changes only
        angular.z inside the terminal assist region.
        """

        delta = self.base_xy - np.asarray(pose[:2], dtype=float)
        distance = float(np.linalg.norm(delta))
        signed_yaw_error = wrap_to_pi(float(pose[2]) - self.base_yaw)
        angular_command = _table_nav_angular_override(
            distance,
            signed_yaw_error,
        )
        was_active = self._table_nav_yaw_assist_active
        previous_command = self._table_nav_yaw_assist_command_radps

        self._nav_goal_distance_m = distance
        self._nav_goal_yaw_error_rad = abs(signed_yaw_error)
        self._table_nav_yaw_assist_error_rad = signed_yaw_error
        self._table_nav_yaw_assist_command_radps = angular_command
        self._table_nav_yaw_assist_active = angular_command is not None
        self._set_nav_angular_override(angular_command)

        if self._table_nav_yaw_assist_active and (
            not was_active or previous_command != angular_command
        ):
            self.get_logger().info(
                "TABLE_NAV_YAW_ASSIST "
                f"distance={distance:.3f}m "
                f"yaw_error={signed_yaw_error:.3f}rad "
                f"fixed_angular={angular_command:.3f}rad/s "
                "linear_source=nav2"
            )
        elif was_active and not self._table_nav_yaw_assist_active:
            self.get_logger().info(
                "TABLE_NAV_YAW_ASSIST_RELEASED linear_and_angular=nav2"
            )

        if not _table_nav_pose_fast_acceptable(
            distance,
            signed_yaw_error,
        ):
            return False

        # Do not wait for Nav2 success or the generic one-second near-goal
        # dwell.  STOP_TABLE publishes zero immediately and retains the short
        # odometry-confirmed stop needed before extending a loaded arm.
        self._set_nav_angular_override(None)
        self._table_nav_yaw_assist_active = False
        self._table_nav_yaw_assist_command_radps = None
        self.nav2.cancel()
        self.nav2.stop_robot()
        self.get_logger().info(
            "TABLE_NAV_FAST_ACCEPT "
            f"distance={distance:.3f}/"
            f"{TABLE_NEAR_GOAL_POSITION_TOL_M:.3f}m "
            f"yaw_error={abs(signed_yaw_error):.3f}/"
            f"{TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD:.3f}rad "
            "skip_nav_terminal_dwell=true"
        )
        self.mission.transition(
            CycleState.STOP_TABLE,
            "table_pose_ready_without_nav_terminal_dwell",
        )
        return True

    def _wait_navigation(
        self,
        success_state: CycleState,
        leg: str,
        pose: tuple[float, float, float],
        near_position_tolerance: float,
        near_yaw_tolerance: float = NAV_NEAR_GOAL_YAW_TOL_RAD,
    ) -> None:
        """Accept a safe near-goal pose after three seconds of real stillness."""

        result = self.nav2.wait_result()
        if result == NavResult.SUCCEEDED:
            self.mission.transition(success_state)
            return

        distance, yaw_error = self._navigation_pose_error(pose)
        near_goal = (
            distance <= near_position_tolerance
            and yaw_error <= near_yaw_tolerance
        )
        stopped = self._odom_stopped()
        if near_goal and stopped:
            if self._nav_near_goal_since is None:
                self._nav_near_goal_since = self.now()
                self.get_logger().info(
                    f"NAV_NEAR_GOAL_SETTLING leg={leg} "
                    f"distance={distance:.3f}/"
                    f"{near_position_tolerance:.3f} "
                    f"yaw_error={yaw_error:.3f}/"
                    f"{near_yaw_tolerance:.3f} "
                    f"dwell={NAV_NEAR_GOAL_SETTLE_SEC:.1f}s"
                )
            elif (
                self.now() - self._nav_near_goal_since
                >= NAV_NEAR_GOAL_SETTLE_SEC
            ):
                self.get_logger().warning(
                    f"NAV_NEAR_GOAL_ACCEPTED leg={leg} "
                    f"distance={distance:.3f} yaw_error={yaw_error:.3f}; "
                    "continuing after 3s stationary dwell"
                )
                self.nav2.cancel()
                self.nav2.stop_robot()
                self.mission.transition(success_state, "near_goal_settled")
                return
        else:
            self._nav_near_goal_since = None

        # If Nav2 itself aborts inside the accepted neighbourhood, hold still
        # and finish the same dwell instead of converting a harmless terminal
        # centimetre-level miss into a mission failure.
        if result in (NavResult.FAILED, NavResult.CANCELLED):
            if near_goal:
                if not self._nav_terminal_hold_commanded:
                    self.nav2.stop_robot()
                    self._nav_terminal_hold_commanded = True
                return
            if (
                result == NavResult.FAILED
                and leg == "delivery_table_drop"
                and self._footprint_mode == "carry"
                and not self._delivery_nav_chassis_retry_used
            ):
                # 当前紧凑载物姿态的双臂关节均位于底盘宽度内。先保留载物
                # 轮廓规划；只有它穷尽 Nav2 恢复仍失败时，才切换真实底盘
                # 轮廓重试一次。这里不修改全局膨胀参数，也不关闭碰撞检测。
                self._delivery_nav_chassis_retry_used = True
                self.nav2.stop_robot()
                self.get_logger().warning(
                    "DELIVERY_NAV_RETRY_WITH_CHASSIS_FOOTPRINT "
                    f"distance={distance:.3f} yaw_error={yaw_error:.3f} "
                    "reason=carry_footprint_no_valid_path"
                )
                self.mission.transition(
                    CycleState.RETRY_TABLE_CHASSIS_FOOTPRINT,
                    "retry_delivery_with_chassis_footprint",
                )
                return
            if (
                getattr(self, "_random_shelf_mode", False)
                and result == NavResult.FAILED
                and leg.startswith("shelf_")
            ):
                failed_shelf = self._active_shelf
                self._random_unreachable_shelves.add(failed_shelf)
                self._random_planned_marker_id = None
                self._random_planned_kind = None
                self.nav2.stop_robot()
                self.get_logger().warning(
                    "RANDOM_SHELF_NAV_RESCHEDULE "
                    f"unreachable_shelf={failed_shelf} "
                    f"distance={distance:.3f} yaw_error={yaw_error:.3f}"
                )
                self._schedule_random_target(
                    start_return_stow=self._nav_pick_prepared,
                    already_at_scan=False,
                )
                return
            self._fail(f"{leg}:{result.name}")

    def _stopped_and_settled(self) -> bool:
        return self.mission.elapsed >= STOP_SETTLE_SEC and self._odom_stopped()

    def _timed_out(self, label: str, timeout: float = MANIPULATION_TIMEOUT_SEC) -> bool:
        if self.mission.elapsed < timeout:
            return False
        self._fail(f"{label}_timeout")
        return True

    def _begin_footprint_change(self, mode: str) -> bool:
        if mode not in {"normal", "carry"}:
            raise ValueError(f"invalid footprint mode: {mode}")

        # The wide package and its two deployed arms are deliberately excluded
        # from Nav2 collision checking.  Modelling that overhang as the robot
        # footprint makes every MPPI sample invalid while leaving E beside the
        # partition, so the robot cannot begin the delivery leg.  Physical
        # contact remains visible in the simulator; Nav2 plans for the chassis
        # only, exactly as requested for this two-hand transport mode.
        requested_mode = mode
        if self._uses_tissue_two_hand_grasp and mode == "carry":
            mode = "normal"
            self.get_logger().info(
                "TISSUE_NAV_CHASSIS_FOOTPRINT "
                f"requested={requested_mode} effective={mode}"
            )
        if self._footprint_mode == mode:
            self._footprint_pending_mode = None
            self._footprint_futures = []
            return True
        unavailable = [
            client.srv_name
            for client in self._footprint_clients
            if not client.wait_for_service(timeout_sec=0.0)
        ]
        if unavailable:
            self.get_logger().error(
                f"COSTMAP_FOOTPRINT services unavailable: {unavailable}"
            )
            return False
        if mode == "normal":
            footprint = NORMAL_FOOTPRINT
        else:
            footprint = CARRY_FOOTPRINT
        parameter = Parameter(
            name="footprint",
            value=ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value=footprint,
            ),
        )
        self._footprint_futures = []
        for client in self._footprint_clients:
            request = SetParameters.Request()
            request.parameters = [parameter]
            self._footprint_futures.append(client.call_async(request))
        self._footprint_pending_mode = mode
        self.get_logger().info(
            f"COSTMAP_FOOTPRINT_REQUEST mode={mode} footprint={footprint}"
        )
        return True

    def _poll_footprint_change(self) -> bool | None:
        if self._footprint_pending_mode is None:
            return True
        if not all(future.done() for future in self._footprint_futures):
            return None
        failures = []
        for future in self._footprint_futures:
            try:
                response = future.result()
            except Exception as exc:
                failures.append(str(exc))
                continue
            if response is None or not response.results:
                failures.append("empty SetParameters response")
                continue
            for result in response.results:
                if not result.successful:
                    failures.append(result.reason or "parameter rejected")
        if failures:
            self.get_logger().error(
                f"COSTMAP_FOOTPRINT_REJECTED reasons={failures}"
            )
            self._footprint_pending_mode = None
            self._footprint_futures = []
            return False
        self._footprint_mode = self._footprint_pending_mode
        self._footprint_pending_mode = None
        self._footprint_futures = []
        self.get_logger().info(f"COSTMAP_FOOTPRINT_READY mode={self._footprint_mode}")
        return True

    def _begin_controller_angular_limit(self, limit_radps: float) -> bool:
        """Dynamically cap MPPI rotation before a loaded tissue goal."""

        limit_radps = max(0.01, float(limit_radps))
        if (
            self._controller_angular_limit_future is None
            and abs(
                self._controller_angular_limit_current - limit_radps
            )
            <= 1e-9
        ):
            self._controller_angular_limit_target = None
            return True
        if self._controller_angular_limit_future is not None:
            self.get_logger().error(
                "CONTROLLER_ANGULAR_LIMIT refusing overlapping request"
            )
            return False
        if not self._controller_parameter_client.wait_for_service(
            timeout_sec=0.0
        ):
            self.get_logger().error(
                "CONTROLLER_ANGULAR_LIMIT service unavailable"
            )
            return False
        parameter = Parameter(
            name="FollowPath.wz_max",
            value=ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE,
                double_value=limit_radps,
            ),
        )
        request = SetParameters.Request()
        request.parameters = [parameter]
        self._controller_angular_limit_target = limit_radps
        self._controller_angular_limit_future = (
            self._controller_parameter_client.call_async(request)
        )
        self.get_logger().info(
            "CONTROLLER_ANGULAR_LIMIT_REQUEST "
            f"wz_max={limit_radps:.3f}rad/s"
        )
        return True

    def _poll_controller_angular_limit(self) -> bool | None:
        """Return True only after controller_server accepts the angular cap."""

        future = self._controller_angular_limit_future
        if future is None:
            return True
        if not future.done():
            return None
        target = self._controller_angular_limit_target
        try:
            response = future.result()
        except Exception as exc:
            response = None
            failure = str(exc)
        else:
            failure = ""
        if response is None or not response.results:
            failure = failure or "empty SetParameters response"
        elif not response.results[0].successful:
            failure = response.results[0].reason or "parameter rejected"
        self._controller_angular_limit_future = None
        self._controller_angular_limit_target = None
        if failure:
            self.get_logger().error(
                f"CONTROLLER_ANGULAR_LIMIT_REJECTED reason={failure}"
            )
            return False
        self._controller_angular_limit_current = float(target)
        self.get_logger().info(
            "CONTROLLER_ANGULAR_LIMIT_READY "
            f"wz_max={self._controller_angular_limit_current:.3f}rad/s"
        )
        return True

    def _fixed_template_world(self) -> np.ndarray:
        lateral = (
            0.0
            if self._uses_tissue_two_hand_grasp
            else float(HAND_WORKPOINT_XY_M[1])
        )
        forward = (
            TISSUE_HAND_CENTER_X_M
            if self._uses_tissue_two_hand_grasp
            else float(HAND_WORKPOINT_XY_M[0])
        )
        point = np.array(
            [
                forward,
                lateral,
                HAND_Z_PLUS_SLIDE_M - self._target_slide,
            ]
        )
        return self.footprint_to_world(point)

    def _table_drop_index(self, fallback_index: int) -> int:
        """Resolve the reserved table slot for the active/pending target."""

        marker_id = None
        if self._current_target is not None:
            marker_id = int(self._current_target["aruco_id"])
        elif self._pending_verification is not None:
            marker_id = int(self._pending_verification["aruco_id"])
        elif self._target_queue:
            marker_id = int(self._target_queue[0])
        if marker_id is not None and marker_id in self._target_drop_slots:
            return int(self._target_drop_slots[marker_id])
        if self._uses_tissue_two_hand_grasp:
            return 2
        return max(
            0,
            min(int(fallback_index), len(TABLE_DROP_X_OFFSETS_M) - 1),
        )

    def _table_approach_pose(self, drop_index: int) -> tuple[float, float, float]:
        """Return a table pose that preserves the established hand drop point."""

        slot_index = self._table_drop_index(drop_index)
        pose = table_approach_pose(slot_index)
        if not self._uses_tissue_two_hand_grasp:
            # At TABLE_YAW=-pi/2 the right hand advances toward world -Y.
            # Protect the *complete product*, not merely the hand endpoint,
            # from crossing the physical tabletop edge.  This also makes an
            # accidentally aggressive future waypoint offset fail safe.
            forward_m = table_place_forward_distances(
                slot_index, self.target_kind
            )[0]
            minimum_safe_base_y = (
                TABLE_Y_MIN_M
                + self.target_geometry.radius_m
                + SINGLE_HAND_TABLE_EDGE_MARGIN_M
                + float(CARRY_HAND_WORKPOINT_XY_M[0])
                + float(forward_m)
            )
            return (
                float(pose[0]),
                max(float(pose[1]), minimum_safe_base_y),
                float(pose[2]),
            )
        # The normal table pose includes the single right hand's -0.15 m local
        # lateral offset.  A symmetric two-arm clamp has zero lateral offset,
        # so remove that compensation or slot 3 ends up on the east table edge.
        # The goal faces -Y: also compensate the longer straight arms, then
        # apply the requested southward inset.  Clamp the base goal so the
        # package centre remains at least one package half-depth plus 15 mm
        # inside the table's negative-Y edge.
        requested_base_y = (
            float(pose[1])
            + TISSUE_TABLE_BASE_Y_OFFSET_M
            - TISSUE_TABLE_INSET_M
        )
        minimum_safe_center_y = (
            TABLE_Y_MIN_M
            + self.target_geometry.radius_m
            + TISSUE_TABLE_EDGE_MARGIN_M
        )
        minimum_safe_base_y = (
            minimum_safe_center_y + TISSUE_HAND_CENTER_X_M
        )
        return (
            float(pose[0]) + float(CARRY_HAND_WORKPOINT_XY_M[1]),
            max(requested_base_y, minimum_safe_base_y),
            float(pose[2]),
        )

    def _send_delivery_table_goal(self, *, replace_active: bool = False) -> None:
        """Send the final table goal, optionally replacing the active goal."""

        pose = self._table_approach_pose(self._picked_count)
        slot_index = self._table_drop_index(self._picked_count)
        if self._uses_tissue_two_hand_grasp:
            ordinary_pose = table_approach_pose(slot_index)
            previous_base_y = (
                float(ordinary_pose[1])
                + TISSUE_TABLE_BASE_Y_OFFSET_M
                - TISSUE_TABLE_BASE_INSET_M
            )
            applied_shift = previous_base_y - float(pose[1])
            self.get_logger().info(
                "TISSUE_TABLE_NEGATIVE_Y_SHIFT "
                f"requested={TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M:.3f}m "
                f"applied={applied_shift:.3f}m "
                f"safe_base_y={pose[1]:.3f}"
            )
        else:
            nominal_pose = table_approach_pose(slot_index)
            if float(pose[1]) > float(nominal_pose[1]) + 1e-6:
                self.get_logger().info(
                    "TABLE_SINGLE_HAND_EDGE_CLAMP "
                    f"kind={self.target_kind} "
                    f"requested_base_y={nominal_pose[1]:.3f} "
                    f"safe_base_y={pose[1]:.3f} "
                    f"margin={SINGLE_HAND_TABLE_EDGE_MARGIN_M:.3f}m"
                )
        self.get_logger().info(
            f"TABLE_DROP_SLOT index={slot_index + 1}/"
            f"{len(TABLE_DROP_X_OFFSETS_M)} "
            f"base_pose=({pose[0]:.3f},{pose[1]:.3f},"
            f"{pose[2]:.3f}) replace_active={replace_active}"
        )
        self._send_goal(
            pose,
            "delivery_table_drop",
            CycleState.WAIT_TABLE_NAV,
            replace_active=replace_active,
        )

    def _mark_delivery_attempt(self) -> None:
        if self._current_target is None:
            return
        self._delivery_attempt_count += 1
        self._pending_verification = {
            "aruco_id": int(self._current_target["aruco_id"]),
            "kind": str(self._current_target.get("kind", self.target_kind)),
            "shelf": str(
                self._current_target.get("shelf", self._active_shelf)
            ),
            "level": str(self._current_target.get("level", "")),
            "column": str(self._current_target.get("column", "")),
            "product_world": np.asarray(
                self._current_target["product_world"], dtype=float
            ).copy(),
            "grasp_attempt": int(self._current_target["grasp_attempt"]),
        }
        self.get_logger().info(
            f"{self._event_prefix}_DELIVERY_ATTEMPT "
            f"count={self._delivery_attempt_count} "
            f"id={self._current_target['aruco_id']}; "
            + (
                "confirm after completed table release"
                if self._random_shelf_mode
                else "verify on next E scan"
            )
        )
        self._current_target = None

    def _pick_left_support_ready(self) -> bool:
        return bool(
            not self._pick_deploy_left_restore_active
            and float(
                np.max(
                    np.abs(
                        self.larm_meas
                        - np.asarray(INIT_ARM_L, dtype=float)
                    )
                )
            )
            < LEFT_ARM_TRANSPORT_TOL
        )

    def _start_pick_left_support_restore(self) -> None:
        """Raise the unused left arm before lowering the torso for a pick."""

        initial = np.asarray(INIT_ARM_L, dtype=float)
        if (
            float(np.max(np.abs(self.larm_meas - initial)))
            < LEFT_ARM_TRANSPORT_TOL
        ):
            self._pick_deploy_left_restore_active = False
            self._pick_deploy_left_restore_stage = len(
                LEFT_ARM_SCAN_WAYPOINTS
            ) - 1
            self._pick_deploy_left_restore_ready_since = None
            self._pick_deploy_held_slide_m = None
            self.tc[5:11] = initial
            self.tc[2] = self._target_slide
            self.joint_slew = PICK_DEPLOY_BODY_LOWER_JOINT_SLEW
            return

        # Select the nearest point on the already collision-checked reverse
        # stow route.  This also handles any post-first rolling handoff that
        # preempts the return.
        # arm sequence midway instead of assuming the left arm reached its
        # final low transport pose.
        measured = np.asarray(self.larm_meas, dtype=float)
        self._pick_deploy_left_restore_stage = int(
            np.argmin(
                [
                    float(np.linalg.norm(measured - waypoint))
                    for waypoint in LEFT_ARM_SCAN_WAYPOINTS
                ]
            )
        )
        self._pick_deploy_left_restore_active = True
        self._pick_deploy_left_restore_ready_since = None
        self._pick_deploy_held_slide_m = float(self.action[2])
        self.tc[2] = self._pick_deploy_held_slide_m
        self.tc[5:11] = LEFT_ARM_SCAN_WAYPOINTS[
            self._pick_deploy_left_restore_stage
        ]
        self.joint_slew = PICK_DEPLOY_FAST_JOINT_SLEW
        self.get_logger().info(
            "PICK_LEFT_SUPPORT_RESTORE_START "
            f"stage={self._pick_deploy_left_restore_stage + 1}/"
            f"{len(LEFT_ARM_SCAN_WAYPOINTS)} "
            f"held_slide={self._pick_deploy_held_slide_m:.3f}; "
            "torso_lowering=false base_visual_control_continues=true"
        )

    def _update_pick_left_support_restore(self) -> None:
        if not self._pick_deploy_left_restore_active:
            return
        stage = self._pick_deploy_left_restore_stage
        target = LEFT_ARM_SCAN_WAYPOINTS[stage]
        self.tc[5:11] = target
        self.tc[2] = float(self._pick_deploy_held_slide_m)
        self.joint_slew = PICK_DEPLOY_FAST_JOINT_SLEW
        ready = bool(
            float(np.max(np.abs(self.larm_meas - target)))
            < LEFT_ARM_TRANSPORT_TOL
        )
        if ready:
            if self._pick_deploy_left_restore_ready_since is None:
                self._pick_deploy_left_restore_ready_since = self.now()
            elif (
                self.now() - self._pick_deploy_left_restore_ready_since
                >= FAST_ARM_STAGE_DWELL_SEC
            ):
                next_stage = stage + 1
                if next_stage < len(LEFT_ARM_SCAN_WAYPOINTS):
                    self._pick_deploy_left_restore_stage = next_stage
                    self._pick_deploy_left_restore_ready_since = None
                    self.tc[5:11] = LEFT_ARM_SCAN_WAYPOINTS[next_stage]
                    self.get_logger().info(
                        "PICK_LEFT_SUPPORT_RESTORE "
                        f"stage={next_stage + 1}/"
                        f"{len(LEFT_ARM_SCAN_WAYPOINTS)}"
                    )
                else:
                    self._pick_deploy_left_restore_active = False
                    self._pick_deploy_left_restore_ready_since = None
                    self._pick_deploy_held_slide_m = None
                    self.tc[5:11] = INIT_ARM_L
                    self.tc[2] = self._target_slide
                    self.joint_slew = PICK_DEPLOY_BODY_LOWER_JOINT_SLEW
                    self.get_logger().info(
                        "PICK_LEFT_SUPPORT_READY "
                        f"target_slide={self._target_slide:.3f} "
                        f"body_slew={PICK_DEPLOY_BODY_LOWER_JOINT_SLEW:.2f}; "
                        "torso_lowering=true"
                    )
        else:
            self._pick_deploy_left_restore_ready_since = None

    def _command_pick_template(self) -> bool:
        """Command the selected product's grasp posture without owning the base."""

        self.joint_slew = PICK_DEPLOY_FAST_JOINT_SLEW
        if self._uses_tissue_two_hand_grasp:
            self.tc[2] = self._target_slide
            if not self._prepare_tissue_grasp_arms():
                return False
            left, right = self._tissue_deploy_arms
            self.tc[5:11] = left
            self.tc[12:18] = right
            # 双臂夹持商品靠两臂间距保持受力，两个宽接触面持续闭合。
            self.tc[11] = GRIP_CLOSE
            self.tc[18] = GRIP_CLOSE
        else:
            self.tc[12:18] = FIXED_GRASP_ARM
            self.tc[18] = GRIP_OPEN
            self._start_pick_left_support_restore()
        self.arm_target_set = True
        self.DEPLOY_WORLD = self._fixed_template_world()
        self.state_t0 = self.now()
        self.get_logger().info(
            (
                "DEPLOY_TISSUE_TWO_HAND_TEMPLATE "
                if self._uses_tissue_two_hand_grasp
                else "DEPLOY_FIXED_TEMPLATE "
            )
            + f"slide={self._target_slide:.3f} "
            f"world={np.round(self.DEPLOY_WORLD, 3).tolist()} "
            + (
                f"half_width={TISSUE_HALF_WIDTH_M:.3f}m "
                f"open_half_separation="
                f"{TISSUE_DEPLOY_HALF_SEPARATION_M:.3f}m"
                if self._uses_tissue_two_hand_grasp
                else ""
            )
        )
        return True

    def _enable_pick_fine_base(
        self, rolling_handoff_speed_mps: float = 0.0
    ) -> None:
        """Give the local visual controller the base, preserving a rolling handoff."""

        rolling_handoff_speed_mps = max(
            0.0, float(rolling_handoff_speed_mps)
        )
        if rolling_handoff_speed_mps <= 0.0:
            self._enable_baseline_base()
            return
        if self.nav2.is_active:
            raise RuntimeError(
                "cannot hand rolling base control over while Nav2 is active"
            )
        # Nav2 has reached a terminal cancellation result.  Do not publish the
        # zero Twist used by a stationary handoff; initialize the local ramp at
        # the bounded incoming speed and continue in the same control cycle.
        self.nav2.cancel()
        # Seed the ramp with odometry's actual incoming velocity.  Clipping
        # cur_lin directly to the much lower visual cruise speed would create
        # a discontinuous command at a rolling handoff; the first visual-control
        # tick instead sets des_lin and ramp_twist() decelerates continuously.
        speed = min(
            rolling_handoff_speed_mps,
            PICK_FINE_HANDOFF_MAX_LINEAR_MPS,
        )
        self.des_lin = self.cur_lin = speed
        self.des_ang = self.cur_ang = 0.0
        self.tc[0] = speed
        self.tc[1] = 0.0
        self.base_control_enabled = True
        self.get_logger().info(
            "CMD_VEL_OWNER baseline_fine_control "
            f"rolling_speed={speed:.3f}m/s stop_command=false"
        )

    def _pick_template_ready(self) -> bool:
        """Check measured torso/arm/gripper convergence for grasping."""

        if self._uses_tissue_two_hand_grasp:
            return bool(
                self._tissue_deploy_arms is not None
                and abs(self.slide_meas - self._target_slide) <= SLIDE_TOL_M
                and self._both_arms_at_target(self._tissue_deploy_arms)
                and self._tissue_grippers_closed
                and self.now() - self.state_t0 >= ARM_SETTLE_SEC
            )
        deploy_slide_tolerance = (
            CHENGZI_DEPLOY_SLIDE_TOL_M
            if self.target_kind == "chengzi"
            else SLIDE_TOL_M
        )
        if self.now() - self.state_t0 < ARM_SETTLE_SEC:
            return False
        # ``DEPLOY_WORLD`` was captured when predeployment started.  During a
        # rolling Nav2 handoff that world point moves with the chassis, so the
        # inherited deploy_done() world-frame Cartesian check becomes stale
        # even though the arm has reached the correct body-relative grasp
        # template.  Validate the same endpoint in base_footprint instead.
        joint_error = float(
            np.max(np.abs(self.rarm_meas - self.tc[12:18]))
        )
        measured_pose = self.ee_footprint_pose()
        expected_point = np.array(
            [
                float(HAND_WORKPOINT_XY_M[0]),
                float(HAND_WORKPOINT_XY_M[1]),
                float(HAND_Z_PLUS_SLIDE_M - self._target_slide),
            ],
            dtype=float,
        )
        cart_error = float(
            np.linalg.norm(measured_pose[:3, 3] - expected_point)
        )
        rotation_delta = measured_pose[:3, :3].T @ GRASP_ROT
        rotation_error = float(
            math.acos(
                np.clip(
                    (np.trace(rotation_delta) - 1.0) * 0.5,
                    -1.0,
                    1.0,
                )
            )
        )
        return bool(
            self._pick_left_support_ready()
            and
            abs(self.slide_meas - self._target_slide)
            <= deploy_slide_tolerance
            and (
                joint_error < DEPLOY_JOINT_TOL
                or (
                    cart_error < DEPLOY_CART_TOL
                    and rotation_error < DEPLOY_ROT_TOL
                )
            )
        )


__all__ = ["EProductCycleMotionMixin"]
