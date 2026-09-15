"""Ordered mission state transitions for the shelf-picking cycle.

The control loop remains byte-for-byte equivalent to the validated monolithic
implementation; it is isolated here because transition ordering is safety and
timing critical.
"""

from __future__ import annotations

import math
import numpy as np

from .baseline_grasp_controller import (
    GRIP_CLOSE,
    GRIP_OPEN,
    wrap_to_pi,
)
from .nav2_manipulation_client import (
    BASE_STOP_EPS,
    PLACE_LOWER_SETTLE_SEC,
    PLACE_RELEASE_SETTLE_SEC,
    STOP_SETTLE_SEC,
)
from .navigation.sorting_geometry import (
    E_SCAN_POSE,
    FIXED_CARRY_ARM,
    SLIDE_MIN_M,
    TABLE_DROP_X_OFFSETS_M,
    TABLE_TOP_Z_M,
    lifted_slide,
    table_place_forward_distances,
)

from .sorting_config import (
    ARM_SETTLE_SEC,
    BROAD_RELEASE_KINDS,
    BROAD_RELEASE_SETTLE_SEC,
    CHENGZI_GRASP_CLOSE_DWELL_SEC,
    CHENGZI_LIFT_JOINT_SLEW,
    CHENGZI_PLACE_LOWER_SETTLE_SEC,
    CHENGZI_PLACE_LOWER_TIMEOUT_SEC,
    COSTMAP_PARAMETER_TIMEOUT_SEC,
    COUPLED_ALIGNMENT_FINAL_TOL_M,
    COUPLED_ALIGNMENT_KINDS,
    E_SCAN_NEAR_GOAL_POSITION_TOL_M,
    FINE_APPROACH_ABSOLUTE_MAX_TRAVEL_M,
    FINE_APPROACH_DEPTH_OVERSHOOT_LIMIT_M,
    FINE_APPROACH_INITIAL_HEADING_TOL_RAD,
    FINE_APPROACH_LATERAL_PROGRESS_EPS_M,
    FINE_APPROACH_MAX_TIMEOUT_SEC,
    FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS,
    FINE_APPROACH_MIN_TARGET_FORWARD_M,
    FINE_APPROACH_MIN_TIMEOUT_SEC,
    FINE_APPROACH_MIN_TRAVEL_LIMIT_M,
    FINE_APPROACH_NEAR_STOP_TOL_M,
    FINE_APPROACH_PROGRESS_EPS_M,
    FINE_APPROACH_STALLED_NEAR_STOP_TOL_M,
    FINE_APPROACH_STOP_TOL_M,
    FINE_APPROACH_TIMEOUT_MARGIN_SEC,
    FINE_APPROACH_TIME_ABORTS_ENABLED,
    FINE_APPROACH_TRAVEL_BRAKE_MARGIN_M,
    FINE_APPROACH_TRAVEL_MARGIN_M,
    FINE_VISION_LOSS_TIMEOUT_SEC,
    FIRST_PICK_HANDOFF_EXTRA_TRAVEL_M,
    GRASP_CLOSE_DWELL_SEC,
    HEAD_PITCH_TOL_RAD,
    IMAGE_SERVO_HARD_FREEZE_REMAINING_M,
    KOUXIANGTANG_FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS,
    L1_FINE_APPROACH_MIN_TARGET_FORWARD_M,
    L3_FINE_APPROACH_MIN_TARGET_FORWARD_M,
    L3_GRASP_CLOSE_DWELL_SEC,
    LIVE_TARGET_MAX_AGE_SEC,
    NORMAL_MPPI_MAX_ANGULAR_RADPS,
    PICK_RETREAT_SPEED_MPS,
    PINGGUO_INSERTED_GRASP_DEPTH_TOL_M,
    PINGGUO_INSERTED_GRASP_LATERAL_TOL_M,
    PLACE_ADVANCE_GENTLE_JOINT_SLEW,
    PLACE_ADVANCE_JOINT_SLEW_OVERRIDES,
    PLACE_ADVANCE_TO_LOWER_SETTLE_SEC,
    PLACE_LOWER_GENTLE_JOINT_SLEW,
    PLACE_LOWER_JOINT_SLEW_OVERRIDES,
    PLACE_RELEASE_JOINT_SLEW,
    PLACE_RELEASE_JOINT_SLEW_OVERRIDES,
    PLACE_UNFOLD_GENTLE_JOINT_SLEW,
    PLACE_UNFOLD_JOINT_SLEW_OVERRIDES,
    POST_RELEASE_CLEARANCE_LIFT_M,
    POST_RELEASE_CLEARANCE_SETTLE_SEC,
    POST_RELEASE_SLIDE_TOL_M,
    RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD,
    RANDOM_ROLLING_HANDOFF_EXTRA_TRAVEL_M,
    RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS,
    SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD,
    SEARCH_MIN_CLUSTER_SAMPLES,
    SEARCH_MIN_DETECTION_FRAMES,
    SEARCH_OBSERVATION_TIMEOUT_SEC,
    SEARCH_STAGE_SETTLE_SEC,
    SLIDE_TOL_M,
    STOP_HANDOFF_TIMEOUT_MARGIN_SEC,
    SUBSEQUENT_FINE_APPROACH_INITIAL_HEADING_TOL_RAD,
    TABLE_NEAR_GOAL_POSITION_TOL_M,
    TABLE_RETREAT_DISTANCE_M,
    TABLE_RETREAT_SPEED_MPS,
    TABLE_RETREAT_TIMEOUT_SEC,
    TABLE_RETURN_FOLD_MIN_Y_M,
    TABLE_STOP_SETTLE_SEC,
    TISSUE_CLAMP_HALF_SEPARATION_M,
    TISSUE_CLAMP_JOINT_SLEW,
    TISSUE_CLAMP_SETTLE_SEC,
    TISSUE_CONTACT_GRASP_DEPTH_TOL_M,
    TISSUE_CONTACT_GRASP_LATERAL_TOL_M,
    TISSUE_HAND_CENTER_X_M,
    TISSUE_INITIAL_LIFT_JOINT_SLEW,
    TISSUE_LOADED_JOINT_SLEW,
    TISSUE_MOTION_TIMEOUT_SEC,
    TISSUE_NAV_MAX_ANGULAR_RADPS,
    TISSUE_NAV_SPEED_LIMIT_MPS,
    TISSUE_PLACE_LOWER_FORCE_RELEASE_SEC,
    TISSUE_PLACE_LOWER_SETTLE_SEC,
    TISSUE_PRETURN_MAX_ANGULAR_RADPS,
    TISSUE_PRETURN_STOP_TIMEOUT_SEC,
    TISSUE_PRETURN_TIMEOUT_SEC,
    TISSUE_PRETURN_YAW_TOL_RAD,
    TISSUE_RELEASE_HALF_SEPARATION_M,
    TISSUE_RELEASE_FORCE_LIFT_SEC,
    TISSUE_RELEASE_LIFT_FORCE_RETREAT_SEC,
    TISSUE_RELEASE_JOINT_SLEW,
    TISSUE_RELEASE_MIN_SEPARATION_M,
    TISSUE_RELEASE_SETTLE_SEC,
    TISSUE_TABLE_BOTTOM_CLEARANCE_M,
    TISSUE_TABLE_RETREAT_SPEED_MPS,
)
from .tissue_handling import _tissue_preturn_signed_angle_rad
from .grasp_profiles import (
    _edge_row_cylinder_slot_lateral_locked,
    _fine_approach_scheduled_speed,
    _fine_approach_stall_timeout,
    _fine_lateral_tolerance,
    _fine_terminal_ee_control_zone,
    _fine_visual_alignment_pending,
    _grip_close_command,
    _pingguo_inserted_grasp_envelope,
    _precision_grasp_profile,
    _random_pick_predeploy_distance,
    _random_rolling_handoff_distance,
    _rolling_pick_handoff_ready,
    _stationary_predeploy_handoff_ready,
    _tissue_contact_grasp_envelope,
)
from .sorting_state import CycleState


class EProductCycleStateMachineMixin:
    # ---- state machine ----
    def tick(self) -> None:
        if getattr(self, "_shutdown_started", False):
            return
        if self.base_xy is None or self.jpos is None:
            return

        state = self.mission.state
        if self._pick_stow_active:
            self._update_concurrent_pick_stow()
            state = self.mission.state
        if self._return_stow_active:
            self._update_concurrent_return_stow()
            state = self.mission.state
        if getattr(self, "_pick_deploy_left_restore_active", False):
            self._update_pick_left_support_restore()
            state = self.mission.state
        if getattr(self, "_return_nav_footprint_restore_active", False):
            self._update_return_navigation_footprint()
            state = self.mission.state

        if state == CycleState.WAIT_COMMAND:
            self.mission.consume_entry()

        elif state == CycleState.NAV_E_SCAN:
            if self.mission.consume_entry():
                if self._random_shelf_mode:
                    self._route_to_random_shelf("E")
                else:
                    self._send_goal(
                        E_SCAN_POSE,
                        "e_shelf_scan",
                        CycleState.WAIT_E_SCAN_NAV,
                    )

        elif state == CycleState.WAIT_E_SCAN_NAV:
            scan_pose = (
                self._active_random_scan_pose()
                if self._random_shelf_mode
                else E_SCAN_POSE
            )
            scan_leg = (
                f"shelf_{self._active_shelf.lower()}_scan"
                if self._random_shelf_mode
                else "e_shelf_scan"
            )
            handled_first_e_direct = bool(
                self._random_shelf_mode and self._first_e_direct_active
            )
            if handled_first_e_direct:
                self._update_first_e_direct_drive(scan_pose)
                state = self.mission.state
            if self._random_shelf_mode and not handled_first_e_direct:
                distance, yaw_error = self._navigation_pose_error(scan_pose)
                policy = self._nav_pick_handoff_mode
                predeploy_distance = _random_pick_predeploy_distance(
                    self._picked_count, policy, self._active_shelf
                )
                rolling_handoff_distance = _random_rolling_handoff_distance(
                    self._picked_count, self._active_shelf
                )
                if (
                    policy in {"rolling", "stationary_predeploy"}
                    and distance <= predeploy_distance
                    and yaw_error <= RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD
                ):
                    self._prepare_random_pick_during_navigation()

                if self._nav_pick_prepared:
                    # Once a concrete item is known, keep the camera on that
                    # item while Nav2 finishes the coarse approach. A fresh
                    # frame improves the remembered pose when available, but
                    # it is not a handoff gate: the task inventory remains
                    # authoritative until a confirmed delivery removes it.
                    self._track_pick_head()
                    self._reacquire_target(tracking=True)
                else:
                    # Until then keep the shelf centred without lowering arms
                    # into the laser plane.
                    self._track_random_scan_head()

                rolling_speed = float(
                    getattr(self, "odom_linear_speed", math.inf)
                )
                stationary_pick_ready = bool(
                    self._nav_pick_prepared
                    and self.target_locked
                    and self._pick_template_ready()
                )
                if _stationary_predeploy_handoff_ready(
                    policy,
                    self._active_shelf,
                    distance,
                    yaw_error,
                    stationary_pick_ready,
                ):
                    self._nav_pick_handoff_speed_mps = max(
                        0.0,
                        min(
                            rolling_speed,
                            RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS,
                        ),
                    )
                    self._fine_first_pick_early_handoff = False
                    self.nav2.cancel()
                    self.get_logger().info(
                        "RANDOM_PICK_ADE_NAV_NEAR_HANDOFF "
                        f"shelf={self._active_shelf} "
                        f"distance={distance:.3f}m "
                        f"yaw_error={yaw_error:.3f}rad "
                        "pick_pose_ready=true exact_nav_yaw_required=false "
                        "wait_for_terminal_dwell=false "
                        "alignment=visual_servo"
                    )
                    self.mission.transition(
                        CycleState.HANDOFF_PICK_CONTROL,
                        "ade_observation_pose_near_continue_visual_pick",
                    )
                    state = self.mission.state
                elif _rolling_pick_handoff_ready(
                    policy,
                    distance,
                    rolling_handoff_distance,
                    yaw_error,
                    self._nav_pick_prepared,
                ):
                    # Cancel the remaining observation-point Nav2 leg without
                    # issuing an explicit stop.  The next state waits only for
                    # command ownership, then continues with visual alignment.
                    self._nav_pick_handoff_speed_mps = max(
                        0.0,
                        min(
                            rolling_speed,
                            RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS,
                        ),
                    )
                    self._fine_first_pick_early_handoff = bool(
                        self._picked_count == 0
                    )
                    self.nav2.cancel()
                    self.get_logger().info(
                        "RANDOM_PICK_ROLLING_HANDOFF "
                        f"shelf={self._active_shelf} "
                        f"distance={distance:.3f}m "
                        f"yaw_error={yaw_error:.3f}rad "
                        f"speed={rolling_speed:.3f}m/s "
                        "wait_for_full_stop=false waypoint_tracking=false "
                        "alignment=visual_servo"
                    )
                    self.mission.transition(
                        CycleState.HANDOFF_PICK_CONTROL,
                        "near_scan_continue_with_visual_pick",
                    )
                    state = self.mission.state
                else:
                    self._wait_navigation(
                        CycleState.STOP_E_SCAN,
                        scan_leg,
                        scan_pose,
                        E_SCAN_NEAR_GOAL_POSITION_TOL_M,
                    )
            elif not self._random_shelf_mode:
                self._wait_navigation(
                    CycleState.STOP_E_SCAN,
                    scan_leg,
                    scan_pose,
                    E_SCAN_NEAR_GOAL_POSITION_TOL_M,
                )

        elif state == CycleState.HANDOFF_PICK_CONTROL:
            if self.mission.consume_entry():
                # cancel() was issued at the rolling handoff boundary.  Do not
                # call stop_robot(): preserving low forward momentum avoids a
                # stop-and-go pause at the observation point.
                self.get_logger().info(
                    "RANDOM_PICK_CONTROL_HANDOFF_WAIT "
                    f"shelf={self._active_shelf} stop_command=false"
                )
            self._track_pick_head()
            self._reacquire_target(tracking=True)
            if not self.nav2.is_active:
                self._fine_started_from_rolling_handoff = True
                self.mission.transition(
                    CycleState.FINE_APPROACH,
                    "nav2_released_continue_visual_or_inventory_memory",
                )

        elif state == CycleState.STOP_E_SCAN:
            if self.mission.consume_entry():
                self.nav2.cancel()
                self.nav2.stop_robot()
            if self._stopped_and_settled() and not self.nav2.is_active:
                if self._nav_pick_prepared and self.target_locked:
                    # This is now only a discovery/legacy fallback: normal
                    # post-first routes hand off before reaching this
                    # state.  If no target was available during the route, the
                    # robot may still observe and activate it while stationary.
                    self._track_pick_head()
                    self._reacquire_target(tracking=True)
                    stationary_pose_ready = bool(
                        self._nav_pick_handoff_mode
                        != "stationary_predeploy"
                        or self._pick_template_ready()
                    )
                    if stationary_pose_ready:
                        self._fine_started_from_rolling_handoff = False
                        self.mission.transition(
                            CycleState.FINE_APPROACH,
                            "observation_stop_complete_visual_or_inventory_memory",
                        )
                elif self._return_stow_active:
                    # Nav2 may reach E before the collision-checked arm route
                    # has finished.  Keep the base still here, but continue
                    # advancing that route at the top of every tick.
                    pass
                elif (
                    self._return_arms_stowed
                    and not self._return_nav_footprint_restore_active
                ):
                    self.mission.transition(CycleState.RESTORE_RETURN_FOOTPRINT)
                else:
                    self.mission.transition(CycleState.SEARCH_E)

        elif state == CycleState.SEARCH_E:
            if self.mission.consume_entry():
                self._configure_e_search()
            if self._search_arms_restore_active:
                self._update_concurrent_search_restore()
            if self.mission.state != CycleState.SEARCH_E:
                state = self.mission.state
            if (
                self.mission.state == CycleState.SEARCH_E
                and self._active_search_poses
            ):
                target_slide, target_pitch = self._active_search_poses[
                    self._search_pose_index
                ]
                self.tc[2] = target_slide
                self.tc[4] = target_pitch
                measured_pitch = float(
                    self.jpos.get("head_pitch_joint", self.tc[4])
                )
                # World-frame detections remain valid while the slide rises,
                # so begin observing as soon as the camera faces the shelf.
                # Selection still waits for the final observation height and
                # both arms, allowing all three motions to overlap safely.
                camera_ready = (
                    abs(measured_pitch - target_pitch) <= HEAD_PITCH_TOL_RAD
                )
                pose_ready = (
                    abs(self.slide_meas - target_slide) <= SLIDE_TOL_M
                    and camera_ready
                )
            else:
                camera_ready = False
                pose_ready = False
            if camera_ready:
                if self._stage_ready_since is None:
                    # Start a clean observation window while the torso and
                    # both arms continue toward their observation poses.
                    self._search_points.clear()
                    self._search_detection_frames = 0
                    self._search_last_frame_at = None
                    self._search_collecting = True
                    self._stage_ready_since = self.now()
                observation_age = self.now() - self._stage_ready_since
                enough_frames = (
                    self._search_detection_frames
                    >= SEARCH_MIN_DETECTION_FRAMES
                )
                if (
                    observation_age >= SEARCH_STAGE_SETTLE_SEC
                    and enough_frames
                    and pose_ready
                    and self._search_arms_ready
                ):
                    self._search_collecting = False
                    self._process_e_search_pose()
                elif (
                    observation_age >= SEARCH_OBSERVATION_TIMEOUT_SEC
                    and pose_ready
                    and self._search_arms_ready
                ):
                    self._search_collecting = False
                    if (
                        self._search_detection_frames
                        >= SEARCH_MIN_CLUSTER_SAMPLES
                    ):
                        self.get_logger().warning(
                            f"{self._event_prefix}_OBSERVATION_PARTIAL "
                            f"frames={self._search_detection_frames}/"
                            f"{SEARCH_MIN_DETECTION_FRAMES}; processing "
                            "available stable detections"
                        )
                        self._process_e_search_pose()
                    else:
                        self._fail(
                            "e_observation_insufficient_detection_frames_"
                            f"{self._search_detection_frames}"
                        )
            else:
                self._search_collecting = False
                self._search_points.clear()
                self._search_detection_frames = 0
                self._search_last_frame_at = None
                self._stage_ready_since = None

        elif state == CycleState.DEPLOY_TEMPLATE:
            if self.mission.consume_entry():
                if not self._command_pick_template():
                    self._fail("tissue_dual_deploy_unreachable")
            # Torso and arm deployment change the camera/object geometry even
            # though the base is stationary.  Track throughout deployment so
            # the selected product remains in view before base motion starts.
            if self.mission.state == CycleState.DEPLOY_TEMPLATE:
                self._track_pick_head()
                if self._uses_tissue_two_hand_grasp:
                    self.tc[11] = GRIP_CLOSE
                    self.tc[18] = GRIP_CLOSE
                if self._pick_template_ready():
                    self.mission.transition(CycleState.FINE_APPROACH)
                else:
                    self._timed_out("deploy_template")

        elif state == CycleState.FINE_APPROACH:
            entered = self.mission.consume_entry()
            if entered:
                rolling_handoff_speed = (
                    self._nav_pick_handoff_speed_mps
                    if self._fine_started_from_rolling_handoff
                    else 0.0
                )
                self._enable_pick_fine_base(rolling_handoff_speed)
                self._fine_start_xy = self.base_xy.copy()
                self._fine_vision_lost_at = None
                self._fine_near_latched = False
                self._fine_near_latched_at = None
                self._fine_near_mode = None
                self._fine_near_tolerance_m = None
                self._fine_near_since = None
                self._fine_precision_aligned = False
                self._fine_terminal_heading_target_rad = None
                self._fine_terminal_heading_error_rad = None
                self._fine_target_forward_m = None
                self._fine_forward_remaining_m = None
                self._fine_forward_error_m = None
                self._fine_braking_speed_mps = None
                self._fine_braking_distance_m = None
                self._fine_dynamic_depth_tolerance_m = None
                self._fine_alignment_reserve_m = None
                self._fine_nominal_speed_mps = None
                self._fine_speed_cap_mps = None
                self._fine_initial_heading_aligned = False
                self._fine_initial_heading_tolerance_rad = (
                    SUBSEQUENT_FINE_APPROACH_INITIAL_HEADING_TOL_RAD
                    if self._picked_count > 0
                    else FINE_APPROACH_INITIAL_HEADING_TOL_RAD
                )
                self._fine_heading_error_rad = None
                self._fine_grasp_yaw_error_rad = None
                self._fine_angular_limit_radps = None
                self._fine_control_mode = "continuous_curve_start"
                self._fine_signed_lateral_error_m = None
                self._fine_live_updates_frozen = False
            # 第一段全程接收当前图像：若商品框和夹爪投影像素都可用，后面的
            # 局部控制器直接以两者的横向像素差纠偏；只有像素缺失/过期时才
            # 使用这里保留的三维视觉线或观察点记忆。进入末端区后，为避免
            # 黑色夹爪遮挡并带偏商品框，锁定最后一次可信视觉几何。
            visual_alignment_pending = _fine_visual_alignment_pending(
                self.now(),
                self._fine_pixel_error_px,
                self._fine_pixel_observation_at,
            )
            live_updates_allowed = bool(
                not self._fine_near_latched
                and not self._fine_live_updates_frozen
                and (
                    self._fine_forward_remaining_m is None
                    or self._fine_forward_remaining_m
                    > _fine_terminal_ee_control_zone(self.target_kind)
                    or (
                        visual_alignment_pending
                        and self._fine_forward_remaining_m
                        > IMAGE_SERVO_HARD_FREEZE_REMAINING_M
                    )
                )
            )
            if live_updates_allowed:
                self._reacquire_target(tracking=True)
            elif (
                not self._fine_near_latched
                and not self._fine_live_updates_frozen
            ):
                self._fine_live_updates_frozen = True
                source = (
                    "last live visual line"
                    if self._fine_last_live_target_world is not None
                    else "observation visual memory"
                )
                self.get_logger().info(
                    f"{self._event_prefix}_FINE_VISUAL_LINE_LOCKED "
                    f"source={source}; "
                    + (
                        "terminal visual-heading hold"
                        if self.target_kind == "sanmingzhi"
                        else "terminal end-effector control"
                    )
                )
            self._track_pick_head()
            if self._uses_tissue_two_hand_grasp:
                self.tc[11] = GRIP_CLOSE
                self.tc[18] = GRIP_CLOSE
            # A symmetric clamp is controlled by the midpoint between both hands;
            # single-hand products retain the measured right endpoint.
            end_effector = self._grasp_reference_world()
            if entered:
                self._fine_best_ee_y = float(end_effector[1])
                self._fine_best_lateral_error_m = abs(
                    float(self.OBJECT_WORLD[0] - end_effector[0])
                )
                self._fine_progress_at = self.now()
                self._fine_required_distance_m = max(
                    0.0, self.CREEP_STOP_Y - float(end_effector[1])
                )
                # Convert the required +Y end-effector progress into an
                # approximate chassis path length at the current heading.
                # Steering can lengthen that path, hence the explicit margin.
                y_progress_per_m = max(0.75, math.sin(self.base_yaw))
                estimated_base_travel = (
                    self._fine_required_distance_m / y_progress_per_m
                )
                absolute_travel_limit = FINE_APPROACH_ABSOLUTE_MAX_TRAVEL_M
                if self._fine_started_from_rolling_handoff:
                    absolute_travel_limit += (
                        FIRST_PICK_HANDOFF_EXTRA_TRAVEL_M
                        if self._fine_first_pick_early_handoff
                        else RANDOM_ROLLING_HANDOFF_EXTRA_TRAVEL_M
                    )
                self._fine_travel_limit_m = float(
                    np.clip(
                        estimated_base_travel + FINE_APPROACH_TRAVEL_MARGIN_M,
                        FINE_APPROACH_MIN_TRAVEL_LIMIT_M,
                        absolute_travel_limit,
                    )
                )
                self._fine_timeout_sec = (
                    float(
                        np.clip(
                            self._fine_travel_limit_m
                            / (
                                KOUXIANGTANG_FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS
                                if self.target_kind == "kouxiangtang"
                                else FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS
                            )
                            + FINE_APPROACH_TIMEOUT_MARGIN_SEC,
                            FINE_APPROACH_MIN_TIMEOUT_SEC,
                            FINE_APPROACH_MAX_TIMEOUT_SEC,
                        )
                    )
                    if FINE_APPROACH_TIME_ABORTS_ENABLED
                    else None
                )
                self._fine_travel_m = 0.0
                self.get_logger().info(
                    "FINE_APPROACH_START "
                    f"ee_y={end_effector[1]:.3f} "
                    f"stop_y={self.CREEP_STOP_Y:.3f} "
                    f"remaining={self._fine_required_distance_m:.3f} "
                    f"travel_limit={self._fine_travel_limit_m:.3f} "
                    f"timeout="
                    f"{f'{self._fine_timeout_sec:.1f}s' if self._fine_timeout_sec is not None else 'disabled'} "
                    f"rolling_handoff="
                    f"{self._fine_started_from_rolling_handoff}"
                )
            vision_fresh = (
                self._target_last_seen_at is not None
                and self.now() - self._target_last_seen_at
                <= LIVE_TARGET_MAX_AGE_SEC
            )
            target_footprint = self.world_to_footprint(self.OBJECT_WORLD)
            self._fine_target_forward_m = float(target_footprint[0])
            remembered_guidance = bool(
                self.target_locked
                and self.OBJECT_WORLD is not None
                and self.CREEP_STOP_Y is not None
            )
            guidance_available = vision_fresh or remembered_guidance
            if self._fine_live_updates_frozen and remembered_guidance:
                self._fine_guidance_source = (
                    "terminal_live_visual_lock"
                    if self._fine_last_live_target_world is not None
                    else "terminal_observation_memory"
                )
            elif vision_fresh:
                level = (
                    ""
                    if self._current_target is None
                    else str(self._current_target.get("level", ""))
                )
                self._fine_guidance_source = (
                    "live_vision_slot_lateral_lock"
                    if _edge_row_cylinder_slot_lateral_locked(
                        self.target_kind, level
                    )
                    else "live_vision"
                )
            elif self._fine_last_live_target_world is not None:
                self._fine_guidance_source = "last_live_visual_memory"
            elif remembered_guidance:
                self._fine_guidance_source = "observation_visual_memory"
            else:
                self._fine_guidance_source = "none"
            if not guidance_available:
                self.set_twist(0.0, 0.0)
                if self._fine_vision_lost_at is None:
                    self._fine_vision_lost_at = self.now()
                    self.get_logger().error(
                        f"{self._event_prefix}_FINE_GUIDANCE_INVALID "
                        "neither live vision nor a locked observation point "
                        "is available"
                    )
                if self._fine_near_latched:
                    self._settle_latched_fine_approach(vision_fresh=False)
                elif (
                    FINE_APPROACH_TIME_ABORTS_ENABLED
                    and self.now() - self._fine_vision_lost_at
                    >= FINE_VISION_LOSS_TIMEOUT_SEC
                ):
                    self._fine_near_since = None
                    self._fail("fine_approach_guidance_invalid")
            else:
                if self._fine_live_updates_frozen:
                    # This is an intentional terminal lock, not a vision-loss
                    # event.  Do not emit a misleading occlusion warning while
                    # the endpoint controller finishes on the frozen line.
                    self._fine_vision_lost_at = None
                elif not vision_fresh and self._fine_vision_lost_at is None:
                    self._fine_vision_lost_at = self.now()
                    self.get_logger().info(
                        f"{self._event_prefix}_FINE_MEMORY_GUIDANCE "
                        "live target is not currently visible; continuing "
                        "on the last visual/observation memory"
                    )
                elif vision_fresh and self._fine_vision_lost_at is not None:
                    self.get_logger().info(
                        f"{self._event_prefix}_FINE_VISION_RESUMED "
                        "live corrections resumed"
                    )
                    self._fine_progress_at = self.now()
                    self._fine_vision_lost_at = None

                if self._fine_near_latched:
                    self._settle_latched_fine_approach(
                        vision_fresh=vision_fresh
                    )

            if guidance_available and not self._fine_near_latched:
                level = str(self._current_target.get("level", ""))
                edge_row = level in {"L1", "L3"}
                precision_profile = _precision_grasp_profile(
                    self.target_kind
                )
                precision_grasp = precision_profile is not None
                coupled_alignment = (
                    self.target_kind in COUPLED_ALIGNMENT_KINDS
                )
                if precision_profile is None:
                    precision_enter_tol = None
                    precision_exit_tol = None
                else:
                    (
                        precision_enter_tol,
                        precision_exit_tol,
                        _,
                    ) = precision_profile
                lateral_remaining = float(
                    self.OBJECT_WORLD[0] - end_effector[0]
                )
                self._fine_signed_lateral_error_m = lateral_remaining
                if (
                    end_effector[1]
                    >= self._fine_best_ee_y + FINE_APPROACH_PROGRESS_EPS_M
                ):
                    self._fine_best_ee_y = float(end_effector[1])
                    self._fine_progress_at = self.now()
                lateral_error = abs(lateral_remaining)
                self._fine_lateral_error_m = lateral_error
                final_lateral_tolerance = _fine_lateral_tolerance(
                    self.target_kind, level
                )
                if precision_grasp:
                    if (
                        not self._fine_precision_aligned
                        and lateral_error <= precision_enter_tol
                    ):
                        self._fine_precision_aligned = True
                        # Rotating the long deployed arm into lateral
                        # alignment can move its endpoint backward in world Y.
                        # Start the insertion progress reference at the newly
                        # aligned pose; otherwise a genuinely moving chassis
                        # can be reported stalled while it is only recovering
                        # the endpoint displacement caused by that rotation.
                        self._fine_best_ee_y = float(end_effector[1])
                        self._fine_progress_at = self.now()
                        self.get_logger().info(
                            f"{self._event_prefix}_LATERAL_ALIGNED "
                            f"error={lateral_error:.3f}m "
                            f"ee_y_reference={self._fine_best_ee_y:.3f}; "
                            "continue coupled insertion"
                        )
                    elif (
                        self._fine_precision_aligned
                        and lateral_error >= precision_exit_tol
                    ):
                        self._fine_precision_aligned = False
                        self._fine_progress_at = self.now()
                        self.get_logger().warning(
                            f"{self._event_prefix}_LATERAL_DRIFT "
                            f"error={lateral_error:.3f}m; slow coupled correction"
                        )
                if (
                    self._fine_best_lateral_error_m is None
                    or lateral_error
                    <= self._fine_best_lateral_error_m
                    - FINE_APPROACH_LATERAL_PROGRESS_EPS_M
                ):
                    self._fine_best_lateral_error_m = lateral_error
                    self._fine_progress_at = self.now()
                grasp_yaw_error = wrap_to_pi(
                    self._target_grasp_yaw() - self.base_yaw
                )
                self._fine_grasp_yaw_error_rad = grasp_yaw_error
                if (
                    self.target_kind == "sanmingzhi"
                    and self._fine_terminal_heading_target_rad is not None
                ):
                    self._fine_terminal_heading_error_rad = wrap_to_pi(
                        self._fine_terminal_heading_target_rad
                        - self.base_yaw
                    )
                    grasp_heading_aligned = bool(
                        abs(self._fine_terminal_heading_error_rad)
                        <= SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD
                    )
                else:
                    grasp_heading_aligned = True
                travel = float(
                    np.linalg.norm(self.base_xy - self._fine_start_xy)
                )
                self._fine_travel_m = travel
                self._fine_forward_error_m = (
                    self.CREEP_STOP_Y - float(end_effector[1])
                )
                self._fine_forward_remaining_m = max(
                    0.0, self._fine_forward_error_m
                )
                self._fine_braking_distance_m = (
                    self._fine_predictive_braking_distance()
                )
                self._fine_dynamic_depth_tolerance_m = (
                    FINE_APPROACH_STOP_TOL_M
                    + self._fine_braking_distance_m
                )
                if level == "L1":
                    min_target_forward = (
                        L1_FINE_APPROACH_MIN_TARGET_FORWARD_M
                    )
                elif level == "L3":
                    min_target_forward = (
                        L3_FINE_APPROACH_MIN_TARGET_FORWARD_M
                    )
                else:
                    min_target_forward = FINE_APPROACH_MIN_TARGET_FORWARD_M
                depth_reached = (
                    end_effector[1]
                    >= self.CREEP_STOP_Y - FINE_APPROACH_STOP_TOL_M
                )
                if precision_grasp:
                    lateral_aligned = (
                        lateral_error <= precision_enter_tol
                    )
                elif coupled_alignment:
                    lateral_aligned = (
                        lateral_error <= COUPLED_ALIGNMENT_FINAL_TOL_M
                    )
                else:
                    # The direct controller aligns every product, including
                    # L2.  The old Nav2 pick waypoint supplied most of this X
                    # correction implicitly, so accepting arbitrary L2 error
                    # after removing that waypoint would be unsafe.
                    lateral_aligned = (
                        lateral_error <= final_lateral_tolerance
                    )
                reached = (
                    depth_reached
                    and lateral_aligned
                    and grasp_heading_aligned
                )
                predictive_brake_ready = (
                    lateral_aligned
                    and grasp_heading_aligned
                    and self._fine_forward_error_m
                    <= self._fine_dynamic_depth_tolerance_m
                )
                excessive_overshoot = (
                    self._fine_forward_error_m
                    < -FINE_APPROACH_DEPTH_OVERSHOOT_LIMIT_M
                )
                if excessive_overshoot:
                    self._fine_near_since = None
                    self._fail("fine_approach_depth_overshoot")
                elif predictive_brake_ready:
                    if self._fine_target_forward_m <= min_target_forward:
                        self._fine_near_since = None
                        self._fail("fine_approach_shelf_guard")
                    else:
                        # A zero desired velocity still takes distance to
                        # become a zero published velocity because the base
                        # command is acceleration-limited.  Latch before the
                        # exact geometric plane using current-speed braking,
                        # then require real odometry stillness before closing.
                        self._fine_near_latched = True
                        self._fine_near_latched_at = self.now()
                        self._fine_near_mode = "predictive_brake"
                        self._fine_near_tolerance_m = (
                            self._fine_dynamic_depth_tolerance_m
                        )
                        self.get_logger().info(
                            "FINE_APPROACH_BRAKE_LOCKED "
                            f"signed_remaining="
                            f"{self._fine_forward_error_m:.3f} "
                            f"braking_distance="
                            f"{self._fine_braking_distance_m:.3f} "
                            f"dynamic_tolerance="
                            f"{self._fine_dynamic_depth_tolerance_m:.3f} "
                            f"speed={max(0.0, self.cur_lin):.3f} "
                            f"lateral_error={lateral_error:.3f}; "
                            "stopping before grasp"
                        )
                        self._settle_latched_fine_approach(
                            vision_fresh=vision_fresh
                        )
                elif not reached:
                    stall_timeout = _fine_approach_stall_timeout(
                        self.target_kind,
                        precision_aligned=self._fine_precision_aligned,
                    )
                    progress_stalled = (
                        self.now() - self._fine_progress_at
                        >= stall_timeout
                    )
                    travel_budget_low = (
                        travel
                        >= self._fine_travel_limit_m
                        - FINE_APPROACH_TRAVEL_BRAKE_MARGIN_M
                    )
                    pingguo_inserted_ready = (
                        _pingguo_inserted_grasp_envelope(
                            self.target_kind,
                            self._fine_forward_remaining_m,
                            lateral_error,
                        )
                    )
                    tissue_contact_ready = bool(
                        progress_stalled
                        and self._odom_stopped()
                        and _tissue_contact_grasp_envelope(
                            self.target_kind,
                            self._fine_forward_remaining_m,
                            lateral_error,
                        )
                    )
                    if pingguo_inserted_ready:
                        near_mode = "inserted_apple_grasp_envelope"
                    elif tissue_contact_ready:
                        near_mode = "tissue_contact_grasp_envelope"
                    elif travel_budget_low:
                        near_mode = "travel_budget"
                    elif (
                        FINE_APPROACH_TIME_ABORTS_ENABLED
                        and progress_stalled
                    ):
                        near_mode = "stalled"
                    else:
                        near_mode = "normal"
                    near_stop_tolerance = (
                        FINE_APPROACH_STALLED_NEAR_STOP_TOL_M
                        if progress_stalled or travel_budget_low
                        else FINE_APPROACH_NEAR_STOP_TOL_M
                    )
                    if pingguo_inserted_ready:
                        near_stop_tolerance = max(
                            near_stop_tolerance,
                            PINGGUO_INSERTED_GRASP_DEPTH_TOL_M,
                        )
                    elif tissue_contact_ready:
                        near_stop_tolerance = max(
                            near_stop_tolerance,
                            TISSUE_CONTACT_GRASP_DEPTH_TOL_M,
                        )
                    near_depth = (
                        self._fine_forward_remaining_m
                        <= near_stop_tolerance
                    )
                    if precision_grasp:
                        near_lateral_tolerance = (
                            PINGGUO_INSERTED_GRASP_LATERAL_TOL_M
                            if pingguo_inserted_ready
                            else (
                                TISSUE_CONTACT_GRASP_LATERAL_TOL_M
                                if tissue_contact_ready
                                else precision_enter_tol
                            )
                        )
                        near_lateral = (
                            lateral_error <= near_lateral_tolerance
                        )
                    elif coupled_alignment:
                        near_lateral = (
                            lateral_error <= COUPLED_ALIGNMENT_FINAL_TOL_M
                        )
                    else:
                        near_lateral = (
                            lateral_error <= final_lateral_tolerance
                        )
                    near_safe = (
                        near_depth
                        and near_lateral
                        and grasp_heading_aligned
                        and self._fine_target_forward_m > min_target_forward
                    )
                    if near_safe:
                        self._fine_near_latched = True
                        self._fine_near_latched_at = self.now()
                        self._fine_near_mode = near_mode
                        self._fine_near_tolerance_m = near_stop_tolerance
                        self.get_logger().warning(
                            "FINE_APPROACH_NEAR_LOCKED "
                            f"mode={near_mode} "
                            f"remaining={self._fine_forward_remaining_m:.3f} "
                            f"tolerance={near_stop_tolerance:.3f} "
                            f"lateral_error={lateral_error:.3f} "
                            f"terminal_heading_error="
                            f"{self._fine_terminal_heading_error_rad or 0.0:.3f}; "
                            "freezing last live target geometry"
                        )
                        self._settle_latched_fine_approach(
                            vision_fresh=vision_fresh
                        )
                    elif (
                        not depth_reached
                        and self._fine_target_forward_m
                        <= min_target_forward
                    ):
                        self._fine_near_since = None
                        self._fail("fine_approach_shelf_guard")
                    elif travel >= self._fine_travel_limit_m:
                        self._fine_near_since = None
                        self._fail("fine_approach_travel_limit")
                    elif (
                        self._fine_timeout_sec is not None
                        and self.mission.elapsed >= self._fine_timeout_sec
                    ):
                        self._fine_near_since = None
                        self._fail("fine_approach_timeout")
                    elif (
                        FINE_APPROACH_TIME_ABORTS_ENABLED
                        and progress_stalled
                    ):
                        self._fine_near_since = None
                        self._fail("fine_approach_stalled")
                    else:
                        self._fine_near_since = None
                        # Every class follows the same continuous curve and
                        # distance-based speed schedule.  Steering begins on
                        # the first translating tick; speed is high in open
                        # space and blends smoothly to the alignment speed as
                        # grasp depth approaches.  The lateral reserve and
                        # braking governor inside the controller can only
                        # reduce this nominal request further.
                        fine_speed = _fine_approach_scheduled_speed(
                            self._fine_forward_remaining_m,
                            self.target_kind,
                        )
                        self._fine_nominal_speed_mps = fine_speed
                        linear_command, angular, _, _ = (
                            self._e_fine_approach_steering(
                                end_effector,
                                linear_speed=fine_speed,
                                edge_row=edge_row,
                            )
                        )
                        self.set_twist(linear_command, angular)
                else:
                    # Exact depth+alignment is a subset of the predictive
                    # brake condition above.  Keep this defensive branch from
                    # ever closing the gripper in the same tick as a moving
                    # base command if future tolerances are changed.
                    self.set_twist(0.0, 0.0)
                    self._fine_near_since = None
                    self._fine_near_latched = True
                    self._fine_near_latched_at = self.now()
                    self._fine_near_mode = "exact_depth_brake"
                    self._fine_near_tolerance_m = FINE_APPROACH_STOP_TOL_M
                    self._settle_latched_fine_approach(
                        vision_fresh=vision_fresh
                    )

        elif state == CycleState.GRASP:
            if self._uses_tissue_two_hand_grasp:
                if self.mission.consume_entry():
                    self.joint_slew = TISSUE_CLAMP_JOINT_SLEW
                    self._tissue_clamp_ready_since = None
                    left, right = self._tissue_clamp_arms
                    self.tc[5:11] = left
                    self.tc[12:18] = right
                    self.tc[11] = GRIP_CLOSE
                    self.tc[18] = GRIP_CLOSE
                    self.get_logger().info(
                        "TISSUE_TWO_HAND_CLAMP_START "
                        f"half_separation="
                        f"{TISSUE_CLAMP_HALF_SEPARATION_M:.3f}m "
                        "left_and_right_simultaneous=true"
                    )
                self.set_twist(0.0, 0.0)
                self.tc[11] = GRIP_CLOSE
                self.tc[18] = GRIP_CLOSE
                if self._both_arms_at_target(self._tissue_clamp_arms):
                    if self._tissue_clamp_ready_since is None:
                        self._tissue_clamp_ready_since = self.now()
                    elif (
                        self.now() - self._tissue_clamp_ready_since
                        >= TISSUE_CLAMP_SETTLE_SEC
                    ):
                        self.mission.transition(CycleState.LIFT)
                else:
                    self._tissue_clamp_ready_since = None
                if self.mission.state == CycleState.GRASP:
                    self._timed_out("tissue_two_hand_clamp")
            else:
                if self.mission.consume_entry():
                    self.tc[18] = _grip_close_command(self.target_kind)
                    self.get_logger().info(
                        "GRASP_CLOSE_START "
                        f"kind={self.target_kind} "
                        f"command={self.tc[18]:.3f}"
                    )
                self.set_twist(0.0, 0.0)
                self.tc[18] = _grip_close_command(self.target_kind)
                if self.target_kind == "chengzi":
                    grasp_dwell = CHENGZI_GRASP_CLOSE_DWELL_SEC
                else:
                    grasp_dwell = (
                        L3_GRASP_CLOSE_DWELL_SEC
                        if str(self._current_target.get("level", "")) == "L3"
                        else GRASP_CLOSE_DWELL_SEC
                    )
                if self.mission.elapsed >= grasp_dwell:
                    if (
                        self.target_kind == "chengzi"
                        and not self._capture_chengzi_grasp_geometry()
                    ):
                        self._fail("chengzi_grasp_geometry_unavailable")
                    else:
                        self.mission.transition(CycleState.LIFT)

        elif state == CycleState.LIFT:
            if self.mission.consume_entry():
                self.tc[2] = lifted_slide(self._target_slide)
                if self.target_kind == "chengzi":
                    self.joint_slew = CHENGZI_LIFT_JOINT_SLEW
                elif self._uses_tissue_two_hand_grasp:
                    self.joint_slew = TISSUE_INITIAL_LIFT_JOINT_SLEW
            self.set_twist(0.0, 0.0)
            if self._uses_tissue_two_hand_grasp:
                self.tc[11] = GRIP_CLOSE
            self.tc[18] = _grip_close_command(self.target_kind)
            lift_ready = abs(self.slide_meas - self.tc[2]) <= SLIDE_TOL_M
            if self._uses_tissue_two_hand_grasp:
                lift_ready = bool(
                    lift_ready
                    and self._both_arms_at_target(self._tissue_clamp_arms)
                )
            if lift_ready:
                self.mission.transition(CycleState.RETREAT)
            else:
                self._timed_out("lift")

        elif state == CycleState.RETREAT:
            if self.mission.consume_entry():
                # Prepare the loaded-navigation footprint while the Baseline
                # is already backing out.  It is then ready when Nav2 and all
                # three manipulation groups start together after the stop.
                if not self._begin_footprint_change("carry"):
                    self._fail("pick_retreat_footprint_service_unavailable")
                else:
                    self.get_logger().info(
                        "PICK_RETREAT_START "
                        f"target_y={self._pick_retreat_y:.3f} "
                        f"speed={PICK_RETREAT_SPEED_MPS:.2f}m/s "
                        "preparing_carry_footprint=true"
                    )
            if self.mission.state == CycleState.RETREAT:
                footprint_result = self._poll_footprint_change()
                if footprint_result is False:
                    self._fail("pick_retreat_footprint_rejected")
            if self.mission.state == CycleState.RETREAT:
                if self._uses_tissue_two_hand_grasp:
                    self.tc[11] = GRIP_CLOSE
                self.tc[18] = _grip_close_command(self.target_kind)
                yaw_error = (
                    self._grasp_heading - self.base_yaw + math.pi
                ) % (2 * math.pi) - math.pi
                if self.base_xy[1] > self._pick_retreat_y + self.pos_tol:
                    self.set_twist(-PICK_RETREAT_SPEED_MPS, yaw_error)
                else:
                    self.set_twist(0.0, 0.0)
                    self.mission.transition(CycleState.STOP_RETREAT)

        elif state == CycleState.STOP_RETREAT:
            self.mission.consume_entry()
            self.set_twist(0.0, 0.0)
            footprint_result = self._poll_footprint_change()
            if footprint_result is False:
                self._fail("pick_retreat_footprint_rejected")
            elif (
                abs(self.cur_lin) <= BASE_STOP_EPS
                and abs(self.cur_ang) <= BASE_STOP_EPS
                and self._odom_stopped()
                and footprint_result is True
            ):
                if self._uses_tissue_two_hand_grasp:
                    # Raise the wide horizontal shelf grasp in open space with
                    # both arms frozen.  The conservative loaded-arm footprint
                    # already covers this pose, so no loaded roll is needed.
                    self._release_baseline_base()
                    self.mission.transition(
                        CycleState.STOW_PICK_ARMS_TOGETHER,
                        "tissue_body_lift_before_navigation",
                    )
                    return
                # Start the table route, torso lift and both arm motions in the
                # same control tick.  The concurrent updater remains active in
                # WAIT_TABLE_NAV and never owns or suppresses base velocity.
                self._release_baseline_base()
                pose = self._table_approach_pose(self._picked_count)
                slot_index = self._table_drop_index(self._picked_count)
                self.get_logger().info(
                    f"TABLE_DROP_SLOT index={slot_index + 1}/"
                    f"{len(TABLE_DROP_X_OFFSETS_M)} "
                    f"base_pose=({pose[0]:.3f},{pose[1]:.3f},"
                    f"{pose[2]:.3f})"
                )
                self._send_goal(
                    pose, "delivery_table_drop", CycleState.WAIT_TABLE_NAV
                )
                if self.mission.state == CycleState.WAIT_TABLE_NAV:
                    self._start_concurrent_pick_stow()
                    self.get_logger().info(
                        "PICK_TRANSPORT_MOTION_TOGETHER "
                        "nav2=true torso=true left_arm=true right_arm=true"
                    )
            else:
                stop_timeout = max(
                    STOP_SETTLE_SEC + STOP_HANDOFF_TIMEOUT_MARGIN_SEC,
                    COSTMAP_PARAMETER_TIMEOUT_SEC,
                )
                self._timed_out("pick_retreat_stop", stop_timeout)

        elif state == CycleState.STOW_PICK_ARMS_TOGETHER:
            if self._uses_tissue_two_hand_grasp:
                if self.mission.consume_entry():
                    if not self._start_tissue_transport_fold():
                        self._fail("tissue_transport_fold_unreachable")
                self.set_twist(0.0, 0.0)
                if (
                    self.mission.state
                    == CycleState.STOW_PICK_ARMS_TOGETHER
                    and self._update_tissue_motion()
                ):
                    self.mission.transition(CycleState.SET_CARRY_FOOTPRINT)
            else:
                # Compatibility fallback for an externally restored/older
                # state snapshot.  New single-hand cycles bypass this wait
                # and start the updater concurrently from STOP_RETREAT.
                if self.mission.consume_entry():
                    self._start_concurrent_pick_stow()
                self.set_twist(0.0, 0.0)
                if (
                    not self._pick_stow_active
                    and self._pick_transport_is_ready()
                ):
                    self._release_baseline_base()
                    self.mission.transition(CycleState.SET_CARRY_FOOTPRINT)

        elif state == CycleState.SET_CARRY_FOOTPRINT:
            if self.mission.consume_entry():
                # 每件商品各自最多获得一次窄通道重规划机会。
                self._delivery_nav_chassis_retry_used = False
                if self._uses_tissue_two_hand_grasp:
                    if not self._begin_controller_angular_limit(
                        TISSUE_NAV_MAX_ANGULAR_RADPS
                    ):
                        self._fail(
                            "tissue_controller_angular_limit_unavailable"
                        )
                if (
                    self.mission.state == CycleState.SET_CARRY_FOOTPRINT
                    and not self._begin_footprint_change("carry")
                ):
                    self._fail("carry_footprint_service_unavailable")
            if self.mission.state == CycleState.SET_CARRY_FOOTPRINT:
                footprint_result = self._poll_footprint_change()
                angular_result = self._poll_controller_angular_limit()
                if footprint_result is True and angular_result is True:
                    if self._uses_tissue_two_hand_grasp:
                        # Updating FollowPath.wz_max makes MPPI reload its
                        # velocity constraints.  Publish the translational
                        # speed limit only after that reload; doing this in
                        # the opposite order silently restored vx_max and made
                        # the loaded two-arm leg run at ordinary Nav2 speed.
                        self._set_navigation_speed_limit(
                            TISSUE_NAV_SPEED_LIMIT_MPS
                        )
                        self.mission.transition(CycleState.TISSUE_PRETURN)
                    else:
                        self.mission.transition(CycleState.NAV_TABLE)
                elif footprint_result is False:
                    self._fail("carry_footprint_rejected")
                elif angular_result is False:
                    self._fail("tissue_controller_angular_limit_rejected")
                else:
                    self._timed_out(
                        "carry_safety_parameters",
                        COSTMAP_PARAMETER_TIMEOUT_SEC,
                    )

        elif state == CycleState.TISSUE_PRETURN:
            if self.mission.consume_entry():
                current_target = getattr(self, "_current_target", None)
                source_shelf = (
                    current_target.get("shelf")
                    if isinstance(current_target, dict)
                    else None
                )
                if not source_shelf:
                    source_shelf = getattr(self, "_active_shelf", None)
                try:
                    signed_angle = _tissue_preturn_signed_angle_rad(
                        source_shelf
                    )
                except ValueError as exc:
                    self.set_twist(0.0, 0.0)
                    self._fail(f"tissue_preturn_source_invalid: {exc}")
                    return
                self._tissue_preturn_source_shelf = str(
                    source_shelf
                ).strip().upper()
                self._tissue_preturn_angle_rad = signed_angle
                self._tissue_preturn_last_yaw = float(self.base_yaw)
                self._tissue_preturn_accumulated_rad = 0.0
                self._tissue_preturn_target_yaw = wrap_to_pi(
                    self.base_yaw + signed_angle
                )
                self._enable_baseline_base()
                initial_error = signed_angle
                self._tissue_preturn_yaw_error = initial_error
                direction = "left" if signed_angle > 0.0 else "right"
                self.get_logger().info(
                    "TISSUE_PRETURN_START "
                    f"shelf={self._tissue_preturn_source_shelf} "
                    f"direction={direction} "
                    f"relative={abs(math.degrees(signed_angle)):.1f}deg "
                    f"signed_relative={math.degrees(signed_angle):+.1f}deg "
                    f"yaw={self.base_yaw:.3f} "
                    f"target={self._tissue_preturn_target_yaw:.3f} "
                    f"error={initial_error:.3f}rad "
                    f"fixed_speed={TISSUE_PRETURN_MAX_ANGULAR_RADPS:.2f}rad/s"
                )

            target_yaw = self._tissue_preturn_target_yaw
            signed_angle = self._tissue_preturn_angle_rad
            last_yaw = self._tissue_preturn_last_yaw
            if target_yaw is None or signed_angle is None or last_yaw is None:
                self._fail("tissue_preturn_target_missing")
            elif self.mission.state == CycleState.TISSUE_PRETURN:
                # Integrate short per-tick yaw deltas instead of recomputing a
                # shortest absolute error.  This preserves the commanded side
                # across +/-pi and makes B's requested right 180-degree turn
                # unambiguous for the whole motion.
                yaw_delta = wrap_to_pi(float(self.base_yaw) - last_yaw)
                self._tissue_preturn_accumulated_rad += yaw_delta
                self._tissue_preturn_last_yaw = float(self.base_yaw)
                yaw_error = (
                    signed_angle - self._tissue_preturn_accumulated_rad
                )
                self._tissue_preturn_yaw_error = yaw_error
                direction_sign = 1.0 if signed_angle > 0.0 else -1.0
                directed_progress = (
                    direction_sign * self._tissue_preturn_accumulated_rad
                )
                if (
                    abs(signed_angle) - directed_progress
                    <= TISSUE_PRETURN_YAW_TOL_RAD
                ):
                    self.set_twist(0.0, 0.0)
                    self.get_logger().info(
                        "TISSUE_PRETURN_ALIGNED "
                        f"shelf={self._tissue_preturn_source_shelf} "
                        f"turned={math.degrees(self._tissue_preturn_accumulated_rad):+.1f}deg "
                        f"error={yaw_error:.3f}rad; stopping before Nav2"
                    )
                    self.mission.transition(CycleState.STOP_TISSUE_PRETURN)
                else:
                    # The direction comes from the source cabinet and never
                    # flips near the +/-pi boundary or after a small overrun.
                    angular = (
                        direction_sign * TISSUE_PRETURN_MAX_ANGULAR_RADPS
                    )
                    self.set_twist(0.0, angular)
                    self._timed_out(
                        "tissue_preturn", TISSUE_PRETURN_TIMEOUT_SEC
                    )

        elif state == CycleState.STOP_TISSUE_PRETURN:
            self.mission.consume_entry()
            self.set_twist(0.0, 0.0)
            if (
                abs(self.cur_lin) <= BASE_STOP_EPS
                and abs(self.cur_ang) <= BASE_STOP_EPS
                and self._odom_stopped()
            ):
                self._release_baseline_base()
                # Refresh immediately before FollowPath starts as a guard
                # against a late controller reconfiguration or a dropped
                # volatile message while the manual pre-turn owned the base.
                self._set_navigation_speed_limit(TISSUE_NAV_SPEED_LIMIT_MPS)
                self.get_logger().info(
                    "TISSUE_PRETURN_STOPPED nav2_handoff=true "
                    f"nav_linear_limit={TISSUE_NAV_SPEED_LIMIT_MPS:.3f}m/s "
                    f"nav_angular_limit={TISSUE_NAV_MAX_ANGULAR_RADPS:.2f}rad/s"
                )
                # 双臂负载预转并停稳后直接规划到最终桌边放置点，不再经过
                # 中间导流航点，也不再进行途中目标预占。
                self.mission.transition(CycleState.NAV_TABLE)
            else:
                self._timed_out(
                    "tissue_preturn_stop",
                    TISSUE_PRETURN_STOP_TIMEOUT_SEC,
                )

        elif state == CycleState.NAV_TABLE:
            if self.mission.consume_entry():
                # One Nav2 goal goes straight to the final table-drop base pose.
                # The short PLACE_ADVANCE below is an arm-only template motion.
                self._send_delivery_table_goal()

        elif state == CycleState.WAIT_TABLE_NAV:
            pose = self._table_approach_pose(self._picked_count)
            if not self._update_table_nav_yaw_assist(pose):
                self._wait_navigation(
                    CycleState.STOP_TABLE,
                    "delivery_table_drop",
                    pose,
                    TABLE_NEAR_GOAL_POSITION_TOL_M,
                )

        elif state == CycleState.RETRY_TABLE_CHASSIS_FOOTPRINT:
            if self.mission.consume_entry():
                if not self._begin_footprint_change("normal"):
                    self._fail("delivery_chassis_footprint_service_unavailable")
            if self.mission.state == CycleState.RETRY_TABLE_CHASSIS_FOOTPRINT:
                result = self._poll_footprint_change()
                if result is True:
                    self.get_logger().warning(
                        "DELIVERY_CHASSIS_FOOTPRINT_READY retrying_nav_once=true"
                    )
                    self.mission.transition(CycleState.NAV_TABLE)
                elif result is False:
                    self._fail("delivery_chassis_footprint_rejected")
                else:
                    self._timed_out(
                        "delivery_chassis_footprint",
                        COSTMAP_PARAMETER_TIMEOUT_SEC,
                    )

        elif state == CycleState.STOP_TABLE:
            if self.mission.consume_entry():
                self._set_nav_angular_override(None)
                self._table_nav_yaw_assist_active = False
                self._table_nav_yaw_assist_command_radps = None
                self.nav2.cancel()
                self.nav2.stop_robot()
                if self._tissue_nav_speed_limit_active:
                    self._set_navigation_speed_limit(0.0)
                if (
                    self._uses_tissue_two_hand_grasp
                    and not self._begin_controller_angular_limit(
                        NORMAL_MPPI_MAX_ANGULAR_RADPS
                    )
                ):
                    self._fail(
                        "tissue_controller_angular_restore_unavailable"
                    )
                self.get_logger().info(
                    f"TABLE_STOP_SETTLING dwell={TABLE_STOP_SETTLE_SEC:.2f}s "
                    "waiting_for_loaded_stow_if_needed=true"
                )
            angular_restore_result = (
                self._poll_controller_angular_limit()
                if self.mission.state == CycleState.STOP_TABLE
                else None
            )
            table_stopped = (
                self.mission.elapsed >= TABLE_STOP_SETTLE_SEC
                and self._odom_stopped()
            )
            if (
                table_stopped
                and not self.nav2.is_active
                and self._pick_transport_is_ready()
                and angular_restore_result is True
            ):
                self.mission.transition(CycleState.RESTORE_TABLE_FOOTPRINT)
            elif angular_restore_result is False:
                self._fail("tissue_controller_angular_restore_rejected")
            elif angular_restore_result is None:
                self._timed_out(
                    "tissue_controller_angular_restore",
                    COSTMAP_PARAMETER_TIMEOUT_SEC,
                )
            elif self.nav2.is_active:
                self._timed_out(
                    "delivery_table_stop",
                    COSTMAP_PARAMETER_TIMEOUT_SEC,
                )

        elif state == CycleState.RESTORE_TABLE_FOOTPRINT:
            if self.mission.consume_entry() and not self._begin_footprint_change(
                "normal"
            ):
                self._fail("table_footprint_service_unavailable")
            if self.mission.state == CycleState.RESTORE_TABLE_FOOTPRINT:
                result = self._poll_footprint_change()
                if result is True:
                    self.mission.transition(CycleState.RESTORE_TABLE_CARRY)
                elif result is False:
                    self._fail("table_footprint_rejected")
                else:
                    self._timed_out(
                        "table_footprint", COSTMAP_PARAMETER_TIMEOUT_SEC
                    )

        elif state == CycleState.RESTORE_TABLE_CARRY:
            if self._uses_tissue_two_hand_grasp:
                if self.mission.consume_entry():
                    if not self._start_tissue_table_unfold():
                        self._fail("tissue_table_unfold_unreachable")
                self.set_twist(0.0, 0.0)
                if (
                    self.mission.state == CycleState.RESTORE_TABLE_CARRY
                    and self._update_tissue_motion()
                ):
                    # The rigid two-hand pose has no independent arm
                    # advance.  Start lowering on the next control tick instead
                    # of inserting a fake PLACE_ADVANCE state and 0.35 s pause.
                    self.mission.transition(
                        CycleState.PLACE_LOWER,
                        "tissue_continuous_hold_to_lower",
                    )
            else:
                if self.mission.consume_entry():
                    # Expand the right elbow only after Nav2 has reached the
                    # table and the chassis is stationary.  Placement then
                    # starts from the exact handoff pose used for the drop
                    # slots.
                    self.tc[12:18] = FIXED_CARRY_ARM
                    self.tc[18] = _grip_close_command(self.target_kind)
                    self.joint_slew = PLACE_UNFOLD_JOINT_SLEW_OVERRIDES.get(
                        self.target_kind,
                        PLACE_UNFOLD_GENTLE_JOINT_SLEW,
                    )
                    self.get_logger().info(
                        "RESTORE_TABLE_CARRY unfolding stationary table handoff pose "
                        f"slew={self.joint_slew:.2f}"
                    )
                self.set_twist(0.0, 0.0)
                ready = (
                    self._right_arm_at_target(FIXED_CARRY_ARM)
                    and self.mission.elapsed >= ARM_SETTLE_SEC
                )
                if ready:
                    self.mission.transition(CycleState.PLACE_ADVANCE)
                else:
                    self._timed_out("restore_table_carry")

        elif state == CycleState.PLACE_ADVANCE:
            if self._uses_tissue_two_hand_grasp:
                if self.mission.consume_entry():
                    self._reset_place_settle()
                    self.tc[11] = GRIP_CLOSE
                    self.tc[18] = GRIP_CLOSE
                    self.get_logger().info(
                        "TISSUE_PLACE_HORIZONTAL_READY "
                        "arm_advance=0.000m; table_pose_supplies_depth"
                    )
                ready = self._both_arms_at_target(
                    self._tissue_table_arms
                )
                if self._place_target_settled(
                    ready,
                    PLACE_ADVANCE_TO_LOWER_SETTLE_SEC,
                    "TISSUE_PLACE_HORIZONTAL",
                ):
                    self.mission.transition(CycleState.PLACE_LOWER)
                elif self._place_ready_since is None:
                    self._timed_out("tissue_place_horizontal")
            else:
                if self.mission.consume_entry():
                    self.joint_slew = PLACE_ADVANCE_JOINT_SLEW_OVERRIDES.get(
                        self.target_kind,
                        PLACE_ADVANCE_GENTLE_JOINT_SLEW,
                    )
                    self._reset_place_settle()
                    self.tc[18] = _grip_close_command(self.target_kind)
                    if not self._command_place_advance(
                        table_place_forward_distances(
                            self._table_drop_index(self._picked_count),
                            self.target_kind,
                        )
                    ):
                        self._fail("place_advance_unreachable")
                if self.mission.state == CycleState.PLACE_ADVANCE:
                    ready = self._right_arm_at_target(
                        self._place_advanced_arm
                    )
                    if self._place_target_settled(
                        ready,
                        PLACE_ADVANCE_TO_LOWER_SETTLE_SEC,
                        "PLACE_ADVANCE",
                    ):
                        if (
                            self.target_kind == "chengzi"
                            and not self._calibrate_chengzi_table_drop_slide()
                        ):
                            self._fail("chengzi_table_drop_calibration_failed")
                        else:
                            self.mission.transition(CycleState.PLACE_LOWER)
                    else:
                        self._timed_out("place_advance")

        elif state == CycleState.PLACE_LOWER:
            if self.mission.consume_entry():
                self._reset_place_settle()
                if self._uses_tissue_two_hand_grasp:
                    if (
                        self._tissue_carry_center_z_m is None
                        or self._tissue_transport_slide_m is None
                    ):
                        self._fail("tissue_place_height_unavailable")
                    else:
                        desired_center_z = (
                            TABLE_TOP_Z_M
                            + self.target_geometry.half_height_m
                            + TISSUE_TABLE_BOTTOM_CLEARANCE_M
                        )
                        self._tissue_table_drop_slide_m = float(
                            np.clip(
                                self._tissue_carry_center_z_m
                                + self._tissue_transport_slide_m
                                - desired_center_z,
                                SLIDE_MIN_M,
                                0.87,
                            )
                        )
                        self._table_drop_slide_m = (
                            self._tissue_table_drop_slide_m
                        )
                        self.tc[2] = self._table_drop_slide_m
                        self.tc[11] = GRIP_CLOSE
                        self.tc[18] = GRIP_CLOSE
                        self.joint_slew = TISSUE_LOADED_JOINT_SLEW
                else:
                    self.joint_slew = PLACE_LOWER_JOINT_SLEW_OVERRIDES.get(
                        self.target_kind,
                        PLACE_LOWER_GENTLE_JOINT_SLEW,
                    )
                    self.tc[2] = self._table_drop_slide_m
                    self.tc[18] = _grip_close_command(self.target_kind)
                self.get_logger().info(
                    f"PLACE_LOWER_DYNAMIC slide={self._table_drop_slide_m:.3f} "
                    f"kind={self.target_kind} "
                    f"slew={self.joint_slew:.2f}"
                )
            # PLACE_ADVANCE already verified and settled the arm pose.  During
            # vertical lowering the simulated joint feedback can briefly cross
            # the arm tolerance as the held bottle contacts the table, which
            # used to reset the settle timer indefinitely even though both the
            # slide and arm ultimately ended exactly on target.
            ready = (
                abs(self.slide_meas - self._table_drop_slide_m) <= SLIDE_TOL_M
            )
            if self.target_kind == "chengzi":
                ready = self._update_chengzi_table_lower()
            if self._uses_tissue_two_hand_grasp:
                self.tc[11] = GRIP_CLOSE
                self.tc[18] = GRIP_CLOSE
                ready = bool(
                    ready
                    and self._both_arms_at_target(
                        self._tissue_table_arms
                    )
                )
            lower_settle = (
                TISSUE_PLACE_LOWER_SETTLE_SEC
                if self._uses_tissue_two_hand_grasp
                else CHENGZI_PLACE_LOWER_SETTLE_SEC
                if self.target_kind == "chengzi"
                else PLACE_LOWER_SETTLE_SEC
            )
            if self._place_target_settled(
                ready, lower_settle, "PLACE_LOWER"
            ):
                self.mission.transition(CycleState.PLACE_RELEASE)
            elif (
                self._uses_tissue_two_hand_grasp
                and self.mission.elapsed
                >= TISSUE_PLACE_LOWER_FORCE_RELEASE_SEC
            ):
                # 接触桌面后，关节/滑轨反馈可能长期停在容差边缘。继续保持
                # 下放和夹紧命令，但不能让反馈门槛阻断必须执行的分臂、抬升
                # 与后退链路。
                self.get_logger().warning(
                    "TISSUE_PLACE_LOWER_FORCE_RELEASE "
                    f"elapsed={self.mission.elapsed:.2f}s "
                    f"slide_error="
                    f"{abs(self.slide_meas - self._table_drop_slide_m):.3f}m "
                    "continuing_mandatory_release_chain=true"
                )
                self.mission.transition(
                    CycleState.PLACE_RELEASE,
                    "two_hand_table_contact_feedback_fallback",
                )
            elif self.target_kind == "chengzi":
                # A timer that is repeatedly reset by motion must not leave
                # this stage hanging indefinitely. Timeout never opens jaws.
                if self._timed_out("place_lower", CHENGZI_PLACE_LOWER_TIMEOUT_SEC):
                    self.tc[2] = float(self.action[2])
            elif self._place_ready_since is None:
                self._timed_out("place_lower")

        elif state == CycleState.PLACE_RELEASE:
            if (
                self._uses_tissue_two_hand_grasp
                and self.mission.consume_entry()
            ):
                self.joint_slew = TISSUE_RELEASE_JOINT_SLEW
                self._reset_place_settle()
                self._release_clearance_slide_m = None
                center_z = (
                    float(self._tissue_carry_center_z_m)
                    + float(self._tissue_transport_slide_m)
                    - float(self._table_drop_slide_m)
                )
                release = self._solve_tissue_arm_targets(
                    slide=float(self._table_drop_slide_m),
                    center_x=TISSUE_HAND_CENTER_X_M,
                    center_z=center_z,
                    half_separation=TISSUE_RELEASE_HALF_SEPARATION_M,
                    roll_rad=0.0,
                    reference=self._tissue_table_arms,
                )
                if release is None:
                    release = self._tissue_table_arms
                    if release is None:
                        release = (
                            np.asarray(self.tc[5:11], dtype=float).copy(),
                            np.asarray(self.tc[12:18], dtype=float).copy(),
                        )
                    self.get_logger().warning(
                        "TISSUE_PLACE_RELEASE_IK_FALLBACK "
                        "holding_current_arm_pose=true "
                        "continuing_mandatory_release_chain=true"
                    )
                self._tissue_release_arms = release
                # 商品已经由桌面承托。保持两只夹爪闭合，只让完整手臂向外
                # 分离来解除侧向夹持；这样爪指不会在纸巾表面展开或勾住包装。
                # 即使外移 IK 只能沿用当前落桌姿态，也继续执行抬升、后退链路。
                self.tc[5:11] = self._tissue_release_arms[0]
                self.tc[12:18] = self._tissue_release_arms[1]
                self.tc[11] = GRIP_CLOSE
                self.tc[18] = GRIP_CLOSE
                self.get_logger().info(
                    "TISSUE_PLACE_RELEASE_START "
                    f"half_separation="
                    f"{TISSUE_RELEASE_HALF_SEPARATION_M:.3f}m "
                    "both_grippers_held_closed=true "
                    "release_by_arm_separation=true"
                )
            elif (
                not self._uses_tissue_two_hand_grasp
                and self.mission.consume_entry()
            ):
                # Recheck orange feedback on the actual release tick. Do not
                # open if motion/stale feedback invalidated the lowering gate.
                if self.target_kind == "chengzi":
                    if not self._update_chengzi_table_lower():
                        self.tc[2] = float(self.action[2])
                        self._fail("chengzi_release_height_guard")
                        return
                    self._log_chengzi_release_height()
                # 单臂商品在下降已经停止且高度门槛满足后快速松爪；
                # 双臂夹持流程使用上方独立的分臂速度，不经过这里。
                self.joint_slew = PLACE_RELEASE_JOINT_SLEW_OVERRIDES.get(
                    self.target_kind,
                    PLACE_RELEASE_JOINT_SLEW,
                )
                self._reset_place_settle()
                self._release_clearance_slide_m = None
                self.tc[18] = GRIP_OPEN
            if self._uses_tissue_two_hand_grasp:
                if self.mission.state != CycleState.PLACE_RELEASE:
                    return
                self.tc[11] = GRIP_CLOSE
                self.tc[18] = GRIP_CLOSE
                self.tc[5:11] = self._tissue_release_arms[0]
                self.tc[12:18] = self._tissue_release_arms[1]
                left_pose, right_pose = self._tissue_hand_poses()
                actual_separation = float(
                    np.linalg.norm(
                        left_pose[:3, 3] - right_pose[:3, 3]
                    )
                )
                required_separation = TISSUE_RELEASE_MIN_SEPARATION_M
                release_ready = self._tissue_release_actuators_ready(
                    actual_separation
                )
                # Do not retreat while the two hands still overlap the box.
                # Once their measured separation is clear, raise the torso
                # with the separated arm pose held, then permit base motion.
                if self._place_target_settled(
                    release_ready,
                    TISSUE_RELEASE_SETTLE_SEC,
                    "TISSUE_PLACE_RELEASE",
                ):
                    self.get_logger().info(
                        "TISSUE_PLACE_RELEASE_READY_FOR_LIFT "
                        f"elapsed={self.mission.elapsed:.2f}s "
                        f"separation={actual_separation:.3f}m "
                        f"minimum={required_separation:.3f}m "
                        "both_grippers_held_closed=true"
                    )
                    self._mark_delivery_attempt()
                    self.mission.transition(
                        CycleState.PLACE_RELEASE_LIFT,
                        "tissue_released_raise_before_retreat",
                    )
                elif (
                    self.mission.elapsed >= TISSUE_RELEASE_FORCE_LIFT_SEC
                ):
                    # Do not turn a completed table release into a terminal
                    # mission failure merely because contact-loaded feedback
                    # did not cross an ideal clearance threshold.  Continue
                    # the mandatory vertical lift while holding the same
                    # outward arm target, then perform the direct base exit.
                    self.get_logger().warning(
                        "TISSUE_PLACE_RELEASE_FORCE_LIFT "
                        f"elapsed={self.mission.elapsed:.2f}s "
                        f"separation={actual_separation:.3f}m "
                        f"minimum={required_separation:.3f}m "
                        "both_grippers_held_closed=true "
                        "continuing_mandatory_lift=true"
                    )
                    self._mark_delivery_attempt()
                    self.mission.transition(
                        CycleState.PLACE_RELEASE_LIFT,
                        "tissue_release_feedback_fallback_raise_before_retreat",
                    )
                return
            broad_release = self.target_kind in BROAD_RELEASE_KINDS
            release_ready = (
                self._broad_release_gripper_is_open
                if broad_release
                else self._right_gripper_is_open
            )
            if self.target_kind == "chengzi":
                # A held sphere can prop the jaw near its open reading even
                # before the opening command has ramped. Do not start the lift
                # timer from that misleading feedback.
                release_ready = bool(
                    release_ready and abs(float(self.action[18]) - GRIP_OPEN) <= 0.01
                )
            release_settle = (
                BROAD_RELEASE_SETTLE_SEC
                if broad_release
                else PLACE_RELEASE_SETTLE_SEC
            )
            if self._place_target_settled(
                release_ready, release_settle, "PLACE_RELEASE"
            ):
                # Keep the released arm pose unchanged.  Every product
                # except heweidao first gets a torso-only vertical separation,
                # then the base backs away.  This prevents an open finger from
                # dragging a released item toward the table edge.
                self._mark_delivery_attempt()
                if self._post_release_lift_required():
                    self.mission.transition(
                        CycleState.PLACE_RELEASE_LIFT,
                        "released_raise_clear_before_retreat",
                    )
                else:
                    self.mission.transition(
                        CycleState.RETREAT_TABLE,
                        "released_direct_base_retreat",
                    )
            else:
                self._timed_out("place_release")

        elif state == CycleState.PLACE_RELEASE_LIFT:
            # Heweidao is explicitly a release-then-retreat product.  Use the
            # pending delivery record (which survives _mark_delivery_attempt)
            # as the authority and recover straight to retreat if stale state
            # ever routes it here; do not issue even one torso-lift command.
            if (
                not self._uses_tissue_two_hand_grasp
                and not self._post_release_lift_required()
            ):
                self._hold_direct_retreat_release_pose()
                self.mission.transition(
                    CycleState.RETREAT_TABLE,
                    "lift_exempt_product_direct_retreat_guard",
                )
                return
            if self.mission.consume_entry():
                if (
                    self._uses_tissue_two_hand_grasp
                    and self._tissue_release_arms is None
                ):
                    self._tissue_release_arms = self._tissue_table_arms
                    if self._tissue_release_arms is None:
                        self._tissue_release_arms = (
                            np.asarray(self.tc[5:11], dtype=float).copy(),
                            np.asarray(self.tc[12:18], dtype=float).copy(),
                        )
                    self.get_logger().warning(
                        "TISSUE_RELEASE_LIFT_POSE_FALLBACK "
                        "holding_current_arm_pose=true "
                        "continuing_mandatory_lift=true"
                    )
                self.joint_slew = (
                    TISSUE_LOADED_JOINT_SLEW
                    if self._uses_tissue_two_hand_grasp
                    else PLACE_LOWER_GENTLE_JOINT_SLEW
                )
                self._reset_place_settle()
                self._release_clearance_slide_m = max(
                    SLIDE_MIN_M,
                    self._table_drop_slide_m
                    - POST_RELEASE_CLEARANCE_LIFT_M,
                )
                self.tc[2] = self._release_clearance_slide_m
                if self._uses_tissue_two_hand_grasp:
                    self.tc[5:11] = self._tissue_release_arms[0]
                    self.tc[12:18] = self._tissue_release_arms[1]
                    self.tc[11] = GRIP_CLOSE
                    self.tc[18] = GRIP_CLOSE
                else:
                    self.tc[18] = GRIP_OPEN
                self.get_logger().info(
                    "POST_RELEASE_CLEARANCE_LIFT "
                    f"kind={self.target_kind} "
                    f"slide={self._release_clearance_slide_m:.3f} "
                    f"lift={POST_RELEASE_CLEARANCE_LIFT_M:.3f}m; "
                    "arm fixed and base stopped"
                )
            self.set_twist(0.0, 0.0)
            if self._uses_tissue_two_hand_grasp:
                self.tc[5:11] = self._tissue_release_arms[0]
                self.tc[12:18] = self._tissue_release_arms[1]
                self.tc[11] = GRIP_CLOSE
                self.tc[18] = GRIP_CLOSE
                # PLACE_RELEASE has already commanded the outward arm target.
                # Do not reintroduce a gripper-feedback gate during the
                # mandatory vertical clearance lift.
                release_actuator_ready = True
            else:
                self.tc[18] = GRIP_OPEN
                release_actuator_ready = self._right_gripper_is_open
            release_lift_ready = (
                self._release_clearance_slide_m is not None
                and abs(
                    self.slide_meas - self._release_clearance_slide_m
                ) <= POST_RELEASE_SLIDE_TOL_M
                and release_actuator_ready
            )
            if self._place_target_settled(
                release_lift_ready,
                POST_RELEASE_CLEARANCE_SETTLE_SEC,
                "PLACE_RELEASE_LIFT",
            ):
                self.mission.transition(
                    CycleState.RETREAT_TABLE,
                    "released_item_clear_direct_base_retreat",
                )
            elif (
                self._uses_tissue_two_hand_grasp
                and self.mission.elapsed
                >= TISSUE_RELEASE_LIFT_FORCE_RETREAT_SEC
            ):
                self.get_logger().warning(
                    "TISSUE_RELEASE_LIFT_FORCE_RETREAT "
                    f"elapsed={self.mission.elapsed:.2f}s "
                    f"slide_error="
                    f"{abs(self.slide_meas - self._release_clearance_slide_m):.3f}m "
                    f"release_actuator_ready={release_actuator_ready} "
                    "continuing_mandatory_retreat=true"
                )
                self.mission.transition(
                    CycleState.RETREAT_TABLE,
                    "two_hand_lift_feedback_fallback_direct_retreat",
                )
            elif self._place_ready_since is None:
                self._timed_out("place_release_lift")

        elif state == CycleState.RETREAT_TABLE:
            table_retreat_speed = (
                TISSUE_TABLE_RETREAT_SPEED_MPS
                if self._uses_tissue_two_hand_grasp
                else TABLE_RETREAT_SPEED_MPS
            )
            # For heweidao, hold the released hand and torso at the exact table
            # pose throughout the straight retreat.  The normal return-stow
            # lift starts only after the base has cleared the table.
            self._hold_direct_retreat_release_pose()
            if self.mission.consume_entry():
                # MPPI is configured for forward-only motion, so a waypoint
                # behind the robot makes it turn and loop. Keep the released
                # arm pose fixed over the table and back the whole base
                # straight out.  Prepare the carry footprint during this
                # straight retreat so it cannot add a stationary pause when
                # Nav2, torso lift, and arm lowering start together afterward.
                footprint_started = self._begin_footprint_change("carry")
                if not footprint_started and not self._uses_tissue_two_hand_grasp:
                    self._fail("return_footprint_service_unavailable")
                else:
                    if not footprint_started:
                        self.get_logger().warning(
                            "TISSUE_RETREAT_FOOTPRINT_START_FALLBACK "
                            "continuing_mandatory_retreat=true"
                        )
                    self._enable_baseline_base()
                    self._table_retreat_start_xy = self.base_xy.copy()
                    self._table_retreat_heading = self.base_yaw
                    self._table_exit_reached = False
                    # This is an explicit table-clearance velocity command,
                    # not a Nav2 goal. The item has already been released and
                    # the required body clearance completed, so command the
                    # product-specific reverse speed immediately instead of
                    # spending most of the short 25 cm move in acceleration.
                    self.des_lin = self.cur_lin = -table_retreat_speed
                    self.des_ang = self.cur_ang = 0.0
                    self.tc[0] = -table_retreat_speed
                    self.tc[1] = 0.0
                    self.get_logger().info(
                        f"TABLE_RETREAT_START distance={TABLE_RETREAT_DISTANCE_M:.2f}m "
                        f"speed={table_retreat_speed:.2f}m/s "
                        f"heading={self._table_retreat_heading:.3f}; "
                        "controller=direct_cmd_vel ramp=false "
                        "preparing_return_footprint=true"
                    )
            if self.mission.state == CycleState.RETREAT_TABLE:
                # Keep the already separated, closed hands at their release
                # targets while the base leaves the table.
                if (
                    self._uses_tissue_two_hand_grasp
                    and self._tissue_release_arms is not None
                ):
                    self.joint_slew = TISSUE_RELEASE_JOINT_SLEW
                    self.tc[5:11] = self._tissue_release_arms[0]
                    self.tc[12:18] = self._tissue_release_arms[1]
                    self.tc[11] = GRIP_CLOSE
                    self.tc[18] = GRIP_CLOSE
                footprint_result = self._poll_footprint_change()
                if footprint_result is False:
                    if self._uses_tissue_two_hand_grasp:
                        footprint_result = True
                        self.get_logger().warning(
                            "TISSUE_RETREAT_FOOTPRINT_REJECTED_FALLBACK "
                            "continuing_mandatory_retreat=true"
                        )
                    else:
                        self._fail("return_footprint_rejected")
                elif (
                    footprint_result is None
                    and self.mission.elapsed >= COSTMAP_PARAMETER_TIMEOUT_SEC
                ):
                    if self._uses_tissue_two_hand_grasp:
                        self._footprint_pending_mode = None
                        self._footprint_futures = []
                        footprint_result = True
                        self.get_logger().warning(
                            "TISSUE_RETREAT_FOOTPRINT_TIMEOUT_FALLBACK "
                            "continuing_mandatory_retreat=true"
                        )
                    else:
                        self._fail("return_footprint_timeout")

            if self.mission.state == CycleState.RETREAT_TABLE:
                travel = float(
                    np.linalg.norm(self.base_xy - self._table_retreat_start_xy)
                )
                travel_ready = travel >= TABLE_RETREAT_DISTANCE_M
                fold_clearance_ready = (
                    float(self.base_xy[1]) >= TABLE_RETURN_FOLD_MIN_Y_M
                )
                if not (travel_ready and fold_clearance_ready):
                    yaw_error = (
                        self._table_retreat_heading - self.base_yaw + math.pi
                    ) % (2.0 * math.pi) - math.pi
                    self.set_twist(-table_retreat_speed, yaw_error)
                    if self.mission.elapsed >= TABLE_RETREAT_TIMEOUT_SEC:
                        if self._uses_tissue_two_hand_grasp:
                            self.set_twist(0.0, 0.0)
                            self._table_exit_reached = True
                            self.get_logger().warning(
                                "TISSUE_TABLE_RETREAT_ODOMETRY_FALLBACK "
                                f"elapsed={self.mission.elapsed:.2f}s "
                                f"travel={travel:.3f}m "
                                "mandatory_reverse_command_completed=true"
                            )
                            self.mission.transition(
                                CycleState.STOP_TABLE_RETREAT,
                                "two_hand_retreat_odometry_timeout_stop",
                            )
                        else:
                            self._fail("table_retreat_timeout")
                else:
                    self.set_twist(0.0, 0.0)
                    self._table_exit_reached = True
                    self.get_logger().info(
                        f"TABLE_RETREAT_REACHED travel={travel:.3f}m "
                        f"y={self.base_xy[1]:.3f}/"
                        f"{TABLE_RETURN_FOLD_MIN_Y_M:.3f}"
                    )
                    final_delivery = self._pending_delivery_is_final()
                    if not final_delivery and footprint_result is True:
                        # End the direct command and create the next Nav2 goal
                        # in this control tick. We intentionally do not wait
                        # for the separate odometry-still state: Nav2 consumes
                        # real odometry and continues smoothly from here.
                        self.des_lin = self.des_ang = 0.0
                        self.cur_lin = self.cur_ang = 0.0
                        self.tc[0] = self.tc[1] = 0.0
                        self._start_navigation_after_table_retreat()
                    else:
                        # The final delivery must remain stationary by design.
                        # A rare still-pending footprint service also falls
                        # back here until ownership can be handed over safely.
                        self.mission.transition(CycleState.STOP_TABLE_RETREAT)

        elif state == CycleState.STOP_TABLE_RETREAT:
            self.mission.consume_entry()
            self.set_twist(0.0, 0.0)
            footprint_result = self._poll_footprint_change()
            retreat_stopped = bool(
                abs(self.cur_lin) <= BASE_STOP_EPS
                and abs(self.cur_ang) <= BASE_STOP_EPS
                and self._odom_stopped()
            )
            final_delivery = self._pending_delivery_is_final()
            table_exit_complete = bool(
                self._table_exit_reached and retreat_stopped
            )
            # 比赛总时间止于最后一件商品完成释放和抬身，并实际达到桌边
            # 退出位置且由里程计确认停止。后续原地收臂属于安全复位，
            # 不计入任务用时。
            if table_exit_complete and final_delivery:
                self._freeze_mission_total_time()
            if footprint_result is False:
                self._fail("return_footprint_rejected")
            elif (
                table_exit_complete
                and footprint_result is True
            ):
                if final_delivery:
                    self._release_baseline_base()
                    # 最后一件不再发送 E_SCAN_POSE 或其他 Nav2 目标。保持当前
                    # 底盘位置，只恢复普通轮廓并完成躯干/双臂安全收回。
                    if not self._begin_footprint_change("normal"):
                        self._fail(
                            "final_normal_footprint_service_unavailable"
                        )
                    else:
                        self._start_concurrent_return_stow(
                            navigation_active=False
                        )
                        self.mission.transition(
                            CycleState.FINAL_STATIONARY_RESTORE,
                            "last_item_exit_complete_no_return_navigation",
                        )
                else:
                    self._start_navigation_after_table_retreat()
            else:
                stop_timeout = max(
                    STOP_SETTLE_SEC + STOP_HANDOFF_TIMEOUT_MARGIN_SEC,
                    COSTMAP_PARAMETER_TIMEOUT_SEC,
                )
                self._timed_out("table_retreat_stop", stop_timeout)

        elif state == CycleState.FINAL_STATIONARY_RESTORE:
            if self.mission.consume_entry():
                self.nav2.cancel()
                self.nav2.stop_robot()
                self.get_logger().info(
                    "FINAL_STATIONARY_RESTORE_START "
                    "navigation=false hold_current_base_pose=true"
                )
            footprint_result = self._poll_footprint_change()
            if footprint_result is False:
                self._fail("final_normal_footprint_rejected")
            elif (
                footprint_result is True
                and self._return_arms_stowed
                and not self._return_stow_active
            ):
                if not self._confirm_final_delivery_at_table():
                    self._fail("final_delivery_state_invalid")
                else:
                    self.get_logger().info(
                        "FINAL_STATIONARY_RESTORE_COMPLETE "
                        "navigation=false robot_stays_at_table_exit"
                    )
                    self.mission.transition(
                        CycleState.DONE,
                        "final_item_complete_stationary_at_table_exit",
                    )
            elif (
                footprint_result is None
                and self.mission.elapsed >= COSTMAP_PARAMETER_TIMEOUT_SEC
            ):
                self._fail("final_normal_footprint_timeout")

        elif state == CycleState.RESTORE_RETURN_FOOTPRINT:
            if self.mission.consume_entry() and not self._begin_footprint_change(
                "normal"
            ):
                self._fail("return_normal_footprint_service_unavailable")
            if self.mission.state == CycleState.RESTORE_RETURN_FOOTPRINT:
                result = self._poll_footprint_change()
                if result is True:
                    # Start vision and the two-arm waiting-pose restore in one
                    # state instead of serially spending another arm cycle.
                    self.mission.transition(
                        CycleState.SEARCH_E, "observe_while_restoring_arms"
                    )
                elif result is False:
                    self._fail("return_normal_footprint_rejected")
                else:
                    self._timed_out(
                        "return_normal_footprint",
                        COSTMAP_PARAMETER_TIMEOUT_SEC,
                    )

        elif state in {CycleState.DONE, CycleState.STOPPED, CycleState.FAILED}:
            if self.mission.consume_entry():
                self._stop_motion()
                self.get_logger().info(
                    f"{self._mission_event_prefix}_CYCLE_{state.name} "
                    f"picked={self._picked_count}"
                )

        if self.base_control_enabled:
            self.ramp_twist()
        if self.manipulation_enabled:
            self.smooth_step()
        self._publish_outputs()

        if self.now() - self._last_state_log >= 1.0:  # 状态日志每秒最多一条
            self.get_logger().info(
                f"e_{self.target_kind}_phase={self.mission.state.name.lower()} "
                f"picked={self._picked_count} "
                f"delivery_attempts={self._delivery_attempt_count} "
                f"queue={list(self._target_queue)} "
                f"base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={self.base_yaw:.2f} slide={self.slide_meas:.3f} "
                f"footprint={self._footprint_mode}"
            )
            self._last_state_log = self.now()


__all__ = ["EProductCycleStateMachineMixin"]
