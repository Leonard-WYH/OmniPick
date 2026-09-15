"""Mission-status serialization kept outside the control loop."""

from __future__ import annotations

import json
import math
import numpy as np

from .baseline_grasp_controller import GRIP_CLOSE, GRIP_OPEN
from .nav2_manipulation_client import LEFT_ARM_STOW_WAYPOINTS
from .navigation.sorting_geometry import (
    CHENGZI_TABLE_RELEASE_CLEARANCE_M,
    TABLE_APPROACH_X_NEGATIVE_SHIFT_M,
    TABLE_DROP_X_OFFSETS_M,
    grasp_height_offset_for_product,
)
from std_msgs.msg import String

from .sorting_config import (
    ADE_OBSERVATION_HANDOFF_POSITION_TOL_M,
    ADE_OBSERVATION_HANDOFF_YAW_TOL_RAD,
    BC_RANDOM_PICK_PREDEPLOY_DISTANCE_M,
    BC_RANDOM_ROLLING_HANDOFF_DISTANCE_M,
    CHENGZI_DEPLOY_SLIDE_TOL_M,
    CHENGZI_PLACE_ADVANCE_JOINT_SLEW,
    CHENGZI_PLACE_LOWER_JOINT_SLEW,
    CHENGZI_PLACE_UNFOLD_JOINT_SLEW,
    COMPACT_CARRY_SAFE_SETTLE_SEC,
    COUPLED_ALIGNMENT_FINAL_TOL_M,
    COUPLED_ALIGNMENT_KINDS,
    DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS,
    DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS,
    DIRECT_FINE_APPROACH_MAX_ANGULAR,
    DIRECT_FINE_APPROACH_TAPER_END_M,
    DIRECT_FINE_APPROACH_TAPER_START_M,
    DIRECT_POSE_GUIDANCE_LATERAL_KP,
    EDGE_FINE_APPROACH_MAX_ANGULAR,
    EDGE_GRASP_LATERAL_TOL_M,
    FAST_ARM_STAGE_DWELL_SEC,
    FINE_APPROACH_NEAR_SETTLE_SEC,
    FINE_APPROACH_TIME_ABORTS_ENABLED,
    FINE_VISION_LOSS_TIMEOUT_SEC,
    FIRST_E_DIRECT_CRUISE_SPEED_MPS,
    FIRST_PICK_PREDEPLOY_DISTANCE_M,
    IMAGE_SERVO_MAX_AGE_SEC,
    IMAGE_SERVO_PIXEL_DEADBAND_PX,
    IMAGE_SERVO_TERMINAL_LOCK_TOL_PX,
    LEFT_ARM_SCAN_WAYPOINTS,
    LIVE_LATERAL_MAX_SLOT_OFFSET_M,
    LIVE_TARGET_MAX_AGE_SEC,
    LIVE_TRACK_MIN_SAMPLES,
    NAV_NEAR_GOAL_SETTLE_SEC,
    OBSERVATION_HEAD_PITCH,
    PICK_DEPLOY_BODY_LOWER_JOINT_SLEW,
    PICK_DEPLOY_FAST_JOINT_SLEW,
    PICK_FINE_HANDOFF_MAX_LINEAR_MPS,
    PICK_RETREAT_SPEED_MPS,
    PICK_TRACK_HEAD_PITCH,
    PICK_TRACK_HEAD_YAW_LIMIT_RAD,
    PINGGUO_PLACE_ADVANCE_JOINT_SLEW,
    PINGGUO_PLACE_LOWER_JOINT_SLEW,
    PINGGUO_PLACE_UNFOLD_JOINT_SLEW,
    PLACE_ADVANCE_GENTLE_JOINT_SLEW,
    PLACE_ADVANCE_JOINT_SLEW_OVERRIDES,
    PLACE_ADVANCE_TO_LOWER_SETTLE_SEC,
    PLACE_LOWER_GENTLE_JOINT_SLEW,
    PLACE_LOWER_JOINT_SLEW_OVERRIDES,
    PLACE_UNFOLD_GENTLE_JOINT_SLEW,
    PLACE_UNFOLD_JOINT_SLEW_OVERRIDES,
    POST_RELEASE_CLEARANCE_LIFT_M,
    POST_RELEASE_SLIDE_TOL_M,
    PRECISION_SLOT_ANCHORED_KINDS,
    RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD,
    RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS,
    RANDOM_ROLLING_HANDOFF_YAW_TOL_RAD,
    RETURN_STOW_FAST_JOINT_SLEW,
    RIGHT_ARM_DOWN_WAYPOINTS,
    SANMINGZHI_FINE_APPROACH_MAX_ANGULAR,
    SANMINGZHI_INSERTION_REDUCTION_M,
    SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD,
    SEARCH_MIN_DETECTION_FRAMES,
    SEARCH_RESTORE_FAST_JOINT_SLEW,
    SLIDE_TOL_M,
    STATIONARY_PICK_PREDEPLOY_DISTANCE_M,
    TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD,
    TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS,
    TABLE_NAV_YAW_ASSIST_START_DISTANCE_M,
    TABLE_RETREAT_SPEED_MPS,
    TABLE_RETURN_FOLD_MIN_Y_M,
    TABLE_STOP_SETTLE_SEC,
    TISSUE_CLAMP_HALF_SEPARATION_M,
    TISSUE_EXTRA_INSERTION_M,
    TISSUE_HAND_CENTER_X_M,
    TISSUE_NAV_MAX_ANGULAR_RADPS,
    TISSUE_NAV_SPEED_LIMIT_MPS,
    TISSUE_PRETURN_MAX_ANGULAR_RADPS,
    TISSUE_RELEASE_FORCE_LIFT_SEC,
    TISSUE_RELEASE_MIN_SEPARATION_M,
    TISSUE_TABLE_BASE_Y_OFFSET_M,
    TISSUE_TABLE_EDGE_MARGIN_M,
    TISSUE_TABLE_INSET_M,
    TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M,
    TISSUE_TABLE_RETREAT_SPEED_MPS,
)
from .grasp_profiles import (
    _edge_row_cylinder_slot_lateral_locked,
    _fine_approach_stall_timeout,
    _fine_lateral_tolerance,
    _fine_terminal_ee_control_zone,
    _fine_visual_alignment_pending,
    _grip_close_command,
    _precision_grasp_profile,
    _random_pick_predeploy_distance,
    _random_rolling_handoff_distance,
)
from .sorting_state import CycleState


class EProductCycleStatusMixin:
    def publish_status(self) -> None:
        compact = self._compact_carry_metrics()
        mission_total_time = self._mission_total_time_sec
        if (
            mission_total_time is None
            and self._mission_motion_started_at is not None
            and self._mission_motion_finished_at is None
        ):
            mission_total_time = max(
                0.0, self.now() - self._mission_motion_started_at
            )
        tissue_hands = None
        if self._uses_tissue_two_hand_grasp and self.jpos is not None:
            left_hand, right_hand = self._tissue_hand_poses()
            left_point = np.asarray(left_hand[:3, 3], dtype=float)
            right_point = np.asarray(right_hand[:3, 3], dtype=float)
            tissue_hands = {
                "left": [float(value) for value in left_point],
                "right": [float(value) for value in right_point],
                "midpoint": [
                    float(value) for value in 0.5 * (left_point + right_point)
                ],
                "separation": float(np.linalg.norm(left_point - right_point)),
            }
        target = None
        if self._current_target is not None:
            target = {
                "aruco_id": int(self._current_target["aruco_id"]),
                "shelf": str(self._current_target.get("shelf", "E")),
                "kind": str(
                    self._current_target.get("kind", self.target_kind)
                ),
                "level": self._current_target["level"],
                "column": self._current_target["column"],
                "match_source": self._current_target["match_source"],
                "product_world": [
                    float(value) for value in self._current_target["product_world"]
                ],
                "vision_product_z": float(
                    self._current_target.get(
                        "vision_product_z",
                        self._current_target["product_world"][2],
                    )
                ),
            }
            observation_anchor = self._current_target.get(
                "observation_product_world"
            )
            if observation_anchor is not None:
                target["observation_product_world"] = [
                    float(value) for value in observation_anchor
                ]
            observation_visual = self._current_target.get(
                "observation_visual_product_world"
            )
            if observation_visual is not None:
                target["observation_visual_product_world"] = [
                    float(value) for value in observation_visual
                ]
        queued_targets = []
        for marker_id in self._target_queue:
            remembered = self._known_targets.get(int(marker_id))
            if remembered is None:
                continue
            queued_targets.append(
                {
                    "aruco_id": int(marker_id),
                    "shelf": str(remembered.get("shelf", "E")),
                    "kind": str(remembered.get("kind", self.target_kind)),
                    "level": remembered["level"],
                    "column": remembered["column"],
                    "match_source": remembered["match_source"],
                    "product_world": [
                        float(value) for value in remembered["product_world"]
                    ],
                    "vision_product_z": float(
                        remembered.get(
                            "vision_product_z", remembered["product_world"][2]
                        )
                    ),
                }
            )
        active_search_pose = None
        if self._active_search_poses:
            slide, pitch = self._active_search_poses[self._search_pose_index]
            active_search_pose = {
                "index": self._search_pose_index + 1,
                "count": len(self._active_search_poses),
                "slide": float(slide),
                "head_pitch": float(pitch),
            }
        payload = {
            "mission": self.mission_name,
            "target_kind": self.target_kind,
            "target_mode": (
                "random"
                if self._random_shelf_mode
                else ("mixed" if self._mixed_mode else self.target_kind)
            ),
            "target_sequence": list(self._mission_kind_sequence),
            "remaining_target_kinds": (
                list(self._random_remaining_kinds)
                if self._random_shelf_mode
                else [
                    str(
                        self._known_targets.get(int(marker_id), {}).get(
                            "kind", self.target_kind
                        )
                    )
                    for marker_id in self._target_queue
                ]
            ),
            "state": self.mission.state.name.lower(),
            "active_command": self._active_command_key or None,
            "failure_reason": self._last_failure_reason,
            "mission_timer_started": (
                self._mission_motion_started_at is not None
            ),
            "mission_timer_finished": (
                self._mission_motion_finished_at is not None
            ),
            "mission_total_time_sec": mission_total_time,
            "mission_timer_start_event": "first_base_motion",
            "mission_timer_end_event": (
                "final_item_placed_and_table_exit_reached"
            ),
            "final_completion_stays_at_table_exit": True,
            "picked_count": self._picked_count,
            "delivery_attempt_count": self._delivery_attempt_count,
            "next_table_drop_slot": min(
                self._table_drop_index(self._picked_count) + 1,
                len(TABLE_DROP_X_OFFSETS_M),
            ),
            "table_drop_slot_count": len(TABLE_DROP_X_OFFSETS_M),
            "target_table_drop_slots": {
                str(marker_id): int(slot_index) + 1
                for marker_id, slot_index in self._target_drop_slots.items()
            },
            "next_table_drop_pose": [
                float(value)
                for value in self._table_approach_pose(self._picked_count)
            ],
            "table_approach_x_negative_shift": (
                TABLE_APPROACH_X_NEGATIVE_SHIFT_M
            ),
            "required_target_count": self._required_target_count,
            "required_kind_counts": {
                kind: self._mission_kind_sequence.count(kind)
                for kind in dict.fromkeys(self._mission_kind_sequence)
            },
            "required_kele_count": self._mission_kind_sequence.count("kele"),
            "required_maidong_count": self._mission_kind_sequence.count(
                "maidong"
            ),
            "initial_discovered_count": self._initial_discovered_count,
            "mission_target_count": self._mission_target_count,
            "queue_initialized": self._queue_initialized,
            "queued_targets": queued_targets,
            "autonomous_shelf_selection": self._random_shelf_mode,
            "active_shelf": self._active_shelf,
            "random_scheduler_phase": (
                "first_cde"
                if self._random_shelf_mode and self._picked_count == 0
                else (
                    "after_first_abc"
                    if self._random_shelf_mode
                    else None
                )
            ),
            "random_visited_shelves": sorted(self._random_visited_shelves),
            "random_unreachable_shelves": sorted(
                self._random_unreachable_shelves
            ),
            "random_shelf_sweep_retry_count": self._random_sweep_retry_count,
            "random_post_first_abc_scan_pending": (
                self._random_post_first_abc_scan_pending
            ),
            "random_grasp_requires_current_live_observation": False,
            "random_grasp_prefers_live_tracking": True,
            "random_pick_handoff_policy": self._nav_pick_handoff_mode,
            "random_pick_predeployed": self._nav_pick_prepared,
            "random_pick_predeploy_distance": (
                _random_pick_predeploy_distance(
                    self._picked_count,
                    self._nav_pick_handoff_mode,
                    self._active_shelf,
                )
            ),
            "random_stationary_pick_predeploy_distance": (
                STATIONARY_PICK_PREDEPLOY_DISTANCE_M
            ),
            "ade_nav_near_handoff_position_tolerance": (
                ADE_OBSERVATION_HANDOFF_POSITION_TOL_M
            ),
            "ade_nav_near_handoff_yaw_tolerance": (
                ADE_OBSERVATION_HANDOFF_YAW_TOL_RAD
            ),
            "random_rolling_handoff_distance": (
                _random_rolling_handoff_distance(
                    self._picked_count, self._active_shelf
                )
            ),
            "bc_random_pick_predeploy_distance": (
                BC_RANDOM_PICK_PREDEPLOY_DISTANCE_M
            ),
            "bc_random_rolling_handoff_distance": (
                BC_RANDOM_ROLLING_HANDOFF_DISTANCE_M
            ),
            "random_rolling_handoff_max_speed": (
                RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS
            ),
            "random_pick_predeploy_yaw_tolerance": (
                RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD
            ),
            "random_rolling_handoff_yaw_tolerance": (
                RANDOM_ROLLING_HANDOFF_YAW_TOL_RAD
            ),
            "random_rolling_handoff_entry_speed": (
                self._nav_pick_handoff_speed_mps
            ),
            "first_e_direct_drive_used": self._first_e_direct_used,
            "first_e_direct_drive_active": self._first_e_direct_active,
            "first_e_direct_drive_cruise_speed": (
                FIRST_E_DIRECT_CRUISE_SPEED_MPS
            ),
            "first_pick_predeploy_distance": (
                FIRST_PICK_PREDEPLOY_DISTANCE_M
            ),
            "pick_fine_handoff_max_speed": (
                PICK_FINE_HANDOFF_MAX_LINEAR_MPS
            ),
            "fine_started_from_rolling_handoff": (
                self._fine_started_from_rolling_handoff
            ),
            "fine_first_pick_early_handoff": (
                self._fine_first_pick_early_handoff
            ),
            "pick_left_support_restore_active": (
                self._pick_deploy_left_restore_active
            ),
            "pick_left_support_restore_stage": (
                self._pick_deploy_left_restore_stage + 1
                if self._pick_deploy_left_restore_active
                else None
            ),
            "pick_left_support_restore_stage_count": len(
                LEFT_ARM_SCAN_WAYPOINTS
            ),
            "pick_left_support_ready": (
                self._pick_left_support_ready()
                if (
                    self.jpos is not None
                    and not self._uses_tissue_two_hand_grasp
                )
                else None
            ),
            "pick_body_lower_joint_slew": (
                PICK_DEPLOY_BODY_LOWER_JOINT_SLEW
            ),
            "random_completed_marker_ids": sorted(
                self._random_completed_marker_ids
            ),
            "random_removed_inventory_slots": [
                "/".join(slot)
                for slot in sorted(self._random_removed_inventory_slots)
            ],
            "random_rejected_inventory_pairs": sorted(
                [marker_id, kind]
                for marker_id, kind in self._random_rejected_inventory_pairs
            ),
            "random_empty_shelf_kinds": sorted(
                [shelf, kind]
                for shelf, kind in self._random_empty_shelf_kinds
            ),
            "random_last_decision": self._random_last_decision or None,
            "inventory_stable_slot_count": sum(
                1
                for slot in self._latest_inventory.get("slots", [])
                if isinstance(slot, dict) and slot.get("stable")
            ),
            "search_mode": self._search_mode,
            "active_search_pose": active_search_pose,
            "observation_collecting": self._search_collecting,
            "observation_detection_frames": self._search_detection_frames,
            "observation_last_frame_age": (
                None
                if self._search_last_frame_at is None
                else max(0.0, self.now() - self._search_last_frame_at)
            ),
            "observation_required_frames": SEARCH_MIN_DETECTION_FRAMES,
            "search_arms_restore_active": self._search_arms_restore_active,
            "search_arms_ready": self._search_arms_ready,
            "search_arms_restore_concurrent": True,
            "pending_removal_verification": (
                self._pending_verification is not None
            ),
            "target": target,
            "pick_target_visible": (
                self._target_last_seen_at is not None
                and self.now() - self._target_last_seen_at
                <= LIVE_TARGET_MAX_AGE_SEC
            ),
            "pick_target_last_seen_age": (
                None
                if self._target_last_seen_at is None
                else max(0.0, self.now() - self._target_last_seen_at)
            ),
            "observation_slide": float(self.scan_slide),
            "observation_head_pitch": float(OBSERVATION_HEAD_PITCH),
            "pick_head_pitch": float(self.tc[4]),
            "pick_head_pitch_fallback": float(PICK_TRACK_HEAD_PITCH),
            "pick_head_yaw_limit": float(PICK_TRACK_HEAD_YAW_LIMIT_RAD),
            "commanded_pick_head_yaw": float(self.tc[3]),
            "commanded_pick_head_pitch": float(self.tc[4]),
            "measured_pick_head_yaw": (
                None
                if self.jpos is None
                else float(
                    self.jpos.get("head_yaw_joint", self.tc[3])
                )
            ),
            "measured_pick_head_pitch": (
                None
                if self.jpos is None
                else float(
                    self.jpos.get("head_pitch_joint", self.tc[4])
                )
            ),
            "pick_head_dynamic_tracking": True,
            "pick_waypoint_enabled": False,
            "pick_nav_distance": None,
            "pick_nav_yaw_error": None,
            "pick_nav_position_tolerance": None,
            "pick_nav_yaw_tolerance": None,
            "pick_nav_stationary_position_tolerance": None,
            "pick_nav_stationary_yaw_tolerance": None,
            "nav_goal_distance": self._nav_goal_distance_m,
            "nav_goal_yaw_error": self._nav_goal_yaw_error_rad,
            "table_nav_yaw_assist_active": (
                self._table_nav_yaw_assist_active
            ),
            "table_nav_yaw_assist_error": (
                self._table_nav_yaw_assist_error_rad
            ),
            "table_nav_yaw_assist_command": (
                self._table_nav_yaw_assist_command_radps
            ),
            "table_nav_yaw_assist_start_distance": (
                TABLE_NAV_YAW_ASSIST_START_DISTANCE_M
            ),
            "table_nav_yaw_assist_fixed_speed": (
                TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS
            ),
            "table_nav_fast_accept_yaw_tolerance": (
                TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD
            ),
            "nav_near_goal_leg": self._nav_near_goal_leg,
            "nav_near_goal_settle_elapsed": (
                None
                if self._nav_near_goal_since is None
                else max(0.0, self.now() - self._nav_near_goal_since)
            ),
            "nav_near_goal_settle_required": NAV_NEAR_GOAL_SETTLE_SEC,
            "fine_required_distance": self._fine_required_distance_m,
            "fine_forward_remaining": self._fine_forward_remaining_m,
            "fine_forward_error": self._fine_forward_error_m,
            "fine_lateral_error": self._fine_lateral_error_m,
            "fine_signed_lateral_error": (
                self._fine_signed_lateral_error_m
            ),
            "fine_braking_speed": self._fine_braking_speed_mps,
            "fine_braking_distance": self._fine_braking_distance_m,
            "fine_dynamic_depth_tolerance": (
                self._fine_dynamic_depth_tolerance_m
            ),
            "fine_alignment_reserve": self._fine_alignment_reserve_m,
            "fine_nominal_speed": self._fine_nominal_speed_mps,
            "fine_speed_cap": self._fine_speed_cap_mps,
            "fine_initial_heading_aligned": (
                self._fine_initial_heading_aligned
            ),
            "fine_continuous_alignment": True,
            "fine_initial_heading_tolerance": (
                self._fine_initial_heading_tolerance_rad
            ),
            "fine_heading_error": self._fine_heading_error_rad,
            "fine_grasp_yaw_error": self._fine_grasp_yaw_error_rad,
            "fine_terminal_heading_target": (
                self._fine_terminal_heading_target_rad
            ),
            "fine_terminal_heading_error": (
                self._fine_terminal_heading_error_rad
            ),
            "fine_terminal_heading_tolerance": (
                SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD
                if self.target_kind == "sanmingzhi"
                else None
            ),
            "fine_global_lateral_kp": (
                DIRECT_POSE_GUIDANCE_LATERAL_KP
            ),
            "fine_dynamic_angular_limit": (
                self._fine_angular_limit_radps
            ),
            "fine_control_mode": self._fine_control_mode,
            "fine_terminal_ee_control_zone": (
                _fine_terminal_ee_control_zone(self.target_kind)
            ),
            "grasp_insertion": self._grasp_insertion_m,
            "sanmingzhi_insertion_reduction": (
                SANMINGZHI_INSERTION_REDUCTION_M
                if self.target_kind == "sanmingzhi"
                else None
            ),
            "grasp_height_offset": grasp_height_offset_for_product(
                self.target_kind,
                (
                    None
                    if self._current_target is None
                    else self._current_target.get("level")
                ),
            ),
            "deploy_slide_tolerance": (
                CHENGZI_DEPLOY_SLIDE_TOL_M
                if self.target_kind == "chengzi"
                else SLIDE_TOL_M
            ),
            "fine_near_latched": self._fine_near_latched,
            "fine_near_latched_age": (
                None
                if self._fine_near_latched_at is None
                else max(0.0, self.now() - self._fine_near_latched_at)
            ),
            "fine_near_mode": self._fine_near_mode,
            "fine_near_tolerance": self._fine_near_tolerance_m,
            "fine_near_settle_elapsed": (
                None
                if self._fine_near_since is None
                else max(0.0, self.now() - self._fine_near_since)
            ),
            "fine_near_settle_required": FINE_APPROACH_NEAR_SETTLE_SEC,
            "edge_grasp_lateral_tolerance": EDGE_GRASP_LATERAL_TOL_M,
            "effective_grasp_lateral_tolerance": (
                _fine_lateral_tolerance(
                    self.target_kind,
                    "" if self._current_target is None else str(
                        self._current_target.get("level", "")
                    ),
                )
            ),
            "edge_cylinder_slot_lateral_lock": (
                self._current_target is not None
                and _edge_row_cylinder_slot_lateral_locked(
                    self.target_kind,
                    str(self._current_target.get("level", "")),
                )
            ),
            "precision_alignment_required": (
                _precision_grasp_profile(self.target_kind) is not None
            ),
            "precision_alignment_locked": self._fine_precision_aligned,
            "precision_alignment_tolerance": (
                None
                if _precision_grasp_profile(self.target_kind) is None
                else _precision_grasp_profile(self.target_kind)[0]
            ),
            "coupled_alignment_required": (
                self.target_kind in COUPLED_ALIGNMENT_KINDS
            ),
            "coupled_alignment_tolerance": (
                COUPLED_ALIGNMENT_FINAL_TOL_M
                if self.target_kind in COUPLED_ALIGNMENT_KINDS
                else None
            ),
            "fine_time_aborts_enabled": FINE_APPROACH_TIME_ABORTS_ENABLED,
            "fine_stall_abort_timeout": (
                _fine_approach_stall_timeout(
                    self.target_kind,
                    precision_aligned=self._fine_precision_aligned,
                )
                if FINE_APPROACH_TIME_ABORTS_ENABLED
                else None
            ),
            "fine_stall_recovery_threshold": _fine_approach_stall_timeout(
                self.target_kind,
                precision_aligned=self._fine_precision_aligned,
            ),
            "fine_guidance_loss_abort_timeout": (
                FINE_VISION_LOSS_TIMEOUT_SEC
                if FINE_APPROACH_TIME_ABORTS_ENABLED
                else None
            ),
            "fine_approach_mode": (
                "direct_from_observation_continuous_curve"
            ),
            "fine_guidance_source": self._fine_guidance_source,
            "fine_guidance_target_world": (
                None
                if self.OBJECT_WORLD is None
                else [float(value) for value in self.OBJECT_WORLD]
            ),
            "fine_pixel_servo_active": self._fine_pixel_servo_active,
            "fine_product_pixel": (
                None
                if self._fine_product_pixel is None
                else [float(value) for value in self._fine_product_pixel]
            ),
            "fine_gripper_pixel": (
                None
                if self._fine_gripper_pixel is None
                else [float(value) for value in self._fine_gripper_pixel]
            ),
            "fine_pixel_error_px": self._fine_pixel_error_px,
            "fine_pixel_error_raw_px": self._fine_pixel_error_raw_px,
            "fine_pixel_correction_radps": (
                self._fine_pixel_correction_radps
            ),
            "fine_pixel_observation_age": (
                None
                if self._fine_pixel_observation_at is None
                else max(
                    0.0, self.now() - self._fine_pixel_observation_at
                )
            ),
            "fine_pixel_linear_scale": self._fine_pixel_linear_scale,
            "fine_pixel_servo_max_age": IMAGE_SERVO_MAX_AGE_SEC,
            "fine_pixel_servo_deadband": IMAGE_SERVO_PIXEL_DEADBAND_PX,
            "fine_pixel_terminal_lock_tolerance": (
                IMAGE_SERVO_TERMINAL_LOCK_TOL_PX
            ),
            "fine_pixel_alignment_pending": (
                _fine_visual_alignment_pending(
                    self.now(),
                    self._fine_pixel_error_px,
                    self._fine_pixel_observation_at,
                )
            ),
            "fine_pixel_servo_memory_fallback": True,
            "fine_live_tracking_min_samples": LIVE_TRACK_MIN_SAMPLES,
            "fine_live_lateral_refinement": not (
                self._current_target is not None
                and _edge_row_cylinder_slot_lateral_locked(
                    self.target_kind,
                    str(self._current_target.get("level", "")),
                )
            ),
            "fine_live_lateral_max_slot_offset": (
                LIVE_LATERAL_MAX_SLOT_OFFSET_M
            ),
            "fine_live_updates_frozen": self._fine_live_updates_frozen,
            "observation_memory_guidance": True,
            "fine_longitudinal_observation_lock": True,
            "fine_slot_anchor_enabled": (
                self.target_kind in PRECISION_SLOT_ANCHORED_KINDS
                or (
                    self._current_target is not None
                    and _edge_row_cylinder_slot_lateral_locked(
                        self.target_kind,
                        str(self._current_target.get("level", "")),
                    )
                )
            ),
            "fine_cruise_speed": DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS,
            "fine_alignment_speed": (
                DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS
                if _precision_grasp_profile(self.target_kind) is None
                else _precision_grasp_profile(self.target_kind)[2]
            ),
            "fine_speed_taper_start": (
                DIRECT_FINE_APPROACH_TAPER_START_M
            ),
            "fine_speed_taper_end": DIRECT_FINE_APPROACH_TAPER_END_M,
            "fine_requested_linear": float(self.des_lin),
            "fine_commanded_linear": float(self.tc[0]),
            "fine_commanded_angular": float(self.tc[1]),
            "fine_max_angular": (
                SANMINGZHI_FINE_APPROACH_MAX_ANGULAR
                if (
                    _precision_grasp_profile(self.target_kind) is not None
                    or self.target_kind in COUPLED_ALIGNMENT_KINDS
                )
                else (
                    EDGE_FINE_APPROACH_MAX_ANGULAR
                    if self._current_target is not None
                    and str(self._current_target.get("level", ""))
                    in {"L1", "L3"}
                    else DIRECT_FINE_APPROACH_MAX_ANGULAR
                )
            ),
            "fine_travel": self._fine_travel_m,
            "fine_travel_limit": self._fine_travel_limit_m,
            "fine_timeout": self._fine_timeout_sec,
            "fine_elapsed": (
                self.mission.elapsed
                if self.mission.state == CycleState.FINE_APPROACH
                else None
            ),
            "fine_target_forward": self._fine_target_forward_m,
            "base_stationary": bool(self.base_xy is not None and self._odom_stopped()),
            "slide": None if self.jpos is None else float(self.slide_meas),
            "commanded_slide": float(self.tc[2]),
            "right_gripper": (
                None
                if self.jpos is None
                else float(
                    self.jpos.get(
                        "right_arm_eef_gripper_joint", self.tc[18]
                    )
                )
            ),
            "commanded_right_gripper": float(self.tc[18]),
            "right_gripper_open_limit": float(GRIP_OPEN),
            "active_grip_close_target": float(
                _grip_close_command(self.target_kind)
            ),
            "left_gripper": (
                None
                if self.jpos is None
                else float(
                    self.jpos.get(
                        "left_arm_eef_gripper_joint", self.tc[11]
                    )
                )
            ),
            "commanded_left_gripper": float(self.tc[11]),
            "dual_arm_grasp": self._uses_tissue_two_hand_grasp,
            "tissue_clamp_half_separation": (
                TISSUE_CLAMP_HALF_SEPARATION_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_hand_center_x": (
                TISSUE_HAND_CENTER_X_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_table_base_y_offset": (
                TISSUE_TABLE_BASE_Y_OFFSET_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_grasp_extra_insertion": (
                TISSUE_EXTRA_INSERTION_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_table_inset_requested": (
                TISSUE_TABLE_INSET_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_table_extra_negative_y_request": (
                TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_table_edge_margin": (
                TISSUE_TABLE_EDGE_MARGIN_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_release_min_separation": (
                TISSUE_RELEASE_MIN_SEPARATION_M
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_release_smoothed_gripper_commands": (
                [float(self.action[11]), float(self.action[18])]
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_release_grippers_held_closed": (
                bool(
                    max(
                        abs(float(self.action[11]) - GRIP_CLOSE),
                        abs(float(self.action[18]) - GRIP_CLOSE),
                    ) <= 0.15
                )
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_release_force_lift_sec": (
                TISSUE_RELEASE_FORCE_LIFT_SEC
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_nav_speed_limit": (
                TISSUE_NAV_SPEED_LIMIT_MPS
                if self._tissue_nav_speed_limit_active
                else None
            ),
            "tissue_nav_max_angular": (
                TISSUE_NAV_MAX_ANGULAR_RADPS
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_preturn_max_angular": (
                TISSUE_PRETURN_MAX_ANGULAR_RADPS
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_preturn_source_shelf": (
                self._tissue_preturn_source_shelf
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_preturn_direction": (
                (
                    "left"
                    if self._tissue_preturn_angle_rad is not None
                    and self._tissue_preturn_angle_rad > 0.0
                    else "right"
                )
                if self._uses_tissue_two_hand_grasp
                and self._tissue_preturn_angle_rad is not None
                else None
            ),
            "tissue_preturn_signed_angle_deg": (
                math.degrees(self._tissue_preturn_angle_rad)
                if self._uses_tissue_two_hand_grasp
                and self._tissue_preturn_angle_rad is not None
                else None
            ),
            "tissue_preturn_accumulated_deg": (
                math.degrees(self._tissue_preturn_accumulated_rad)
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_preturn_target_yaw": (
                self._tissue_preturn_target_yaw
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_preturn_yaw_error": (
                self._tissue_preturn_yaw_error
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "controller_max_angular": (
                self._controller_angular_limit_current
            ),
            "controller_max_angular_pending": (
                self._controller_angular_limit_target
            ),
            "tissue_transport_strategy": (
                "straight_arm_horizontal_clamp_minimum_body_lift_no_roll"
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_grippers_closed": (
                self._tissue_grippers_closed
                if self._uses_tissue_two_hand_grasp
                else None
            ),
            "tissue_hands": tissue_hands,
            "tissue_motion": self._tissue_motion_label,
            "tissue_motion_stage": (
                self._tissue_motion_index + 1
                if self._tissue_motion_waypoints
                else None
            ),
            "tissue_motion_stage_count": len(
                self._tissue_motion_waypoints
            ),
            "tissue_carry_center_z": self._tissue_carry_center_z_m,
            "tissue_transport_slide": self._tissue_transport_slide_m,
            "tissue_table_drop_slide": self._tissue_table_drop_slide_m,
            "right_arm_max_error_rad": (
                None
                if self.jpos is None
                else float(np.max(np.abs(self.rarm_meas - self.tc[12:18])))
            ),
            "right_arm_compact": bool(compact["strict"]),
            "right_arm_compact_safe": bool(compact["safe"]),
            "right_arm_compact_non_wrist_max_error_rad": compact[
                "non_wrist_max_error"
            ],
            "right_arm_compact_wrist_error_rad": compact["wrist_error"],
            "right_arm_compact_ee": (
                None
                if compact["ee"] is None
                else [float(value) for value in compact["ee"]]
            ),
            "right_arm_compact_tilt_rad": compact["tilt"],
            "right_arm_compact_lateral_range": (
                None
                if compact["lateral_min"] is None
                else [
                    float(compact["lateral_min"]),
                    float(compact["lateral_max"]),
                ]
            ),
            "pick_transport_arms_together": True,
            "pick_transport_nav_concurrent": True,
            "pick_stow_active": self._pick_stow_active,
            "pick_stow_elapsed": (
                None
                if self._pick_stow_started_at is None
                else max(0.0, self.now() - self._pick_stow_started_at)
            ),
            "pick_left_stow_stage": (
                self._left_arm_stow_stage + 1
                if self._pick_stow_active and not self._pick_left_stow_complete
                else None
            ),
            "pick_left_stow_stage_count": len(LEFT_ARM_STOW_WAYPOINTS),
            "pick_left_stow_complete": self._pick_left_stow_complete,
            "pick_right_carry_ready": self._pick_right_carry_ready,
            "pick_transport_ready": self._pick_transport_is_ready(),
            "compact_carry_safe_settle_elapsed": (
                None
                if self._compact_carry_safe_since is None
                else max(
                    0.0, self.now() - self._compact_carry_safe_since
                )
            ),
            "compact_carry_safe_settle_required": (
                COMPACT_CARRY_SAFE_SETTLE_SEC
            ),
            "left_arm_max_error_rad": (
                None
                if self.jpos is None
                else float(np.max(np.abs(self.larm_meas - self.tc[5:11])))
            ),
            "return_arms_stowed": self._return_arms_stowed,
            "return_stow_active": self._return_stow_active,
            "return_stow_both_arms": self._return_stow_both_arms,
            "nav_scan_height_deferred_until_stationary": (
                self._random_shelf_mode and not self._nav_pick_prepared
            ),
            "return_stow_stage": (
                self._return_arm_stage + 1
                if self._return_stow_active
                else None
            ),
            "return_stow_stage_count": len(RIGHT_ARM_DOWN_WAYPOINTS),
            "search_arms_restore_together": True,
            "return_stow_fast_joint_slew": RETURN_STOW_FAST_JOINT_SLEW,
            "search_restore_fast_joint_slew": (
                SEARCH_RESTORE_FAST_JOINT_SLEW
            ),
            "pick_deploy_fast_joint_slew": PICK_DEPLOY_FAST_JOINT_SLEW,
            "fast_arm_stage_dwell": FAST_ARM_STAGE_DWELL_SEC,
            "pick_retreat_y": float(self._pick_retreat_y),
            "pick_retreat_speed": PICK_RETREAT_SPEED_MPS,
            "table_fold_clearance_y": TABLE_RETURN_FOLD_MIN_Y_M,
            "table_fold_clearance_ready": (
                self.base_xy is not None
                and float(self.base_xy[1]) >= TABLE_RETURN_FOLD_MIN_Y_M
            ),
            "table_stop_settle_required": TABLE_STOP_SETTLE_SEC,
            "place_joint_slew": float(self.joint_slew),
            "place_unfold_gentle_slew": PLACE_UNFOLD_GENTLE_JOINT_SLEW,
            "place_advance_gentle_slew": PLACE_ADVANCE_GENTLE_JOINT_SLEW,
            "place_advance_to_lower_settle_required": (
                PLACE_ADVANCE_TO_LOWER_SETTLE_SEC
            ),
            "place_lower_gentle_slew": PLACE_LOWER_GENTLE_JOINT_SLEW,
            "chengzi_place_unfold_slew": (
                CHENGZI_PLACE_UNFOLD_JOINT_SLEW
            ),
            "chengzi_place_advance_slew": (
                CHENGZI_PLACE_ADVANCE_JOINT_SLEW
            ),
            "chengzi_place_lower_slew": CHENGZI_PLACE_LOWER_JOINT_SLEW,
            "pingguo_place_unfold_slew": (
                PINGGUO_PLACE_UNFOLD_JOINT_SLEW
            ),
            "pingguo_place_advance_slew": (
                PINGGUO_PLACE_ADVANCE_JOINT_SLEW
            ),
            "pingguo_place_lower_slew": PINGGUO_PLACE_LOWER_JOINT_SLEW,
            "active_place_unfold_slew": (
                PLACE_UNFOLD_JOINT_SLEW_OVERRIDES.get(
                    self.target_kind,
                    PLACE_UNFOLD_GENTLE_JOINT_SLEW,
                )
            ),
            "active_place_advance_slew": (
                PLACE_ADVANCE_JOINT_SLEW_OVERRIDES.get(
                    self.target_kind,
                    PLACE_ADVANCE_GENTLE_JOINT_SLEW,
                )
            ),
            "active_place_lower_slew": (
                PLACE_LOWER_JOINT_SLEW_OVERRIDES.get(
                    self.target_kind,
                    PLACE_LOWER_GENTLE_JOINT_SLEW,
                )
            ),
            "chengzi_table_release_clearance": (
                CHENGZI_TABLE_RELEASE_CLEARANCE_M
                if self.target_kind == "chengzi"
                else None
            ),
            "chengzi_place_feedback": (
                self._chengzi_place_feedback
                if self.target_kind == "chengzi" else None
            ),
            "table_retreat_speed": (
                TISSUE_TABLE_RETREAT_SPEED_MPS
                if self._uses_tissue_two_hand_grasp
                else TABLE_RETREAT_SPEED_MPS
            ),
            "table_drop_slide": float(self._table_drop_slide_m),
            "post_release_lift_required": self._post_release_lift_required(),
            "post_release_clearance_lift": (
                POST_RELEASE_CLEARANCE_LIFT_M
            ),
            "post_release_slide_tolerance": (
                POST_RELEASE_SLIDE_TOL_M
            ),
            "post_release_target_slide": self._release_clearance_slide_m,
            "place_forward_distance": float(self._place_forward_distance),
            "costmap_footprint_mode": self._footprint_mode,
            "delivery_nav_chassis_retry_used": (
                self._delivery_nav_chassis_retry_used
            ),
        }
        self.status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )


__all__ = ["EProductCycleStatusMixin"]
