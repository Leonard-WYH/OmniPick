"""Mission commands, visual inventory and shelf scheduling."""

from __future__ import annotations

import json
import math
import numpy as np

from .baseline_grasp_controller import GRIP_OPEN
from .nav2_manipulation_client import RIGHT_ARM_RETURN_JOINT_SLEW
from .navigation.sorting_geometry import (
    SHELF_NAMES,
    bottle_geometry,
    e_shelf_contains,
    parse_mission_command,
    shelf_contains,
)
from std_msgs.msg import String
from typing import Any
from vision_msgs.msg import Detection3DArray

from .sorting_config import (
    FINE_APPROACH_INITIAL_HEADING_TOL_RAD,
    TISSUE_KIND,
)
from .grasp_profiles import (
    _detection_image_center,
)
from .sorting_state import CycleState


class EProductCycleInventoryMixin:
    # ---- commands and perception ----
    def task_cb(self, msg: String) -> None:
        """Remember which target products the Server requested this run."""

        try:
            payload = json.loads(msg.data)
            targets = payload.get("targets", [])
            if not isinstance(targets, list):
                raise TypeError("targets must be a list")
            task_kinds = [
                str(target.get("kind", "")).strip().lower()
                for target in targets
                if isinstance(target, dict)
            ]
            if self._random_shelf_mode:
                unsupported = [
                    kind
                    for kind in task_kinds
                    if kind not in self._allowed_target_kinds
                ]
                if unsupported:
                    raise ValueError(
                        "random cycle contains unsupported "
                        f"targets: {unsupported}"
                    )
                if len(task_kinds) != 5:
                    raise ValueError(
                        "random task must contain exactly five physical "
                        f"products: count={len(task_kinds)}"
                    )
                # Mission order from the Server is only an order description,
                # not a cabinet assignment. Preserve every occurrence so
                # repeated classes remain separate inventory requirements.
                # If tissue occurs, promote only its first occurrence. It then
                # receives table slot 3 while all remaining physical products
                # consume slots 1/2/4/5 in their actual delivery order.
                ordered_kinds = list(task_kinds)
                if TISSUE_KIND in ordered_kinds:
                    tissue_index = ordered_kinds.index(TISSUE_KIND)
                    ordered_kinds.insert(
                        0, ordered_kinds.pop(tissue_index)
                    )
                    self._forced_first_target_kind = TISSUE_KIND
                else:
                    self._forced_first_target_kind = str(
                        getattr(
                            self,
                            "_configured_forced_first_target_kind",
                            "",
                        )
                    )
                self._mission_kind_sequence = ordered_kinds
                self._random_remaining_kinds = list(ordered_kinds)
                required = len(ordered_kinds)
            elif self._mixed_mode:
                mission_sequence = [
                    kind
                    for kind in task_kinds
                    if kind in self._allowed_target_kinds
                ]
                expected = list(self._configured_kind_sequence)
                if mission_sequence != expected:
                    raise ValueError(
                        "mixed task sequence mismatch: "
                        f"expected={expected} received={mission_sequence}"
                    )
                self._mission_kind_sequence = mission_sequence
                required = len(mission_sequence)
            else:
                required = sum(
                    kind == self.target_kind for kind in task_kinds
                )
                self._mission_kind_sequence = [self.target_kind] * required
        except (json.JSONDecodeError, TypeError, AttributeError) as exc:
            self.get_logger().warning(f"{self._event_prefix}_TASK_REJECT {exc}")
            return
        except ValueError as exc:
            self.get_logger().error(f"{self._mission_event_prefix}_TASK_REJECT {exc}")
            return

        incoming_run_prefix = str(payload.get("run_prefix", "")).strip()
        if self._random_shelf_mode and incoming_run_prefix:
            previous_run_prefix = getattr(
                self, "_inventory_ledger_run_prefix", None
            )
            if previous_run_prefix != incoming_run_prefix:
                self._persistent_inventory_slots = {}
                self._inventory_conflict_log_keys = set()
                self._inventory_conflicted_slots = set()
                self._latest_inventory = {}
                self._inventory_ledger_run_prefix = incoming_run_prefix
                self.get_logger().info(
                    "RANDOM_VISUAL_INVENTORY_NEW_RUN "
                    f"run_prefix={incoming_run_prefix} ledger_cleared=true"
                )

        self.task_payload = payload
        self.task_received = True
        self._required_target_count = required
        self.get_logger().info(
            f"{self._mission_event_prefix}_TASK_REQUIREMENT count={required} "
            f"sequence={self._mission_kind_sequence} "
            f"run_prefix={payload.get('run_prefix', 'unknown')}"
        )

    def command_cb(self, msg: String) -> None:
        try:
            command = parse_mission_command(msg.data, self.mission_name)
        except ValueError as exc:
            self.get_logger().warning(f"MISSION_COMMAND_REJECT {exc}")
            return

        if command.command == "stop":
            if self.mission.state not in {
                CycleState.WAIT_COMMAND,
                CycleState.DONE,
                CycleState.STOPPED,
                CycleState.FAILED,
            }:
                self._stop_motion()
                self.mission.transition(CycleState.STOPPED, "external_stop")
            return

        key = command.command_id or msg.data.strip()
        if key in self._accepted_start_keys:
            return
        if self.mission.state not in {
            CycleState.WAIT_COMMAND,
            CycleState.DONE,
            CycleState.STOPPED,
            CycleState.FAILED,
        }:
            self.get_logger().warning(
                f"MISSION_COMMAND_BUSY state={self.mission.state.name}"
            )
            return

        self._accepted_start_keys.add(key)
        self._active_command_key = key
        if self._random_shelf_mode and not self.task_received:
            self.get_logger().warning(
                "RANDOM_MISSION_COMMAND_WAITING task topic not received yet"
            )
            self._accepted_start_keys.discard(key)
            return
        self._reset_cycle()
        self.manipulation_enabled = True
        self.get_logger().info(
            f"MISSION_COMMAND_ACCEPT key={key} mission={self.mission_name}"
        )
        self.mission.transition(CycleState.NAV_E_SCAN, "start_command")

    def inventory_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return

        if not self._random_shelf_mode:
            self._latest_inventory = dict(payload)
            return

        payload_run_prefix = str(payload.get("run_prefix", "")).strip()
        task_payload = getattr(self, "task_payload", None) or {}
        expected_run_prefix = str(
            task_payload.get("run_prefix", "")
        ).strip()
        # A transient latched DDS message from the preceding randomized run
        # must not repopulate the new mission's database.
        if (
            payload_run_prefix
            and expected_run_prefix
            and payload_run_prefix != expected_run_prefix
        ):
            return
        ledger_run_prefix = getattr(
            self, "_inventory_ledger_run_prefix", None
        )
        if payload_run_prefix and ledger_run_prefix != payload_run_prefix:
            self._persistent_inventory_slots = {}
            self._inventory_conflict_log_keys = set()
            self._inventory_conflicted_slots = set()
            self._inventory_ledger_run_prefix = payload_run_prefix

        ledger = getattr(self, "_persistent_inventory_slots", None)
        if ledger is None:
            ledger = {}
            self._persistent_inventory_slots = ledger
        conflict_keys = getattr(self, "_inventory_conflict_log_keys", None)
        if conflict_keys is None:
            conflict_keys = set()
            self._inventory_conflict_log_keys = conflict_keys
        conflicted_slots = getattr(
            self, "_inventory_conflicted_slots", None
        )
        if conflicted_slots is None:
            conflicted_slots = set()
            self._inventory_conflicted_slots = conflicted_slots

        slots = payload.get("slots", [])
        if isinstance(slots, list):
            for slot in slots:
                if (
                    not isinstance(slot, dict)
                    or not bool(slot.get("stable"))
                    or self._random_inventory_entry_removed(slot)
                ):
                    continue
                slot_key = self._inventory_slot_key(slot)
                if slot_key is None:
                    continue
                remembered = ledger.get(slot_key)
                if remembered is None:
                    ledger[slot_key] = dict(slot)
                    self.get_logger().info(
                        "RANDOM_VISUAL_INVENTORY_LATCH "
                        f"slot={'/'.join(slot_key)} "
                        f"kind={slot.get('kind')} "
                        f"id={slot.get('aruco_id')}"
                    )
                    continue
                remembered_kind = str(remembered.get("kind", ""))
                observed_kind = str(slot.get("kind", ""))
                if remembered_kind != observed_kind:
                    conflicted_slots.add(slot_key)
                    conflict_key = (
                        slot_key,
                        remembered_kind,
                        observed_kind,
                    )
                    if conflict_key not in conflict_keys:
                        conflict_keys.add(conflict_key)
                        self.get_logger().warning(
                            "RANDOM_VISUAL_INVENTORY_CONFLICT "
                            f"slot={'/'.join(slot_key)} "
                            f"latched={remembered_kind} "
                            f"current={observed_kind}; "
                            "latched_record_retained=true "
                            "route_candidate_quarantined=true"
                        )
                    continue
                conflicted_slots.discard(slot_key)
                # Refresh evidence quality for the same latched identity while
                # retaining the original physical slot and product kind.
                refreshed = dict(remembered)
                for key in (
                    "votes",
                    "vote_ratio",
                    "mean_confidence",
                    "last_seen",
                    "product_world",
                    "marker_world",
                ):
                    if key in slot:
                        refreshed[key] = slot[key]
                ledger[slot_key] = refreshed

        # No negative camera frame deletes a ledger entry. Only delivery
        # tombstones below are applied to the accumulated map.
        for slot_key, slot in list(ledger.items()):
            if self._random_inventory_entry_removed(slot):
                ledger.pop(slot_key, None)
        filtered_slots = list(ledger.values())
        filtered = dict(payload)
        filtered["slots"] = filtered_slots
        filtered["mapped_slot_count"] = len(filtered_slots)
        filtered["persistent_ledger"] = True
        self._latest_inventory = filtered

    @staticmethod
    def _inventory_slot_key(
        item: dict[str, Any],
    ) -> tuple[str, str, str] | None:
        """Return the physical shelf/level/column identity of one product."""

        shelf = str(item.get("shelf", "")).strip().upper()
        level = str(item.get("level", "")).strip().upper()
        column = str(item.get("column", "")).strip().upper()
        if shelf not in SHELF_NAMES:
            return None
        if level not in {"L1", "L2", "L3"}:
            return None
        if column not in {"C1", "C2", "C3"}:
            return None
        return shelf, level, column

    def _random_inventory_entry_removed(self, item: dict[str, Any]) -> bool:
        """Reject completed inventory by both marker ID and physical slot."""

        try:
            marker_id = int(item.get("aruco_id", -1))
        except (TypeError, ValueError):
            marker_id = -1
        slot_key = self._inventory_slot_key(item)
        return bool(
            marker_id
            in getattr(self, "_random_completed_marker_ids", set())
            or (
                slot_key is not None
                and slot_key
                in getattr(self, "_random_removed_inventory_slots", set())
            )
        )

    def _remove_random_target_from_visual_inventory(
        self, target: dict[str, Any]
    ) -> None:
        """Purge a delivered target locally and from perception memory."""

        marker_id = int(target["aruco_id"])
        slot_key = self._inventory_slot_key(target)
        if slot_key is not None:
            self._random_removed_inventory_slots.add(slot_key)
            getattr(self, "_inventory_conflicted_slots", set()).discard(
                slot_key
            )
            getattr(self, "_persistent_inventory_slots", {}).pop(
                slot_key, None
            )

        slots = self._latest_inventory.get("slots", [])
        if isinstance(slots, list):
            kept_slots = [
                slot
                for slot in slots
                if isinstance(slot, dict)
                and not self._random_inventory_entry_removed(slot)
            ]
            self._latest_inventory = {
                **self._latest_inventory,
                "slots": kept_slots,
                "mapped_slot_count": sum(
                    bool(slot.get("stable")) for slot in kept_slots
                ),
            }
        self._random_last_live_candidates = [
            item
            for item in self._random_last_live_candidates
            if not self._random_inventory_entry_removed(item)
        ]

        task_payload = getattr(self, "task_payload", None) or {}
        run_prefix = str(task_payload.get("run_prefix", ""))
        removal_payload = {
            "run_prefix": run_prefix,
            "removed_marker_ids": sorted(
                self._random_completed_marker_ids
            ),
            "removed_slots": [
                {"shelf": shelf, "level": level, "column": column}
                for shelf, level, column in sorted(
                    self._random_removed_inventory_slots
                )
            ],
        }
        publisher = getattr(self, "_inventory_remove_pub", None)
        if publisher is not None:
            publisher.publish(
                String(
                    data=json.dumps(
                        removal_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            )
        slot_text = (
            "/".join(slot_key) if slot_key is not None else "unknown"
        )
        self.get_logger().info(
            "RANDOM_VISUAL_INVENTORY_REMOVED "
            f"id={marker_id} slot={slot_text} "
            f"remaining_slots={len(self._latest_inventory.get('slots', []))}; "
            "local_memory=true perception_memory=true"
        )

    def _vision_to_object_center(
        self, p_world: np.ndarray, kind: str | None = None
    ) -> np.ndarray:
        """Convert the visible front surface using the selected package depth."""

        footprint = self.world_to_footprint(p_world)
        geometry = bottle_geometry(kind or self.target_kind)
        footprint[0] += geometry.radius_m
        return self.footprint_to_world(footprint)

    def product_cb(self, msg: Detection3DArray) -> None:
        state = self.mission.state
        search_observation = state == CycleState.SEARCH_E
        rolling_route_observation = bool(
            state == CycleState.WAIT_E_SCAN_NAV
            and self._random_shelf_mode
            and self._nav_pick_handoff_mode == "rolling"
            and not self.target_locked
        )
        target_tracking = bool(
            state
            in {
                CycleState.DEPLOY_TEMPLATE,
                CycleState.FINE_APPROACH,
                CycleState.HANDOFF_PICK_CONTROL,
                CycleState.STOP_E_SCAN,
            }
            or (
                state == CycleState.WAIT_E_SCAN_NAV
                and self.target_locked
            )
        )
        if not (
            search_observation
            or rolling_route_observation
            or target_tracking
        ):
            return
        if search_observation:
            # Exclude detections received while the torso/head is still moving
            # into its fixed observation posture.
            if not self._search_collecting:
                return
            self._search_detection_frames += 1
            self._search_last_frame_at = self.now()
        elif rolling_route_observation:
            # All rolling legs must identify the concrete product
            # while Nav2 is still approaching.  Previously an unlocked route
            # discarded every image here, so it could only stop at the
            # observation point and run SEARCH_E before deploying the arm.
            self._search_detection_frames += 1
            self._search_last_frame_at = self.now()
        for detection in msg.detections:
            if not detection.results:
                continue
            hypothesis = detection.results[0].hypothesis
            detected_kind = str(hypothesis.class_id).strip().lower()
            if search_observation or rolling_route_observation:
                if detected_kind not in self._allowed_target_kinds:
                    continue
            elif detected_kind != self.target_kind:
                continue
            position = detection.results[0].pose.pose.position
            point = np.array([position.x, position.y, position.z], dtype=float)
            on_active_shelf = (
                shelf_contains(point, self._active_shelf)
                if self._random_shelf_mode
                else e_shelf_contains(point)
            )
            if on_active_shelf:
                if search_observation or rolling_route_observation:
                    self._search_points.append((detected_kind, point))
                else:
                    received_at = self.now()
                    self._reacquire_points.append((received_at, point))
                    image_center = _detection_image_center(detection.id)
                    if image_center is not None and self.jpos is not None:
                        gripper_image_center = (
                            self._project_world_to_head_image(
                                self._grasp_reference_world()
                            )
                        )
                        # The pixel and world surface point came from the same
                        # RGB-D inference; project the measured gripper at
                        # receipt time as well.  Preserving the pair avoids both
                        # adjacent-slot association and moving-camera skew.
                        if gripper_image_center is not None:
                            self._image_reacquire_points.append(
                                (
                                    received_at,
                                    point.copy(),
                                    image_center,
                                    gripper_image_center,
                                )
                            )

    def _reset_cycle(self) -> None:
        self._mission_motion_started_at = None
        self._mission_motion_finished_at = None
        self._mission_total_time_sec = None
        self._chengzi_grasp_center_in_hand = None
        self._chengzi_place_initial_slide = None
        self._chengzi_place_heights.clear()
        self._chengzi_place_feedback = {}
        self._active_shelf = "E"
        self._random_active_scan_pose = None
        self._random_planned_marker_id = None
        self._random_planned_kind = None
        self._random_remaining_kinds = list(self._mission_kind_sequence)
        self._random_completed_marker_ids.clear()
        self._random_failed_marker_ids.clear()
        self._random_removed_inventory_slots.clear()
        self._random_rejected_inventory_pairs.clear()
        self._random_empty_shelf_kinds.clear()
        self._random_last_live_candidates.clear()
        self._random_visited_shelves.clear()
        self._random_unreachable_shelves.clear()
        self._random_sweep_retry_count = 0
        self._random_last_decision = {}
        self._random_post_first_abc_scan_pending = False
        self._random_post_first_abc_transit_scan_active = False
        self._random_scan_head_mode = None
        if self._mission_kind_sequence:
            self._set_active_target_kind(self._mission_kind_sequence[0])
        self._target_attempts.clear()
        self._grasp_attempts.clear()
        self._pending_verification = None
        self._current_target = None
        self._picked_count = 0
        self._delivery_attempt_count = 0
        self._mission_target_count = (
            len(self._mission_kind_sequence)
            if self._random_shelf_mode
            else 0
        )
        self._initial_discovered_count = 0
        self._queue_initialized = False
        self._known_targets.clear()
        self._target_drop_slots.clear()
        self._target_queue.clear()
        self._search_points.clear()
        self._reacquire_points.clear()
        self._image_reacquire_points.clear()
        self._search_mode = "idle"
        self._search_pose_index = 0
        self._active_search_poses = []
        self._search_collecting = False
        self._search_detection_frames = 0
        self._search_last_frame_at = None
        self._target_last_seen_at = None
        self._fine_vision_lost_at = None
        self._fine_guidance_source = "none"
        self._fine_observation_target_world = None
        self._fine_last_live_target_world = None
        self._fine_last_vision_update_at = None
        self._fine_live_updates_frozen = False
        self._fine_pixel_servo_active = False
        self._fine_product_pixel = None
        self._fine_gripper_pixel = None
        self._fine_pixel_error_px = None
        self._fine_pixel_error_raw_px = None
        self._fine_pixel_correction_radps = None
        self._fine_pixel_linear_scale = 1.0
        self._fine_pixel_observation_at = None
        self._fine_pixel_filter_observation_at = None
        self._last_live_track_log = -math.inf
        self._pick_nav_distance_m = None
        self._pick_nav_yaw_error_rad = None
        self._nav_goal_distance_m = None
        self._nav_goal_yaw_error_rad = None
        self._nav_near_goal_since = None
        self._nav_near_goal_leg = None
        self._nav_terminal_hold_commanded = False
        self._nav_pick_prepared = False
        self._nav_pick_handoff_mode = "traditional"
        self._nav_pick_handoff_speed_mps = 0.0
        self._fine_started_from_rolling_handoff = False
        self._fine_first_pick_early_handoff = False
        self._pick_deploy_left_restore_active = False
        self._pick_deploy_left_restore_stage = 0
        self._pick_deploy_left_restore_ready_since = None
        self._pick_deploy_held_slide_m = None
        self._first_e_direct_used = False
        self._first_e_direct_active = False
        self._footprint_pending_mode = None
        self._footprint_futures = []
        self._delivery_nav_chassis_retry_used = False
        self.deploy_set = False
        self.target_locked = False
        self.OBJECT_WORLD = None
        self.DEPLOY_WORLD = None
        self.CREEP_STOP_Y = None
        self._fine_start_xy = None
        self._fine_best_ee_y = None
        self._fine_best_lateral_error_m = None
        self._fine_terminal_heading_target_rad = None
        self._fine_terminal_heading_error_rad = None
        self._fine_lateral_error_m = None
        self._fine_progress_at = None
        self._fine_required_distance_m = None
        self._fine_travel_m = None
        self._fine_travel_limit_m = None
        self._fine_timeout_sec = None
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
            FINE_APPROACH_INITIAL_HEADING_TOL_RAD
        )
        self._fine_heading_error_rad = None
        self._fine_grasp_yaw_error_rad = None
        self._fine_angular_limit_radps = None
        self._fine_control_mode = "idle"
        self._fine_signed_lateral_error_m = None
        self._grasp_insertion_m = None
        self._fine_near_latched = False
        self._fine_near_latched_at = None
        self._fine_near_mode = None
        self._fine_near_tolerance_m = None
        self._fine_near_since = None
        self._fine_precision_aligned = False
        self._last_failure_reason = None
        self._compact_carry_safe_since = None
        self._left_arm_stow_stage = 0
        self._left_arm_stow_stage_started_at = 0.0
        self._pick_left_stow_complete = False
        self._pick_right_carry_ready = False
        self._pick_stow_active = False
        self._pick_stow_started_at = None
        self._return_arm_stage = 0
        self._return_arm_stage_ready_since = None
        self._return_arms_stowed = False
        self._return_stow_active = False
        self._return_stow_both_arms = False
        self._return_stow_started_at = None
        self._search_arms_restore_active = False
        self._search_arms_ready = False
        self._search_arms_restore_started_at = None
        self._table_retreat_start_xy = None
        self._table_retreat_heading = 0.0
        self._table_exit_reached = False
        self._release_clearance_slide_m = None
        self._tissue_deploy_arms = None
        self._tissue_clamp_arms = None
        self._tissue_transport_arms = None
        self._tissue_table_arms = None
        self._tissue_release_arms = None
        self._tissue_motion_waypoints = []
        self._tissue_motion_index = 0
        self._tissue_motion_ready_since = None
        self._tissue_motion_started_at = None
        self._tissue_motion_label = None
        self._tissue_carry_center_z_m = None
        self._tissue_transport_slide_m = None
        self._tissue_table_drop_slide_m = None
        self._tissue_clamp_ready_since = None
        self._tissue_preturn_source_shelf = None
        self._tissue_preturn_angle_rad = None
        self._tissue_preturn_last_yaw = None
        self._tissue_preturn_accumulated_rad = 0.0
        self._tissue_preturn_target_yaw = None
        self._tissue_preturn_yaw_error = None
        self._table_nav_yaw_assist_active = False
        self._table_nav_yaw_assist_error_rad = None
        self._table_nav_yaw_assist_command_radps = None
        self._set_nav_angular_override(None)
        self.tc[11] = GRIP_OPEN
        self.tc[18] = GRIP_OPEN
        self.joint_slew = RIGHT_ARM_RETURN_JOINT_SLEW



__all__ = ["EProductCycleInventoryMixin"]
