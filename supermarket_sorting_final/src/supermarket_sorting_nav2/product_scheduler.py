"""Candidate scoring, cross-shelf routing and target activation."""

from __future__ import annotations

import math
import numpy as np

from .baseline_grasp_controller import GRIP_OPEN
from .navigation.sorting_geometry import (
    E_SCAN_POSE,
    SHELF_COLUMNS_M,
    SHELF_FIRST_ARUCO_ID,
    SHELF_NAMES,
    SHELF_PRODUCT_CENTER_Y_M,
    TABLE_DROP_X_OFFSETS_M,
    bottle_geometry,
    cluster_points,
    e_product_center_z,
    grasp_slide_for_product_z,
    nearest_e_slot,
    nearest_shelf_slot,
    shelf_scan_pose,
    shelf_target_observation_pose,
    table_approach_pose,
)
from collections import (
    Counter,
    deque,
)
from typing import Any

from .sorting_config import (
    ADE_NEAR_HANDOFF_SHELVES,
    CONFIRMED_SLOT_DISTANCE_M,
    FIRST_E_DIRECT_CRUISE_SPEED_MPS,
    FIRST_PICK_PREDEPLOY_DISTANCE_M,
    GRASP_RETRY_LIMIT,
    LIVE_LATERAL_MAX_SLOT_OFFSET_M,
    NORMAL_MPPI_MAX_ANGULAR_RADPS,
    OBSERVATION_HEAD_PITCH,
    PICK_RETREAT_CLEAR_Y_M,
    PRECISION_SLOT_ANCHORED_KINDS,
    RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD,
    SEARCH_MIN_CLUSTER_SAMPLES,
    SEARCH_MIN_DETECTION_FRAMES,
    TARGET_MEMORY_MATCH_RADIUS_M,
    TISSUE_KIND,
    TISSUE_PICK_RETREAT_Y_M,
)
from .grasp_profiles import (
    _edge_row_cylinder_slot_lateral_locked,
    _first_e_direct_velocity_command,
    _grasp_insertion_for_product,
    _grasp_lateral_target_x,
)
from .sorting_state import CycleState


class EProductCycleSchedulerMixin:
    def _random_schedulable_kind_counts(self) -> Counter[str]:
        """Return every outstanding class count, including repetitions."""

        return Counter(getattr(self, "_random_remaining_kinds", ()))

    def _matching_inventory_slot(
        self, surface_point: np.ndarray, kind: str | None = None
    ) -> dict[str, Any] | None:
        target_kind = kind or self.target_kind
        best = None
        best_distance = CONFIRMED_SLOT_DISTANCE_M
        for slot in self._latest_inventory.get("slots", []):
            if (
                not isinstance(slot, dict)
                or not slot.get("stable")
                or slot.get("shelf")
                != (self._active_shelf if self._random_shelf_mode else "E")
                or slot.get("kind") != target_kind
            ):
                continue
            try:
                mapped_point = np.asarray(slot["product_world"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            distance = float(np.linalg.norm(mapped_point - surface_point))
            if distance < best_distance:
                best_distance = distance
                best = slot
        return best

    @staticmethod
    def _copy_target(target: dict[str, Any]) -> dict[str, Any]:
        copied = dict(target)
        for key in (
            "surface_world",
            "product_world",
            "observation_product_world",
            "observation_visual_product_world",
        ):
            if key in copied:
                copied[key] = np.asarray(copied[key], dtype=float).copy()
        return copied

    def _search_candidates(self) -> list[dict[str, Any]]:
        """Build one stable live candidate per ArUco slot from the current scan."""

        candidates_by_id: dict[int, dict[str, Any]] = {}
        for target_kind in self._allowed_target_kinds:
            kind_points = [
                point
                for detected_kind, point in self._search_points
                if detected_kind == target_kind
            ]
            for cluster in cluster_points(kind_points):
                if len(cluster) < SEARCH_MIN_CLUSTER_SAMPLES:
                    continue
                surface = np.median(cluster, axis=0)
                center = self._vision_to_object_center(surface, target_kind)
                if self._random_shelf_mode:
                    (
                        inferred_id,
                        inferred_shelf,
                        inferred_level,
                        inferred_column,
                    ) = nearest_shelf_slot(
                        center, target_kind, self._active_shelf
                    )
                else:
                    inferred_id, inferred_level, inferred_column = nearest_e_slot(
                        center, target_kind
                    )
                    inferred_shelf = "E"
                slot = self._matching_inventory_slot(surface, target_kind)
                if slot is not None and int(slot["aruco_id"]) == inferred_id:
                    marker_id = int(slot["aruco_id"])
                    level = str(slot["level"])
                    column = str(slot["column"])
                    source = "aruco_confirmed"
                else:
                    marker_id = inferred_id
                    level = inferred_level
                    column = inferred_column
                    source = (
                        "geometry_corrected"
                        if slot is not None
                        else "shelf_inferred"
                    )
                    if slot is not None:
                        rejected_pair = (int(slot["aruco_id"]), target_kind)
                        rejected_pairs = getattr(
                            self, "_random_rejected_inventory_pairs", None
                        )
                        first_rejection = bool(
                            rejected_pairs is None
                            or rejected_pair not in rejected_pairs
                        )
                        if rejected_pairs is not None:
                            rejected_pairs.add(rejected_pair)
                        # A close-range depth sample can disagree with the
                        # earlier slot association. Use shelf geometry for
                        # this live frame, but keep the latched inventory
                        # record: a single contradictory view must never erase
                        # a product that was already observed this mission.
                        # The set is diagnostic/log de-duplication only.
                        if first_rejection:
                            self.get_logger().warning(
                                f"{inferred_shelf}_{target_kind.upper()}_SLOT_ASSOCIATION_REJECT "
                                f"vision_id={inferred_id} "
                                f"aruco_association_id={slot['aruco_id']} "
                                f"vision_z={center[2]:.3f}; "
                                "using shelf geometry for live tracking; "
                                "persistent_inventory_retained=true"
                            )

                # RGB-D samples the visible box centre and is consistently a
                # few centimetres high for a low-row bottle.  Once its slot is
                # known, use that product's intrinsic half-height.
                vision_product_z = float(center[2])
                center[2] = e_product_center_z(level, target_kind)
                # Preserve the unobstructed multi-frame visual X before the
                # product is tied to its ArUco slot.  Shelf Y and Z remain
                # geometry-anchored: using noisy close-range depth there could
                # command the gripper into the shelf.
                observation_visual_center = center.copy()
                edge_cylinder_slot_lock = (
                    _edge_row_cylinder_slot_lateral_locked(
                        target_kind, level
                    )
                )
                if (
                    target_kind in PRECISION_SLOT_ANCHORED_KINDS
                    or edge_cylinder_slot_lock
                ):
                    first_id = SHELF_FIRST_ARUCO_ID[inferred_shelf]
                    columns = SHELF_COLUMNS_M[inferred_shelf]
                    column_index = (
                        int(marker_id) - first_id
                    ) % len(columns)
                    center[0] = columns[column_index]
                    center[1] = SHELF_PRODUCT_CENTER_Y_M
                    observation_visual_center[1] = center[1]
                    if edge_cylinder_slot_lock:
                        # Do not seed an edge-row bottle approach from the
                        # repeatable oblique-view centroid bias.  A later live
                        # frame still confirms visibility, while the known
                        # ArUco slot remains the lateral grasp reference.
                        observation_visual_center[0] = center[0]
                stable_surface = np.asarray(surface, dtype=float).copy()
                stable_surface[2] = center[2]
                candidate = {
                    "surface_world": stable_surface,
                    "product_world": center,
                    # Preserve the fixed, unobstructed observation separately
                    # from later close-range tracking updates.  Box packages
                    # use this as their precision alignment anchor.
                    "observation_product_world": center.copy(),
                    # Lateral memory recorded while the camera view is still
                    # unobstructed.  It is the fallback when live detections
                    # disappear during the approach.
                    "observation_visual_product_world": (
                        observation_visual_center
                    ),
                    "aruco_id": marker_id,
                    "shelf": inferred_shelf,
                    "kind": target_kind,
                    "level": level,
                    "column": column,
                    "match_source": source,
                    "samples": len(cluster),
                    "vision_product_z": vision_product_z,
                }
                previous = candidates_by_id.get(marker_id)
                rank = (source == "aruco_confirmed", len(cluster))
                previous_rank = (
                    False,
                    -1,
                ) if previous is None else (
                    previous["match_source"] == "aruco_confirmed",
                    int(previous["samples"]),
                )
                if rank > previous_rank:
                    candidates_by_id[marker_id] = candidate

        candidates = list(candidates_by_id.values())
        candidates.sort(
            key=lambda item: (
                item["match_source"] != "aruco_confirmed",
                -item["samples"],
                abs(
                    float(item["product_world"][0])
                    - shelf_scan_pose(
                        str(item.get("shelf", self._active_shelf))
                    )[0]
                ),
            )
        )
        return candidates

    def _random_inventory_candidates(self) -> list[dict[str, Any]]:
        """Convert the persistent inventory map into schedulable targets.

        The detector owns multi-frame voting and ArUco association.  This
        client intentionally consumes only stable slots and keeps their world
        pose as memory when a target temporarily leaves the camera image.
        """

        if not self._random_shelf_mode:
            return []
        needed = self._random_schedulable_kind_counts()
        candidates: list[dict[str, Any]] = []
        for slot in self._latest_inventory.get("slots", []):
            if not isinstance(slot, dict) or not slot.get("stable"):
                continue
            kind = str(slot.get("kind", "")).strip().lower()
            shelf = str(slot.get("shelf", "")).strip().upper()
            if kind not in needed or needed[kind] <= 0 or shelf not in SHELF_NAMES:
                continue
            try:
                marker_id = int(slot["aruco_id"])
                center = np.asarray(slot["product_world"], dtype=float).copy()
            except (KeyError, TypeError, ValueError):
                continue
            if (
                self._random_inventory_entry_removed(slot)
                or marker_id in self._random_failed_marker_ids
            ):
                continue
            slot_key = self._inventory_slot_key(slot)
            if slot_key in getattr(
                self, "_inventory_conflicted_slots", set()
            ):
                # Keep the task-scoped memory, but do not dispatch directly
                # from a slot for which two stable classes disagree. A live
                # candidate can still be selected when the robot observes the
                # shelf, and seeing the original class clears the quarantine.
                continue
            level = str(slot.get("level", ""))
            column = str(slot.get("column", ""))
            if level not in {"L1", "L2", "L3"}:
                continue
            # ArUco truth is the safety anchor for shelf depth and height.
            center[1] = SHELF_PRODUCT_CENTER_Y_M
            center[2] = e_product_center_z(level, kind)
            try:
                column_index = int(column.removeprefix("C")) - 1
                center[0] = SHELF_COLUMNS_M[shelf][column_index]
            except (ValueError, IndexError):
                continue
            surface = center.copy()
            # Approximate the visible front surface so live reacquisition can
            # match this remembered candidate before a new frame replaces it.
            if self.base_xy is not None:
                footprint = self.world_to_footprint(center)
                footprint[0] -= bottle_geometry(kind).radius_m
                surface = self.footprint_to_world(footprint)
            candidates.append(
                {
                    "surface_world": surface,
                    "product_world": center,
                    "observation_product_world": center.copy(),
                    "observation_visual_product_world": center.copy(),
                    "aruco_id": marker_id,
                    "shelf": shelf,
                    "kind": kind,
                    "level": level,
                    "column": column,
                    "match_source": "inventory_memory",
                    "samples": int(slot.get("votes", 0)),
                    "vision_product_z": float(
                        np.asarray(slot.get("product_world", center))[2]
                    ),
                    "vote_ratio": float(slot.get("vote_ratio", 0.0)),
                    "mean_confidence": float(
                        slot.get("mean_confidence", 0.0)
                    ),
                }
            )
        return candidates

    def _random_candidates_with_live(
        self, live_candidates: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Merge global memory with current-shelf vision, preferring vision."""

        by_id = {
            int(item["aruco_id"]): item
            for item in self._random_inventory_candidates()
        }
        for item in live_candidates or []:
            marker_id = int(item["aruco_id"])
            if self._random_inventory_entry_removed(item):
                continue
            kind = str(item.get("kind", "")).strip().lower()
            shelf = str(item.get("shelf", "")).strip().upper()
            # Current geometry-backed vision is newer than persistent map
            # memory. If a product reappears, restore that exact candidate and
            # its shelf/category immediately.
            getattr(
                self, "_random_rejected_inventory_pairs", set()
            ).discard((marker_id, kind))
            getattr(self, "_random_empty_shelf_kinds", set()).discard(
                (shelf, kind)
            )
            by_id[marker_id] = item
        needed = self._random_schedulable_kind_counts()
        # Keep all alternative instances of a requested kind.  Even when the
        # order needs only one, the scheduler must be able to choose a nearer
        # or reachable cabinet instead of inheriting inventory insertion order.
        return [
            item
            for item in by_id.values()
            if needed[str(item.get("kind", ""))] > 0
        ]

    def _record_random_shelf_kind_visibility(
        self,
        live_candidates: list[dict[str, Any]],
        kinds: set[str],
        *,
        reason: str,
    ) -> None:
        """Log a current-view absence without invalidating task inventory.

        Occlusion, head angle and detector jitter make negative observations
        unreliable.  A product stays latched in the mission database until a
        confirmed table delivery removes its marker and physical slot.
        """

        if self._search_detection_frames < SEARCH_MIN_DETECTION_FRAMES:
            return
        shelf = str(self._active_shelf).upper()
        visible_kinds = {
            str(item.get("kind", "")).strip().lower()
            for item in live_candidates
            if str(item.get("shelf", "")).strip().upper() == shelf
            and str(item.get("match_source", "")) != "inventory_memory"
        }
        for kind in kinds:
            normalized_kind = str(kind).strip().lower()
            pair = (shelf, normalized_kind)
            if normalized_kind in visible_kinds:
                self._random_empty_shelf_kinds.discard(pair)
                continue
            if pair in self._random_empty_shelf_kinds:
                continue
            self._random_empty_shelf_kinds.add(pair)
            self.get_logger().info(
                "RANDOM_SHELF_KIND_NOT_CURRENTLY_VISIBLE "
                f"shelf={shelf} kind={normalized_kind} "
                f"frames={self._search_detection_frames} reason={reason}; "
                "persistent_inventory_retained=true"
            )

    def _random_candidate_key(self, item: dict[str, Any]) -> tuple[float, ...]:
        """Score one slot using phase priority and travel geometry.

        Cabinet priority is a hard first key.  Distance then breaks ties using
        robot-to-scan travel plus a smaller estimate of the later shelf-to-table
        leg.  Nav2 remains the authority on obstacle feasibility; an aborted
        scan leg blacklists that shelf for the current sweep and reschedules.
        """

        shelf = str(item.get("shelf", "E"))
        marker_id = int(item["aruco_id"])
        if self._picked_count == 0:
            priority = {"E": 0, "D": 1, "C": 2, "B": 3, "A": 3}
        else:
            priority = {"B": 0, "A": 1, "C": 1, "D": 2, "E": 2}
        forced_first_kind = str(
            getattr(self, "_forced_first_target_kind", "")
        ).strip().lower()
        if self._picked_count == 0 and forced_first_kind:
            # The fixed regression layout can request one specific first
            # class while retaining the normal multi-shelf scheduler for all
            # later products.  A hard score tier guarantees that the direct
            # E run locks this class instead of another simultaneously visible
            # item; the route itself still uses live vision for final alignment.
            item_kind = str(item.get("kind", "")).strip().lower()
            if item_kind == forced_first_kind:
                priority[shelf] = -1
            else:
                priority[shelf] += len(SHELF_NAMES)
        scan = np.asarray(shelf_scan_pose(shelf)[:2], dtype=float)
        robot_distance = (
            float(np.linalg.norm(scan - self.base_xy))
            if self.base_xy is not None
            else 0.0
        )
        table_slot = self._random_drop_slot_for_kind(
            str(item.get("kind", ""))
        )
        table = np.asarray(table_approach_pose(table_slot)[:2], dtype=float)
        table_distance = float(np.linalg.norm(scan - table))
        confidence_penalty = 1.0 - float(item.get("mean_confidence", 0.0))
        return (
            float(priority[shelf]),
            robot_distance + 0.35 * table_distance + 0.10 * confidence_penalty,
            abs(float(item["product_world"][0]) - scan[0]),
            float(marker_id),
        )

    def _select_random_candidate(
        self,
        live_candidates: list[dict[str, Any]] | None = None,
        *,
        excluded_shelves: set[str] | None = None,
    ) -> dict[str, Any] | None:
        excluded = excluded_shelves or set()
        candidates = [
            item
            for item in self._random_candidates_with_live(live_candidates)
            if str(item.get("shelf", ""))
            not in self._random_unreachable_shelves
            and str(item.get("shelf", "")) not in excluded
        ]
        candidates = self._first_pick_kind_candidates(candidates)
        if not candidates:
            return None
        if self._random_planned_marker_id is not None:
            for item in candidates:
                if int(item["aruco_id"]) == self._random_planned_marker_id:
                    return item
        return min(candidates, key=self._random_candidate_key)

    def _first_pick_kind_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Prevent another requested class from overtaking forced item one."""

        forced_kind = str(
            getattr(self, "_forced_first_target_kind", "")
        ).strip().lower()
        if self._picked_count != 0 or not forced_kind:
            return candidates
        return [
            item
            for item in candidates
            if str(item.get("kind", "")).strip().lower() == forced_kind
        ]

    def _select_live_candidate_at_active_shelf(
        self, live_candidates: list[dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        """Prefer a target from the current stationary observation window.

        The caller falls back to the task-scoped inventory ledger when this
        window is empty. Live vision therefore refines a remembered target but
        is no longer required to authorize the Nav2-to-grasp handoff.
        """

        needed = self._random_schedulable_kind_counts()
        candidates = [
            item
            for item in live_candidates or []
            if str(item.get("shelf", "")).upper() == self._active_shelf
            and str(item.get("kind", "")).lower() in needed
            and needed[str(item.get("kind", "")).lower()] > 0
            and not self._random_inventory_entry_removed(item)
            and int(item["aruco_id"]) not in self._random_failed_marker_ids
            and str(item.get("match_source", "")) != "inventory_memory"
        ]
        candidates = self._first_pick_kind_candidates(candidates)
        if not candidates:
            return None
        if self._random_planned_marker_id is not None:
            for item in candidates:
                if int(item["aruco_id"]) == self._random_planned_marker_id:
                    return item
        return min(candidates, key=self._random_candidate_key)

    def _random_scan_order(self) -> tuple[str, ...]:
        # The first outbound leg deliberately starts at E and falls back
        # toward D/C.  After the first delivery, B is observed first, then A/C;
        # D/E are revisited only when the preferred group has no requested item.
        if self._picked_count == 0:
            return ("E", "D", "C", "B", "A")
        return ("B", "A", "C", "D", "E")

    def _random_pick_handoff_policy(self) -> str:
        """Return the requested observation-point handoff for this shelf leg."""

        if not self._random_shelf_mode:
            return "traditional"
        if self._picked_count == 0 and self._active_shelf in {"E", "D"}:
            return "rolling"
        # After the first delivery, A/D/E keep Nav2 only until they are roughly
        # aligned near the observation point while deploying the grasp posture
        # concurrently. B/C retain their earlier rolling visual handoff.
        if (
            self._picked_count > 0
            and self._active_shelf in ADE_NEAR_HANDOFF_SHELVES
        ):
            return "stationary_predeploy"
        if self._picked_count > 0:
            return "rolling"
        return "traditional"

    def _active_shelf_pick_candidate(self) -> dict[str, Any] | None:
        """Choose an en-route live target, falling back to stable inventory."""

        live_candidates = self._search_candidates()
        self._random_last_live_candidates = [
            self._copy_target(item) for item in live_candidates
        ]
        candidates = [
            item
            for item in self._random_candidates_with_live(
                live_candidates
            )
            if str(item.get("shelf", "")).upper() == self._active_shelf
        ]
        candidates = self._first_pick_kind_candidates(candidates)
        if not candidates:
            return None
        if self._random_planned_marker_id is not None:
            for item in candidates:
                if int(item["aruco_id"]) == self._random_planned_marker_id:
                    return item
        return min(candidates, key=self._random_candidate_key)

    def _prepare_random_pick_during_navigation(self) -> bool:
        """Lock a stable target and deploy its grasp posture while Nav2 moves."""

        if self._nav_pick_prepared:
            return True
        selected = self._active_shelf_pick_candidate()
        if selected is None:
            return False
        planned_kind = getattr(self, "_random_planned_kind", None)
        if (
            planned_kind is not None
            and str(selected.get("kind", "")).strip().lower()
            != planned_kind
        ):
            # The route was chosen from old inventory, but a full rolling
            # camera window found another requested category and no planned
            # one. Preserve that contradiction across the intervening table
            # delivery so the scheduler does not choose this cabinet again
            # for the same already-absent product.
            self._record_random_shelf_kind_visibility(
                self._random_last_live_candidates,
                {planned_kind},
                reason="rolling_planned_target_not_live",
            )
        rolling_pick = bool(
            self._nav_pick_handoff_mode == "rolling"
            or self._first_e_direct_active
        )
        if rolling_pick and self._return_stow_active:
            # At the shelf-specific handoff the next grasp takes priority over a
            # return-to-transport sequence that has not quite completed.  The
            # old serial gate let Nav2 reach and stop at the observation pose
            # before the arm was permitted to deploy.  Stop advancing that
            # obsolete arm target and command the concrete pick target below;
            # base ownership is handed off in the same control tick.
            self._return_stow_active = False
            self._return_stow_navigation_active = False
            self._return_stow_started_at = None
            self._return_arm_stage_ready_since = None
            self._return_arms_stowed = False
            self.get_logger().info(
                "RETURN_STOW_PREEMPTED_BY_ROLLING_PICK "
                f"shelf={self._active_shelf} stop_command=false"
            )
        elif self._return_stow_active:
            return False
        # A rolling pick immediately releases Nav2 and may safely retain the
        # conservative carry footprint for its last control tick.  Only a
        # non-rolling legacy path requires the normal footprint here.
        if not rolling_pick and (
            self._return_nav_footprint_restore_active
            or self._footprint_mode != "normal"
        ):
            return False
        self._activate_random_candidate(selected, transition=False)
        if (
            getattr(self, "_random_post_first_abc_scan_pending", False)
            and self._active_shelf in {"A", "B", "C"}
        ):
            self._random_post_first_abc_scan_pending = False
            self.get_logger().info(
                "RANDOM_POST_FIRST_ABC_OBSERVATION_COMPLETE "
                f"shelf={self._active_shelf} mode=rolling_inventory"
            )
        if not self._command_pick_template():
            return False
        self._nav_pick_prepared = True
        coarse_controller = (
            "direct_cmd_vel"
            if self._first_e_direct_active
            else "nav2"
        )
        self.get_logger().info(
            "RANDOM_PICK_PREDEPLOY_START "
            f"policy={self._nav_pick_handoff_mode} "
            f"shelf={self._active_shelf} "
            f"kind={self.target_kind} "
            f"id={selected['aruco_id']} "
            f"base_controller={coarse_controller} continues=true"
        )
        return True

    def _should_use_first_e_direct_drive(self, shelf: str) -> bool:
        """Use direct velocity control only for the first outbound E leg."""

        return bool(
            self._random_shelf_mode
            and self._picked_count == 0
            and str(shelf).upper() == "E"
            and not self._first_e_direct_used
        )

    def _start_first_e_direct_drive(
        self, pose: tuple[float, float, float]
    ) -> bool:
        """Take base ownership without creating a Nav2 goal."""

        if self.nav2.is_active:
            return False
        self.nav2.cancel()
        self.nav2.stop_robot()
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0] = self.tc[1] = 0.0
        self.base_control_enabled = True
        self._first_e_direct_used = True
        self._first_e_direct_active = True
        self._nav_goal_distance_m = None
        self._nav_goal_yaw_error_rad = None
        self._nav_near_goal_since = None
        self._nav_near_goal_leg = "shelf_e_scan_direct_cmd_vel"
        self._nav_terminal_hold_commanded = False
        self.get_logger().info(
            "FIRST_E_DIRECT_DRIVE_START "
            f"goal=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) "
            f"cruise={FIRST_E_DIRECT_CRUISE_SPEED_MPS:.2f}m/s "
            "nav2=false"
        )
        self.mission.transition(
            CycleState.WAIT_E_SCAN_NAV,
            "first_e_direct_velocity_control",
        )
        return True

    def _update_first_e_direct_drive(
        self, pose: tuple[float, float, float]
    ) -> None:
        """Drive quickly toward E, then hand straight to the visual grasp."""

        pose = self._update_first_e_direct_column_target(pose)
        linear, angular, distance, yaw_error, arrived = (
            _first_e_direct_velocity_command(
                self.base_xy, self.base_yaw, pose
            )
        )
        self._nav_goal_distance_m = distance
        self._nav_goal_yaw_error_rad = yaw_error

        if (
            distance <= FIRST_PICK_PREDEPLOY_DISTANCE_M
            and yaw_error <= RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD
        ):
            self._prepare_random_pick_during_navigation()

        if self._nav_pick_prepared:
            self._track_pick_head()
            incoming_speed = float(
                getattr(self, "odom_linear_speed", self.cur_lin)
            )
            if not math.isfinite(incoming_speed):
                incoming_speed = float(self.cur_lin)
            self._nav_pick_handoff_speed_mps = float(
                np.clip(
                    incoming_speed,
                    0.0,
                    FIRST_E_DIRECT_CRUISE_SPEED_MPS,
                )
            )
            self._first_e_direct_active = False
            self._fine_started_from_rolling_handoff = True
            self._fine_first_pick_early_handoff = True
            self.get_logger().info(
                "FIRST_E_DIRECT_VISUAL_HANDOFF "
                f"distance={distance:.3f}m "
                f"yaw_error={yaw_error:.3f}rad "
                f"speed={self._nav_pick_handoff_speed_mps:.3f}m/s "
                "stop_command=false"
            )
            self.mission.transition(
                CycleState.FINE_APPROACH,
                "first_e_half_metre_visual_pick",
            )
            return

        self._track_random_scan_head()
        # Direct-drive commands intentionally bypass the Baseline class's
        # legacy 0.45 m/s route clamp; the dedicated constants above bound this
        # one initial leg and ramp_twist() still enforces acceleration limits.
        self.des_lin = float(linear)
        self.des_ang = float(angular)
        if arrived and self._odom_stopped():
            self._first_e_direct_active = False
            self.get_logger().info(
                "FIRST_E_DIRECT_DRIVE_COMPLETE target_available=false"
            )
            self.mission.transition(
                CycleState.STOP_E_SCAN,
                "first_e_direct_observation_pose_reached",
            )

    def _update_first_e_direct_column_target(
        self, fallback_pose: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Lock the first E direct-drive goal to the selected product column."""

        if self._random_planned_marker_id is not None:
            return tuple(float(value) for value in fallback_pose)
        selected = self._active_shelf_pick_candidate()
        if selected is None or str(selected.get("shelf", "")).upper() != "E":
            return tuple(float(value) for value in fallback_pose)
        try:
            marker_id = int(selected["aruco_id"])
        except (KeyError, TypeError, ValueError):
            return tuple(float(value) for value in fallback_pose)

        pose = shelf_target_observation_pose(
            "E", marker_id, shift_edge_columns=True
        )
        self._random_planned_marker_id = marker_id
        self._random_planned_kind = str(selected.get("kind", "")).lower() or None
        self._random_active_scan_pose = pose
        self.get_logger().info(
            "FIRST_E_DIRECT_COLUMN_TARGET "
            f"marker_id={marker_id} "
            f"column={selected.get('column', 'unknown')} "
            f"goal=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f})"
        )
        return pose

    def _next_random_scan_shelf(self) -> str | None:
        for shelf in self._random_scan_order():
            if (
                shelf not in self._random_visited_shelves
                and shelf not in self._random_unreachable_shelves
            ):
                return shelf
        return None

    def _route_to_random_shelf(
        self,
        shelf: str,
        *,
        marker_id: int | None = None,
        planned_kind: str | None = None,
        start_return_stow: bool = False,
        stow_both_arms: bool = False,
    ) -> bool:
        shelf = str(shelf).strip().upper()
        if shelf not in SHELF_NAMES:
            return False
        # A rare Nav2 abort can happen after the next grasp posture was already
        # prepared.  Drop only that transient activation before routing to a
        # different shelf; persistent inventory remains available to reschedule.
        if self._nav_pick_prepared:
            self._current_target = None
            self._target_queue.clear()
            self.target_locked = False
            self.OBJECT_WORLD = None
            self.DEPLOY_WORLD = None
            self.CREEP_STOP_Y = None
            self.arm_target_set = False
        self._active_shelf = shelf
        self._random_planned_marker_id = marker_id
        self._random_planned_kind = (
            None
            if planned_kind is None
            else str(planned_kind).strip().lower()
        )
        self._nav_pick_prepared = False
        self._nav_pick_handoff_mode = self._random_pick_handoff_policy()
        self._nav_pick_handoff_speed_mps = 0.0
        self._random_scan_head_mode = None
        self._fine_started_from_rolling_handoff = False
        self._fine_first_pick_early_handoff = False
        self._pick_deploy_left_restore_active = False
        self._pick_deploy_left_restore_ready_since = None
        self._pick_deploy_held_slide_m = None
        # Every cabinet route gets an isolated camera window.  For rolling
        # legs product_cb fills this buffer while the base moves, so a missing
        # inventory-memory target can still be locked before the shelf-specific
        # handoff rather than forcing a stop at the observation point.
        self._search_points.clear()
        self._search_detection_frames = 0
        self._search_last_frame_at = None
        # Every route with a concrete inventory slot, including the first E
        # route, shifts the observation point toward C1/C3.  If the first route
        # starts before a marker is known, the direct-drive loop updates this
        # pose as soon as en-route vision identifies the selected product.
        pose = shelf_target_observation_pose(
            shelf,
            marker_id,
            shift_edge_columns=True,
        )
        self._random_active_scan_pose = pose
        fixed_pose = shelf_scan_pose(shelf)
        observation_x_shift = float(pose[0] - fixed_pose[0])
        self._random_last_decision = {
            "shelf": shelf,
            "aruco_id": marker_id,
            "kind": self._random_planned_kind,
            "phase": "first_cde" if self._picked_count == 0 else "after_first_abc",
            "pose": [float(value) for value in pose],
            "observation_x_shift_m": observation_x_shift,
        }
        self.get_logger().info(
            "RANDOM_SHELF_ROUTE "
            f"shelf={shelf} marker_id={marker_id} "
            f"kind={self._random_planned_kind} "
            f"observation_x_shift={observation_x_shift:+.3f}m "
            f"remaining={self._random_remaining_kinds} "
            f"visited={sorted(self._random_visited_shelves)}"
        )
        if (
            self._should_use_first_e_direct_drive(shelf)
            and self._start_first_e_direct_drive(pose)
        ):
            self._random_last_decision["controller"] = "direct_cmd_vel"
        else:
            self._send_goal(
                pose,
                f"shelf_{shelf.lower()}_scan",
                CycleState.WAIT_E_SCAN_NAV,
            )
            self._random_last_decision["controller"] = "nav2"
        if (
            start_return_stow
            and self.mission.state == CycleState.WAIT_E_SCAN_NAV
        ):
            self._start_concurrent_return_stow(
                both_arms_from_waiting=stow_both_arms
            )
        return self.mission.state == CycleState.WAIT_E_SCAN_NAV

    def _active_random_scan_pose(self) -> tuple[float, float, float]:
        """Return the exact observation goal used by the active shelf route."""

        pose = getattr(self, "_random_active_scan_pose", None)
        if pose is None:
            return shelf_scan_pose(self._active_shelf)
        return tuple(float(value) for value in pose)

    def _activate_random_candidate(
        self, selected: dict[str, Any], *, transition: bool = True
    ) -> None:
        marker_id = int(selected["aruco_id"])
        copied = self._copy_target(selected)
        self._known_targets[marker_id] = self._copy_target(copied)
        self._target_queue = deque((marker_id,))
        self._target_drop_slots[marker_id] = self._random_drop_slot_for_kind(
            str(copied["kind"])
        )
        self._queue_initialized = True
        self._set_active_target_kind(str(copied["kind"]))
        attempt = self._grasp_attempts.get(marker_id, 0) + 1
        self._grasp_attempts[marker_id] = attempt
        copied["grasp_attempt"] = attempt
        self._random_planned_marker_id = marker_id
        self._random_planned_kind = str(copied["kind"])
        self.get_logger().info(
            "RANDOM_TARGET_SELECTED "
            f"kind={copied['kind']} id={marker_id} "
            f"slot={copied.get('shelf')}/{copied['level']}/{copied['column']} "
            f"score={self._random_candidate_key(copied)[:2]} "
            f"source={copied['match_source']}"
        )
        self._activate_search_target(copied, transition=transition)

    def _random_drop_slot_for_kind(self, kind: str) -> int:
        """Reserve slot 3 for the first tissue in an otherwise ordered task."""

        normalized_kind = str(kind).strip().lower()
        mission_kinds = [
            str(item).strip().lower()
            for item in getattr(self, "_mission_kind_sequence", ())
        ]
        if TISSUE_KIND not in mission_kinds:
            return min(self._picked_count, len(TABLE_DROP_X_OFFSETS_M) - 1)

        remaining_kinds = [
            str(item).strip().lower()
            for item in getattr(self, "_random_remaining_kinds", ())
        ]
        completed_tissue_count = max(
            0,
            mission_kinds.count(TISSUE_KIND)
            - remaining_kinds.count(TISSUE_KIND),
        )
        if normalized_kind == TISSUE_KIND and completed_tissue_count == 0:
            return 2

        ordinary_slots = tuple(
            index
            for index in range(len(TABLE_DROP_X_OFFSETS_M))
            if index != 2
        )
        ordinary_completed_count = max(
            0,
            self._picked_count - min(completed_tissue_count, 1),
        )
        return ordinary_slots[
            min(ordinary_completed_count, len(ordinary_slots) - 1)
        ]

    @staticmethod
    def _table_drop_slots_for_sequence(kinds: list[str]) -> list[int]:
        """Build fixed-queue slots with the same tissue reservation policy."""

        normalized = [str(kind).strip().lower() for kind in kinds]
        if TISSUE_KIND not in normalized:
            return [
                min(index, len(TABLE_DROP_X_OFFSETS_M) - 1)
                for index in range(len(normalized))
            ]

        ordinary_slots = iter(
            index
            for index in range(len(TABLE_DROP_X_OFFSETS_M))
            if index != 2
        )
        tissue_slot_assigned = False
        slots: list[int] = []
        for kind in normalized:
            if kind == TISSUE_KIND and not tissue_slot_assigned:
                slots.append(2)
                tissue_slot_assigned = True
                continue
            slots.append(next(ordinary_slots, len(TABLE_DROP_X_OFFSETS_M) - 1))
        return slots

    def _schedule_random_target(
        self,
        live_candidates: list[dict[str, Any]] | None = None,
        *,
        start_return_stow: bool = False,
        already_at_scan: bool = False,
    ) -> bool:
        """Choose a shelf from memory, with live-vision fallback at the shelf."""

        # On the first post-delivery route prefer a concrete A/B/C product
        # already surveyed by the moving camera.  Keeping marker_id=None here
        # forced the robot to finish the B observation goal, stop, collect a
        # second stationary window and only then deploy.  A stable inventory
        # target can instead be refined continuously by the same live image
        # servo over the shelf-specific final segment.  If no usable A/B/C
        # memory exists, retain the central B observation as the discovery
        # fallback.
        if self._random_post_first_abc_scan_pending and not already_at_scan:
            remembered = self._select_random_candidate()
            if (
                remembered is not None
                and str(remembered.get("shelf", "")).upper()
                in {"A", "B", "C"}
            ):
                shelf = str(remembered["shelf"]).upper()
                marker_id = int(remembered["aruco_id"])
                self._random_post_first_abc_scan_pending = False
                self.get_logger().info(
                    "RANDOM_POST_FIRST_ABC_MEMORY_TARGET "
                    f"shelf={shelf} marker_id={marker_id}; "
                    "live_refinement_during_route=true"
                )
                return self._route_to_random_shelf(
                    shelf,
                    marker_id=marker_id,
                    planned_kind=str(remembered.get("kind", "")),
                    start_return_stow=start_return_stow,
                )
            for shelf in ("B", "A", "C"):
                if shelf not in self._random_unreachable_shelves:
                    return self._route_to_random_shelf(
                        shelf,
                        start_return_stow=start_return_stow,
                    )
            self._random_post_first_abc_scan_pending = False

        if already_at_scan:
            live_selected = self._select_live_candidate_at_active_shelf(
                live_candidates
            )
            if live_selected is not None:
                self._activate_random_candidate(live_selected)
                return True
            # A current frame is optional refinement, not proof of absence.
            # Prefer the exact planned slot, then any requested product that
            # the persistent ledger previously latched in this cabinet.
            shelf_memory = [
                item
                for item in self._random_candidates_with_live(live_candidates)
                if str(item.get("shelf", "")).upper() == self._active_shelf
            ]
            memory_selected = None
            if self._random_planned_marker_id is not None:
                memory_selected = next(
                    (
                        item
                        for item in shelf_memory
                        if int(item["aruco_id"])
                        == self._random_planned_marker_id
                    ),
                    None,
                )
            if memory_selected is None and shelf_memory:
                memory_selected = min(
                    shelf_memory,
                    key=self._random_candidate_key,
                )
            if memory_selected is not None:
                self.get_logger().info(
                    "RANDOM_PLANNED_MEMORY_RETAINED "
                    f"shelf={self._active_shelf} "
                    f"marker_id={memory_selected['aruco_id']} "
                    "live_visible=false proceed_to_visual_servo=true"
                )
                self._activate_random_candidate(memory_selected)
                return True
            self._random_planned_marker_id = None
            self._random_planned_kind = None

        selected = self._select_random_candidate(
            live_candidates,
            excluded_shelves=(
                self._random_visited_shelves if already_at_scan else None
            ),
        )
        if selected is not None:
            shelf = str(selected["shelf"])
            return self._route_to_random_shelf(
                shelf,
                marker_id=int(selected["aruco_id"]),
                planned_kind=str(selected.get("kind", "")),
                start_return_stow=(start_return_stow or already_at_scan),
                stow_both_arms=already_at_scan,
            )
        shelf = self._next_random_scan_shelf()
        if shelf is not None:
            return self._route_to_random_shelf(
                shelf,
                start_return_stow=(start_return_stow or already_at_scan),
                stow_both_arms=already_at_scan,
            )
        if self._random_sweep_retry_count < 1:
            self._random_sweep_retry_count += 1
            self._random_visited_shelves.clear()
            shelf = self._next_random_scan_shelf()
            if shelf is not None:
                self.get_logger().warning(
                    "RANDOM_SHELF_SWEEP_RETRY "
                    f"attempt={self._random_sweep_retry_count + 1}/2 "
                    f"remaining={self._random_remaining_kinds}"
                )
                return self._route_to_random_shelf(
                    shelf,
                    start_return_stow=(start_return_stow or already_at_scan),
                    stow_both_arms=already_at_scan,
                )
        self._fail(
            "random_required_products_not_found_after_full_shelf_sweep"
        )
        return False

    def _candidate_for_queue_target(
        self, candidates: list[dict[str, Any]], marker_id: int
    ) -> dict[str, Any] | None:
        """Match fresh vision to a queued slot, with position as a safe fallback."""

        remembered = self._known_targets.get(marker_id)
        if remembered is None:
            return None
        target_kind = str(remembered.get("kind", self.target_kind))
        exact = [
            item
            for item in candidates
            if int(item["aruco_id"]) == marker_id
            and str(item.get("kind", target_kind)) == target_kind
        ]
        if exact:
            return exact[0]
        expected = np.asarray(remembered["product_world"], dtype=float)
        nearby = [
            (
                float(
                    np.linalg.norm(
                        np.asarray(item["product_world"], dtype=float) - expected
                    )
                ),
                item,
            )
            for item in candidates
            if str(item.get("kind", target_kind)) == target_kind
        ]
        if nearby:
            distance, nearest = min(nearby, key=lambda pair: pair[0])
            if distance <= TARGET_MEMORY_MATCH_RADIUS_M:
                return nearest
        return None

    def _remember_fresh_target(
        self, marker_id: int, fresh: dict[str, Any]
    ) -> dict[str, Any]:
        """Update queued coordinates without allowing a noisy inference to change slots."""

        remembered = self._known_targets[marker_id]
        target_kind = str(remembered.get("kind", self.target_kind))
        updated = self._copy_target(fresh)
        updated["aruco_id"] = marker_id
        updated["kind"] = target_kind
        updated["level"] = remembered["level"]
        updated["column"] = remembered["column"]
        updated["product_world"][2] = e_product_center_z(
            updated["level"], target_kind
        )
        updated["surface_world"][2] = updated["product_world"][2]
        if int(fresh["aruco_id"]) != marker_id:
            updated["match_source"] = "queue_position_matched"
        self._known_targets[marker_id] = self._copy_target(updated)
        return updated

    def _activate_queue_head(
        self, fresh: dict[str, Any] | None = None
    ) -> None:
        if not self._target_queue:
            self.mission.transition(CycleState.DONE, "persistent_queue_empty")
            return
        marker_id = int(self._target_queue[0])
        if fresh is not None:
            selected = self._remember_fresh_target(marker_id, fresh)
        else:
            selected = self._copy_target(self._known_targets[marker_id])
        self._set_active_target_kind(
            str(selected.get("kind", self.target_kind))
        )
        attempt = self._grasp_attempts.get(marker_id, 0) + 1
        self._grasp_attempts[marker_id] = attempt
        selected["grasp_attempt"] = attempt
        self.get_logger().info(
            f"{self._event_prefix}_SELECTED "
            f"source={selected['match_source']} id={selected['aruco_id']} "
            f"slot={selected.get('shelf', 'E')}/"
            f"{selected['level']}/{selected['column']} "
            f"world={np.round(selected['product_world'], 3).tolist()} "
            f"samples={selected['samples']} attempt={attempt}/{GRASP_RETRY_LIMIT}"
        )
        self._activate_search_target(selected)

    def _activate_first_visible_queue_target(
        self, candidates: list[dict[str, Any]]
    ) -> bool:
        """Select a queued product only after the fixed observation sees it.

        A later visible queue entry may be chosen before an occluded head entry;
        the deque is reordered so verification still applies to the product that
        is actually being handled.
        """

        queue_ids = (
            tuple(self._target_queue)[:1]
            if self._mixed_mode
            else tuple(self._target_queue)
        )
        for marker_id in queue_ids:
            fresh = self._candidate_for_queue_target(candidates, int(marker_id))
            if fresh is None:
                continue
            if int(self._target_queue[0]) != int(marker_id):
                self._target_queue.remove(marker_id)
                self._target_queue.appendleft(marker_id)
                self.get_logger().info(
                    f"{self._event_prefix}_QUEUE_REORDER visible_id={marker_id} "
                    f"queue={list(self._target_queue)}"
                )
            self._activate_queue_head(fresh)
            return True
        return False

    def _activate_queued_target_with_memory_fallback(
        self, candidates: list[dict[str, Any]]
    ) -> bool:
        """Activate a queued target without treating occlusion as completion.

        The initial multi-frame observation freezes an ArUco-backed world pose
        for every queued product.  On later returns an arm, shelf edge, or a
        slightly low torso can temporarily hide the strict mixed-sequence head
        while another product remains visible.  Fresh vision is preferred, but
        absence from this one observation window is not evidence that the
        remaining mission is complete.
        """

        if self._activate_first_visible_queue_target(candidates):
            return True
        if not self._target_queue:
            return False
        marker_id = int(self._target_queue[0])
        remembered = self._known_targets.get(marker_id)
        if remembered is None:
            self._fail(f"queued_target_{marker_id}_memory_missing")
            return True
        self.get_logger().warning(
            f"{self._event_prefix}_QUEUE_HEAD_MEMORY_FALLBACK "
            f"id={marker_id} kind={remembered.get('kind')} "
            f"slot=E/{remembered.get('level')}/{remembered.get('column')}; "
            "not visible in the return observation, continuing from the "
            "initial multi-frame lock"
        )
        self._activate_queue_head()
        return True

    def _initialize_target_queue(
        self, candidates: list[dict[str, Any]]
    ) -> None:
        """Freeze the first full E scan into the persistent mission queue."""

        self._known_targets = {
            int(candidate["aruco_id"]): self._copy_target(candidate)
            for candidate in candidates
        }
        self._initial_discovered_count = len(candidates)
        required = self._required_target_count
        if self._mixed_mode:
            available = list(candidates)
            active = []
            for desired_kind in self._mission_kind_sequence:
                matches = [
                    item
                    for item in available
                    if str(item.get("kind", "")) == desired_kind
                ]
                if not matches:
                    self.get_logger().warning(
                        f"{self._mission_event_prefix}_QUEUE_MISSING_KIND "
                        f"kind={desired_kind} sequence={self._mission_kind_sequence}"
                    )
                    continue
                # Repeated kinds are consumed bottom-to-top by ArUco ID.  This
                # gives every deterministic mixed test a stable identity even
                # when multiple requested products share a class.
                selected_index = min(
                    (
                        index
                        for index, item in enumerate(available)
                        if str(item.get("kind", "")) == desired_kind
                    ),
                    key=lambda index: (
                        int(available[index]["aruco_id"]),
                        available[index]["match_source"]
                        != "aruco_confirmed",
                    ),
                )
                selected = available.pop(selected_index)
                active.append(selected)
        else:
            active_count = (
                len(candidates)
                if required is None
                else min(required, len(candidates))
            )
            active = candidates[:active_count]
        drop_slots = self._table_drop_slots_for_sequence(
            [str(item.get("kind", "")) for item in active]
        )
        self._target_drop_slots = {
            int(item["aruco_id"]): drop_slots[index]
            for index, item in enumerate(active)
        }
        self._target_queue = deque(int(item["aruco_id"]) for item in active)
        self._mission_target_count = len(self._target_queue)
        self._queue_initialized = True

        requested_text = "all_discovered" if required is None else str(required)
        self.get_logger().info(
            f"{self._event_prefix}_QUEUE_READY requested={requested_text} "
            f"discovered={len(candidates)} active={len(self._target_queue)} "
            f"kinds={[item.get('kind') for item in active]} "
            f"ids={list(self._target_queue)} "
            f"table_slots={{"
            f"{', '.join(f'{marker_id}:{slot + 1}' for marker_id, slot in self._target_drop_slots.items())}"
            f"}}"
        )
        if required is not None and len(active) < required:
            self.get_logger().warning(
                f"{self._event_prefix}_QUEUE_SHORTFALL requested={required} "
                f"matched={len(active)} discovered={len(candidates)}; "
                "finish after matched targets"
            )
        if not self._target_queue:
            reason = (
                f"no_required_{self.target_kind}"
                if required == 0
                else f"no_{self.target_kind}_discovered"
            )
            self.mission.transition(CycleState.DONE, reason)
            return
        self._activate_queue_head()

    def _activate_search_target(
        self, selected: dict[str, Any], *, transition: bool = True
    ) -> None:
        """Lock the observation geometry and start a direct visual approach."""

        self._chengzi_grasp_center_in_hand = None
        self._chengzi_place_initial_slide = None
        self._chengzi_place_heights.clear()
        self._chengzi_place_feedback = {}
        self._current_target = selected
        self._search_mode = "target_locked"
        self._active_search_poses = []
        self._stage_ready_since = None
        self._target_last_seen_at = (
            None
            if str(selected.get("match_source", "")) == "inventory_memory"
            else (self._search_last_frame_at or self.now())
        )
        self._fine_vision_lost_at = None
        self._fine_guidance_source = "observation_memory"
        self._grasp_insertion_m = None
        self._fine_best_lateral_error_m = None
        self._fine_terminal_heading_target_rad = None
        self._fine_terminal_heading_error_rad = None
        self._fine_lateral_error_m = None
        # 每个商品必须从“尚未计算前进余量”开始。否则第二个商品会沿用上一个
        # 商品结束时约 1 cm 的余量，在第一帧就误判进入末端遮挡区并锁掉视觉。
        self._fine_target_forward_m = None
        self._fine_forward_remaining_m = None
        self._fine_forward_error_m = None
        center = np.asarray(selected["product_world"], dtype=float)
        level = str(selected.get("level", ""))
        observation_visual = np.asarray(
            selected.get("observation_visual_product_world", center),
            dtype=float,
        ).copy()
        # Only visual X is consumed by the steering loop.  The slot-validated
        # shelf depth and intrinsic product height remain fixed.
        observation_visual[0] = float(
            np.clip(
                observation_visual[0],
                center[0] - LIVE_LATERAL_MAX_SLOT_OFFSET_M,
                center[0] + LIVE_LATERAL_MAX_SLOT_OFFSET_M,
            )
        )
        observation_visual[1:] = center[1:]
        observation_visual[0] = _grasp_lateral_target_x(
            self.target_kind, level, center[0]
        )
        self._fine_observation_target_world = observation_visual.copy()
        self._fine_last_live_target_world = None
        self._fine_last_vision_update_at = None
        self._fine_live_updates_frozen = False
        self._target_slide = grasp_slide_for_product_z(
            center[2], self.target_kind, level
        )
        # Observation itself is fixed.  Only after choosing a visible target do
        # the torso and head move to the geometry-checked tracking posture.
        self.tc[2] = self._target_slide
        self._reacquire_points.clear()
        self._image_reacquire_points.clear()
        self._fine_pixel_servo_active = False
        self._fine_product_pixel = None
        self._fine_gripper_pixel = None
        self._fine_pixel_error_px = None
        self._fine_pixel_error_raw_px = None
        self._fine_pixel_correction_radps = None
        self._fine_pixel_linear_scale = 1.0
        self._fine_pixel_observation_at = None
        self._fine_pixel_filter_observation_at = None
        self.OBJECT_WORLD = observation_visual.copy()
        self._track_pick_head()
        self._grasp_insertion_m = _grasp_insertion_for_product(
            self.target_kind, level
        )
        self.CREEP_STOP_Y = float(center[1] + self._grasp_insertion_m)
        # Only back far enough to clear the shelf before folding both arms.
        # Returning to the full observation pose here added unnecessary travel
        # to every pick-table cycle.
        self._pick_retreat_y = (
            TISSUE_PICK_RETREAT_Y_M
            if self._uses_tissue_two_hand_grasp
            else PICK_RETREAT_CLEAR_Y_M
        )
        self.target_locked = True
        self.get_logger().info(
            f"{self._event_prefix}_DIRECT_PICK_FROM_OBSERVATION "
            f"shelf={selected.get('shelf', self._active_shelf)} "
            f"slot_world={np.round(center, 3).tolist()} "
            f"visual_guidance_world="
            f"{np.round(observation_visual, 3).tolist()} "
            f"stop_y={self.CREEP_STOP_Y:.3f} "
            f"insertion={self._grasp_insertion_m:.3f}; "
            f"retreat_y={self._pick_retreat_y:.3f}; "
            "per-product Nav2 waypoint disabled"
        )
        if transition:
            self.mission.transition(
                CycleState.DEPLOY_TEMPLATE, "direct_from_observation"
            )

    def _retry_pending_if_visible(
        self, candidates: list[dict[str, Any]]
    ) -> bool:
        """Retry the queue head immediately when the just-grasped bottle remains."""

        pending = self._pending_verification
        if pending is None:
            return False
        marker_id = int(pending["aruco_id"])
        fresh = self._candidate_for_queue_target(candidates, marker_id)
        if fresh is None:
            return False
        attempt = int(pending["grasp_attempt"])
        self._pending_verification = None
        if attempt >= GRASP_RETRY_LIMIT:
            self._fail(
                f"{self.target_kind}_id_{marker_id}_still_present_after_"
                f"{attempt}_attempts"
            )
            return True
        self.get_logger().warning(
            f"{self._event_prefix}_GRASP_NOT_CONFIRMED id={marker_id} "
            f"attempt={attempt}/{GRASP_RETRY_LIMIT}; bottle still visible"
        )
        self._activate_queue_head(fresh)
        return True

    def _confirm_pending_removed(self) -> bool:
        """Pop a delivered target after its complete return scan finds no bottle."""

        pending = self._pending_verification
        if pending is None:
            return False
        marker_id = int(pending["aruco_id"])
        attempt = int(pending["grasp_attempt"])
        self._pending_verification = None
        if self._target_queue and int(self._target_queue[0]) == marker_id:
            self._target_queue.popleft()
        else:
            try:
                self._target_queue.remove(marker_id)
            except ValueError:
                pass
        self._known_targets.pop(marker_id, None)
        self._picked_count += 1
        self.get_logger().info(
            f"{self._event_prefix}_REMOVAL_CONFIRMED count={self._picked_count}/"
            f"{self._mission_target_count} id={marker_id} "
            f"after_attempt={attempt} remaining={list(self._target_queue)}"
        )
        if (
            self._picked_count >= self._mission_target_count
            or not self._target_queue
        ):
            self.mission.transition(CycleState.DONE, "persistent_queue_complete")
            return True
        return False

    def _pending_delivery_is_final(self) -> bool:
        """Return whether the released item is the last queued mission item."""

        pending = self._pending_verification
        if pending is None or self._mission_target_count <= 0:
            return False
        if getattr(self, "_random_shelf_mode", False):
            return self._picked_count + 1 >= self._mission_target_count
        marker_id = int(pending["aruco_id"])
        return bool(
            len(self._target_queue) == 1
            and int(self._target_queue[0]) == marker_id
            and self._picked_count + 1 >= self._mission_target_count
        )

    def _confirm_random_delivery_at_table(self) -> bool:
        """Consume a category/count target after its completed table release."""

        pending = self._pending_verification
        if pending is None:
            return False
        marker_id = int(pending["aruco_id"])
        kind = str(pending.get("kind", ""))
        attempt = int(pending["grasp_attempt"])
        remembered = self._known_targets.get(marker_id, {})
        removed_target = {**remembered, **pending}
        self._pending_verification = None
        if self._target_queue and int(self._target_queue[0]) == marker_id:
            self._target_queue.popleft()
        self._known_targets.pop(marker_id, None)
        self._random_completed_marker_ids.add(marker_id)
        self._remove_random_target_from_visual_inventory(removed_target)
        try:
            self._random_remaining_kinds.remove(kind)
        except ValueError:
            self.get_logger().error(
                f"RANDOM_DELIVERY_KIND_MISMATCH id={marker_id} kind={kind} "
                f"remaining={self._random_remaining_kinds}"
            )
            return False
        self._picked_count += 1
        self._random_planned_marker_id = None
        self._random_planned_kind = None
        if self._picked_count == 1 and self._random_remaining_kinds:
            self._random_post_first_abc_scan_pending = True
            self._random_post_first_abc_transit_scan_active = True
        # A new delivery leg starts a new reachability sweep.  Persistent
        # inventory remains available, but shelves that were blocked from the
        # previous robot pose may now be reachable from the table side.
        self._random_visited_shelves.clear()
        self._random_unreachable_shelves.clear()
        self._random_sweep_retry_count = 0
        self.get_logger().info(
            "RANDOM_DELIVERY_CONFIRMED "
            f"count={self._picked_count}/{self._mission_target_count} "
            f"kind={kind} id={marker_id} after_attempt={attempt} "
            f"remaining={self._random_remaining_kinds} "
            "source=table_release_lift_and_retreat"
        )
        return True

    def _start_navigation_after_table_retreat(self) -> bool:
        """Hand the cleared base directly from fixed reverse to Nav2.

        The caller has already reached the table-exit distance and forced the
        locally published Twist to zero.  Do not add an odometry-still dwell
        for non-final deliveries: Nav2 receives the next goal immediately and
        uses the current measured velocity as its initial state.
        """

        if self._uses_tissue_two_hand_grasp:
            # STOP_TABLE normally restores both controller limits before the
            # placement motion starts. Reassert the unlimited linear command
            # and verify normal MPPI angular authority once more at the exact
            # handoff to the next product, so no tissue carry limit can leak
            # into the following Nav2 route.
            pending = self._pending_verification
            if (
                isinstance(pending, dict)
                and not pending.get("normal_nav_limits_reasserted")
            ):
                self._set_navigation_speed_limit(0.0)
                pending["normal_nav_limits_reasserted"] = True
            angular_result = self._poll_controller_angular_limit()
            if angular_result is False:
                self._fail("post_tissue_angular_restore_rejected")
                return False
            if angular_result is None:
                return False
            if abs(
                self._controller_angular_limit_current
                - NORMAL_MPPI_MAX_ANGULAR_RADPS
            ) > 1e-9:
                if not self._begin_controller_angular_limit(
                    NORMAL_MPPI_MAX_ANGULAR_RADPS
                ):
                    self._fail("post_tissue_angular_restore_unavailable")
                return False
            self.get_logger().info(
                "TISSUE_NAV_LIMITS_RESTORED "
                "linear_limit=unlimited "
                f"wz_max={NORMAL_MPPI_MAX_ANGULAR_RADPS:.2f}rad/s "
                "next_product_nav2=true"
            )

        self._release_baseline_base()
        if getattr(self, "_random_shelf_mode", False):
            if not self._confirm_random_delivery_at_table():
                self._fail("random_delivery_state_invalid")
                return False
            self._schedule_random_target(
                start_return_stow=True,
                already_at_scan=False,
            )
        else:
            # Fixed E-cabinet regression mode still returns to E for removal
            # verification, but starts that Nav2 action in this same tick.
            self._send_goal(
                E_SCAN_POSE,
                "e_shelf_scan",
                CycleState.WAIT_E_SCAN_NAV,
            )
            if self.mission.state == CycleState.WAIT_E_SCAN_NAV:
                self._start_concurrent_return_stow()
        if self.mission.state == CycleState.WAIT_E_SCAN_NAV:
            self.get_logger().info(
                "TABLE_RETREAT_NAV2_IMMEDIATE_HANDOFF "
                "nav2=true torso=true arms=true stop_dwell=false"
            )
            return True
        return False

    def _confirm_final_delivery_at_table(self) -> bool:
        """Complete the last delivery without making another E-shelf trip.

        Earlier deliveries retain the return observation and retry check.  The
        final item has no next pick, so after its verified release/lift/retreat
        sequence the mission intentionally accepts that delivery at the table.
        """

        if not self._pending_delivery_is_final():
            return False
        if getattr(self, "_random_shelf_mode", False):
            return self._confirm_random_delivery_at_table()
        pending = self._pending_verification
        marker_id = int(pending["aruco_id"])
        attempt = int(pending["grasp_attempt"])
        self._pending_verification = None
        self._target_queue.popleft()
        self._known_targets.pop(marker_id, None)
        self._picked_count += 1
        self.get_logger().info(
            f"{self._event_prefix}_FINAL_DELIVERY_CONFIRMED "
            f"count={self._picked_count}/{self._mission_target_count} "
            f"id={marker_id} after_attempt={attempt} "
            "source=table_release_lift_and_retreat"
        )
        return True

    def _configure_e_search(self) -> None:
        """Hold one fixed observation posture before every pick decision."""

        self._search_points.clear()
        self._search_pose_index = 0
        self._stage_ready_since = None
        self._search_collecting = False
        self._search_detection_frames = 0
        self._search_last_frame_at = None
        if self._random_shelf_mode:
            self._search_mode = "random_shelf_observation"
        elif not self._queue_initialized:
            self._search_mode = "initial_observation"
        elif self._pending_verification is not None:
            self._search_mode = "verify_delivery_observation"
        elif self._target_queue:
            self._search_mode = "queue_observation"
        else:
            self._active_search_poses = []
            self.mission.transition(CycleState.DONE, "persistent_queue_empty")
            return

        # Do not sweep the torso or head at the observation point.  The default
        # 0.30 m slide puts all three E-row centres inside the camera FOV; an
        # explicit SUPERMARKET_SCAN_SLIDE override remains fixed here as well.
        self._active_search_poses = [
            (float(self.scan_slide), float(OBSERVATION_HEAD_PITCH))
        ]

        slide, pitch = self._active_search_poses[0]
        self.tc[2] = slide
        self.tc[3] = 0.0
        self.tc[4] = pitch
        self.tc[11] = GRIP_OPEN
        self.tc[18] = GRIP_OPEN
        # Camera/torso observation and the collision-checked two-arm restore
        # begin in the same state.  Target selection waits for both results,
        # but neither operation serially blocks the other.
        self._start_concurrent_search_restore()
        self.get_logger().info(
            f"{self._event_prefix}_SEARCH_START mode={self._search_mode} "
            f"poses={self._active_search_poses} "
            f"required_frames={SEARCH_MIN_DETECTION_FRAMES} "
            f"queue={list(self._target_queue)} arms_concurrent=true"
        )

    def _process_e_search_pose(self) -> None:
        candidates = self._search_candidates()

        if self._random_shelf_mode:
            self._random_visited_shelves.add(self._active_shelf)
            self._record_random_shelf_kind_visibility(
                candidates,
                set(self._random_remaining_kinds),
                reason="stationary_shelf_observation",
            )
            if (
                self._random_post_first_abc_scan_pending
                and self._active_shelf in {"A", "B", "C"}
            ):
                self._random_post_first_abc_scan_pending = False
                self.get_logger().info(
                    "RANDOM_POST_FIRST_ABC_OBSERVATION_COMPLETE "
                    f"shelf={self._active_shelf}"
                )
            self._initial_discovered_count = max(
                self._initial_discovered_count,
                len(self._random_inventory_candidates()),
            )
            self.get_logger().info(
                "RANDOM_SHELF_OBSERVED "
                f"shelf={self._active_shelf} live={len(candidates)} "
                f"inventory={len(self._random_inventory_candidates())} "
                f"remaining={self._random_remaining_kinds}"
            )
            self._schedule_random_target(
                candidates,
                already_at_scan=True,
            )
            return

        if self._search_mode == "initial_observation":
            self._initialize_target_queue(candidates)
            return

        if self._search_mode == "verify_delivery_observation":
            if self._retry_pending_if_visible(candidates):
                return
            if self._confirm_pending_removed():
                return

            # Prefer a fresh match, but keep the strict mixed-task order and
            # continue from the initial ArUco/slot lock when the next product
            # is temporarily outside the return-observation image.
            if not self._activate_queued_target_with_memory_fallback(
                candidates
            ):
                self.mission.transition(CycleState.DONE, "persistent_queue_empty")
            return

        if self._search_mode == "queue_observation":
            if not self._target_queue:
                self.mission.transition(CycleState.DONE, "persistent_queue_empty")
                return
            if not self._activate_queued_target_with_memory_fallback(
                candidates
            ):
                self.mission.transition(CycleState.DONE, "persistent_queue_empty")


__all__ = ["EProductCycleSchedulerMixin"]
