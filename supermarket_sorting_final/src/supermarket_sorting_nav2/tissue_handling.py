"""Two-arm tissue grasp, transport and release helpers."""

from __future__ import annotations

import math
import numpy as np

from .baseline_grasp_controller import (
    GRIP_CLOSE,
    GRIP_OPEN,
)
from .navigation.sorting_geometry import (
    GRASP_YAW,
    HAND_Z_PLUS_SLIDE_M,
    SLIDE_MIN_M,
    TABLE_TOP_Z_M,
)

from .sorting_config import (
    BROAD_RELEASE_COMMAND_OPEN_MIN,
    PLACE_RELEASE_JOINT_SLEW,
    POST_RELEASE_LIFT_EXEMPT_KINDS,
    SLIDE_TOL_M,
    TISSUE_ARM_TOL_RAD,
    TISSUE_CLAMP_HALF_SEPARATION_M,
    TISSUE_DEPLOY_HALF_SEPARATION_M,
    TISSUE_GRASP_YAW_RAD,
    TISSUE_HAND_CENTER_X_M,
    TISSUE_KIND,
    TISSUE_LOADED_JOINT_SLEW,
    TISSUE_MOTION_TIMEOUT_SEC,
    TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF,
    TISSUE_RELEASE_MIN_SEPARATION_M,
    TISSUE_STAGE_SETTLE_SEC,
    TISSUE_TRANSPORT_BOTTOM_CLEARANCE_M,
)


def _tissue_preturn_signed_angle_rad(shelf: str) -> float:
    """Return the required signed shelf-exit turn for a tissue.

    Positive angles turn left and negative angles turn right.  Rejecting an
    unknown shelf is intentional: silently falling back to E could send the
    wide two-arm carry pose into the wrong wall.
    """

    normalized = str(shelf).strip().upper()
    try:
        return float(TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF[normalized])
    except KeyError as exc:
        raise ValueError(f"unsupported tissue source shelf: {shelf!r}") from exc


class EProductCycleTissueMixin:
    @property
    def _uses_tissue_two_hand_grasp(self) -> bool:
        return self.target_kind == TISSUE_KIND

    def _released_product_kind(self) -> str:
        """Return the item being released even after current-target cleanup."""

        pending = getattr(self, "_pending_verification", None)
        if isinstance(pending, dict) and pending.get("kind"):
            return str(pending["kind"]).strip().lower()
        current = getattr(self, "_current_target", None)
        if isinstance(current, dict) and current.get("kind"):
            return str(current["kind"]).strip().lower()
        return str(self.target_kind).strip().lower()

    def _post_release_lift_required(self) -> bool:
        return self._released_product_kind() not in POST_RELEASE_LIFT_EXEMPT_KINDS

    def _hold_direct_retreat_release_pose(self) -> None:
        """Keep a lift-exempt product at table height while backing away."""

        if self._post_release_lift_required():
            return
        self.joint_slew = PLACE_RELEASE_JOINT_SLEW
        self.tc[2] = float(self._table_drop_slide_m)
        self.tc[18] = GRIP_OPEN

    def _target_grasp_yaw(self) -> float:
        # The common single-hand template uses an 11-degree diagonal so its
        # small lateral offset points at the target from the observation X.
        # A symmetric two-arm clamp must finish square to the shelf because
        # both arms contact the package sides.  A sandwich stays image-centred with the common
        # single-hand yaw and freezes that observed heading only at handoff.
        if self._uses_tissue_two_hand_grasp:
            return TISSUE_GRASP_YAW_RAD
        return GRASP_YAW

    def _both_arms_at_target(
        self, targets: tuple[np.ndarray, np.ndarray] | None
    ) -> bool:
        if targets is None or self.jpos is None:
            return False
        left, right = targets
        return bool(
            float(np.max(np.abs(self.larm_meas - left)))
            < TISSUE_ARM_TOL_RAD
            and float(np.max(np.abs(self.rarm_meas - right)))
            < TISSUE_ARM_TOL_RAD
        )

    @property
    def _left_gripper_is_open(self) -> bool:
        measured = self.jpos.get(
            "left_arm_eef_gripper_joint", self.tc[11]
        )
        return float(measured) >= BROAD_RELEASE_COMMAND_OPEN_MIN

    def _tissue_release_actuators_ready(
        self, actual_separation: float
    ) -> bool:
        """Accept release once the two closed hands have moved far enough apart."""

        return bool(float(actual_separation) >= TISSUE_RELEASE_MIN_SEPARATION_M)

    @property
    def _tissue_grippers_closed(self) -> bool:
        """Require both empty jaws to close before approaching the tissue.

        Tissue clamping is produced by moving the complete hands inward, not
        by closing the individual jaws.  Waiting for both broad contact pads
        to form first makes their geometry deterministic and prevents a still
        opening finger from reaching the shelf before the other hand.
        """

        if self.jpos is None:
            return False
        left = self.jpos.get(
            "left_arm_eef_gripper_joint", self.tc[11]
        )
        right = self.jpos.get(
            "right_arm_eef_gripper_joint", self.tc[18]
        )
        return bool(
            max(
                float(left),
                float(right),
                float(self.tc[11]),
                float(self.tc[18]),
            )
            <= 0.15
        )

    def _tissue_hand_poses(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        left, right = self.kdl.forward_kinematics(
            np.concatenate(
                [
                    [float(self.slide_meas)],
                    np.asarray(self.larm_meas, dtype=float),
                    np.asarray(self.rarm_meas, dtype=float),
                ]
            )
        )
        return left, right

    def _tissue_midpoint_footprint(self) -> np.ndarray:
        left, right = self._tissue_hand_poses()
        return 0.5 * (
            np.asarray(left[:3, 3], dtype=float)
            + np.asarray(right[:3, 3], dtype=float)
        )

    def _grasp_reference_world(self) -> np.ndarray:
        if self._uses_tissue_two_hand_grasp:
            return self.footprint_to_world(
                self._tissue_midpoint_footprint()
            )
        return self.ee_world()

    @staticmethod
    def _tissue_roll_rotation(roll_rad: float) -> np.ndarray:
        cosine = math.cos(float(roll_rad))
        sine = math.sin(float(roll_rad))
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine, -sine],
                [0.0, sine, cosine],
            ],
            dtype=float,
        )

    def _solve_tissue_arm_targets(
        self,
        *,
        slide: float,
        center_x: float,
        center_z: float,
        half_separation: float,
        roll_rad: float,
        reference: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Solve a symmetric rigid two-hand tissue pose in base_footprint."""

        rotation = self._tissue_roll_rotation(roll_rad)
        offset = rotation @ np.array(
            [0.0, float(half_separation), 0.0], dtype=float
        )
        center = np.array([float(center_x), 0.0, float(center_z)])
        left_pose = np.eye(4)
        right_pose = np.eye(4)
        left_pose[:3, :3] = rotation
        right_pose[:3, :3] = rotation
        left_pose[:3, 3] = center + offset
        right_pose[:3, 3] = center - offset
        if reference is None:
            reference = (
                np.asarray(self.larm_meas, dtype=float),
                np.asarray(self.rarm_meas, dtype=float),
            )
        ref_pos = np.concatenate(
            [[float(slide)], reference[0], reference[1]]
        )
        solutions = self.kdl.inverse_kinematics(
            T_left=left_pose,
            T_right=right_pose,
            ref_pos=ref_pos,
            target_height=float(slide),
        )
        if not solutions:
            return None
        solution = np.asarray(solutions[0], dtype=float)
        targets = (solution[1:7].copy(), solution[7:13].copy())
        solved_left, solved_right = self.kdl.forward_kinematics(solution)
        position_error = max(
            float(
                np.linalg.norm(
                    solved_left[:3, 3] - left_pose[:3, 3]
                )
            ),
            float(
                np.linalg.norm(
                    solved_right[:3, 3] - right_pose[:3, 3]
                )
            ),
        )
        if position_error > 0.005:
            self.get_logger().error(
                "TISSUE_DUAL_IK rejected position error "
                f"{position_error:.4f}m"
            )
            return None
        return targets

    def _prepare_tissue_grasp_arms(self) -> bool:
        center_z = HAND_Z_PLUS_SLIDE_M - float(self._target_slide)
        deploy = self._solve_tissue_arm_targets(
            slide=float(self._target_slide),
            center_x=TISSUE_HAND_CENTER_X_M,
            center_z=center_z,
            half_separation=TISSUE_DEPLOY_HALF_SEPARATION_M,
            roll_rad=0.0,
        )
        if deploy is None:
            return False
        clamp = self._solve_tissue_arm_targets(
            slide=float(self._target_slide),
            center_x=TISSUE_HAND_CENTER_X_M,
            center_z=center_z,
            half_separation=TISSUE_CLAMP_HALF_SEPARATION_M,
            roll_rad=0.0,
            reference=deploy,
        )
        if clamp is None:
            return False
        self._tissue_deploy_arms = deploy
        self._tissue_clamp_arms = clamp
        return True

    def _start_tissue_motion(
        self,
        label: str,
        waypoints: list[tuple[float, np.ndarray, np.ndarray]],
    ) -> bool:
        if not waypoints:
            return False
        self._tissue_motion_label = label
        self._tissue_motion_waypoints = waypoints
        self._tissue_motion_index = 0
        self._tissue_motion_ready_since = None
        self._tissue_motion_started_at = self.now()
        slide, left, right = waypoints[0]
        self.tc[2] = slide
        self.tc[5:11] = left
        self.tc[12:18] = right
        self.tc[11] = GRIP_CLOSE
        self.tc[18] = GRIP_CLOSE
        self.joint_slew = TISSUE_LOADED_JOINT_SLEW
        self.get_logger().info(
            f"TISSUE_{label.upper()}_START stages={len(waypoints)} "
            f"center_z={self._tissue_carry_center_z_m:.3f}m "
            "rigid_two_hand=true"
        )
        return True

    def _start_tissue_transport_fold(self) -> bool:
        """Raise the torso while preserving the successful shelf clamp."""

        midpoint = self._tissue_midpoint_footprint()
        shelf_center_z = float(midpoint[2])
        current_slide = float(self.slide_meas)
        # Reducing the slide coordinate raises the torso.  Command exactly the
        # measured arm joints and move only the torso.  Do not subsequently
        # roll the load: runtime feedback showed the left arm finishing 0.076
        # rad behind while the right arm was within 0.021 rad, moving the hand
        # midpoint 6.5 mm off-centre before the package fell.  The horizontal
        # clamp already fits inside TISSUE_CARRY_FOOTPRINT, so no loaded arm
        # reconfiguration is necessary.
        hold_arms = (
            np.asarray(self.larm_meas, dtype=float),
            np.asarray(self.rarm_meas, dtype=float),
        )
        minimum_carry_center_z = (
            TABLE_TOP_Z_M
            + self.target_geometry.half_height_m
            + TISSUE_TRANSPORT_BOTTOM_CLEARANCE_M
        )
        required_rise = max(
            0.0, minimum_carry_center_z - shelf_center_z
        )
        transport_slide = float(
            np.clip(
                current_slide - required_rise,
                SLIDE_MIN_M,
                current_slide,
            )
        )
        carry_center_z = (
            shelf_center_z + current_slide - transport_slide
        )
        self._tissue_transport_slide_m = transport_slide
        self._tissue_carry_center_z_m = carry_center_z
        waypoints: list[tuple[float, np.ndarray, np.ndarray]] = [
            (transport_slide, hold_arms[0], hold_arms[1])
        ]
        self._tissue_transport_arms = hold_arms
        self.get_logger().info(
            "TISSUE_HORIZONTAL_RIGID_CARRY_PLAN "
            f"slide={current_slide:.3f}->{transport_slide:.3f}m "
            f"center_z={shelf_center_z:.3f}->{carry_center_z:.3f}m "
            f"required_rise={required_rise:.3f}m "
            "arms_frozen_during_lift=true loaded_roll=false"
        )
        return self._start_tissue_motion("transport_lift", waypoints)

    def _start_tissue_table_unfold(self) -> bool:
        """Verify the unchanged horizontal clamp at the table."""

        if (
            self._tissue_carry_center_z_m is None
            or self._tissue_transport_slide_m is None
            or self._tissue_transport_arms is None
        ):
            return False
        self._tissue_table_arms = self._tissue_transport_arms
        return self._start_tissue_motion(
            "table_horizontal_hold",
            [
                (
                    self._tissue_transport_slide_m,
                    self._tissue_table_arms[0],
                    self._tissue_table_arms[1],
                )
            ],
        )

    def _update_tissue_motion(self) -> bool:
        """Advance one collision-smooth rigid two-hand waypoint sequence."""

        if not self._tissue_motion_waypoints:
            return False
        slide, left, right = self._tissue_motion_waypoints[
            self._tissue_motion_index
        ]
        self.tc[2] = slide
        self.tc[5:11] = left
        self.tc[12:18] = right
        self.tc[11] = GRIP_CLOSE
        self.tc[18] = GRIP_CLOSE
        ready = bool(
            self._both_arms_at_target((left, right))
            and abs(float(self.slide_meas) - slide) <= SLIDE_TOL_M
        )
        if ready:
            if self._tissue_motion_ready_since is None:
                self._tissue_motion_ready_since = self.now()
            elif (
                self.now() - self._tissue_motion_ready_since
                >= TISSUE_STAGE_SETTLE_SEC
            ):
                next_index = self._tissue_motion_index + 1
                if next_index >= len(self._tissue_motion_waypoints):
                    label = self._tissue_motion_label or "motion"
                    self._tissue_motion_waypoints = []
                    self._tissue_motion_started_at = None
                    self._tissue_motion_ready_since = None
                    self.get_logger().info(
                        f"TISSUE_{label.upper()}_COMPLETE "
                        "hand_distance_preserved=true"
                    )
                    return True
                self._tissue_motion_index = next_index
                self._tissue_motion_ready_since = None
                next_slide, next_left, next_right = (
                    self._tissue_motion_waypoints[next_index]
                )
                self.tc[2] = next_slide
                self.tc[5:11] = next_left
                self.tc[12:18] = next_right
        else:
            self._tissue_motion_ready_since = None
        if (
            self._tissue_motion_started_at is not None
            and self.now() - self._tissue_motion_started_at
            >= TISSUE_MOTION_TIMEOUT_SEC
        ):
            label = self._tissue_motion_label or "motion"
            self._tissue_motion_waypoints = []
            self._tissue_motion_started_at = None
            self._fail(f"tissue_{label}_timeout")
        return False

    def _tissue_transport_is_ready(self) -> bool:
        return bool(
            self._tissue_transport_arms is not None
            and self._tissue_transport_slide_m is not None
            and not self._tissue_motion_waypoints
            and self._both_arms_at_target(self._tissue_transport_arms)
            and abs(
                float(self.slide_meas)
                - float(self._tissue_transport_slide_m)
            )
            <= SLIDE_TOL_M
        )


__all__ = [
    "EProductCycleTissueMixin",
    "_tissue_preturn_signed_angle_rad",
]
