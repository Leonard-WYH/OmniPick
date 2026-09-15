"""Pure geometry and control-profile helpers for the shelf cycle."""

from __future__ import annotations

import math

import numpy as np

from .baseline_grasp_controller import GRIP_CLOSE, wrap_to_pi
from .sorting_config import *  # calibrated profile constants

def _grasp_insertion_for_product(product_kind: str, level: str) -> float:
    insertion = GRASP_INSERTION_BEYOND_CENTER_M
    if product_kind in CYLINDER_DEEP_GRASP_KINDS:
        insertion += CYLINDER_EXTRA_INSERTION_M
    if product_kind == "kele":
        insertion += KELE_ADDITIONAL_INSERTION_M
    if product_kind == "maidong":
        insertion += MAIDONG_ADDITIONAL_INSERTION_M
    if product_kind in BROAD_DEEP_GRASP_KINDS:
        insertion += BROAD_PACKAGE_EXTRA_INSERTION_M
    if product_kind == "sanmingzhi":
        insertion -= SANMINGZHI_INSERTION_REDUCTION_M
    if product_kind == "pingguo":
        insertion -= PINGGUO_INSERTION_REDUCTION_M
    if product_kind == "heweidao":
        insertion += HEWEIDAO_ADDITIONAL_INSERTION_M
    if product_kind == "kouxiangtang":
        insertion += KOUXIANGTANG_EXTRA_INSERTION_M
    if product_kind == "shupian":
        insertion += SHUPIAN_EXTRA_INSERTION_M
    if product_kind == "zhijin":
        insertion += TISSUE_EXTRA_INSERTION_M
    if level == "L1":
        insertion += L1_GRASP_EXTRA_INSERTION_M
    elif level == "L3":
        insertion += L3_GRASP_EXTRA_INSERTION_M
    return insertion


def _grip_close_command(product_kind: str) -> float:
    if product_kind == "chengzi":
        return CHENGZI_GRIP_CLOSE_COMMAND
    return GRIP_CLOSE


def _precision_grasp_profile(
    product_kind: str,
) -> tuple[float, float, float] | None:
    return PRECISION_GRASP_ALIGNMENT.get(product_kind)


def _fine_terminal_ee_control_zone(product_kind: str) -> float:
    """Return where live image steering hands off to endpoint control."""

    if product_kind == TISSUE_KIND:
        return TISSUE_TERMINAL_EE_CONTROL_ZONE_M
    if product_kind == "pingguo":
        return PINGGUO_TERMINAL_EE_CONTROL_ZONE_M
    if product_kind == "kele":
        return KELE_TERMINAL_EE_CONTROL_ZONE_M
    if product_kind == "maidong":
        return MAIDONG_TERMINAL_EE_CONTROL_ZONE_M
    if product_kind == "shupian":
        return SHUPIAN_TERMINAL_EE_CONTROL_ZONE_M
    if product_kind == "kouxiangtang":
        return KOUXIANGTANG_TERMINAL_EE_CONTROL_ZONE_M
    if product_kind == "sanmingzhi":
        return SANMINGZHI_TERMINAL_EE_CONTROL_ZONE_M
    return FINE_APPROACH_TERMINAL_EE_CONTROL_ZONE_M


def _pingguo_inserted_grasp_envelope(
    product_kind: str,
    forward_remaining_m: float,
    lateral_error_m: float,
) -> bool:
    """Accept a centred apple already enclosed by the open fingertips."""

    return bool(
        product_kind == "pingguo"
        and forward_remaining_m <= PINGGUO_INSERTED_GRASP_DEPTH_TOL_M
        and lateral_error_m <= PINGGUO_INSERTED_GRASP_LATERAL_TOL_M
    )


def _tissue_contact_grasp_envelope(
    product_kind: str,
    forward_remaining_m: float,
    lateral_error_m: float,
    lateral_tolerance_m: float = TISSUE_CONTACT_GRASP_LATERAL_TOL_M,
) -> bool:
    """Recognize tissue already centred between both hands after contact."""

    return bool(
        product_kind == TISSUE_KIND
        and forward_remaining_m <= TISSUE_CONTACT_GRASP_DEPTH_TOL_M
        and lateral_error_m <= lateral_tolerance_m
    )


def _sanmingzhi_terminal_heading_hold(
    target_yaw_rad: float, current_yaw_rad: float
) -> tuple[float, float]:
    """Return (heading error, bounded correction) for final insertion."""

    error = wrap_to_pi(target_yaw_rad - current_yaw_rad)
    correction = float(
        np.clip(
            SANMINGZHI_TERMINAL_HEADING_HOLD_KP * error,
            -SANMINGZHI_TERMINAL_HEADING_MAX_CORRECTION_RADPS,
            SANMINGZHI_TERMINAL_HEADING_MAX_CORRECTION_RADPS,
        )
    )
    return error, correction


def _detection_image_center(detection_id: str) -> np.ndarray | None:
    """Decode the synchronized YOLO centre appended by product_detect."""

    marker = "|pixel="
    if marker not in detection_id:
        return None
    try:
        encoded = detection_id.rsplit(marker, 1)[1]
        u_text, v_text = encoded.split(",", 1)
        pixel = np.array([float(u_text), float(v_text)], dtype=float)
    except (TypeError, ValueError):
        return None
    return pixel if np.all(np.isfinite(pixel)) else None


def _image_servo_correction(
    product_u: float,
    gripper_u: float,
    focal_x: float,
    deadband_px: float = IMAGE_SERVO_PIXEL_DEADBAND_PX,
) -> tuple[float, float]:
    """Return (horizontal pixel error, yaw-rate correction).

    Image U grows to the right.  At the E shelf a product to the image-right
    of the gripper needs a right turn, which is a negative base yaw command.
    """

    pixel_error = float(product_u - gripper_u)
    if not all(math.isfinite(value) for value in (pixel_error, focal_x)):
        return pixel_error, 0.0
    deadband = max(0.0, float(deadband_px))
    if focal_x <= 1.0 or abs(pixel_error) <= deadband:
        return pixel_error, 0.0
    effective_error = math.copysign(
        abs(pixel_error) - deadband, pixel_error
    )
    angular_error = math.atan2(effective_error, focal_x)
    correction = float(
        np.clip(
            -IMAGE_SERVO_ANGULAR_KP * angular_error,
            -IMAGE_SERVO_MAX_ANGULAR_CORRECTION_RADPS,
            IMAGE_SERVO_MAX_ANGULAR_CORRECTION_RADPS,
        )
    )
    return pixel_error, correction


def _image_servo_angular_command(
    pixel_correction: float, grasp_yaw_error: float
) -> float:
    """Combine dominant live-pixel steering with bounded grasp-yaw feed-forward."""

    heading_correction = float(
        np.clip(
            IMAGE_SERVO_GRASP_HEADING_KP * grasp_yaw_error,
            -IMAGE_SERVO_MAX_HEADING_CORRECTION_RADPS,
            IMAGE_SERVO_MAX_HEADING_CORRECTION_RADPS,
        )
    )
    return float(pixel_correction + heading_correction)


def _fine_visual_alignment_pending(
    now: float,
    pixel_error_px: float | None,
    pixel_observed_at: float | None,
) -> bool:
    """Return whether a fresh image still asks for lateral correction."""

    return bool(
        pixel_error_px is not None
        and pixel_observed_at is not None
        and 0.0 <= float(now) - float(pixel_observed_at)
        <= IMAGE_SERVO_MAX_AGE_SEC
        and abs(float(pixel_error_px))
        > IMAGE_SERVO_TERMINAL_LOCK_TOL_PX
    )


def _random_pick_predeploy_distance(
    picked_count: int, handoff_policy: str, shelf: str
) -> float:
    """Return the policy-specific distance for concurrent grasp deployment."""

    if int(picked_count) == 0:
        return FIRST_PICK_PREDEPLOY_DISTANCE_M
    if str(handoff_policy) == "stationary_predeploy":
        return STATIONARY_PICK_PREDEPLOY_DISTANCE_M
    if str(shelf).strip().upper() in {"B", "C"}:
        return BC_RANDOM_PICK_PREDEPLOY_DISTANCE_M
    return RANDOM_PICK_PREDEPLOY_DISTANCE_M


def _random_rolling_handoff_distance(
    picked_count: int, shelf: str
) -> float:
    """Return the shelf-specific distance where visual servo owns the base."""

    if int(picked_count) == 0:
        return FIRST_PICK_ROLLING_HANDOFF_DISTANCE_M
    if str(shelf).strip().upper() in {"B", "C"}:
        return BC_RANDOM_ROLLING_HANDOFF_DISTANCE_M
    return RANDOM_ROLLING_HANDOFF_DISTANCE_M


def _rolling_pick_handoff_ready(
    handoff_policy: str,
    distance_m: float,
    handoff_distance_m: float,
    yaw_error_rad: float,
    pick_prepared: bool,
) -> bool:
    """Only release Nav2 after a rolling shelf leg is roughly facing in.

    A short lateral move between neighbouring cabinets can enter the distance
    window before Nav2 has completed its roughly 90-degree turn.  Handing the
    base to the forward-only visual controller there makes the remembered
    shelf target appear beside/behind the robot and trips the shelf guard.
    """

    return bool(
        str(handoff_policy) == "rolling"
        and bool(pick_prepared)
        and float(distance_m) <= float(handoff_distance_m)
        and abs(float(yaw_error_rad))
        <= RANDOM_ROLLING_HANDOFF_YAW_TOL_RAD
    )


def _stationary_predeploy_handoff_ready(
    handoff_policy: str,
    shelf: str,
    distance_m: float,
    yaw_error_rad: float,
    pick_ready: bool,
) -> bool:
    """Accept a roughly aligned A/D/E pose with the grasp pose deployed."""

    return bool(
        str(handoff_policy) == "stationary_predeploy"
        and str(shelf).strip().upper() in ADE_NEAR_HANDOFF_SHELVES
        and pick_ready
        and float(distance_m) <= ADE_OBSERVATION_HANDOFF_POSITION_TOL_M
        and abs(float(yaw_error_rad)) <= ADE_OBSERVATION_HANDOFF_YAW_TOL_RAD
    )


def _random_scan_focus_shelf(
    active_shelf: str,
    picked_count: int,
    abc_transit_scan_active: bool,
    distance_to_scan_m: float,
) -> tuple[str, str]:
    """Choose the wide-view cabinet without changing base navigation.

    The first trip after one completed delivery uses B as the geometric centre
    of A/B/C while the robot is still far away.  Close to the selected scan
    pose it switches to that cabinet's full-shelf centre.  Product-specific
    tracking takes over separately once a concrete target is deployed.
    """

    normalized_shelf = str(active_shelf).strip().upper()
    if (
        int(picked_count) > 0
        and bool(abc_transit_scan_active)
        and float(distance_to_scan_m)
        > POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M
    ):
        return "B", "abc_transit_overview"
    return normalized_shelf, "target_shelf_overview"


def _table_nav_angular_override(
    distance_m: float,
    signed_yaw_error_rad: float,
) -> float | None:
    """Return fixed terminal yaw speed while Nav2 retains linear control."""

    if float(distance_m) > TABLE_NAV_YAW_ASSIST_START_DISTANCE_M:
        return None
    error = float(signed_yaw_error_rad)
    if abs(error) <= TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD:
        return 0.0
    return math.copysign(TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS, error)


def _table_nav_pose_fast_acceptable(
    distance_m: float,
    signed_yaw_error_rad: float,
) -> bool:
    """Accept a usable table pose without Nav2's terminal settle dwell."""

    return bool(
        float(distance_m) <= TABLE_NEAR_GOAL_POSITION_TOL_M
        and abs(float(signed_yaw_error_rad))
        <= TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD
    )


def _edge_row_cylinder_slot_lateral_locked(
    product_kind: str, level: str
) -> bool:
    return bool(
        product_kind in EDGE_ROW_CYLINDER_SLOT_LATERAL_KINDS
        and level in {"L1", "L3"}
    )


def _grasp_lateral_target_x(
    product_kind: str, level: str, slot_x: float
) -> float:
    """Return the calibrated final gripper line for a shelf slot."""

    offset = (
        0.0
        if product_kind == TISSUE_KIND
        else RIGHT_HAND_GRASP_X_OFFSET_M
    )
    if _edge_row_cylinder_slot_lateral_locked(product_kind, level):
        offset += EDGE_ROW_CYLINDER_GRASP_X_OFFSET_M
    if product_kind == "shupian":
        offset += SHUPIAN_ADDITIONAL_GRASP_X_OFFSET_M
    if product_kind == "maidong":
        offset += MAIDONG_ADDITIONAL_GRASP_X_OFFSET_M
    if product_kind == "pingguo":
        offset += PINGGUO_ADDITIONAL_GRASP_X_OFFSET_M
    return float(slot_x) + offset


def _first_e_direct_velocity_command(
    base_xy: np.ndarray,
    base_yaw: float,
    pose: tuple[float, float, float],
) -> tuple[float, float, float, float, bool]:
    """Compute the fast odometry-closed-loop command for the first E leg."""

    delta = np.asarray(pose[:2], dtype=float) - np.asarray(
        base_xy, dtype=float
    )
    distance = float(np.linalg.norm(delta))
    final_yaw_error = wrap_to_pi(float(pose[2]) - float(base_yaw))
    if distance <= FIRST_E_DIRECT_POSITION_TOL_M:
        if abs(final_yaw_error) <= FIRST_E_DIRECT_YAW_TOL_RAD:
            return 0.0, 0.0, distance, abs(final_yaw_error), True
        angular = float(
            np.clip(
                FIRST_E_DIRECT_HEADING_KP * final_yaw_error,
                -FIRST_E_DIRECT_MAX_ANGULAR_RADPS,
                FIRST_E_DIRECT_MAX_ANGULAR_RADPS,
            )
        )
        return 0.0, angular, distance, abs(final_yaw_error), False

    bearing = math.atan2(float(delta[1]), float(delta[0]))
    heading_error = wrap_to_pi(bearing - float(base_yaw))
    angular = float(
        np.clip(
            FIRST_E_DIRECT_HEADING_KP * heading_error,
            -FIRST_E_DIRECT_MAX_ANGULAR_RADPS,
            FIRST_E_DIRECT_MAX_ANGULAR_RADPS,
        )
    )
    distance_speed = float(
        np.clip(
            FIRST_E_DIRECT_POSITION_KP * distance,
            FIRST_E_DIRECT_MIN_LINEAR_MPS,
            FIRST_E_DIRECT_CRUISE_SPEED_MPS,
        )
    )
    # Keep describing an arc while correcting heading.  A zero cosine gate
    # made the chassis visibly stop in the aisle to rotate even though this
    # first E leg has ample free space and the user requested continuous
    # motion.  The goal-distance loop and the 0.5 m visual handoff still bound
    # this command before reaching the shelf.
    alignment = max(
        FINE_APPROACH_MIN_FORWARD_ALIGNMENT_SCALE,
        math.cos(heading_error),
    )
    linear = distance_speed * alignment
    return linear, angular, distance, abs(final_yaw_error), False


def _fine_lateral_tolerance(product_kind: str, level: str) -> float:
    if _edge_row_cylinder_slot_lateral_locked(product_kind, level):
        return EDGE_ROW_CYLINDER_GRASP_LATERAL_TOL_M
    if product_kind == "shupian":
        return SHUPIAN_GRASP_LATERAL_TOL_M
    return EDGE_GRASP_LATERAL_TOL_M


def _fine_approach_stall_timeout(
    product_kind: str, *, precision_aligned: bool
) -> float:
    if (
        _precision_grasp_profile(product_kind) is not None
        and not precision_aligned
    ):
        return SANMINGZHI_ALIGNMENT_STALL_TIMEOUT_SEC
    if product_kind in COUPLED_ALIGNMENT_KINDS:
        return KOUXIANGTANG_FINE_APPROACH_STALL_TIMEOUT_SEC
    return FINE_APPROACH_STALL_TIMEOUT_SEC


def _fine_approach_scheduled_speed(
    forward_remaining_m: float, product_kind: str | None = None
) -> float:
    """Smoothly reduce approach speed as the hand nears grasp depth."""

    span = (
        DIRECT_FINE_APPROACH_TAPER_START_M
        - DIRECT_FINE_APPROACH_TAPER_END_M
    )
    fraction = float(
        np.clip(
            (float(forward_remaining_m) - DIRECT_FINE_APPROACH_TAPER_END_M)
            / span,
            0.0,
            1.0,
        )
    )
    # Smoothstep has zero slope at both ends, avoiding a command jerk when the
    # controller enters or leaves the deceleration region.
    profile = _precision_grasp_profile(product_kind or "")
    alignment_speed = (
        DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS
        if profile is None
        else float(profile[2])
    )
    blend = fraction * fraction * (3.0 - 2.0 * fraction)
    return float(
        alignment_speed
        + (
            DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS
            - alignment_speed
        )
        * blend
    )

__all__ = [
    '_grasp_insertion_for_product',
    '_grip_close_command',
    '_precision_grasp_profile',
    '_fine_terminal_ee_control_zone',
    '_pingguo_inserted_grasp_envelope',
    '_tissue_contact_grasp_envelope',
    '_sanmingzhi_terminal_heading_hold',
    '_detection_image_center',
    '_image_servo_correction',
    '_image_servo_angular_command',
    '_fine_visual_alignment_pending',
    '_random_pick_predeploy_distance',
    '_random_rolling_handoff_distance',
    '_rolling_pick_handoff_ready',
    '_stationary_predeploy_handoff_ready',
    '_random_scan_focus_shelf',
    '_table_nav_angular_override',
    '_table_nav_pose_fast_acceptable',
    '_edge_row_cylinder_slot_lateral_locked',
    '_grasp_lateral_target_x',
    '_first_e_direct_velocity_command',
    '_fine_lateral_tolerance',
    '_fine_approach_stall_timeout',
    '_fine_approach_scheduled_speed',
]
