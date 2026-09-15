"""Target tracking and the observation-to-shelf visual servo."""

from __future__ import annotations

import math
import numpy as np

from .baseline_grasp_controller import (
    DETECT_MIN_SAMPLES,
    wrap_to_pi,
)
from .nav2_manipulation_client import BASE_STOP_EPS
from .navigation.sorting_geometry import (
    HAND_WORKPOINT_XY_M,
    SHELF_COLUMNS_M,
    SHELF_PRODUCT_CENTER_Y_M,
    cluster_points,
    grasp_slide_for_product_z,
)
from scipy.spatial.transform import Rotation

from .sorting_config import (
    COUPLED_ALIGNMENT_FINAL_TOL_M,
    COUPLED_ALIGNMENT_KINDS,
    DIRECT_FINE_APPROACH_MAX_ANGULAR,
    DIRECT_POSE_GUIDANCE_ALPHA_KP,
    DIRECT_POSE_GUIDANCE_BETA_KP,
    DIRECT_POSE_GUIDANCE_DISTANCE_KP,
    DIRECT_POSE_GUIDANCE_LATERAL_KP,
    EDGE_FINE_APPROACH_MAX_ANGULAR,
    FINE_APPROACH_ALIGNMENT_CRAWL_SPEED_MPS,
    FINE_APPROACH_ALIGNMENT_RESERVE_BASE_M,
    FINE_APPROACH_ALIGNMENT_RESERVE_GAIN,
    FINE_APPROACH_ALIGNMENT_RESERVE_MAX_M,
    FINE_APPROACH_ALIGNMENT_RESERVE_RAMP_M,
    FINE_APPROACH_ANGULAR_ACCEL_RADPS2,
    FINE_APPROACH_BRAKE_MARGIN_M,
    FINE_APPROACH_BRAKE_REACTION_SEC,
    FINE_APPROACH_INITIAL_HEADING_TOL_RAD,
    FINE_APPROACH_MIN_FORWARD_ALIGNMENT_SCALE,
    FINE_APPROACH_MIN_NEAR_ANGULAR_RADPS,
    FINE_APPROACH_NEAR_ANGULAR_ZONE_M,
    FINE_APPROACH_NEAR_LATERAL_ZONE_M,
    FINE_APPROACH_NEAR_SETTLE_SEC,
    FINE_APPROACH_TERMINAL_EE_KP,
    HEAD_CAMERA_FIXED_ELEVATION_RAD,
    HEAD_CAMERA_FORWARD_AT_ZERO_PITCH_M,
    HEAD_CAMERA_FORWARD_PER_PITCH_M,
    HEAD_CAMERA_LATERAL_M,
    HEAD_CAMERA_Z_PLUS_SLIDE_AT_ZERO_PITCH_M,
    HEAD_CAMERA_Z_PLUS_SLIDE_PER_PITCH_M,
    IMAGE_SERVO_FILTER_ALPHA,
    IMAGE_SERVO_HARD_FREEZE_REMAINING_M,
    IMAGE_SERVO_MAX_AGE_SEC,
    IMAGE_SERVO_MIN_LINEAR_SCALE,
    IMAGE_SERVO_PIXEL_DEADBAND_PX,
    IMAGE_SERVO_SLOWDOWN_FULL_PX,
    IMAGE_SERVO_SLOWDOWN_START_PX,
    LIVE_LATERAL_FILTER_ALPHA,
    LIVE_LATERAL_MAX_SLOT_OFFSET_M,
    LIVE_LATERAL_MAX_STEP_M,
    LIVE_TARGET_MAX_AGE_SEC,
    LIVE_TRACK_LOG_INTERVAL_SEC,
    LIVE_TRACK_MIN_SAMPLES,
    PICK_TRACK_HEAD_PITCH,
    PICK_TRACK_HEAD_PITCH_MAX_RAD,
    PICK_TRACK_HEAD_PITCH_MIN_RAD,
    PICK_TRACK_HEAD_YAW_LIMIT_RAD,
    PICK_TRACK_MIN_CAMERA_RANGE_M,
    PINGGUO_IMAGE_SERVO_PIXEL_DEADBAND_PX,
    POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M,
    PRECISION_ALIGNMENT_RESERVE_BASE_M,
    PRECISION_ALIGNMENT_RESERVE_GAIN,
    PRECISION_ALIGNMENT_RESERVE_MAX_M,
    REACQUIRE_MAX_TARGET_DISTANCE_M,
    SANMINGZHI_FINE_APPROACH_MAX_ANGULAR,
    SHELF_OVERVIEW_CENTER_Z_M,
    SUBSEQUENT_FINE_APPROACH_INITIAL_HEADING_TOL_RAD,
    TISSUE_HAND_CENTER_X_M,
)
from .grasp_profiles import (
    _edge_row_cylinder_slot_lateral_locked,
    _fine_lateral_tolerance,
    _fine_terminal_ee_control_zone,
    _fine_visual_alignment_pending,
    _grasp_insertion_for_product,
    _grasp_lateral_target_x,
    _image_servo_angular_command,
    _image_servo_correction,
    _precision_grasp_profile,
    _random_scan_focus_shelf,
    _sanmingzhi_terminal_heading_hold,
)
from .sorting_state import CycleState


class EProductCycleVisionMixin:
    def _reacquire_target(self, *, tracking: bool = False) -> bool:
        """Refresh lateral guidance from live vision for the selected slot."""

        if self._current_target is None:
            return False
        marker_id = int(self._current_target["aruco_id"])
        expected = np.asarray(self._current_target["surface_world"], dtype=float)
        viable = []
        now = self.now()
        recent_points = [
            point
            for observed_at, point in self._reacquire_points
            if now - observed_at <= LIVE_TARGET_MAX_AGE_SEC
        ]
        min_samples = (
            LIVE_TRACK_MIN_SAMPLES if tracking else DETECT_MIN_SAMPLES
        )
        for cluster in cluster_points(recent_points):
            if len(cluster) < min_samples:
                continue
            surface = np.median(cluster, axis=0)
            distance = float(np.linalg.norm(surface - expected))
            if distance <= REACQUIRE_MAX_TARGET_DISTANCE_M:
                viable.append((distance, -len(cluster), surface))
        if not viable:
            return False
        _, _, surface = min(viable, key=lambda item: (item[0], item[1]))
        supporting_timestamps = [
            timestamp
            for timestamp, point in self._reacquire_points
            if np.linalg.norm(point - surface) <= 0.12
        ]
        if not supporting_timestamps:
            return False
        observed_at = max(supporting_timestamps)
        center = self._vision_to_object_center(surface)
        # Keep the height established by the shelf observation.  Once the arm
        # enters the image, RGB-D often samples the shelf/arm behind a narrow
        # bottle and can move Z by 5-10 cm even though the physical bottle did
        # not move.  Fresh vision still corrects lateral X during the open
        # first segment of the approach.
        remembered_z = float(self._current_target["product_world"][2])
        measured_z = float(center[2])
        center[2] = remembered_z
        level = str(self._current_target.get("level", ""))
        slot_anchor = np.asarray(
            self._current_target.get(
                "observation_product_world",
                self._current_target["product_world"],
            ),
            dtype=float,
        )
        observation_target = (
            self._fine_observation_target_world
            if self._fine_observation_target_world is not None
            else slot_anchor
        )
        raw_measured_lateral_x = float(center[0])
        measured_lateral_x = raw_measured_lateral_x
        if tracking and self.OBJECT_WORLD is not None:
            # Products do not move before contact, whereas close RGB-D depth
            # does move when the deployed black hand occludes their front.
            # Freeze longitudinal depth for every shelf row.  Most products
            # consume live lateral measurements while visible; edge-row
            # cylinders deliberately retain their surveyed slot X because the
            # oblique RGB-D centroid is repeatably biased there.
            center[1] = float(observation_target[1])
            raw_measured_lateral_x = float(
                np.clip(
                    center[0],
                    float(slot_anchor[0])
                    - LIVE_LATERAL_MAX_SLOT_OFFSET_M,
                    float(slot_anchor[0])
                    + LIVE_LATERAL_MAX_SLOT_OFFSET_M,
                )
            )
            # Apply the right-hand aperture calibration to every *new* live
            # measurement, not only to the initial observation target.  The
            # previous filter converged toward the raw product centroid and
            # gradually erased this 5 mm correction; narrow products such as
            # sanmingzhi then ended visibly left of the gripper despite a
            # nominally small controller error.
            measured_lateral_x = _grasp_lateral_target_x(
                self.target_kind, level, raw_measured_lateral_x
            )
            current_lateral_x = float(self.OBJECT_WORLD[0])
            edge_cylinder_slot_lock = (
                _edge_row_cylinder_slot_lateral_locked(
                    self.target_kind, level
                )
            )
            # Do not repeatedly filter the same inference frame at the much
            # faster control-loop rate.  Each accepted camera update changes
            # the target smoothly and by a bounded amount.
            if edge_cylinder_slot_lock:
                # The product is placed at the surveyed slot centre.  Near
                # L1/L3, perspective and partial gripper occlusion bias the
                # visible cylinder surface.  Keep the grasp line on the slot
                # plus the measured rightward gripper-aperture calibration.
                center[0] = _grasp_lateral_target_x(
                    self.target_kind, level, float(slot_anchor[0])
                )
                self._fine_last_vision_update_at = observed_at
            elif (
                self._fine_last_vision_update_at is None
                or observed_at > self._fine_last_vision_update_at + 1.0e-6
            ):
                filtered_delta = float(
                    np.clip(
                        LIVE_LATERAL_FILTER_ALPHA
                        * (measured_lateral_x - current_lateral_x),
                        -LIVE_LATERAL_MAX_STEP_M,
                        LIVE_LATERAL_MAX_STEP_M,
                    )
                )
                center[0] = current_lateral_x + filtered_delta
                self._fine_last_vision_update_at = observed_at
            else:
                center[0] = current_lateral_x
        footprint = self.world_to_footprint(center)
        # Direct approaches begin at the common observation pose, where an
        # edge-column product can be roughly 1.2 m forward and 0.2 m lateral.
        # The E-shelf world-volume and selected-slot distance checks above are
        # still mandatory, so this only enlarges the valid tracking envelope;
        # it does not admit detections from another shelf.
        if not (0.45 <= footprint[0] <= 1.35 and abs(footprint[1]) <= 0.35):
            return False
        stable_surface = np.asarray(surface, dtype=float).copy()
        stable_surface[2] = remembered_z
        self._current_target["surface_world"] = stable_surface
        self._current_target["product_world"] = center
        slot = self._matching_inventory_slot(surface)
        # Never let a nearby bottle change the persistent queue identity.  A
        # confirmed match may only enrich the same ArUco slot.
        if slot is not None and int(slot["aruco_id"]) == marker_id:
            self._current_target.update(
                {
                    "level": str(slot["level"]),
                    "column": str(slot["column"]),
                    "match_source": "aruco_confirmed",
                }
            )
        remembered = self._known_targets.get(marker_id)
        if remembered is not None:
            remembered["surface_world"] = stable_surface.copy()
            remembered["vision_product_z"] = measured_z
        self.OBJECT_WORLD = center
        self._fine_last_live_target_world = center.copy()
        insertion = _grasp_insertion_for_product(self.target_kind, level)
        self._grasp_insertion_m = insertion
        requested_stop_y = float(center[1] + insertion)
        # Keep the stop line fixed throughout each fine-approach attempt.  It
        # is initialized by the unobstructed observation above and must not be
        # pushed into the shelf by close-range occlusion.
        if tracking and self.CREEP_STOP_Y is not None:
            self.CREEP_STOP_Y = float(self.CREEP_STOP_Y)
        else:
            self.CREEP_STOP_Y = requested_stop_y
        self._target_slide = grasp_slide_for_product_z(
            center[2], self.target_kind, level
        )
        self.tc[2] = self._target_slide
        self._track_pick_head()
        self.target_locked = True
        self._target_last_seen_at = observed_at
        should_log = not tracking or (
            self.now() - self._last_live_track_log >= LIVE_TRACK_LOG_INTERVAL_SEC
        )
        if should_log:
            label = (
                f"{self._event_prefix}_TRACK_UPDATED"
                if tracking
                else f"{self._event_prefix}_REACQUIRED"
            )
            self.get_logger().info(
                f"{label} world={np.round(center, 3).tolist()} "
                f"measured_x={raw_measured_lateral_x:.3f} "
                f"calibrated_x={measured_lateral_x:.3f} "
                f"slot_x={slot_anchor[0]:.3f} "
                f"measured_z={measured_z:.3f} remembered_z={remembered_z:.3f} "
                f"footprint={np.round(footprint, 3).tolist()} "
                f"creep_stop_y={self.CREEP_STOP_Y:.3f} "
                f"insertion={insertion:.3f} "
                f"slide={self._target_slide:.3f}"
            )
            self._last_live_track_log = self.now()
        return True

    def _track_pick_head(self) -> float:
        """Keep the camera aimed at the target in both yaw and pitch."""

        if self.OBJECT_WORLD is None or self.base_xy is None:
            self.tc[3] = 0.0
            self.tc[4] = PICK_TRACK_HEAD_PITCH
            return 0.0
        footprint = np.asarray(
            self.world_to_footprint(self.OBJECT_WORLD), dtype=float
        )
        pitch_estimate = float(
            np.clip(
                self.tc[4],
                PICK_TRACK_HEAD_PITCH_MIN_RAD,
                PICK_TRACK_HEAD_PITCH_MAX_RAD,
            )
        )
        camera_forward = (
            HEAD_CAMERA_FORWARD_AT_ZERO_PITCH_M
            + HEAD_CAMERA_FORWARD_PER_PITCH_M * pitch_estimate
        )
        camera_lateral = HEAD_CAMERA_LATERAL_M
        camera_z = (
            HEAD_CAMERA_Z_PLUS_SLIDE_AT_ZERO_PITCH_M
            + HEAD_CAMERA_Z_PLUS_SLIDE_PER_PITCH_M * pitch_estimate
            - float(self.slide_meas)
        )
        target_from_camera_x = max(
            PICK_TRACK_MIN_CAMERA_RANGE_M,
            float(footprint[0]) - camera_forward,
        )
        target_from_camera_y = float(footprint[1]) - camera_lateral

        # footprint[1] is left-positive, matching the head-yaw joint.  Aim
        # from the camera rather than the chassis origin so close shelf targets
        # do not drift toward an image edge as the base approaches.
        desired_yaw = math.atan2(
            target_from_camera_y, target_from_camera_x
        )
        commanded_yaw = float(
            np.clip(
                desired_yaw,
                -PICK_TRACK_HEAD_YAW_LIMIT_RAD,
                PICK_TRACK_HEAD_YAW_LIMIT_RAD,
            )
        )
        horizontal_range = max(
            PICK_TRACK_MIN_CAMERA_RANGE_M,
            math.hypot(target_from_camera_x, target_from_camera_y),
        )
        desired_optical_elevation = math.atan2(
            float(footprint[2]) - camera_z,
            horizontal_range,
        )
        commanded_pitch = float(
            np.clip(
                desired_optical_elevation
                - HEAD_CAMERA_FIXED_ELEVATION_RAD,
                PICK_TRACK_HEAD_PITCH_MIN_RAD,
                PICK_TRACK_HEAD_PITCH_MAX_RAD,
            )
        )
        self.tc[3] = commanded_yaw
        self.tc[4] = commanded_pitch
        return commanded_yaw

    def _track_random_scan_head(self) -> None:
        """Aim the moving head camera at the cabinet being approached."""

        if not self._random_shelf_mode or self.base_xy is None:
            return
        scan_xy = np.asarray(
            self._active_random_scan_pose()[:2], dtype=float
        )
        distance_to_scan = float(
            np.linalg.norm(scan_xy - np.asarray(self.base_xy, dtype=float))
        )
        focus_shelf, head_mode = _random_scan_focus_shelf(
            self._active_shelf,
            self._picked_count,
            self._random_post_first_abc_transit_scan_active,
            distance_to_scan,
        )
        # On the very first E-bound trip, look through D until the chassis is
        # near E.  The wide view then gathers D/C inventory on the way while
        # still centring E before the stationary scan.
        if (
            self._picked_count == 0
            and self._active_shelf == "E"
            and float(self.base_xy[0]) < 1.35
        ):
            focus_shelf = "D"
            head_mode = "first_cde_transit_overview"
        if (
            self._random_post_first_abc_transit_scan_active
            and head_mode == "target_shelf_overview"
        ):
            self._random_post_first_abc_transit_scan_active = False
            self.get_logger().info(
                "RANDOM_CAMERA_TARGET_SHELF_OVERVIEW "
                f"shelf={self._active_shelf} "
                f"distance={distance_to_scan:.3f}m "
                f"switch_distance="
                f"{POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M:.2f}m"
            )
        if head_mode != self._random_scan_head_mode:
            self._random_scan_head_mode = head_mode
            self.get_logger().info(
                "RANDOM_CAMERA_SCAN_MODE "
                f"mode={head_mode} focus_shelf={focus_shelf} "
                f"active_shelf={self._active_shelf} "
                f"distance={distance_to_scan:.3f}m"
            )
        target = np.array(
            [
                SHELF_COLUMNS_M[focus_shelf][1],
                SHELF_PRODUCT_CENTER_Y_M,
                SHELF_OVERVIEW_CENTER_Z_M,
            ],
            dtype=float,
        )
        footprint = np.asarray(self.world_to_footprint(target), dtype=float)
        forward = max(0.10, float(footprint[0]) - HEAD_CAMERA_FORWARD_AT_ZERO_PITCH_M)
        lateral = float(footprint[1]) - HEAD_CAMERA_LATERAL_M
        self.tc[3] = float(np.clip(math.atan2(lateral, forward), -0.90, 0.90))
        camera_z = HEAD_CAMERA_Z_PLUS_SLIDE_AT_ZERO_PITCH_M - float(
            self.slide_meas
        )
        elevation = math.atan2(
            float(footprint[2]) - camera_z,
            max(0.10, math.hypot(forward, lateral)),
        )
        self.tc[4] = float(
            np.clip(
                elevation - HEAD_CAMERA_FIXED_ELEVATION_RAD,
                PICK_TRACK_HEAD_PITCH_MIN_RAD,
                PICK_TRACK_HEAD_PITCH_MAX_RAD,
            )
        )

    def _project_world_to_head_image(
        self, point_world: np.ndarray
    ) -> np.ndarray | None:
        """Project a world point through the measured head-camera pose."""

        if (
            self._camera_k is None
            or self._base_position_xyz is None
            or self._base_quaternion_wxyz is None
            or self.jpos is None
        ):
            return None
        point_world = np.asarray(point_world, dtype=float)
        if point_world.shape != (3,) or not np.all(np.isfinite(point_world)):
            return None
        base_quaternion = self._base_quaternion_wxyz
        if (
            not np.all(np.isfinite(base_quaternion))
            or np.linalg.norm(base_quaternion) <= 1.0e-6
        ):
            return None

        self._head_camera_fk.set_base_pose(
            self._base_position_xyz, base_quaternion
        )
        self._head_camera_fk.set_slide_joint(float(self.slide_meas))
        self._head_camera_fk.set_head_joints(
            [
                float(self.jpos.get("head_yaw_joint", self.action[3])),
                float(self.jpos.get("head_pitch_joint", self.action[4])),
            ]
        )
        camera_position, camera_quaternion_wxyz = (
            self._head_camera_fk.get_head_camera_pose()
        )
        camera_rotation = Rotation.from_quat(
            [
                camera_quaternion_wxyz[1],
                camera_quaternion_wxyz[2],
                camera_quaternion_wxyz[3],
                camera_quaternion_wxyz[0],
            ]
        ).as_matrix()
        point_camera = camera_rotation.T @ (
            point_world - np.asarray(camera_position, dtype=float)
        )
        if not np.all(np.isfinite(point_camera)) or point_camera[2] <= 0.05:
            return None
        focal_x = float(self._camera_k[0, 0])
        focal_y = float(self._camera_k[1, 1])
        centre_x = float(self._camera_k[0, 2])
        centre_y = float(self._camera_k[1, 2])
        return np.array(
            [
                centre_x + focal_x * point_camera[0] / point_camera[2],
                centre_y + focal_y * point_camera[1] / point_camera[2],
            ],
            dtype=float,
        )

    def _update_fine_pixel_servo(self) -> float | None:
        """Use a fresh target box and projected gripper for early yaw control."""

        self._fine_pixel_servo_active = False
        self._fine_pixel_correction_radps = None
        self._fine_pixel_linear_scale = 1.0
        if self._current_target is None or self._camera_k is None:
            return None

        now = self.now()
        expected_surface = np.asarray(
            self._current_target["surface_world"], dtype=float
        )
        viable = [
            (
                observed_at,
                float(np.linalg.norm(point - expected_surface)),
                product_pixel,
                gripper_pixel,
            )
            for (
                observed_at,
                point,
                product_pixel,
                gripper_pixel,
            ) in self._image_reacquire_points
            if 0.0 <= now - observed_at <= IMAGE_SERVO_MAX_AGE_SEC
            and np.linalg.norm(point - expected_surface)
            <= REACQUIRE_MAX_TARGET_DISTANCE_M
        ]
        if not viable:
            return None
        observed_at, _, product_pixel, gripper_pixel = max(
            viable, key=lambda item: (item[0], -item[1])
        )

        pixel_deadband = (
            PINGGUO_IMAGE_SERVO_PIXEL_DEADBAND_PX
            if getattr(self, "target_kind", "") == "pingguo"
            else IMAGE_SERVO_PIXEL_DEADBAND_PX
        )
        raw_pixel_error, _ = _image_servo_correction(
            float(product_pixel[0]),
            float(gripper_pixel[0]),
            float(self._camera_k[0, 0]),
            pixel_deadband,
        )
        if (
            self._fine_pixel_filter_observation_at is None
            or observed_at
            > self._fine_pixel_filter_observation_at + 1.0e-6
        ):
            if (
                self._fine_pixel_error_px is None
                or self._fine_pixel_observation_at is None
                or observed_at - self._fine_pixel_observation_at
                > IMAGE_SERVO_MAX_AGE_SEC
            ):
                filtered_pixel_error = raw_pixel_error
            else:
                filtered_pixel_error = float(
                    self._fine_pixel_error_px
                    + IMAGE_SERVO_FILTER_ALPHA
                    * (raw_pixel_error - self._fine_pixel_error_px)
                )
            self._fine_pixel_error_px = filtered_pixel_error
            self._fine_pixel_error_raw_px = raw_pixel_error
            self._fine_pixel_filter_observation_at = observed_at
        elif self._fine_pixel_error_px is None:
            return None

        _, correction = _image_servo_correction(
            float(product_pixel[0]),
            float(product_pixel[0] - self._fine_pixel_error_px),
            float(self._camera_k[0, 0]),
            pixel_deadband,
        )
        slowdown_fraction = float(
            np.clip(
                (
                    abs(self._fine_pixel_error_px)
                    - IMAGE_SERVO_SLOWDOWN_START_PX
                )
                / (
                    IMAGE_SERVO_SLOWDOWN_FULL_PX
                    - IMAGE_SERVO_SLOWDOWN_START_PX
                ),
                0.0,
                1.0,
            )
        )
        self._fine_pixel_linear_scale = float(
            1.0
            - (1.0 - IMAGE_SERVO_MIN_LINEAR_SCALE) * slowdown_fraction
        )
        self._fine_pixel_servo_active = True
        self._fine_product_pixel = np.asarray(product_pixel, dtype=float).copy()
        self._fine_gripper_pixel = gripper_pixel.copy()
        self._fine_pixel_correction_radps = correction
        self._fine_pixel_observation_at = observed_at
        return correction

    def _e_fine_approach_steering(
        self,
        end_effector: np.ndarray,
        *,
        linear_speed: float,
        edge_row: bool,
    ) -> tuple[float, float, float, float]:
        """Return a coupled base command toward the locked grasp pose."""

        lateral_remaining = float(self.OBJECT_WORLD[0] - end_effector[0])
        forward_remaining = float(self.CREEP_STOP_Y - end_effector[1])
        if (
            _precision_grasp_profile(self.target_kind) is not None
            or self.target_kind in COUPLED_ALIGNMENT_KINDS
        ):
            max_angular = SANMINGZHI_FINE_APPROACH_MAX_ANGULAR
        elif edge_row:
            max_angular = EDGE_FINE_APPROACH_MAX_ANGULAR
        else:
            max_angular = DIRECT_FINE_APPROACH_MAX_ANGULAR

        # Generate the direct observation-to-target geometry in the Baseline
        # controller.  This is not a Nav2 waypoint: the final base pose is
        # recomputed from the remembered/live target and consumed only by this
        # local velocity loop.  Fresh image pixels directly steer the open
        # first segment; this world-pose law becomes its memory fallback and
        # still supplies the final endpoint geometry.
        # Use the measured hand offset rather than assuming the commanded arm
        # template is exact.  Loaded joints can remain several hundredths of a
        # radian short; feeding that real FK residual into the terminal base
        # pose prevents the chassis from stopping at an ideal pose while the
        # physical gripper is still laterally displaced.
        current_world_offset = (
            np.asarray(end_effector[:2], dtype=float) - self.base_xy
        )
        c_current = math.cos(self.base_yaw)
        s_current = math.sin(self.base_yaw)
        measured_hand_local = np.array(
            [
                c_current * current_world_offset[0]
                + s_current * current_world_offset[1],
                -s_current * current_world_offset[0]
                + c_current * current_world_offset[1],
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(measured_hand_local)):
            measured_hand_local = (
                np.array([TISSUE_HAND_CENTER_X_M, 0.0], dtype=float)
                if self._uses_tissue_two_hand_grasp
                else HAND_WORKPOINT_XY_M.copy()
            )
        grasp_yaw = self._target_grasp_yaw()
        c_goal = math.cos(grasp_yaw)
        s_goal = math.sin(grasp_yaw)
        hand_offset = np.array(
            [
                c_goal * measured_hand_local[0]
                - s_goal * measured_hand_local[1],
                s_goal * measured_hand_local[0]
                + c_goal * measured_hand_local[1],
            ],
            dtype=float,
        )
        goal_base = np.array(
            [self.OBJECT_WORLD[0], self.CREEP_STOP_Y], dtype=float
        ) - hand_offset
        delta = goal_base - self.base_xy
        goal_distance = float(np.linalg.norm(delta))
        goal_bearing = math.atan2(float(delta[1]), float(delta[0]))
        alpha = wrap_to_pi(goal_bearing - self.base_yaw)
        beta = wrap_to_pi(grasp_yaw - self.base_yaw - alpha)
        pose_curve_angular = (
            DIRECT_POSE_GUIDANCE_ALPHA_KP * alpha
            + DIRECT_POSE_GUIDANCE_BETA_KP * beta
        )
        self._fine_heading_error_rad = alpha
        self._fine_grasp_yaw_error_rad = wrap_to_pi(
            grasp_yaw - self.base_yaw
        )
        heading_tolerance = (
            SUBSEQUENT_FINE_APPROACH_INITIAL_HEADING_TOL_RAD
            if self._picked_count > 0
            else FINE_APPROACH_INITIAL_HEADING_TOL_RAD
        )
        self._fine_initial_heading_tolerance_rad = heading_tolerance
        if not self._fine_initial_heading_aligned:
            # The observation pose is clear of the shelf, so immediately use
            # the full pose curve.  Forward speed is naturally reduced by
            # cos(alpha) while steering catches up; there is no serial
            # rotate-then-drive phase and therefore no late correction debt.
            self._fine_initial_heading_aligned = True
            self._fine_progress_at = self.now()
            self.get_logger().info(
                "FINE_APPROACH_CONTINUOUS_ALIGNMENT_START "
                f"error={alpha:.3f}rad "
                f"tolerance={heading_tolerance:.3f}rad; "
                "translation_and_steering=true"
            )
        visual_alignment_pending = _fine_visual_alignment_pending(
            self.now(),
            self._fine_pixel_error_px,
            self._fine_pixel_observation_at,
        )
        terminal_endpoint_control = bool(
            self._fine_initial_heading_aligned
            and forward_remaining
            <= _fine_terminal_ee_control_zone(self.target_kind)
            and (
                not visual_alignment_pending
                or forward_remaining
                <= IMAGE_SERVO_HARD_FREEZE_REMAINING_M
            )
        )
        if terminal_endpoint_control:
            self._fine_pixel_servo_active = False
            self._fine_pixel_correction_radps = None
            self._fine_pixel_linear_scale = 1.0
            error_world = np.array(
                [lateral_remaining, forward_remaining], dtype=float
            )
            error_local = np.array(
                [
                    c_current * error_world[0]
                    + s_current * error_world[1],
                    -s_current * error_world[0]
                    + c_current * error_world[1],
                ],
                dtype=float,
            )
            hand_forward_lever = max(
                0.30, abs(float(measured_hand_local[0]))
            )
            angular_raw = (
                FINE_APPROACH_TERMINAL_EE_KP
                * float(error_local[1])
                / hand_forward_lever
            )
            linear_raw = FINE_APPROACH_TERMINAL_EE_KP * float(
                error_local[0]
            ) + angular_raw * float(measured_hand_local[1])
            if self.target_kind == "sanmingzhi":
                if self._fine_terminal_heading_target_rad is None:
                    # The live pixel servo has already centred this narrow
                    # package.  Preserve that observed approach direction;
                    # do not generate a fresh turn merely to chase a generic
                    # shelf-normal chassis pose.
                    self._fine_terminal_heading_target_rad = self.base_yaw
                    self._fine_progress_at = self.now()
                    self.get_logger().info(
                        "SANMINGZHI_VISUAL_HEADING_LOCKED "
                        f"yaw={self.base_yaw:.3f}rad; "
                        "continue endpoint insertion"
                    )
                (
                    self._fine_terminal_heading_error_rad,
                    heading_correction,
                ) = _sanmingzhi_terminal_heading_hold(
                    self._fine_terminal_heading_target_rad,
                    self.base_yaw,
                )
                angular_raw += heading_correction
                self._fine_control_mode = "terminal_visual_heading_hold"
            else:
                self._fine_terminal_heading_error_rad = None
                self._fine_control_mode = "terminal_end_effector"
            commanded_linear = min(
                max(0.0, float(linear_speed)), max(0.0, linear_raw)
            )
        else:
            pixel_correction = self._update_fine_pixel_servo()
            if pixel_correction is not None:
                # Live image-space alignment has priority in the open first
                # segment.  Keep only a small final-yaw feed-forward so the
                # world-pose curve cannot saturate steering in the opposite
                # direction and drown out the current camera measurement.
                angular_raw = _image_servo_angular_command(
                    pixel_correction,
                    wrap_to_pi(grasp_yaw - self.base_yaw),
                )
                self._fine_control_mode = "early_live_pixel_servo"
                self._fine_guidance_source = "live_image_pixel_servo"
            else:
                angular_raw = (
                    pose_curve_angular
                    - DIRECT_POSE_GUIDANCE_LATERAL_KP * lateral_remaining
                )
                self._fine_control_mode = "pose_curve_memory_fallback"
            commanded_linear = min(
                max(0.0, float(linear_speed)),
                DIRECT_POSE_GUIDANCE_DISTANCE_KP * goal_distance,
            ) * max(
                FINE_APPROACH_MIN_FORWARD_ALIGNMENT_SCALE,
                math.cos(alpha),
            ) * self._fine_pixel_linear_scale

        # Do not let the faster observation-to-shelf cruise consume the room
        # still needed to finish lateral alignment.  This governor is based on
        # both geometry and current speed: it reserves more depth for large X
        # error, then caps velocity to what can be stopped within the remaining
        # free distance under the command acceleration limit.
        precision_profile = _precision_grasp_profile(self.target_kind)
        coupled_alignment = self.target_kind in COUPLED_ALIGNMENT_KINDS
        level = (
            ""
            if self._current_target is None
            else str(self._current_target.get("level", ""))
        )
        if precision_profile is not None:
            final_lateral_tolerance = float(precision_profile[0])
        elif coupled_alignment:
            final_lateral_tolerance = COUPLED_ALIGNMENT_FINAL_TOL_M
        else:
            final_lateral_tolerance = _fine_lateral_tolerance(
                self.target_kind, level
            )
        lateral_excess = max(
            0.0, abs(lateral_remaining) - final_lateral_tolerance
        )
        if lateral_excess > 0.0:
            reserve_ramp = min(
                1.0,
                lateral_excess / FINE_APPROACH_ALIGNMENT_RESERVE_RAMP_M,
            )
            if precision_profile is not None or coupled_alignment:
                alignment_reserve = min(
                    PRECISION_ALIGNMENT_RESERVE_MAX_M,
                    PRECISION_ALIGNMENT_RESERVE_BASE_M * reserve_ramp
                    + PRECISION_ALIGNMENT_RESERVE_GAIN * lateral_excess,
                )
            else:
                alignment_reserve = min(
                    FINE_APPROACH_ALIGNMENT_RESERVE_MAX_M,
                    FINE_APPROACH_ALIGNMENT_RESERVE_BASE_M * reserve_ramp
                    + FINE_APPROACH_ALIGNMENT_RESERVE_GAIN * lateral_excess,
                )
        else:
            alignment_reserve = 0.0
        braking_distance = self._fine_predictive_braking_distance()
        available_advance = max(
            0.0,
            forward_remaining - alignment_reserve - braking_distance,
        )
        brake_deceleration = max(0.10, float(self.max_lin_acc))
        speed_cap = math.sqrt(
            2.0 * brake_deceleration * available_advance
        )
        if lateral_excess > 0.0:
            speed_cap = max(
                speed_cap, FINE_APPROACH_ALIGNMENT_CRAWL_SPEED_MPS
            )
        commanded_linear = min(commanded_linear, speed_cap)
        self._fine_alignment_reserve_m = alignment_reserve
        self._fine_braking_distance_m = braking_distance
        self._fine_speed_cap_mps = speed_cap
        depth_fraction = float(
            np.clip(
                max(0.0, forward_remaining)
                / FINE_APPROACH_NEAR_ANGULAR_ZONE_M,
                0.0,
                1.0,
            )
        )
        lateral_fraction = float(
            np.clip(
                lateral_excess
                / FINE_APPROACH_NEAR_LATERAL_ZONE_M,
                0.0,
                1.0,
            )
        )
        # Do not remove steering authority merely because depth is nearly
        # complete.  Authority tapers only when both endpoint axes are near
        # their targets; over the rest of the route steering stays active.
        near_fraction = max(depth_fraction, lateral_fraction)
        angular_limit = (
            FINE_APPROACH_MIN_NEAR_ANGULAR_RADPS
            + (max_angular - FINE_APPROACH_MIN_NEAR_ANGULAR_RADPS)
            * near_fraction
        )
        self._fine_angular_limit_radps = angular_limit
        clipped_angular = float(
            np.clip(angular_raw, -angular_limit, angular_limit)
        )
        angular_step = FINE_APPROACH_ANGULAR_ACCEL_RADPS2 * self.dt
        angular = float(
            np.clip(
                clipped_angular,
                float(self.cur_ang) - angular_step,
                float(self.cur_ang) + angular_step,
            )
        )
        return commanded_linear, angular, lateral_remaining, forward_remaining

    def _fine_predictive_braking_distance(self) -> float:
        """Estimate forward coast after commanding zero at the current speed."""

        # The ramped command predicts the next publish; fresh odometry catches
        # physical coasting or slip that is faster than that command.  Use the
        # larger value so braking is never scheduled from a stale low speed.
        measured_speed = 0.0
        if (
            # 仅用 0.5 s 内的里程计速度参与滑行距离预测。
            self.now() - getattr(self, "odom_received_at", -math.inf)
            <= 0.5
        ):
            measured_speed = float(
                getattr(self, "odom_linear_speed", 0.0)
            )
        forward_speed = max(
            0.0, float(self.cur_lin), measured_speed
        )
        self._fine_braking_speed_mps = forward_speed
        deceleration = max(0.10, float(self.max_lin_acc))
        return (
            forward_speed * forward_speed / (2.0 * deceleration)
            + forward_speed * FINE_APPROACH_BRAKE_REACTION_SEC
            + FINE_APPROACH_BRAKE_MARGIN_M
        )

    def _settle_latched_fine_approach(self, *, vision_fresh: bool) -> None:
        """Hold a confirmed safe-near pose through gripper occlusion."""

        self.set_twist(0.0, 0.0)
        # 滚动交接允许底盘在手臂展开期间就开始视觉前进，但绝不能在机械臂
        # 或升降轴尚未真正到达抓取姿态时闭合夹爪。
        if not self._pick_template_ready():
            self._fine_near_since = None
            return
        stopped = (
            abs(self.cur_lin) <= BASE_STOP_EPS
            and abs(self.cur_ang) <= BASE_STOP_EPS
            and self._odom_stopped()
        )
        if not stopped:
            # Keep the geometric latch, but count the three-second dwell only
            # while the chassis is genuinely stationary.
            self._fine_near_since = None
            return

        now = self.now()
        visibility = "fresh" if vision_fresh else "occluded"
        if self._fine_near_since is None:
            self._fine_near_since = now
            self.get_logger().warning(
                "FINE_APPROACH_NEAR_SETTLING "
                f"mode={self._fine_near_mode} "
                f"vision={visibility} "
                f"remaining={self._fine_forward_remaining_m:.3f} "
                f"tolerance={self._fine_near_tolerance_m:.3f} "
                f"lateral_error={self._fine_lateral_error_m:.3f} "
                f"terminal_heading_error="
                f"{self._fine_terminal_heading_error_rad or 0.0:.3f} "
                f"travel={self._fine_travel_m:.3f}/"
                f"{self._fine_travel_limit_m:.3f}"
            )
        elif now - self._fine_near_since >= FINE_APPROACH_NEAR_SETTLE_SEC:
            self._grasp_heading = self.base_yaw
            self.get_logger().warning(
                "FINE_APPROACH_NEAR_ACCEPTED after 3s "
                f"mode={self._fine_near_mode} "
                f"vision={visibility} "
                f"remaining={self._fine_forward_remaining_m:.3f} "
                f"tolerance={self._fine_near_tolerance_m:.3f} "
                f"lateral_error={self._fine_lateral_error_m:.3f} "
                f"terminal_heading_error="
                f"{self._fine_terminal_heading_error_rad or 0.0:.3f}"
            )
            self.mission.transition(CycleState.GRASP)


__all__ = ["EProductCycleVisionMixin"]
