#!/usr/bin/env python3
"""Move requested shelf products to the delivery table.

This is an explicitly commanded integration-test mission.  It does not replace
the official five-item order topic.  Nav2 owns long travel; the Baseline owns
the observation-point-to-shelf approach and retreat.  There is deliberately no
per-product Nav2 pick waypoint: after the fixed E observation locks a target,
the grasp posture is deployed and the chassis advances while continuously
correcting lateral alignment.  Every product uses its calibrated grasp,
transport and placement profile.  The observation is retained as an
ArUco/slot queue so later targets survive the round trip to the table.
``--target-kind random`` extends the same validated motions to A--E: the order
supplies category counts, stable global inventory chooses concrete slots, and
Nav2 failures trigger a different cabinet choice.  A randomized order contains
five distinct physical items from the full catalogue and may repeat classes.

中文说明：这是货架循环抓取主状态机。固定调试模式保留 E 柜行为；random
模式根据订单品类/数量自主选择 A--E 的具体货位，第一件按 E/D/C，后续优先
B 再考虑 A/C，最后才回 D/E。随机任务首件去 E 使用高速里程计速度闭环，
其他长距离运输交给 Nav2；近柜段由视觉/记忆目标闭环控制底盘，各商品按自身
标定的抓取、运输和放置配置执行。下方名称含
``SETTLE``/``DWELL`` 的量是真正停稳时间，含
``TIMEOUT`` 的量只是故障等待上限，不能把二者混为动作延时。
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from enum import Enum, auto
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from nav2_msgs.msg import SpeedLimit
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from .baseline_grasp_controller import (
    DETECT_MIN_SAMPLES,
    DEPLOY_CART_TOL,
    DEPLOY_JOINT_TOL,
    DEPLOY_ROT_TOL,
    GRIP_CLOSE,
    GRIP_OPEN,
    GRASP_ROT,
    INIT_ARM_L,
    INIT_ARM_R,
    wrap_to_pi,
)
from .nav2_manipulation_client import (
    BASE_STOP_EPS,
    LEFT_ARM_STOW_STAGE_DWELL_SEC,
    LEFT_ARM_STOW_WAYPOINTS,
    LEFT_ARM_TRANSPORT,
    LEFT_ARM_TRANSPORT_TOL,
    PLACE_LOWER_SETTLE_SEC,
    PLACE_RELEASE_SETTLE_SEC,
    PLACE_ARM_TOL,
    RIGHT_ARM_RETURN_JOINT_SLEW,
    STOP_SETTLE_SEC,
    Nav2TaskClient,
)
from .navigation.sorting_geometry import (
    CARRY_FOOTPRINT,
    CARRY_HAND_WORKPOINT_XY_M,
    CARRY_HAND_Z_PLUS_SLIDE_M,
    CHENGZI_TABLE_RELEASE_CLEARANCE_M,
    COMPACT_CARRY_ARM,
    COMPACT_CARRY_HAND_WORKPOINT_XY_M,
    COMPACT_CARRY_HAND_Z_PLUS_SLIDE_M,
    COMPACT_CARRY_WRIST_YAW_RAD,
    E_SCAN_POSE,
    E_SHELF_COLUMNS_M,
    E_SHELF_FIRST_ARUCO_ID,
    E_SHELF_PRODUCT_CENTER_Y_M,
    FIXED_CARRY_ARM,
    FIXED_GRASP_ARM,
    GRASP_YAW,
    HAND_WORKPOINT_XY_M,
    HAND_Z_PLUS_SLIDE_M,
    NORMAL_FOOTPRINT,
    SHELF_COLUMNS_M,
    SHELF_FIRST_ARUCO_ID,
    SHELF_NAMES,
    SHELF_PRODUCT_CENTER_Y_M,
    SLIDE_MIN_M,
    SUPPORTED_PRODUCT_KINDS,
    TABLE_APPROACH_X_NEGATIVE_SHIFT_M,
    TABLE_DROP_X_OFFSETS_M,
    TABLE_TOP_Z_M,
    TABLE_Y_MIN_M,
    TISSUE_HALF_WIDTH_M,
    TISSUE_L3_GRASP_ABOVE_CENTER_M,
    bottle_geometry,
    cluster_points,
    e_product_center_z,
    e_shelf_contains,
    grasp_height_offset_for_product,
    grasp_slide_for_product_z,
    lifted_slide,
    mission_name_for_product,
    nearest_e_slot,
    nearest_shelf_slot,
    parse_mission_command,
    shelf_contains,
    shelf_scan_pose,
    table_approach_pose,
    table_drop_slide_for_product,
    table_place_forward_distances,
)
from .navigation.mission_manager import MissionStateMachine
from .navigation.nav2_manager import NavResult
from .kinematics.mmk2_fk import MMK2FK


from .sorting_config import *  # re-export calibrated public constants
from .grasp_profiles import *  # re-export existing pure helper API
from .sorting_state import CycleState
from .tissue_handling import (
    EProductCycleTissueMixin,
    _tissue_preturn_signed_angle_rad,
)
from .inventory_tracking import EProductCycleInventoryMixin
from .product_scheduler import EProductCycleSchedulerMixin
from .visual_servo import EProductCycleVisionMixin
from .motion_control import EProductCycleMotionMixin
from .status_reporting import EProductCycleStatusMixin
from .sorting_state_machine import EProductCycleStateMachineMixin


class EProductCycleClient(
    EProductCycleTissueMixin,
    EProductCycleInventoryMixin,
    EProductCycleSchedulerMixin,
    EProductCycleVisionMixin,
    EProductCycleMotionMixin,
    EProductCycleStatusMixin,
    EProductCycleStateMachineMixin,
    Nav2TaskClient,
):
    """Command-driven shelf product clearing state machine."""

    def __init__(
        self,
        waypoint_path: str,
        scan_slide: float = 0.30,
        target_kind: str = "kele",
        target_sequence: tuple[str, ...] | None = None,
    ):
        requested_mode = str(target_kind).strip().lower()
        self._random_shelf_mode = requested_mode == "random"
        self._mixed_mode = requested_mode == "mixed"
        forced_first_kind = os.environ.get(
            "SUPERMARKET_FORCE_FIRST_TARGET_KIND", ""
        ).strip().lower()
        if forced_first_kind and forced_first_kind not in SUPPORTED_PRODUCT_KINDS:
            raise ValueError(
                "unsupported forced first product: "
                f"{forced_first_kind}; supported={SUPPORTED_PRODUCT_KINDS}"
            )
        self._configured_forced_first_target_kind = (
            forced_first_kind if self._random_shelf_mode else ""
        )
        self._forced_first_target_kind = (
            self._configured_forced_first_target_kind
        )
        if getattr(self, "_random_shelf_mode", False):
            # Every supported class participates in the same random-order
            # contract; repeated classes are represented by separate bodies.
            self._configured_kind_sequence = ()
            self._mission_kind_sequence = []
            self._allowed_target_kinds = tuple(SUPPORTED_PRODUCT_KINDS)
            self.target_kind = "kele"
            self.mission_name = "clear_random"
            node_mode = "random_cycle"
            self._mission_event_prefix = "RANDOM"
        elif self._mixed_mode:
            configured_sequence = tuple(
                str(kind).strip().lower()
                for kind in (
                    target_sequence or ("maidong", "kele", "maidong")
                )
            )
            if not configured_sequence or any(
                kind not in SUPPORTED_PRODUCT_KINDS
                for kind in configured_sequence
            ):
                raise ValueError(
                    "mixed target sequence contains an unsupported product: "
                    f"supported={SUPPORTED_PRODUCT_KINDS}"
                )
            self._configured_kind_sequence = configured_sequence
            self._mission_kind_sequence = list(configured_sequence)
            self._allowed_target_kinds = tuple(
                dict.fromkeys(configured_sequence)
            )
            self.target_kind = configured_sequence[0]
            self.mission_name = "clear_e_mixed"
            node_mode = "e_mixed_cycle"
            self._mission_event_prefix = "E_MIXED"
        else:
            if requested_mode not in SUPPORTED_PRODUCT_KINDS:
                raise ValueError(f"unsupported E target mode: {requested_mode}")
            self._configured_kind_sequence = (requested_mode,)
            self._mission_kind_sequence = [requested_mode]
            self._allowed_target_kinds = (requested_mode,)
            self.target_kind = requested_mode
            self.mission_name = mission_name_for_product(self.target_kind)
            node_mode = f"e_{self.target_kind}_cycle"
            self._mission_event_prefix = f"E_{self.target_kind.upper()}"
        # The base odometry subscription is created by the parent constructor
        # and binds this class's override.  Initialise the full pose first so a
        # very early callback can safely populate the camera projection state.
        self._base_position_xyz: np.ndarray | None = None
        self._base_quaternion_wxyz: np.ndarray | None = None
        self.target_geometry = bottle_geometry(self.target_kind)
        self._event_prefix = f"E_{self.target_kind.upper()}"
        self._table_drop_slide_m = table_drop_slide_for_product(self.target_kind)
        super().__init__(
            node_mode, waypoint_path, scan_slide=scan_slide
        )
        self.mission = MissionStateMachine(self, CycleState.WAIT_COMMAND)
        self.manipulation_enabled = False
        self.task_received = False

        self._accepted_start_keys: set[str] = set()
        self._active_command_key = ""
        self._mission_motion_started_at: float | None = None
        self._mission_motion_finished_at: float | None = None
        self._mission_total_time_sec: float | None = None
        self._latest_inventory: dict[str, Any] = {}
        # Task-scoped, monotonic visual inventory. Stable YOLO/slot sightings
        # are latched here and disappear only after a confirmed delivery (or
        # when a new randomized run_prefix starts).
        self._inventory_ledger_run_prefix: str | None = None
        self._persistent_inventory_slots: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
        self._inventory_conflict_log_keys: set[
            tuple[tuple[str, str, str], str, str]
        ] = set()
        # A stable contradictory classification does not erase the latched
        # slot, but that disputed memory must not dispatch a blind grasp until
        # the original class is seen again or live shelf vision supplies a
        # concrete replacement.
        self._inventory_conflicted_slots: set[
            tuple[str, str, str]
        ] = set()
        self._active_shelf = "E"
        # Exact per-route observation pose. Edge-column targets may offset the
        # fixed cabinet pose; all handoff/camera distance checks use this copy.
        self._random_active_scan_pose: tuple[float, float, float] | None = None
        self._random_planned_marker_id: int | None = None
        self._random_planned_kind: str | None = None
        self._random_remaining_kinds: list[str] = []
        self._random_completed_marker_ids: set[int] = set()
        self._random_failed_marker_ids: set[int] = set()
        # A completed product is removed by physical shelf slot as well as by
        # ArUco ID. This prevents a stale marker association from restoring the
        # same now-empty visual slot under another ID.
        self._random_removed_inventory_slots: set[
            tuple[str, str, str]
        ] = set()
        # Kept as empty compatibility fields for status consumers from older
        # builds.  Current-view absence is no longer allowed to invalidate the
        # detector's persistent inventory; only a confirmed delivery creates a
        # marker/physical-slot tombstone.
        self._random_rejected_inventory_pairs: set[tuple[int, str]] = set()
        self._random_empty_shelf_kinds: set[tuple[str, str]] = set()
        self._random_last_live_candidates: list[dict[str, Any]] = []
        self._random_visited_shelves: set[str] = set()
        self._random_unreachable_shelves: set[str] = set()
        self._random_sweep_retry_count = 0
        self._random_last_decision: dict[str, Any] = {}
        # After the first table delivery, inventory memory must not immediately
        # pull the robot back to D/E. First visit the central B observation pose
        # so one fresh camera window can update the preferred A/B/C group.
        self._random_post_first_abc_scan_pending = False
        # This is deliberately independent of the routing flag above: choosing
        # a remembered target must not disable the outbound A/B/C camera sweep
        # before the chassis has actually travelled past it.
        self._random_post_first_abc_transit_scan_active = False
        self._random_scan_head_mode: str | None = None
        self._required_target_count: int | None = None
        self._mission_target_count = (
            len(self._mission_kind_sequence)
            if self._random_shelf_mode
            else 0
        )
        self._initial_discovered_count = 0
        self._queue_initialized = False
        self._known_targets: dict[int, dict[str, Any]] = {}
        self._target_drop_slots: dict[int, int] = {}
        self._target_queue: deque[int] = deque()
        self._search_points: deque[tuple[str, np.ndarray]] = deque(maxlen=600)
        self._reacquire_points: deque[tuple[float, np.ndarray]] = deque(maxlen=180)
        self._image_reacquire_points: deque[
            tuple[float, np.ndarray, np.ndarray, np.ndarray]
        ] = deque(maxlen=180)
        self._target_attempts: dict[int, int] = {}
        self._grasp_attempts: dict[int, int] = {}
        self._pending_verification: dict[str, Any] | None = None
        self._current_target: dict[str, Any] | None = None
        self._picked_count = 0
        self._delivery_attempt_count = 0
        self._search_mode = "idle"
        self._search_pose_index = 0
        self._active_search_poses: list[tuple[float, float]] = []
        self._stage_ready_since: float | None = None
        self._search_collecting = False
        self._search_detection_frames = 0
        self._search_last_frame_at: float | None = None
        self._target_last_seen_at: float | None = None
        self._fine_vision_lost_at: float | None = None
        self._last_live_track_log = -math.inf
        self._target_slide = scan_slide
        self._pick_retreat_y = (
            TISSUE_PICK_RETREAT_Y_M
            if self._uses_tissue_two_hand_grasp
            else PICK_RETREAT_CLEAR_Y_M
        )
        self._pick_nav_distance_m: float | None = None
        self._pick_nav_yaw_error_rad: float | None = None
        self._nav_goal_distance_m: float | None = None
        self._nav_goal_yaw_error_rad: float | None = None
        self._nav_near_goal_since: float | None = None
        self._nav_near_goal_leg: str | None = None
        self._nav_terminal_hold_commanded = False
        self._nav_pick_prepared = False
        self._nav_pick_handoff_mode = "traditional"
        self._nav_pick_handoff_speed_mps = 0.0
        self._fine_started_from_rolling_handoff = False
        self._fine_first_pick_early_handoff = False
        self._pick_deploy_left_restore_active = False
        self._pick_deploy_left_restore_stage = 0
        self._pick_deploy_left_restore_ready_since: float | None = None
        self._pick_deploy_held_slide_m: float | None = None
        self._first_e_direct_used = False
        self._first_e_direct_active = False
        self._fine_start_xy: np.ndarray | None = None
        self._fine_best_ee_y: float | None = None
        self._fine_best_lateral_error_m: float | None = None
        self._fine_terminal_heading_target_rad: float | None = None
        self._fine_terminal_heading_error_rad: float | None = None
        self._fine_lateral_error_m: float | None = None
        self._fine_progress_at: float | None = None
        self._fine_required_distance_m: float | None = None
        self._fine_travel_m: float | None = None
        self._fine_travel_limit_m: float | None = None
        self._fine_timeout_sec: float | None = None
        self._fine_target_forward_m: float | None = None
        self._fine_forward_remaining_m: float | None = None
        self._fine_forward_error_m: float | None = None
        self._fine_braking_speed_mps: float | None = None
        self._fine_braking_distance_m: float | None = None
        self._fine_dynamic_depth_tolerance_m: float | None = None
        self._fine_alignment_reserve_m: float | None = None
        self._fine_nominal_speed_mps: float | None = None
        self._fine_speed_cap_mps: float | None = None
        self._fine_initial_heading_aligned = False
        self._fine_initial_heading_tolerance_rad = (
            FINE_APPROACH_INITIAL_HEADING_TOL_RAD
        )
        self._fine_heading_error_rad: float | None = None
        self._fine_grasp_yaw_error_rad: float | None = None
        self._fine_angular_limit_radps: float | None = None
        self._fine_control_mode = "idle"
        self._fine_signed_lateral_error_m: float | None = None
        self._fine_guidance_source = "none"
        self._fine_observation_target_world: np.ndarray | None = None
        self._fine_last_live_target_world: np.ndarray | None = None
        self._fine_last_vision_update_at: float | None = None
        self._fine_live_updates_frozen = False
        self._camera_k: np.ndarray | None = None
        self._camera_width_px: int | None = None
        self._camera_height_px: int | None = None
        self._head_camera_fk = MMK2FK()
        self._fine_pixel_servo_active = False
        self._fine_product_pixel: np.ndarray | None = None
        self._fine_gripper_pixel: np.ndarray | None = None
        self._fine_pixel_error_px: float | None = None
        self._fine_pixel_error_raw_px: float | None = None
        self._fine_pixel_correction_radps: float | None = None
        self._fine_pixel_linear_scale = 1.0
        self._fine_pixel_observation_at: float | None = None
        self._fine_pixel_filter_observation_at: float | None = None
        self._grasp_insertion_m: float | None = None
        self._fine_near_latched = False
        self._fine_near_latched_at: float | None = None
        self._fine_near_mode: str | None = None
        self._fine_near_tolerance_m: float | None = None
        self._fine_near_since: float | None = None
        self._fine_precision_aligned = False
        self._last_failure_reason: str | None = None
        self._compact_carry_safe_since: float | None = None
        self._left_arm_stow_stage = 0
        self._left_arm_stow_stage_started_at = 0.0
        self._pick_left_stow_complete = False
        self._pick_right_carry_ready = False
        self._pick_stow_active = False
        self._pick_stow_started_at: float | None = None
        self._return_arm_stage = 0
        self._return_arm_stage_ready_since: float | None = None
        self._return_arms_stowed = False
        self._return_stow_active = False
        self._return_stow_both_arms = False
        self._return_stow_navigation_active = False
        self._return_nav_footprint_restore_active = False
        self._return_stow_started_at: float | None = None
        self._search_arms_restore_active = False
        self._search_arms_ready = False
        self._search_arms_restore_started_at: float | None = None
        self._table_retreat_start_xy: np.ndarray | None = None
        self._table_retreat_heading = 0.0
        # 只有桌边后退距离和折臂净空条件都满足后才置位。计时结束不能仅凭
        # 状态名推断“已经退出到位”，必须同时检查这个实际运动结果。
        self._table_exit_reached = False
        self._release_clearance_slide_m: float | None = None
        self._chengzi_grasp_center_in_hand: np.ndarray | None = None
        self._chengzi_place_initial_slide: float | None = None
        self._chengzi_place_heights = deque(maxlen=30)
        self._chengzi_place_feedback: dict[str, Any] = {}
        self._joint_feedback_received_at: float | None = None
        self._tissue_deploy_arms: tuple[np.ndarray, np.ndarray] | None = None
        self._tissue_clamp_arms: tuple[np.ndarray, np.ndarray] | None = None
        self._tissue_transport_arms: tuple[np.ndarray, np.ndarray] | None = None
        self._tissue_table_arms: tuple[np.ndarray, np.ndarray] | None = None
        self._tissue_release_arms: tuple[np.ndarray, np.ndarray] | None = None
        self._tissue_motion_waypoints: list[
            tuple[float, np.ndarray, np.ndarray]
        ] = []
        self._tissue_motion_index = 0
        self._tissue_motion_ready_since: float | None = None
        self._tissue_motion_started_at: float | None = None
        self._tissue_motion_label: str | None = None
        self._tissue_carry_center_z_m: float | None = None
        self._tissue_transport_slide_m: float | None = None
        self._tissue_table_drop_slide_m: float | None = None
        self._tissue_clamp_ready_since: float | None = None
        self._tissue_nav_speed_limit_active = False
        self._tissue_preturn_source_shelf: str | None = None
        self._tissue_preturn_angle_rad: float | None = None
        self._tissue_preturn_last_yaw: float | None = None
        self._tissue_preturn_accumulated_rad = 0.0
        self._tissue_preturn_target_yaw: float | None = None
        self._tissue_preturn_yaw_error: float | None = None
        self._table_nav_yaw_assist_active = False
        self._table_nav_yaw_assist_error_rad: float | None = None
        self._table_nav_yaw_assist_command_radps: float | None = None

        self._footprint_mode = "normal"
        self._footprint_pending_mode: str | None = None
        self._footprint_futures: list[Any] = []
        # 载物轮廓优先保证机械臂/商品余量；若去桌子的 Nav2 明确失败，允许每件
        # 商品仅用一次真实底盘轮廓重规划，避免物理上能通过的窄道被永久判死。
        self._delivery_nav_chassis_retry_used = False
        self._controller_angular_limit_current = (
            NORMAL_MPPI_MAX_ANGULAR_RADPS
        )
        self._controller_angular_limit_target: float | None = None
        self._controller_angular_limit_future: Any | None = None
        self._footprint_clients = [
            self.create_client(
                SetParameters, "/local_costmap/local_costmap/set_parameters"
            ),
            self.create_client(
                SetParameters, "/global_costmap/global_costmap/set_parameters"
            ),
        ]
        self._controller_parameter_client = self.create_client(
            SetParameters, "/controller_server/set_parameters"
        )

        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, COMMAND_TOPIC, self.command_cb, command_qos)
        self.create_subscription(
            Detection3DArray, "/product/detections", self.product_cb, 10
        )
        self.create_subscription(
            CameraInfo,
            "/head_camera/color/camera_info",
            self._camera_info_cb,
            10,
        )
        self.create_subscription(String, "/inventory/map", self.inventory_cb, 10)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, status_qos)
        # Publish the complete removed set, rather than just the latest ID.
        # TRANSIENT_LOCAL lets a restarted perception node immediately prune
        # its accumulated visual inventory to the same state.
        self._inventory_remove_pub = self.create_publisher(
            String, INVENTORY_REMOVE_TOPIC, status_qos
        )
        self._speed_limit_pub = self.create_publisher(
            SpeedLimit, "/speed_limit", 10
        )
        # 每 1 s 发布一次任务诊断快照；这只是状态上报周期，不阻塞抓放控制。
        self.create_timer(1.0, self.publish_status)

        self._validate_fixed_template()
        self.get_logger().info(
            f"{self._mission_event_prefix}_CYCLE_READY "
            f"command_topic={COMMAND_TOPIC} "
            f"status_topic={STATUS_TOPIC} mission={self.mission_name} "
            f"sequence={self._mission_kind_sequence} "
            f"radius={self.target_geometry.radius_m:.4f}m "
            f"half_height={self.target_geometry.half_height_m:.4f}m"
        )

    def _set_active_target_kind(self, kind: str) -> None:
        """Switch product-specific grasp/place geometry for a queued target."""

        kind = str(kind).strip().lower()
        if kind not in self._allowed_target_kinds:
            raise ValueError(f"target kind is outside this mission: {kind}")
        changed = kind != self.target_kind
        self.target_kind = kind
        self.target_geometry = bottle_geometry(kind)
        self._table_drop_slide_m = table_drop_slide_for_product(kind)
        event_shelf = (
            self._active_shelf
            if getattr(self, "_random_shelf_mode", False)
            else "E"
        )
        self._event_prefix = f"{event_shelf}_{kind.upper()}"
        if changed and hasattr(self, "mission"):
            self.get_logger().info(
                f"{self._mission_event_prefix}_ACTIVE_KIND kind={kind} "
                f"shelf={event_shelf} "
                f"drop_slide={self._table_drop_slide_m:.3f}"
            )

    def js_cb(self, msg) -> None:
        super().js_cb(msg)
        self._joint_feedback_received_at = self.now()

    def odom_cb(self, msg) -> None:
        super().odom_cb(msg)
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self._base_position_xyz = np.array(
            [position.x, position.y, position.z], dtype=float
        )
        self._base_quaternion_wxyz = np.array(
            [
                orientation.w,
                orientation.x,
                orientation.y,
                orientation.z,
            ],
            dtype=float,
        )
        if (
            self._mission_motion_started_at is None
            and bool(getattr(self, "_active_command_key", ""))
            and hasattr(self, "mission")
            and self.mission.state not in {
                CycleState.WAIT_COMMAND,
                CycleState.DONE,
                CycleState.STOPPED,
                CycleState.FAILED,
            }
            and (
                float(getattr(self, "odom_linear_speed", 0.0))
                >= MISSION_TIMER_LINEAR_START_MPS
                or float(getattr(self, "odom_angular_speed", 0.0))
                >= MISSION_TIMER_ANGULAR_START_RADPS
            )
        ):
            self._mission_motion_started_at = self.now()
            # 静默记录起点；只在任务结束时向 Client 终端打印一次总时长。

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        camera_k = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if np.all(np.isfinite(camera_k)) and camera_k[0, 0] > 1.0:
            self._camera_k = camera_k
            self._camera_width_px = int(msg.width)
            self._camera_height_px = int(msg.height)

    def _capture_chengzi_grasp_geometry(self) -> bool:
        """Remember the shelf-observed sphere centre relative to the held hand.

        This accounts for the actual grasp height/depth and subsequent wrist
        rotation. It is an estimate assuming the fruit remains held, not a
        contact sensor or proof that the fruit cannot slip during transport.
        """
        if (
            self.OBJECT_WORLD is None
            or self._joint_feedback_received_at is None
            # 橙子抓取几何只接受 0.25 s 内的新鲜关节反馈。
            or not 0.0 <= self.now() - self._joint_feedback_received_at <= 0.25
        ):
            return False
        pose = self.ee_footprint_pose()
        centre = self.world_to_footprint(self.OBJECT_WORLD)
        offset = pose[:3, :3].T @ (centre - pose[:3, 3])
        if not np.all(np.isfinite(offset)) or np.linalg.norm(offset) > 0.12:
            return False
        self._chengzi_grasp_center_in_hand = offset.copy()
        self.get_logger().info(
            f"CHENGZI_HELD_CENTRE_CAPTURED offset={np.round(offset, 4)}"
        )
        return True

    def _chengzi_estimated_bottom_z(self) -> float:
        pose = self.ee_footprint_pose()
        centre = (
            pose[:3, 3]
            + pose[:3, :3] @ self._chengzi_grasp_center_in_hand
        )
        return float(centre[2] - self.target_geometry.half_height_m)

    def _calibrate_chengzi_table_drop_slide(self) -> bool:
        """Use measured loaded-arm FK, including its steady servo error."""
        if (
            self._place_advanced_arm is None
            or self._chengzi_grasp_center_in_hand is None
        ):
            self.get_logger().error(
                "CHENGZI_TABLE_DROP_CALIBRATION missing grasp/advanced arm"
            )
            return False
        bottom_z = self._chengzi_estimated_bottom_z()
        desired_bottom_z = TABLE_TOP_Z_M + CHENGZI_TABLE_RELEASE_CLEARANCE_M
        # Increasing slide lowers the hand. Use the currently published slide
        # command, not measured slide: the latter runs ~6 mm ahead under load.
        target_slide = float(self.action[2]) + bottom_z - desired_bottom_z
        if not math.isfinite(target_slide) or not SLIDE_MIN_M <= target_slide <= 0.87:
            return False
        self._table_drop_slide_m = target_slide
        self._chengzi_place_initial_slide = target_slide
        self._chengzi_place_heights.clear()
        self._chengzi_place_feedback = {}
        self.get_logger().info(
            "CHENGZI_TABLE_DROP_CALIBRATED "
            f"slide={target_slide:.3f} estimated_bottom_z={bottom_z:.3f} "
            f"desired_bottom_z={desired_bottom_z:.3f} measured_fk=true"
        )
        return True

    def _update_chengzi_table_lower(self) -> bool:
        """Slow near the table; require settled, fresh, two-sided feedback.

        Only small corrections after the slide has stopped are permitted. A
        blocked actuator times out holding the fruit; it must not release just
        because the lowering timer expired or feedback is above the target.
        """
        if self._chengzi_grasp_center_in_hand is None:
            self.joint_slew = 0.0
            return False
        now = self.now()
        bottom_z = self._chengzi_estimated_bottom_z()
        if not math.isfinite(bottom_z):
            self.joint_slew = 0.0
            self._chengzi_place_feedback = {"ready": False, "reason": "nonfinite_height"}
            return False
        error = bottom_z - TABLE_TOP_Z_M - CHENGZI_TABLE_RELEASE_CLEARANCE_M
        samples = self._chengzi_place_heights
        samples.append((now, bottom_z))
        # 速度估计保留约 0.20 s 窗口；至少跨度 0.15 s 才认为导数可信。
        while len(samples) > 2 and now - samples[1][0] >= 0.20:
            samples.popleft()
        duration = now - samples[0][0]
        speed = (
            abs(bottom_z - samples[0][1]) / duration
            if duration >= 0.15 else math.inf
        )
        fresh = (
            self._joint_feedback_received_at is not None
            # 关节反馈超过 0.25 s 视为陈旧，禁止橙子松爪。
            and 0.0 <= now - self._joint_feedback_received_at <= 0.25
        )
        command_done = (
            abs(float(self.action[2]) - self._table_drop_slide_m)
            <= CHENGZI_PLACE_COMMAND_TOL_M
        )
        stopped = (
            speed <= CHENGZI_PLACE_STOP_SPEED_MPS
            and abs((self.jvel or {}).get("slide_joint", math.inf))
            <= CHENGZI_PLACE_STOP_SPEED_MPS
        )
        fraction = float(np.clip(
            (error - 0.02) / (CHENGZI_PLACE_SLOW_DISTANCE_M - 0.02),
            0.0, 1.0,
        ))
        self.joint_slew = (
            CHENGZI_PLACE_FINAL_LOWER_SLEW
            + fraction * (CHENGZI_PLACE_LOWER_JOINT_SLEW
                          - CHENGZI_PLACE_FINAL_LOWER_SLEW)
        )
        if not fresh:
            self.joint_slew = 0.0  # pause lowering until feedback returns
        height_ready = abs(error) <= CHENGZI_PLACE_EE_Z_TOL_M
        arm_ready = self._right_arm_at_target(self._place_advanced_arm)
        if fresh and command_done and stopped and arm_ready and not height_ready:
            correction = float(np.clip(error, -0.005, 0.005))
            candidate = self._table_drop_slide_m + correction
            if (
                abs(candidate - self._chengzi_place_initial_slide)
                <= CHENGZI_PLACE_CORRECTION_LIMIT_M
                and SLIDE_MIN_M <= candidate <= 0.87
            ):
                self._table_drop_slide_m = candidate
                self.tc[2] = candidate
                command_done = False
        ready = bool(fresh and command_done and stopped and arm_ready and height_ready)
        self._chengzi_place_feedback = {
            "estimated_bottom_z": bottom_z,
            "estimated_table_gap": bottom_z - TABLE_TOP_Z_M,
            "height_error": error,
            "vertical_speed": speed if math.isfinite(speed) else None,
            "feedback_fresh": fresh,
            "command_done": command_done,
            "ready": ready,
        }
        return ready

    def _log_chengzi_release_height(self) -> None:
        self.get_logger().info(
            "CHENGZI_RELEASE_HEIGHT "
            + json.dumps(self._chengzi_place_feedback, separators=(",", ":"))
        )

    def _validate_fixed_template(self) -> None:
        _, pose = self.kdl.forward_kinematics(
            np.concatenate([[0.39], FIXED_GRASP_ARM]), index="right"
        )
        expected = np.array(
            [HAND_WORKPOINT_XY_M[0], HAND_WORKPOINT_XY_M[1], HAND_Z_PLUS_SLIDE_M - 0.39]
        )
        error = float(np.linalg.norm(pose[:3, 3] - expected))
        if error > 0.005:
            raise RuntimeError(f"fixed grasp template FK mismatch: {error:.4f} m")
        self.get_logger().info(
            f"FIXED_GRASP_TEMPLATE_READY fk_error={error:.4f}m "
            f"arm={np.round(FIXED_GRASP_ARM, 4).tolist()}"
        )
        _, carry_pose = self.kdl.forward_kinematics(
            np.concatenate([[TRANSPORT_SLIDE_M], FIXED_CARRY_ARM]), index="right"
        )
        carry_expected = np.array(
            [
                CARRY_HAND_WORKPOINT_XY_M[0],
                CARRY_HAND_WORKPOINT_XY_M[1],
                CARRY_HAND_Z_PLUS_SLIDE_M - TRANSPORT_SLIDE_M,
            ]
        )
        carry_error = float(
            np.linalg.norm(carry_pose[:3, 3] - carry_expected)
        )
        carry_tilt = float(
            math.acos(np.clip(carry_pose[2, 2], -1.0, 1.0))
        )
        if carry_error > 0.005 or carry_tilt > math.radians(2.0):
            raise RuntimeError(
                "fixed carry template mismatch: "
                f"position={carry_error:.4f} m tilt={math.degrees(carry_tilt):.2f} deg"
            )
        self.get_logger().info(
            f"FIXED_CARRY_TEMPLATE_READY fk_error={carry_error:.4f}m "
            f"tilt={math.degrees(carry_tilt):.2f}deg"
        )
        _, compact_pose = self.kdl.forward_kinematics(
            np.concatenate([[TRANSPORT_SLIDE_M], COMPACT_CARRY_ARM]),
            index="right",
        )
        compact_expected = np.array(
            [
                COMPACT_CARRY_HAND_WORKPOINT_XY_M[0],
                COMPACT_CARRY_HAND_WORKPOINT_XY_M[1],
                COMPACT_CARRY_HAND_Z_PLUS_SLIDE_M - TRANSPORT_SLIDE_M,
            ]
        )
        compact_error = float(
            np.linalg.norm(compact_pose[:3, 3] - compact_expected)
        )
        compact_tilt = float(
            math.acos(np.clip(compact_pose[2, 2], -1.0, 1.0))
        )
        compact_yaw = float(
            math.atan2(compact_pose[1, 0], compact_pose[0, 0])
        )
        compact_yaw_error = abs(
            (compact_yaw - COMPACT_CARRY_WRIST_YAW_RAD + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        compact_origins = []
        compact_transform = (
            self.kdl.spine.get_transformation_matrix(TRANSPORT_SLIDE_M)
            @ self.kdl.spine2arm.get_transformation_matrix("right")
        )
        for index, joint in enumerate(COMPACT_CARRY_ARM, start=1):
            compact_transform = (
                compact_transform
                @ self.kdl.right_arm.dh.adjacent_transform(
                    float(joint), index
                )
            )
            compact_origins.append(compact_transform[:3, 3].copy())
        compact_origins.append(compact_pose[:3, 3].copy())
        compact_lateral_min = float(
            np.min(np.asarray(compact_origins)[:, 1])
        )
        compact_lateral_max = float(
            np.max(np.asarray(compact_origins)[:, 1])
        )
        if (
            compact_error > 0.005
            or compact_tilt > math.radians(2.0)
            or compact_yaw_error > math.radians(1.0)
            or compact_lateral_min < -0.205
            or compact_lateral_max > 0.100
        ):
            raise RuntimeError(
                "compact carry template mismatch: "
                f"position={compact_error:.4f} m "
                f"tilt={math.degrees(compact_tilt):.2f} deg "
                f"yaw_error={math.degrees(compact_yaw_error):.2f} deg "
                f"lateral=[{compact_lateral_min:.3f},"
                f"{compact_lateral_max:.3f}] m"
            )
        self.get_logger().info(
            f"COMPACT_CARRY_TEMPLATE_READY fk_error={compact_error:.4f}m "
            f"tilt={math.degrees(compact_tilt):.2f}deg "
            f"yaw={compact_yaw:.3f}rad "
            f"lateral=[{compact_lateral_min:.3f},"
            f"{compact_lateral_max:.3f}]m"
        )

    def _compact_carry_metrics(self) -> dict[str, Any]:
        """Measure strict and load-tolerant readiness of the travel pose."""

        if self.jpos is None:
            return {
                "strict": False,
                "safe": False,
                "max_error": None,
                "non_wrist_max_error": None,
                "wrist_error": None,
                "ee": None,
                "tilt": None,
                "lateral_min": None,
                "lateral_max": None,
            }

        measured = np.asarray(self.rarm_meas, dtype=float)
        errors = np.abs(measured - COMPACT_CARRY_ARM)
        non_wrist_max_error = float(np.max(errors[[0, 1, 2, 3, 5]]))
        wrist_error = float(errors[4])
        max_error = float(np.max(errors))
        _, pose = self.kdl.forward_kinematics(
            np.concatenate([[float(self.slide_meas)], measured]),
            index="right",
        )
        ee = np.asarray(pose[:3, 3], dtype=float)
        tilt = float(math.acos(np.clip(pose[2, 2], -1.0, 1.0)))

        transform = (
            self.kdl.spine.get_transformation_matrix(float(self.slide_meas))
            @ self.kdl.spine2arm.get_transformation_matrix("right")
        )
        lateral = []
        for index, joint in enumerate(measured, start=1):
            transform = (
                transform
                @ self.kdl.right_arm.dh.adjacent_transform(
                    float(joint), index
                )
            )
            lateral.append(float(transform[1, 3]))
        lateral.append(float(ee[1]))
        lateral_min = min(lateral)
        lateral_max = max(lateral)

        slide_ready = (
            abs(float(self.slide_meas) - TRANSPORT_SLIDE_M) <= SLIDE_TOL_M
        )
        safe = bool(
            slide_ready
            and non_wrist_max_error < COMPACT_CARRY_NON_WRIST_TOL_RAD
            and wrist_error < COMPACT_CARRY_WRIST_TOL_RAD
            and COMPACT_CARRY_EE_X_RANGE_M[0]
            <= float(ee[0])
            <= COMPACT_CARRY_EE_X_RANGE_M[1]
            and COMPACT_CARRY_EE_Y_RANGE_M[0]
            <= float(ee[1])
            <= COMPACT_CARRY_EE_Y_RANGE_M[1]
            and COMPACT_CARRY_EE_Z_RANGE_M[0]
            <= float(ee[2])
            <= COMPACT_CARRY_EE_Z_RANGE_M[1]
            and tilt <= COMPACT_CARRY_MAX_TILT_RAD
            and COMPACT_CARRY_LINK_Y_RANGE_M[0]
            <= lateral_min
            and lateral_max
            <= COMPACT_CARRY_LINK_Y_RANGE_M[1]
        )
        return {
            "strict": bool(slide_ready and max_error < PLACE_ARM_TOL),
            "safe": safe,
            "max_error": max_error,
            "non_wrist_max_error": non_wrist_max_error,
            "wrist_error": wrist_error,
            "ee": ee,
            "tilt": tilt,
            "lateral_min": lateral_min,
            "lateral_max": lateral_max,
        }





    def _freeze_mission_total_time(self) -> None:
        """Freeze time after the last item is placed and the base exits fully."""

        if self._mission_motion_finished_at is not None:
            return
        self._mission_motion_finished_at = self.now()
        if self._mission_motion_started_at is None:
            self._mission_total_time_sec = None
            self.get_logger().warning(
                "MISSION_TOTAL_TIME_UNAVAILABLE "
                "reason=no_odom_motion_start_detected"
            )
            return
        self._mission_total_time_sec = max(
            0.0,
            self._mission_motion_finished_at
            - self._mission_motion_started_at,
        )
        total_minutes = int(self._mission_total_time_sec // 60.0)
        remaining_seconds = self._mission_total_time_sec - 60.0 * total_minutes
        summary = (
            "[任务计时完成] MISSION_TOTAL_TIME "
            f"seconds={self._mission_total_time_sec:.3f} "
            f"duration={total_minutes:02d}:{remaining_seconds:06.3f}"
        )
        # 总时长只写入固定文本文件，不再经 stdout 或 ROS logger 输出。
        # 每轮覆盖旧值，读取该文件得到的始终是最近一次正常完成结果。
        try:
            MISSION_TOTAL_TIME_OUTPUT_PATH.write_text(
                f"{summary}\n", encoding="utf-8"
            )
        except OSError as exc:
            self.get_logger().error(
                "MISSION_TOTAL_TIME_FILE_WRITE_FAILED "
                f"path={MISSION_TOTAL_TIME_OUTPUT_PATH} error={exc}"
            )




# Preserve the original import name for any local tooling that constructed the
# cola client directly; the implementation is now product-parameterized.
EKeleCycleClient = EProductCycleClient


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-slide", type=float, default=0.30)
    parser.add_argument(
        "--target-kind",
        choices=(*SUPPORTED_PRODUCT_KINDS, "mixed", "random"),
        default="kele",
        help=(
            "fixed E-shelf product class, ordered mixed cycle, or autonomous "
            "A--E random task"
        ),
    )
    parser.add_argument(
        "--target-sequence",
        default="maidong,kele,maidong",
        help="comma-separated product order used by --target-kind mixed",
    )
    parser.add_argument(
        "--waypoints", default=str(project_root / "config" / "nav_waypoints.yaml")
    )
    args = parser.parse_args()
    if not -0.04 <= args.scan_slide <= 0.87:
        parser.error("--scan-slide must be within -0.04..0.87 m")
    args.target_sequence = tuple(
        kind.strip().lower()
        for kind in args.target_sequence.split(",")
        if kind.strip()
    )
    if args.target_kind == "mixed" and (
        not args.target_sequence
        or any(
            kind not in SUPPORTED_PRODUCT_KINDS
            for kind in args.target_sequence
        )
    ):
        parser.error(
            "--target-sequence contains an unsupported product; supported: "
            + ",".join(SUPPORTED_PRODUCT_KINDS)
        )
    return args


def main() -> None:
    args = parse_args()
    if os.environ.get("FULL_ARM_CLEARANCE_VERIFIED", "0") != "1":
        raise SystemExit(
            f"e_{args.target_kind}_cycle is safety-locked: acknowledge the "
            "current Server MJCF "
            "clearance validation, then set FULL_ARM_CLEARANCE_VERIFIED=1"
        )
    rclpy.init()
    node = EProductCycleClient(
        args.waypoints,
        scan_slide=args.scan_slide,
        target_kind=args.target_kind,
        target_sequence=args.target_sequence,
    )
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
