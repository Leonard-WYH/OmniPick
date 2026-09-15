#!/usr/bin/env python3
"""Nine-class RGB-D product perception with a fixed Server-MJCF ArUco map.

Product classes come from YOLO. Camera-visible ArUco IDs are preferred when
pairing products with shelf slots in image space.  If an RGB-D product has a
valid world point but no same-frame ArUco match, the copied fixed truth table in
``config/aruco_truth.json`` provides a geometry-gated slot fallback.

Published topics
----------------
``/product/detections``
    ``vision_msgs/Detection3DArray`` containing all product surface points in
    the odom frame.
``/aruco/detections``
    JSON marker observations with public shelf/level/column decoding.
``/inventory/observations``
    JSON product-to-marker associations from the current image.
``/inventory/map``
    Multi-frame accumulated slot map. This map starts empty on every launch.
``/inventory/remove``
    Complete mission-local list of shelf products already removed. Matching
    vote histories are deleted and cannot repopulate the inventory map.
``/aruco/map``
    Complete 45-marker fixed truth map, available without camera observations.
``/inventory/waypoints``
    Shelf-line fits plus safe scan and requested-product approach poses.
``/inventory/shelf_scan_poses`` and ``/inventory/target_approach_poses``
    ``PoseArray`` versions of the generated waypoints for RViz.
``/product/result_image``
    RGB debug image containing product boxes, markers and associations.

中文说明：九类商品、ArUco 货位与多帧投票的感知节点。RGB 回调驱动检测；
``ARUCO_FAR_DETECTION_INTERVAL`` 的单位是“帧”而非秒，库存地图另以 1 s
周期发布。深度同步容差只用于拒绝过期帧，不会主动 sleep。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
import math
from pathlib import Path
from typing import Any

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.utilities import remove_ros_args
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from ..kinematics.mmk2_fk import MMK2FK
from ..navigation.sorting_geometry import BOTTLE_GEOMETRY
from .yolo_backend import YoloBackend


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PT_WEIGHTS = PROJECT_ROOT / "weights" / "products.pt"
# 仓库仅保留可跨显卡使用的 PyTorch 权重。目标机器上临时生成的 TensorRT
# 引擎仍可通过启动参数 --weights 显式传入。
DEFAULT_WEIGHTS = DEFAULT_PT_WEIGHTS
DEFAULT_ARUCO_TRUTH = PROJECT_ROOT / "config" / "aruco_truth.json"
PRODUCT_NAMES = [
    "sanmingzhi",
    "heweidao",
    "shupian",
    "zhijin",
    "maidong",
    "kouxiangtang",
    "pingguo",
    "chengzi",
    "kele",
]
SHELF_NAMES = ("A", "B", "C", "D", "E")
INVENTORY_REMOVE_TOPIC = "/inventory/remove"
ARUCO_MARKER_SIZE_M = 0.03
ARUCO_ID_MIN = 0
ARUCO_ID_MAX = 44
ARUCO_DETECTION_SCALES = (1.0, 3.0)
ARUCO_FAR_DETECTION_SCALE = 4.0
ARUCO_FAR_DETECTION_INTERVAL = 3  # 每 3 个 RGB 帧执行一次高倍率远距 ArUco 检测
ARUCO_MIN_SIDE_PX = 7.0
ARUCO_MAX_REPROJECTION_ERROR_PX = 2.5
ARUCO_GRID_RECOVERY_MIN_CURRENT = 5
ARUCO_GRID_RECOVERY_MAX_FIT_ERROR_PX = 3.0
ARUCO_GRID_RECOVERY_MAX_CENTER_ERROR_SIDES = 0.80
ARUCO_GRID_RECOVERY_MIN_SIDE_RATIO = 0.55
ARUCO_GRID_RECOVERY_MAX_SIDE_RATIO = 1.60
DEPTH_MAX_M = 4.0
DEPTH_SYNC_TOLERANCE_SEC = 0.20  # RGB/深度时间差超过 0.20 s 时拒绝配对
VOTE_HISTORY = 120
VOTE_RATIO_MIN = 0.60
MAX_PRODUCT_MARKER_PLANAR_M = 0.48
MIN_PRODUCT_MARKER_VERTICAL_M = -0.05
MAX_PRODUCT_MARKER_VERTICAL_M = 0.50
# The fixed truth marker lies on the rear shelf panel at y=3.168 m while the
# products are spawned with their centres at y=3.243 m.  These conservative
# per-axis gates let a depth-backed YOLO detection select the nearest physical
# slot even when no ArUco is decoded in that RGB frame, without allowing an
# aisle/table detection to enter the shelf inventory.  Adjacent columns are
# 0.22 m apart and adjacent levels are at least 0.338 m apart, so the limits
# remain below the corresponding full grid spacing and the closest candidate
# is unambiguous under normal RGB-D pose noise.
FIXED_SLOT_PRODUCT_DEPTH_OFFSET_M = 0.075
FIXED_SLOT_MAX_X_ERROR_M = 0.13
FIXED_SLOT_MAX_Y_ERROR_M = 0.22
FIXED_SLOT_MAX_Z_ERROR_M = 0.15
MAX_MARKER_TRACK_PLANAR_JUMP_M = 0.45
MAX_MARKER_TRACK_VERTICAL_JUMP_M = 0.35
# Camera ArUco poses are used only for image-space product/slot association.
# The authoritative 45-marker world map comes from aruco_truth.json, so a
# camera observation must not be rejected using the historical (and inverted)
# L1/L2/L3 height convention.  Keep only a broad physical sanity bound here;
# ID validity, reprojection error and per-ID jump rejection remain active.
ARUCO_CAMERA_Z_MIN_M = -0.10
ARUCO_CAMERA_Z_MAX_M = 1.55
PROVISIONAL_CLUSTER_RADIUS_M = 0.90
SINGLE_MARKER_PROVISIONAL_MIN_OBSERVATIONS = 12
SHELF_FRONTAGE_INLIER_M = 0.30
SHELF_FRONTAGE_MIN_MARKERS = 4
SHELF_FRONTAGE_MIN_SHELVES = 2
SHELF_LINE_INLIER_M = 0.18
SHELF_MIN_MARKERS = 3
GRID_COLUMN_STEP_MIN_M = 0.12
GRID_COLUMN_STEP_MAX_M = 0.40
GRID_ROW_STEP_MIN_M = 0.25
GRID_ROW_STEP_MAX_M = 0.60
GRID_STEP_SAMPLE_RESIDUAL_M = 0.10
GRID_MODEL_MAX_ANCHOR_RESIDUAL_M = 0.18
GRID_MIN_STEP_SAMPLES = 2

CLASS_COLOURS = [
    (230, 110, 40),
    (60, 180, 75),
    (255, 210, 45),
    (80, 90, 230),
    (210, 90, 210),
    (220, 170, 55),
    (45, 190, 225),
    (150, 95, 220),
    (70, 220, 150),
]


def stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def slot_from_aruco(marker_id: int) -> dict[str, Any]:
    """Decode the public 0..44 ArUco slot numbering from the rules."""

    shelf_index, shelf_offset = divmod(marker_id, 9)
    level_index, column_index = divmod(shelf_offset, 3)
    return {
        "shelf": chr(ord("A") + shelf_index),
        "level": f"L{level_index + 1}",
        "column": f"C{column_index + 1}",
    }


def load_fixed_aruco_truth(path: Path) -> list[dict[str, Any]]:
    """Expand and validate the fixed 45-marker MJCF truth table."""

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to load ArUco truth table {path}: {exc}") from exc
    if payload.get("frame_id") != "odom":
        raise RuntimeError("ArUco truth table must use the odom frame")
    if payload.get("marker_dictionary") != "DICT_4X4_50":
        raise RuntimeError("ArUco truth table dictionary mismatch")
    if not math.isclose(
        float(payload.get("marker_size_m", 0.0)), ARUCO_MARKER_SIZE_M, abs_tol=1e-9
    ):
        raise RuntimeError("ArUco truth table marker size mismatch")

    y = float(payload["y"])
    levels = payload["levels"]
    normal = [float(value) for value in payload["front_normal_xy"]]
    if len(normal) != 2 or not math.isclose(math.hypot(*normal), 1.0, abs_tol=1e-6):
        raise RuntimeError("ArUco truth table front_normal_xy must be a unit vector")

    markers: list[dict[str, Any]] = []
    for shelf_entry in payload["shelves"]:
        shelf = str(shelf_entry["shelf"])
        first_id = int(shelf_entry["first_id"])
        columns = [float(value) for value in shelf_entry["columns"]]
        if len(columns) != 3:
            raise RuntimeError(f"ArUco truth shelf {shelf} must contain three columns")
        for level_index, level in enumerate(("L1", "L2", "L3")):
            for column_index, x in enumerate(columns):
                marker_id = first_id + level_index * 3 + column_index
                expected_slot = slot_from_aruco(marker_id)
                if expected_slot["shelf"] != shelf:
                    raise RuntimeError(
                        f"ArUco truth shelf/id mismatch: shelf={shelf} id={marker_id}"
                    )
                markers.append(
                    {
                        "id": marker_id,
                        **expected_slot,
                        "stable": True,
                        "observations": 0,
                        "world": [x, y, float(levels[level])],
                        "normal_xy": list(normal),
                        "mean_side_px": 0.0,
                        "direct_observations": 0,
                        "grid_recovered_observations": 0,
                        "last_seen": 0.0,
                        "position_source": "server_mjcf_truth",
                    }
                )
    ids = [marker["id"] for marker in markers]
    if ids != list(range(ARUCO_ID_MIN, ARUCO_ID_MAX + 1)):
        raise RuntimeError(f"ArUco truth table must cover IDs 0..44 exactly; got {ids}")
    return markers


def configure_aruco_parameters(scale: float, far_pass: bool = False) -> Any:
    """Create conservative parameters for 7--15 px simulated markers.

    Detection normally runs on the native image and on a 3x image. A throttled
    4x far pass uses AprilTag corner refinement when supported. Error correction
    remains conservative; accepting extra bit errors produced convincing false
    IDs on the perforated shelf panels.
    """

    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.008
    parameters.minCornerDistanceRate = 0.03
    parameters.minMarkerDistanceRate = 0.08
    parameters.minDistanceToBorder = max(3, round(2.0 * scale))
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_APRILTAG
        if far_pass and hasattr(cv2.aruco, "CORNER_REFINE_APRILTAG")
        else cv2.aruco.CORNER_REFINE_SUBPIX
    )
    parameters.cornerRefinementWinSize = max(3, round(2.0 * scale))
    parameters.cornerRefinementMaxIterations = 50
    parameters.cornerRefinementMinAccuracy = 0.01
    parameters.perspectiveRemovePixelPerCell = 8
    parameters.errorCorrectionRate = 0.60
    parameters.detectInvertedMarker = False
    return parameters


def yaw_pose(x: float, y: float, yaw: float) -> Pose:
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.orientation.z = math.sin(0.5 * yaw)
    pose.orientation.w = math.cos(0.5 * yaw)
    return pose


def dominant_marker_cluster(
    markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Choose the observation-weighted connected marker cluster.

    A shelf is much narrower than the multi-metre jumps produced by a false
    small-code decode. Weighting by observations prevents a four-frame false
    marker from pulling a 120-frame real marker halfway across the map.
    """

    if len(markers) <= 1:
        return markers
    points = np.asarray([marker["world"][:2] for marker in markers])
    remaining = set(range(len(markers)))
    components: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            neighbours = [
                candidate
                for candidate in remaining
                if np.linalg.norm(points[current] - points[candidate])
                <= PROVISIONAL_CLUSTER_RADIUS_M
            ]
            for candidate in neighbours:
                remaining.remove(candidate)
                component.append(candidate)
                frontier.append(candidate)
        components.append(component)
    best = max(
        components,
        key=lambda component: (
            sum(markers[index]["observations"] for index in component),
            len(component),
        ),
    )
    return [markers[index] for index in best]


def frontage_consistent_markers(
    markers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject candidates outside the learned common shelf frontage.

    The fit is enabled only after stable markers from at least two shelves form
    a four-marker consensus. No absolute shelf or ArUco world position is used.
    """

    stable_indices = [
        index for index, marker in enumerate(markers) if marker["stable"]
    ]
    if len(stable_indices) < SHELF_FRONTAGE_MIN_MARKERS:
        return markers

    points = np.asarray([marker["world"][:2] for marker in markers])
    best: tuple[tuple[int, int, int, float], np.ndarray, np.ndarray] | None = None
    for offset, first in enumerate(stable_indices):
        for second in stable_indices[offset + 1 :]:
            if markers[first]["shelf"] == markers[second]["shelf"]:
                continue
            delta = points[second] - points[first]
            span = float(np.linalg.norm(delta))
            if span < 0.30:
                continue
            direction = delta / span
            residuals = np.abs(
                (points[:, 0] - points[first, 0]) * direction[1]
                - (points[:, 1] - points[first, 1]) * direction[0]
            )
            inliers = np.asarray(
                [
                    index
                    for index in stable_indices
                    if residuals[index] <= SHELF_FRONTAGE_INLIER_M
                ],
                dtype=np.int64,
            )
            shelves = {markers[index]["shelf"] for index in inliers}
            if (
                len(inliers) < SHELF_FRONTAGE_MIN_MARKERS
                or len(shelves) < SHELF_FRONTAGE_MIN_SHELVES
            ):
                continue
            score = (
                len(shelves),
                len(inliers),
                sum(markers[index]["observations"] for index in inliers),
                span,
            )
            if best is None or score > best[0]:
                best = (score, direction, points[first])

    if best is None:
        return markers
    _, direction, origin = best
    residuals = np.abs(
        (points[:, 0] - origin[0]) * direction[1]
        - (points[:, 1] - origin[1]) * direction[0]
    )
    return [
        marker
        for marker, residual in zip(markers, residuals)
        if residual <= SHELF_FRONTAGE_INLIER_M
    ]


def _robust_grid_step(
    samples: list[np.ndarray],
    *,
    minimum_norm: float,
    maximum_norm: float,
) -> tuple[np.ndarray, int] | None:
    """Return a median grid step after rejecting inconsistent pair samples."""

    if len(samples) < GRID_MIN_STEP_SAMPLES:
        return None
    values = np.asarray(samples, dtype=np.float64)
    candidate = np.median(values, axis=0)
    residuals = np.linalg.norm(values - candidate, axis=1)
    inliers = values[residuals <= GRID_STEP_SAMPLE_RESIDUAL_M]
    if len(inliers) < GRID_MIN_STEP_SAMPLES:
        return None
    candidate = np.median(inliers, axis=0)
    norm = float(np.linalg.norm(candidate))
    if not minimum_norm <= norm <= maximum_norm:
        return None
    return candidate, len(inliers)


def infer_complete_aruco_grids(
    measured_estimates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Infer missing 3x3 marker positions from a measured common grid basis.

    Absolute positions are never hard-coded.  Column and level step vectors are
    learned from stable, directly localized markers that share a row or column.
    Once both vectors have multiple mutually consistent samples, one stable
    marker is enough to anchor another shelf's otherwise identical 3x3 grid.
    """

    stable = [marker for marker in measured_estimates if marker["stable"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for marker in stable:
        grouped[marker["shelf"]].append(marker)

    column_samples: list[np.ndarray] = []
    row_samples: list[np.ndarray] = []
    basis_support_ids: set[int] = set()
    for markers in grouped.values():
        for first_index, first in enumerate(markers):
            first_offset = first["id"] % 9
            first_row, first_column = divmod(first_offset, 3)
            first_world = np.asarray(first["world"], dtype=np.float64)
            for second in markers[first_index + 1 :]:
                second_offset = second["id"] % 9
                second_row, second_column = divmod(second_offset, 3)
                second_world = np.asarray(second["world"], dtype=np.float64)
                if first_row == second_row and first_column != second_column:
                    column_samples.append(
                        (second_world - first_world)
                        / float(second_column - first_column)
                    )
                    basis_support_ids.update((first["id"], second["id"]))
                if first_column == second_column and first_row != second_row:
                    row_samples.append(
                        (second_world - first_world) / float(second_row - first_row)
                    )
                    basis_support_ids.update((first["id"], second["id"]))

    column_result = _robust_grid_step(
        column_samples,
        minimum_norm=GRID_COLUMN_STEP_MIN_M,
        maximum_norm=GRID_COLUMN_STEP_MAX_M,
    )
    row_result = _robust_grid_step(
        row_samples,
        minimum_norm=GRID_ROW_STEP_MIN_M,
        maximum_norm=GRID_ROW_STEP_MAX_M,
    )
    model_payload: dict[str, Any] = {
        "ready": False,
        "column_sample_count": len(column_samples),
        "row_sample_count": len(row_samples),
        "shelves": {},
    }
    measured_by_id = {marker["id"]: dict(marker) for marker in measured_estimates}
    for marker in measured_by_id.values():
        marker["position_source"] = "measured"

    if column_result is None or row_result is None:
        return sorted(measured_by_id.values(), key=lambda marker: marker["id"]), model_payload

    column_step, column_inliers = column_result
    row_step, row_inliers = row_result
    # Increasing IDs move from L1 toward L3, so their vertical component must
    # point downward.  The column direction should remain nearly horizontal.
    if abs(float(column_step[2])) > 0.15 or float(row_step[2]) > -0.20:
        model_payload["reject_reason"] = "implausible_grid_orientation"
        return sorted(measured_by_id.values(), key=lambda marker: marker["id"]), model_payload

    model_payload.update(
        {
            "ready": True,
            "column_step": [float(value) for value in column_step],
            "row_step": [float(value) for value in row_step],
            "column_inlier_count": column_inliers,
            "row_inlier_count": row_inliers,
            "basis_support_marker_ids": sorted(basis_support_ids),
        }
    )

    for shelf_index, shelf in enumerate("ABCDE"):
        anchors = grouped.get(shelf, [])
        if not anchors:
            continue
        origins = []
        for marker in anchors:
            row, column = divmod(marker["id"] % 9, 3)
            origins.append(
                np.asarray(marker["world"], dtype=np.float64)
                - column * column_step
                - row * row_step
            )
        origin = np.median(np.asarray(origins), axis=0)
        residuals = []
        for marker in anchors:
            row, column = divmod(marker["id"] % 9, 3)
            prediction = origin + column * column_step + row * row_step
            residuals.append(
                float(
                    np.linalg.norm(
                        prediction - np.asarray(marker["world"], dtype=np.float64)
                    )
                )
            )
        max_residual = max(residuals, default=0.0)
        if max_residual > GRID_MODEL_MAX_ANCHOR_RESIDUAL_M:
            model_payload["shelves"][shelf] = {
                "accepted": False,
                "anchor_marker_ids": sorted(marker["id"] for marker in anchors),
                "max_anchor_residual_m": max_residual,
            }
            continue

        normals = [
            np.asarray(marker["normal_xy"], dtype=np.float64)
            for marker in anchors
            if np.linalg.norm(marker["normal_xy"]) > 1.0e-6
        ]
        normal = np.zeros(2, dtype=np.float64)
        if normals:
            reference = normals[0]
            aligned = [
                value if np.dot(value, reference) >= 0.0 else -value
                for value in normals
            ]
            normal = np.median(np.asarray(aligned), axis=0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm > 1.0e-6:
                normal /= normal_norm

        source_ids = sorted(marker["id"] for marker in anchors)
        last_seen = max(float(marker["last_seen"]) for marker in anchors)
        mean_side_px = float(np.mean([marker["mean_side_px"] for marker in anchors]))
        predicted: dict[int, list[float]] = {}
        shelf_predictions_valid = True
        for offset in range(9):
            marker_id = shelf_index * 9 + offset
            row, column = divmod(offset, 3)
            world = origin + column * column_step + row * row_step
            if not ARUCO_CAMERA_Z_MIN_M <= float(world[2]) <= ARUCO_CAMERA_Z_MAX_M:
                shelf_predictions_valid = False
                break
            predicted[marker_id] = [float(value) for value in world]
        if not shelf_predictions_valid:
            model_payload["shelves"][shelf] = {
                "accepted": False,
                "anchor_marker_ids": source_ids,
                "max_anchor_residual_m": max_residual,
                "reject_reason": "predicted_level_height_out_of_range",
            }
            continue

        inferred_count = 0
        for marker_id, world in predicted.items():
            existing = measured_by_id.get(marker_id)
            if existing is not None and existing["stable"]:
                continue
            measured_by_id[marker_id] = {
                "id": marker_id,
                **slot_from_aruco(marker_id),
                "stable": True,
                "observations": 0,
                "world": world,
                "normal_xy": [float(value) for value in normal],
                "mean_side_px": mean_side_px,
                "direct_observations": 0,
                "grid_recovered_observations": 0,
                "last_seen": last_seen,
                "position_source": "grid_inferred",
                "inference_anchor_marker_ids": source_ids,
                "inference_max_residual_m": max_residual,
            }
            inferred_count += 1
        model_payload["shelves"][shelf] = {
            "accepted": True,
            "anchor_marker_ids": source_ids,
            "anchor_count": len(source_ids),
            "inferred_count": inferred_count,
            "max_anchor_residual_m": max_residual,
            "origin": [float(value) for value in origin],
        }

    return sorted(measured_by_id.values(), key=lambda marker: marker["id"]), model_payload


class ProductDetectNode(Node):
    def __init__(
        self,
        weights: Path = DEFAULT_WEIGHTS,
        confidence: float = 0.50,
        device: str = "auto",
        publish_result_image: bool = True,
        min_votes: int = 3,
        scan_stand_off: float = 0.85,
        approach_stand_off: float = 0.70,
        aruco_truth: Path = DEFAULT_ARUCO_TRUTH,
    ) -> None:
        super().__init__("product_detect")
        self.bridge = CvBridge()
        self.publish_result_image = publish_result_image
        self.min_votes = min_votes
        self.scan_stand_off = scan_stand_off
        self.approach_stand_off = approach_stand_off
        self.fixed_markers = load_fixed_aruco_truth(aruco_truth)

        self.K: np.ndarray | None = None
        self.D = np.zeros((5, 1), dtype=np.float32)
        self._depth_msg: Image | None = None
        self.base_pos: list[float] | None = None
        self.base_quat: list[float] | None = None
        self.slide = 0.0
        self.head = [0.0, 0.0]
        self.fk = MMK2FK()
        self.requested_kinds: Counter[str] = Counter()

        self.detector = YoloBackend(weights, confidence=confidence, device=device)
        if self.detector.class_names != PRODUCT_NAMES:
            raise RuntimeError(
                "YOLO class names/order mismatch: "
                f"expected={PRODUCT_NAMES}, actual={self.detector.class_names}"
            )

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._aruco_dictionary = dictionary
        self._aruco_detectors = []
        detection_passes = [
            (scale, False) for scale in ARUCO_DETECTION_SCALES
        ] + [(ARUCO_FAR_DETECTION_SCALE, True)]
        for scale, far_pass in detection_passes:
            parameters = configure_aruco_parameters(scale, far_pass=far_pass)
            detector = (
                cv2.aruco.ArucoDetector(dictionary, parameters)
                if hasattr(cv2.aruco, "ArucoDetector")
                else None
            )
            self._aruco_detectors.append(
                (scale, detector, parameters, far_pass)
            )
        self._aruco_frame_index = 0

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        task_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CameraInfo, "/head_camera/color/camera_info", self.camera_info_cb, 10
        )
        self.create_subscription(
            Image,
            "/head_camera/aligned_depth_to_color/image_raw",
            self.depth_cb,
            image_qos,
        )
        self.create_subscription(
            Image, "/head_camera/color/image_raw", self.rgb_cb, image_qos
        )
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry,
            "/slamware_ros_sdk_server_node/odom",
            self.odom_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/supermarket_sorting/task", self.task_cb, task_qos
        )
        self.create_subscription(
            String,
            INVENTORY_REMOVE_TOPIC,
            self.inventory_remove_cb,
            task_qos,
        )

        self.product_pub = self.create_publisher(
            Detection3DArray, "/product/detections", 10
        )
        self.aruco_pub = self.create_publisher(String, "/aruco/detections", 10)
        self.observation_pub = self.create_publisher(
            String, "/inventory/observations", 10
        )
        self.inventory_pub = self.create_publisher(
            String, "/inventory/map", task_qos
        )
        self.marker_map_pub = self.create_publisher(String, "/aruco/map", task_qos)
        self.waypoint_pub = self.create_publisher(
            String, "/inventory/waypoints", task_qos
        )
        self.shelf_pose_pub = self.create_publisher(
            PoseArray, "/inventory/shelf_scan_poses", task_qos
        )
        self.target_pose_pub = self.create_publisher(
            PoseArray, "/inventory/target_approach_poses", task_qos
        )
        self.image_pub = self.create_publisher(Image, "/product/result_image", 5)

        self._slot_votes: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=VOTE_HISTORY)
        )
        self._removed_marker_ids: set[int] = set()
        self._removed_inventory_slots: set[tuple[str, str, str]] = set()
        self._inventory_run_prefix: str | None = None
        self._marker_tracks: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=VOTE_HISTORY)
        )
        self._shelf_front_normals: dict[str, np.ndarray] = {}
        self._last_aruco_reject_log: dict[tuple[int, str], float] = {}
        self._last_grid_recovery_log: dict[int, float] = {}
        self._last_summary_log = 0.0
        # 每 1 s 发布一次累计库存图；不会降低逐帧商品检测速度。
        self.create_timer(1.0, self.publish_inventory_map)
        self.get_logger().info(
            f"product_detect ready; weights={self.detector.active_weights}; "
            f"backend={self.detector.backend}; classes={PRODUCT_NAMES}; "
            f"aruco_scales={ARUCO_DETECTION_SCALES}; "
            f"aruco_far_scale={ARUCO_FAR_DETECTION_SCALE}x/"
            f"{ARUCO_FAR_DETECTION_INTERVAL}frames; "
            f"scan_stand_off={scan_stand_off:.2f}; "
            f"approach_stand_off={approach_stand_off:.2f}; "
            f"layout_truth={aruco_truth}; fixed_markers={len(self.fixed_markers)}"
        )

    def task_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            targets = payload.get("targets", [])
            kinds = [
                str(target["kind"])
                for target in targets
                if isinstance(target, dict) and target.get("kind") in PRODUCT_NAMES
            ]
        except (json.JSONDecodeError, TypeError, KeyError):
            self.get_logger().warning("ignored malformed task message")
            return
        run_prefix = str(payload.get("run_prefix", "")).strip()
        if run_prefix:
            if (
                self._inventory_run_prefix is not None
                and run_prefix != self._inventory_run_prefix
            ):
                # A new randomized scene starts from new visual evidence; do
                # not carry votes or removed-slot tombstones across runs.
                self._slot_votes.clear()
                self._removed_marker_ids.clear()
                self._removed_inventory_slots.clear()
            self._inventory_run_prefix = run_prefix
        self.requested_kinds = Counter(kinds)
        self.get_logger().info(
            f"scan targets={dict(self.requested_kinds)} "
            f"run_prefix={payload.get('run_prefix', 'unknown')}"
        )

    def inventory_remove_cb(self, msg: String) -> None:
        """Delete grasped slots from the accumulated visual inventory."""

        try:
            payload = json.loads(msg.data)
            marker_ids = {
                int(marker_id)
                for marker_id in payload.get("removed_marker_ids", [])
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            self.get_logger().warning("ignored malformed inventory removal")
            return
        run_prefix = str(payload.get("run_prefix", "")).strip()
        if (
            run_prefix
            and self._inventory_run_prefix is not None
            and run_prefix != self._inventory_run_prefix
        ):
            return
        if run_prefix and self._inventory_run_prefix is None:
            self._inventory_run_prefix = run_prefix
        removed_slots: set[tuple[str, str, str]] = set()
        raw_slots = payload.get("removed_slots", [])
        if isinstance(raw_slots, list):
            for item in raw_slots:
                if not isinstance(item, dict):
                    continue
                slot_key = (
                    str(item.get("shelf", "")).strip().upper(),
                    str(item.get("level", "")).strip().upper(),
                    str(item.get("column", "")).strip().upper(),
                )
                if (
                    slot_key[0] in SHELF_NAMES
                    and slot_key[1] in {"L1", "L2", "L3"}
                    and slot_key[2] in {"C1", "C2", "C3"}
                ):
                    removed_slots.add(slot_key)
        newly_removed = marker_ids - self._removed_marker_ids
        newly_removed_slots = (
            removed_slots - self._removed_inventory_slots
        )
        self._removed_marker_ids.update(marker_ids)
        self._removed_inventory_slots.update(removed_slots)
        for marker_id in marker_ids:
            self._slot_votes.pop(marker_id, None)
        # The marker ID normally maps one-to-one to a physical slot.  Also
        # prune by shelf/level/column so a stale association cannot preserve
        # the now-empty location under a conflicting inventory record.
        for marker_id in tuple(self._slot_votes):
            slot = slot_from_aruco(marker_id)
            slot_key = (
                str(slot["shelf"]),
                str(slot["level"]),
                str(slot["column"]),
            )
            if slot_key in self._removed_inventory_slots:
                self._slot_votes.pop(marker_id, None)
        if newly_removed or newly_removed_slots:
            self.get_logger().info(
                "VISUAL_INVENTORY_REMOVED "
                f"ids={sorted(newly_removed)} "
                f"slots={sorted(newly_removed_slots)} "
                f"total_removed={len(self._removed_marker_ids)}"
            )

    def camera_info_cb(self, msg: CameraInfo) -> None:
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        if msg.d:
            self.D = np.asarray(msg.d, dtype=np.float64).reshape(-1, 1)

    def depth_cb(self, msg: Image) -> None:
        self._depth_msg = msg

    def joint_state_cb(self, msg: JointState) -> None:
        joints = {
            name: msg.position[index]
            for index, name in enumerate(msg.name)
            if index < len(msg.position)
        }
        self.slide = float(joints.get("slide_joint", self.slide))
        self.head = [
            float(joints.get("head_yaw_joint", self.head[0])),
            float(joints.get("head_pitch_joint", self.head[1])),
        ]

    def odom_cb(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.base_pos = [position.x, position.y, position.z]
        self.base_quat = [orientation.w, orientation.x, orientation.y, orientation.z]

    def camera_to_odom(self) -> np.ndarray | None:
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(self.slide)
        self.fk.set_head_joints(self.head)
        position, quaternion = self.fk.get_head_camera_pose()
        transform = np.eye(4)
        transform[:3, 3] = position
        transform[:3, :3] = Rotation.from_quat(
            quaternion[[1, 2, 3, 0]]
        ).as_matrix()
        return transform

    def pixel_to_camera(self, u: float, v: float, depth_m: float) -> np.ndarray:
        assert self.K is not None
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return np.array(
            [(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m]
        )

    @staticmethod
    def bbox_depth_m(depth: np.ndarray, detection: dict[str, Any]) -> float:
        """Estimate the visible front surface from the central part of a box."""

        image_height, image_width = depth.shape[:2]
        x0, y0, x1, y1 = detection["xyxy"]
        box_width = max(1, x1 - x0)
        box_height = max(1, y1 - y0)
        inner_x0 = max(0, min(image_width, round(x0 + 0.25 * box_width)))
        inner_x1 = max(0, min(image_width, round(x1 - 0.25 * box_width)))
        inner_y0 = max(0, min(image_height, round(y0 + 0.20 * box_height)))
        inner_y1 = max(0, min(image_height, round(y1 - 0.20 * box_height)))
        if inner_x1 <= inner_x0 or inner_y1 <= inner_y0:
            return 0.0
        values = depth[inner_y0:inner_y1, inner_x0:inner_x1].astype(np.float32)
        valid = values[(values > 0.0) & (values < DEPTH_MAX_M * 1000.0)]
        if valid.size < 6:
            return 0.0
        return float(np.percentile(valid, 20.0)) * 1.0e-3

    def aruco_pose_record(
        self,
        marker_id: int,
        candidate: dict[str, Any],
        camera_to_odom: np.ndarray,
        *,
        grid_recovered: bool = False,
        grid_prediction_error_px: float | None = None,
    ) -> dict[str, Any] | None:
        """Estimate a marker pose after its ID has been safely established."""

        assert self.K is not None
        half = ARUCO_MARKER_SIZE_M * 0.5
        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )
        image_corners = candidate["corners"]
        success, rotation_vector, translation = cv2.solvePnP(
            object_points,
            image_corners,
            self.K,
            self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None
        camera_point = translation.reshape(3)
        if camera_point[2] <= 0.0 or camera_point[2] > DEPTH_MAX_M:
            return None
        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation,
            self.K,
            self.D,
        )
        reprojection_error = float(
            np.mean(
                np.linalg.norm(
                    projected.reshape(4, 2) - image_corners,
                    axis=1,
                )
            )
        )
        if reprojection_error > ARUCO_MAX_REPROJECTION_ERROR_PX:
            return None
        odom_point = (
            camera_to_odom
            @ np.array(
                [camera_point[0], camera_point[1], camera_point[2], 1.0]
            )
        )[:3]
        marker_rotation, _ = cv2.Rodrigues(rotation_vector)
        normal_camera = marker_rotation[:, 2]
        normal_odom = camera_to_odom[:3, :3] @ normal_camera
        normal_xy = normal_odom[:2]
        normal_norm = float(np.linalg.norm(normal_xy))
        if normal_norm > 1.0e-6:
            normal_xy = normal_xy / normal_norm
        else:
            normal_xy = np.zeros(2, dtype=np.float64)
        center = image_corners.mean(axis=0)
        record = {
            "id": marker_id,
            **slot_from_aruco(marker_id),
            "pixel": [float(center[0]), float(center[1])],
            "corners": image_corners,
            "world": [float(value) for value in odom_point],
            "normal_xy": [float(value) for value in normal_xy],
            "side_px": float(candidate["side_px"]),
            "source_scale": float(candidate["source_scale"]),
            "reprojection_error_px": reprojection_error,
            "grid_recovered": grid_recovered,
        }
        if grid_prediction_error_px is not None:
            record["grid_prediction_error_px"] = grid_prediction_error_px
        return record

    def recover_single_missing_marker(
        self,
        records: list[dict[str, Any]],
        rejected_candidates: list[dict[str, Any]],
        camera_to_odom: np.ndarray,
        stable_ids: set[int],
    ) -> list[dict[str, Any]]:
        """Recover one blurred 8/9 shelf code from its constrained grid slot.

        The ID is never guessed from a generic square. Recovery is enabled only
        when the accumulated map already contains eight stable IDs for a shelf,
        the current image supplies enough decoded grid points for a validated
        homography, and one rejected ArUco quadrilateral closely matches both
        the predicted center and the decoded marker size.
        """

        if not rejected_candidates:
            return []
        recovered: list[dict[str, Any]] = []
        for shelf_index, shelf in enumerate("ABCDE"):
            first_id = shelf_index * 9
            expected_ids = set(range(first_id, first_id + 9))
            found_ids = stable_ids & expected_ids
            missing_ids = expected_ids - found_ids
            if len(found_ids) != 8 or len(missing_ids) != 1:
                continue
            missing_id = next(iter(missing_ids))
            current = [record for record in records if record["shelf"] == shelf]
            if len(current) < ARUCO_GRID_RECOVERY_MIN_CURRENT:
                continue
            rows = {(record["id"] - first_id) // 3 for record in current}
            columns = {(record["id"] - first_id) % 3 for record in current}
            if len(rows) < 2 or len(columns) < 2:
                continue

            grid_points = np.asarray(
                [
                    [
                        (record["id"] - first_id) % 3,
                        (record["id"] - first_id) // 3,
                    ]
                    for record in current
                ],
                dtype=np.float32,
            )
            image_points = np.asarray(
                [record["pixel"] for record in current], dtype=np.float32
            )
            homography, inlier_mask = cv2.findHomography(
                grid_points,
                image_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )
            if homography is None or inlier_mask is None:
                continue
            inliers = inlier_mask.reshape(-1).astype(bool)
            if int(np.count_nonzero(inliers)) < 4:
                continue
            fitted = cv2.perspectiveTransform(
                grid_points.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            fit_errors = np.linalg.norm(fitted - image_points, axis=1)
            if float(np.median(fit_errors[inliers])) > ARUCO_GRID_RECOVERY_MAX_FIT_ERROR_PX:
                continue

            missing_offset = missing_id - first_id
            missing_grid = np.asarray(
                [[[(missing_offset % 3), (missing_offset // 3)]]],
                dtype=np.float32,
            )
            predicted = cv2.perspectiveTransform(
                missing_grid, homography
            ).reshape(2)
            median_side = float(
                np.median([record["side_px"] for record in current])
            )
            maximum_center_error = max(
                5.0,
                median_side * ARUCO_GRID_RECOVERY_MAX_CENTER_ERROR_SIDES,
            )
            decoded_centers = np.asarray(
                [record["pixel"] for record in records], dtype=np.float32
            )
            matches: list[tuple[float, float, dict[str, Any]]] = []
            for candidate in rejected_candidates:
                side_ratio = float(candidate["side_px"]) / median_side
                if not (
                    ARUCO_GRID_RECOVERY_MIN_SIDE_RATIO
                    <= side_ratio
                    <= ARUCO_GRID_RECOVERY_MAX_SIDE_RATIO
                ):
                    continue
                center_error = float(
                    np.linalg.norm(candidate["pixel"] - predicted)
                )
                if center_error > maximum_center_error:
                    continue
                if decoded_centers.size and float(
                    np.min(np.linalg.norm(decoded_centers - candidate["pixel"], axis=1))
                ) < 0.55 * median_side:
                    continue
                matches.append(
                    (center_error, abs(side_ratio - 1.0), candidate)
                )
            if not matches:
                continue
            center_error, _, candidate = min(matches, key=lambda item: item[:2])
            record = self.aruco_pose_record(
                missing_id,
                candidate,
                camera_to_odom,
                grid_recovered=True,
                grid_prediction_error_px=center_error,
            )
            if record is None:
                continue
            recovered.append(record)
            now = self.get_clock().now().nanoseconds * 1.0e-9
            # 同一缺失货位的网格恢复日志每 2 s 最多一条，不影响恢复计算。
            if now - self._last_grid_recovery_log.get(missing_id, -math.inf) >= 2.0:
                self.get_logger().info(
                    f"ARUCO_GRID_RECOVERY id={missing_id} shelf={shelf} "
                    f"center_error={center_error:.2f}px "
                    f"side={candidate['side_px']:.1f}px "
                    f"current_decoded={len(current)} stable=8/9"
                )
                self._last_grid_recovery_log[missing_id] = now
        return recovered

    def detect_aruco(
        self, rgb: np.ndarray, camera_to_odom: np.ndarray
    ) -> list[dict[str, Any]]:
        assert self.K is not None
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        raw_candidates: list[dict[str, Any]] = []
        rejected_candidates: list[dict[str, Any]] = []
        stable_ids = {
            int(marker["id"])
            for marker in self.measured_marker_estimates()
            if marker["stable"]
        }
        grid_recovery_enabled = any(
            len(
                stable_ids
                & set(range(shelf_index * 9, shelf_index * 9 + 9))
            )
            == 8
            for shelf_index in range(5)
        )
        self._aruco_frame_index += 1
        run_far_pass = (
            self._aruco_frame_index % ARUCO_FAR_DETECTION_INTERVAL == 0
        )
        for scale, detector, parameters, far_pass in self._aruco_detectors:
            if far_pass and not run_far_pass:
                continue
            scaled_gray = (
                gray
                if scale == 1.0
                else cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
            if detector is not None:
                corners, ids, rejected = detector.detectMarkers(scaled_gray)
            else:
                corners, ids, rejected = cv2.aruco.detectMarkers(
                    scaled_gray,
                    self._aruco_dictionary,
                    parameters=parameters,
                )
            if grid_recovery_enabled:
                for corner in rejected:
                    image_corners = corner.reshape(4, 2).astype(np.float32) / scale
                    edge_lengths = np.linalg.norm(
                        image_corners - np.roll(image_corners, 1, axis=0), axis=1
                    )
                    side_px = float(np.mean(edge_lengths))
                    if side_px < ARUCO_MIN_SIDE_PX:
                        continue
                    rejected_candidates.append(
                        {
                            "corners": image_corners,
                            "pixel": image_corners.mean(axis=0),
                            "side_px": side_px,
                            "source_scale": float(scale),
                        }
                    )
            if ids is None:
                continue
            for corner, raw_id in zip(corners, ids.flatten()):
                marker_id = int(raw_id)
                if marker_id < ARUCO_ID_MIN or marker_id > ARUCO_ID_MAX:
                    continue
                image_corners = corner.reshape(4, 2).astype(np.float32) / scale
                edge_lengths = np.linalg.norm(
                    image_corners - np.roll(image_corners, 1, axis=0), axis=1
                )
                side_px = float(np.mean(edge_lengths))
                if side_px < ARUCO_MIN_SIDE_PX:
                    continue
                raw_candidates.append(
                    {
                        "id": marker_id,
                        "corners": image_corners,
                        "pixel": image_corners.mean(axis=0),
                        "side_px": side_px,
                        "source_scale": float(scale),
                    }
                )

        # Every ID occurs only once in the competition scene. Prefer a native
        # detection, then the largest candidate, and suppress conflicting
        # multi-scale detections at the same image location.
        raw_candidates.sort(
            key=lambda candidate: (
                candidate["source_scale"] == 1.0,
                candidate["side_px"],
            ),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        for candidate in raw_candidates:
            if candidate["id"] in selected_ids:
                continue
            overlaps = any(
                np.linalg.norm(candidate["pixel"] - other["pixel"])
                < max(5.0, 0.5 * min(candidate["side_px"], other["side_px"]))
                for other in selected
            )
            if overlaps:
                continue
            selected.append(candidate)
            selected_ids.add(candidate["id"])

        records: list[dict[str, Any]] = []
        for candidate in selected:
            record = self.aruco_pose_record(
                candidate["id"], candidate, camera_to_odom
            )
            if record is not None:
                records.append(record)
        records.extend(
            self.recover_single_missing_marker(
                records, rejected_candidates, camera_to_odom, stable_ids
            )
        )
        return records

    def track_marker(self, marker: dict[str, Any], seen_at: float) -> bool:
        """Accumulate a camera marker while rejecting impossible geometry."""

        marker_id = int(marker["id"])
        marker_z = float(marker["world"][2])
        if marker_z < ARUCO_CAMERA_Z_MIN_M or marker_z > ARUCO_CAMERA_Z_MAX_M:
            reason = (
                f"z={marker_z:.3f} expected="
                f"{ARUCO_CAMERA_Z_MIN_M:.2f}..{ARUCO_CAMERA_Z_MAX_M:.2f}"
            )
            log_key = (marker_id, "physical_height")
            # 同一 ArUco 拒绝原因每 5 s 最多告警一次，仅减少刷屏。
            if seen_at - self._last_aruco_reject_log.get(log_key, -math.inf) >= 5.0:
                self.get_logger().warning(
                    f"ARUCO_GEOMETRY_REJECT id={marker_id} {reason}"
                )
                self._last_aruco_reject_log[log_key] = seen_at
            return False

        track = self._marker_tracks[marker_id]
        if len(track) >= self.min_votes:
            previous = np.median(
                np.asarray([sample["world"] for sample in track]), axis=0
            )
            current = np.asarray(marker["world"])
            planar_jump = float(np.linalg.norm(current[:2] - previous[:2]))
            vertical_jump = abs(float(current[2] - previous[2]))
            if (
                planar_jump > MAX_MARKER_TRACK_PLANAR_JUMP_M
                or vertical_jump > MAX_MARKER_TRACK_VERTICAL_JUMP_M
            ):
                self.get_logger().warning(
                    "ARUCO_OUTLIER "
                    f"id={marker_id} planar_jump={planar_jump:.3f} "
                    f"vertical_jump={vertical_jump:.3f}"
                )
                return False
        track.append(
            {
                "world": marker["world"],
                "normal_xy": marker["normal_xy"],
                "side_px": marker["side_px"],
                "grid_recovered": bool(marker.get("grid_recovered", False)),
                "seen_at": seen_at,
            }
        )
        return True

    @staticmethod
    def associate_products_to_markers(
        products: list[dict[str, Any]], markers: list[dict[str, Any]]
    ) -> list[tuple[int, int]]:
        """Greedily match each product to the marker immediately below it."""

        candidates: list[tuple[float, int, int]] = []
        for product_index, product in enumerate(products):
            if product.get("world") is None:
                continue
            x0, _, x1, y1 = product["xyxy"]
            width = max(12.0, float(x1 - x0))
            height = max(12.0, float(product["h"]))
            product_x = float(product["x"])
            for marker_index, marker in enumerate(markers):
                marker_x, marker_y = marker["pixel"]
                horizontal = abs(marker_x - product_x)
                below_box = marker_y - float(y1)
                if horizontal > max(48.0, 0.85 * width):
                    continue
                if below_box < -0.20 * height or below_box > max(95.0, 2.0 * height):
                    continue
                product_world = np.asarray(product["world"])
                marker_world = np.asarray(marker["world"])
                planar_distance = float(
                    np.linalg.norm(product_world[:2] - marker_world[:2])
                )
                vertical_distance = float(product_world[2] - marker_world[2])
                if planar_distance > MAX_PRODUCT_MARKER_PLANAR_M:
                    continue
                if not (
                    MIN_PRODUCT_MARKER_VERTICAL_M
                    <= vertical_distance
                    <= MAX_PRODUCT_MARKER_VERTICAL_M
                ):
                    continue
                score = (
                    horizontal / width
                    + abs(below_box) / height
                    + planar_distance / MAX_PRODUCT_MARKER_PLANAR_M
                )
                candidates.append((score, product_index, marker_index))

        matches: list[tuple[int, int]] = []
        used_products: set[int] = set()
        used_markers: set[int] = set()
        for _, product_index, marker_index in sorted(candidates):
            if product_index in used_products or marker_index in used_markers:
                continue
            used_products.add(product_index)
            used_markers.add(marker_index)
            matches.append((product_index, marker_index))
        return matches

    @staticmethod
    def associate_products_to_fixed_slots(
        products: list[dict[str, Any]],
        fixed_markers: list[dict[str, Any]],
        *,
        used_product_indices: set[int] | None = None,
        used_marker_ids: set[int] | None = None,
        removed_marker_ids: set[int] | None = None,
        removed_inventory_slots: set[tuple[str, str, str]] | None = None,
    ) -> list[tuple[int, int]]:
        """Associate unmatched RGB-D products with fixed physical shelf slots.

        Current-frame image-space ArUco associations are supplied through the
        ``used_*`` sets and always keep priority.  The fallback then compares
        every remaining depth-backed YOLO centre with the expected product
        centre above each fixed marker.  A globally sorted greedy pass makes
        the result one-product/one-slot, while removal tombstones prevent a
        previously cleared slot from being reconstructed by residual YOLO
        detections.

        The returned marker index refers to ``fixed_markers``.
        """

        used_products = set(used_product_indices or ())
        reserved_marker_ids = set(used_marker_ids or ())
        removed_ids = set(removed_marker_ids or ())
        removed_slots = set(removed_inventory_slots or ())
        candidates: list[tuple[float, int, int]] = []

        for product_index, product in enumerate(products):
            if product_index in used_products or product.get("world") is None:
                continue
            kind = str(product.get("class", "")).strip().lower()
            geometry = BOTTLE_GEOMETRY.get(kind)
            if geometry is None:
                continue
            product_world = np.asarray(product["world"], dtype=np.float64)
            if product_world.shape != (3,) or not np.all(np.isfinite(product_world)):
                continue

            for marker_index, marker in enumerate(fixed_markers):
                marker_id = int(marker["id"])
                marker_slot = (
                    str(marker["shelf"]),
                    str(marker["level"]),
                    str(marker["column"]),
                )
                if (
                    marker_id in reserved_marker_ids
                    or marker_id in removed_ids
                    or marker_slot in removed_slots
                ):
                    continue
                marker_world = np.asarray(marker["world"], dtype=np.float64)
                if marker_world.shape != (3,) or not np.all(np.isfinite(marker_world)):
                    continue
                expected_product_world = marker_world + np.asarray(
                    (
                        0.0,
                        FIXED_SLOT_PRODUCT_DEPTH_OFFSET_M,
                        geometry.half_height_m,
                    ),
                    dtype=np.float64,
                )
                errors = np.abs(product_world - expected_product_world)
                if (
                    errors[0] > FIXED_SLOT_MAX_X_ERROR_M
                    or errors[1] > FIXED_SLOT_MAX_Y_ERROR_M
                    or errors[2] > FIXED_SLOT_MAX_Z_ERROR_M
                ):
                    continue
                score = float(
                    errors[0] / FIXED_SLOT_MAX_X_ERROR_M
                    + errors[1] / FIXED_SLOT_MAX_Y_ERROR_M
                    + errors[2] / FIXED_SLOT_MAX_Z_ERROR_M
                )
                candidates.append((score, product_index, marker_index))

        matches: list[tuple[int, int]] = []
        for _, product_index, marker_index in sorted(candidates):
            marker_id = int(fixed_markers[marker_index]["id"])
            if product_index in used_products or marker_id in reserved_marker_ids:
                continue
            used_products.add(product_index)
            reserved_marker_ids.add(marker_id)
            matches.append((product_index, marker_index))
        return matches

    def publish_product_detections(
        self, products: list[dict[str, Any]], stamp: Any
    ) -> None:
        message = Detection3DArray()
        message.header.stamp = stamp
        message.header.frame_id = "odom"
        for product_index, product in enumerate(products):
            if product.get("world") is None:
                continue
            detection = Detection3D()
            detection.header = message.header
            # Detection3D has no standard field for the source-image pixel.
            # Keep the historical ``class:index`` prefix and append the exact
            # YOLO box centre so the grasp controller can run a synchronized
            # image-space servo without subscribing to the debug image.
            detection.id = (
                f"{product['class']}:{product_index}"
                f"|pixel={float(product['x']):.3f},{float(product['y']):.3f}"
            )
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = product["class"]
            hypothesis.hypothesis.score = float(product["conf"])
            hypothesis.pose.pose.position.x = product["world"][0]
            hypothesis.pose.pose.position.y = product["world"][1]
            hypothesis.pose.pose.position.z = product["world"][2]
            hypothesis.pose.pose.orientation.w = 1.0
            detection.results.append(hypothesis)
            detection.bbox.center.position.x = product["world"][0]
            detection.bbox.center.position.y = product["world"][1]
            detection.bbox.center.position.z = product["world"][2]
            detection.bbox.center.orientation.w = 1.0
            depth_m = float(product["depth_m"])
            detection.bbox.size.x = max(
                0.01, float(product["w"]) * depth_m / float(self.K[0, 0])
            )
            detection.bbox.size.y = max(
                0.01, float(product["h"]) * depth_m / float(self.K[1, 1])
            )
            detection.bbox.size.z = 0.05
            message.detections.append(detection)
        self.product_pub.publish(message)

    def rgb_cb(self, msg: Image) -> None:
        if self.K is None or self._depth_msg is None:
            return
        if (
            abs(stamp_seconds(msg.header.stamp) - stamp_seconds(self._depth_msg.header.stamp))
            > DEPTH_SYNC_TOLERANCE_SEC
        ):
            return
        camera_to_odom = self.camera_to_odom()
        if camera_to_odom is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)
        products = self.detector.detect(rgb)
        for product in products:
            depth_m = self.bbox_depth_m(depth, product)
            if depth_m <= 0.0:
                product["world"] = None
                continue
            camera_point = self.pixel_to_camera(product["x"], product["y"], depth_m)
            odom_point = (
                camera_to_odom
                @ np.array(
                    [camera_point[0], camera_point[1], camera_point[2], 1.0]
                )
            )[:3]
            product["depth_m"] = depth_m
            product["world"] = [float(value) for value in odom_point]

        current_time = self.get_clock().now().nanoseconds * 1.0e-9
        detected_markers = self.detect_aruco(rgb, camera_to_odom)
        markers = [
            marker
            for marker in detected_markers
            if self.track_marker(marker, current_time)
        ]
        matches = self.associate_products_to_markers(products, markers)
        # Preserve the stronger current-frame image-space associations first.
        # Only YOLO detections left unmatched by that pass may use the fixed
        # Server-MJCF shelf geometry as a persistent-inventory fallback.
        associations: list[tuple[int, dict[str, Any], str]] = [
            (product_index, markers[marker_index], "image_aruco")
            for product_index, marker_index in matches
        ]
        fixed_slot_matches = self.associate_products_to_fixed_slots(
            products,
            self.fixed_markers,
            used_product_indices={product_index for product_index, _ in matches},
            used_marker_ids={
                int(markers[marker_index]["id"])
                for _, marker_index in matches
            },
            removed_marker_ids=self._removed_marker_ids,
            removed_inventory_slots=self._removed_inventory_slots,
        )
        associations.extend(
            (
                product_index,
                self.fixed_markers[marker_index],
                "fixed_truth_geometry",
            )
            for product_index, marker_index in fixed_slot_matches
        )
        observations: list[dict[str, Any]] = []
        for product_index, marker, association_source in associations:
            product = products[product_index]
            if product.get("world") is None:
                continue
            marker_slot = (
                str(marker["shelf"]),
                str(marker["level"]),
                str(marker["column"]),
            )
            if (
                marker["id"] in self._removed_marker_ids
                or marker_slot in self._removed_inventory_slots
            ):
                # This physical shelf slot has already been carried away.
                # Ignore residual detector/association votes so the empty
                # location cannot become a target again.
                continue
            observation = {
                "aruco_id": marker["id"],
                "shelf": marker["shelf"],
                "level": marker["level"],
                "column": marker["column"],
                "kind": product["class"],
                "confidence": float(product["conf"]),
                "product_world": product["world"],
                "marker_world": marker["world"],
                "planar_distance": float(
                    np.linalg.norm(
                        np.asarray(product["world"][:2])
                        - np.asarray(marker["world"][:2])
                    )
                ),
                "vertical_distance": float(
                    product["world"][2] - marker["world"][2]
                ),
                "association_source": association_source,
                "seen_at": current_time,
            }
            observations.append(observation)
            self._slot_votes[marker["id"]].append(observation)

        marker_payload = [
            {
                key: value
                for key, value in marker.items()
                if key not in {"corners"}
            }
            for marker in markers
        ]
        self.aruco_pub.publish(
            String(
                data=json.dumps(
                    {"frame_id": "odom", "markers": marker_payload},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
        self.observation_pub.publish(
            String(
                data=json.dumps(
                    {"frame_id": "odom", "observations": observations},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
        self.publish_product_detections(products, msg.header.stamp)
        if self.publish_result_image:
            self.publish_debug_image(
                rgb,
                products,
                markers,
                matches,
                msg,
                fixed_truth_match_count=len(fixed_slot_matches),
            )

    def publish_debug_image(
        self,
        rgb: np.ndarray,
        products: list[dict[str, Any]],
        markers: list[dict[str, Any]],
        matches: list[tuple[int, int]],
        source_msg: Image,
        *,
        fixed_truth_match_count: int = 0,
    ) -> None:
        visual = rgb.copy()
        for product in products:
            class_id = int(product["class_id"])
            colour = CLASS_COLOURS[class_id % len(CLASS_COLOURS)]
            x0, y0, x1, y1 = product["xyxy"]
            cv2.rectangle(visual, (x0, y0), (x1, y1), colour, 2, cv2.LINE_AA)
            depth_text = (
                f" {product['depth_m']:.2f}m" if product.get("depth_m") else " no-depth"
            )
            cv2.putText(
                visual,
                f"{product['class']} {product['conf']:.2f}{depth_text}",
                (x0, max(16, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
                cv2.LINE_AA,
            )
        for marker in markers:
            corners = marker["corners"].astype(int)
            cv2.polylines(visual, [corners], True, (0, 190, 255), 2, cv2.LINE_AA)
            center = tuple(np.round(marker["pixel"]).astype(int))
            cv2.putText(
                visual,
                f"ID{marker['id']}{'*' if marker.get('grid_recovered') else ''} "
                f"{marker['shelf']}/{marker['level']}/{marker['column']}",
                (center[0] - 30, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 190, 255),
                1,
                cv2.LINE_AA,
            )
        for product_index, marker_index in matches:
            product_center = (products[product_index]["x"], products[product_index]["y"])
            marker_center = tuple(np.round(markers[marker_index]["pixel"]).astype(int))
            cv2.line(visual, product_center, marker_center, (255, 255, 255), 1, cv2.LINE_AA)

        validated_markers = self.marker_estimates()
        validated_marker_ids = {marker["id"] for marker in validated_markers}
        stable_count = sum(
            1
            for marker_id, votes in self._slot_votes.items()
            if marker_id in validated_marker_ids and len(votes) >= self.min_votes
        )
        stable_marker_count = sum(
            marker["stable"] for marker in validated_markers
        )
        status = (
            f"products={len(products)} aruco={len(markers)} "
            f"matched={len(matches) + fixed_truth_match_count} "
            f"(aruco={len(matches)},truth={fixed_truth_match_count}) "
            f"markers={stable_marker_count}/45 "
            f"slots={stable_count}/45"
        )
        cv2.rectangle(visual, (0, 0), (visual.shape[1], 24), (20, 20, 20), -1)
        cv2.putText(
            visual,
            status,
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        output = self.bridge.cv2_to_imgmsg(visual, "bgr8")
        output.header = source_msg.header
        self.image_pub.publish(output)

    def measured_marker_estimates(self) -> list[dict[str, Any]]:
        """Return median fixed-marker poses independent of product matching."""

        raw_estimates: list[dict[str, Any]] = []
        for marker_id, samples in sorted(self._marker_tracks.items()):
            if not samples:
                continue
            positions = np.asarray([sample["world"] for sample in samples])
            position = np.median(positions, axis=0)
            slot = slot_from_aruco(marker_id)
            if not (
                ARUCO_CAMERA_Z_MIN_M
                <= float(position[2])
                <= ARUCO_CAMERA_Z_MAX_M
            ):
                continue
            normals = [
                np.asarray(sample["normal_xy"], dtype=np.float64)
                for sample in samples
                if np.linalg.norm(sample["normal_xy"]) > 1.0e-6
            ]
            normal = np.zeros(2, dtype=np.float64)
            if normals:
                reference = normals[0]
                aligned = [
                    vector if np.dot(vector, reference) >= 0.0 else -vector
                    for vector in normals
                ]
                normal = np.median(np.asarray(aligned), axis=0)
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm > 1.0e-6:
                    normal /= normal_norm
            raw_estimates.append(
                {
                    "id": marker_id,
                    **slot,
                    "stable": len(samples) >= self.min_votes,
                    "observations": len(samples),
                    "world": [float(value) for value in position],
                    "normal_xy": [float(value) for value in normal],
                    "mean_side_px": float(
                        np.mean([sample["side_px"] for sample in samples])
                    ),
                    "direct_observations": sum(
                        not sample.get("grid_recovered", False)
                        for sample in samples
                    ),
                    "grid_recovered_observations": sum(
                        sample.get("grid_recovered", False)
                        for sample in samples
                    ),
                    "last_seen": max(sample["seen_at"] for sample in samples),
                }
            )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for marker in raw_estimates:
            grouped[marker["shelf"]].append(marker)

        # Keep only the observation-weighted spatial consensus for each shelf.
        # This removes a persistent, geometrically plausible false ID once real
        # markers from that shelf have accumulated more supporting observations.
        estimates: list[dict[str, Any]] = []
        for markers in grouped.values():
            cluster = dominant_marker_cluster(markers)
            if (
                len(cluster) == 1
                and cluster[0]["observations"]
                < SINGLE_MARKER_PROVISIONAL_MIN_OBSERVATIONS
            ):
                continue
            estimates.extend(cluster)
        estimates = frontage_consistent_markers(estimates)
        return sorted(estimates, key=lambda marker: marker["id"])

    def marker_estimates(self) -> list[dict[str, Any]]:
        """Return the complete marker map copied from the Server MJCF truth."""

        return [dict(marker) for marker in self.fixed_markers]

    def fit_shelf_models(
        self, marker_estimates: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Fit robust 2-D shelf lines from at least three fixed markers."""

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for marker in marker_estimates:
            if marker["stable"]:
                grouped[marker["shelf"]].append(marker)

        base_xy = (
            np.asarray(self.base_pos[:2], dtype=np.float64)
            if self.base_pos is not None
            else None
        )
        models: dict[str, dict[str, Any]] = {}
        for shelf, markers in sorted(grouped.items()):
            markers = dominant_marker_cluster(markers)
            points = np.asarray([marker["world"][:2] for marker in markers])
            best: tuple[tuple[int, int, float], np.ndarray, np.ndarray] | None = None
            if len(markers) >= SHELF_MIN_MARKERS:
                for first in range(len(markers)):
                    for second in range(first + 1, len(markers)):
                        if markers[first]["column"] == markers[second]["column"]:
                            continue
                        if markers[first]["level"] != markers[second]["level"]:
                            continue
                        delta = points[second] - points[first]
                        span = float(np.linalg.norm(delta))
                        if span < 0.10:
                            continue
                        direction = delta / span
                        residuals = np.abs(
                            (points[:, 0] - points[first, 0]) * direction[1]
                            - (points[:, 1] - points[first, 1]) * direction[0]
                        )
                        inliers = np.flatnonzero(
                            residuals <= SHELF_LINE_INLIER_M
                        )
                        columns = {markers[index]["column"] for index in inliers}
                        if len(inliers) < SHELF_MIN_MARKERS or len(columns) < 2:
                            continue
                        score = (len(inliers), len(columns), span)
                        if best is None or score > best[0]:
                            best = (score, direction, inliers)
            if best is None:
                # A stable single marker is still enough for a safe refinement
                # pose: move closer along the already visible robot-marker ray.
                # It is never used as a grasp approach pose; only a full line
                # fit can authorize those.
                if base_xy is None:
                    continue
                if (
                    len(markers) == 1
                    and markers[0]["observations"]
                    < SINGLE_MARKER_PROVISIONAL_MIN_OBSERVATIONS
                ):
                    continue
                weights = np.asarray(
                    [max(1, marker["observations"]) for marker in markers],
                    dtype=np.float64,
                )
                center = np.average(points, axis=0, weights=weights)
                normal = base_xy - center
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm < 1.0e-6:
                    continue
                normal /= normal_norm
                direction = np.array([normal[1], -normal[0]], dtype=np.float64)
                models[shelf] = {
                    "shelf": shelf,
                    "quality": "provisional",
                    "marker_ids": sorted(marker["id"] for marker in markers),
                    "center": center,
                    "tangent": direction,
                    "normal": normal,
                    "fit_residual_m": None,
                }
                continue

            _, direction, inliers = best
            if direction[0] < 0.0:
                direction = -direction
            inlier_points = points[inliers]
            center = np.median(inlier_points, axis=0)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)
            truth_normals = [
                np.asarray(markers[index]["normal_xy"], dtype=np.float64)
                for index in inliers
                if np.linalg.norm(markers[index]["normal_xy"]) > 1.0e-6
            ]
            if truth_normals:
                reference = truth_normals[0]
                aligned = [
                    value if np.dot(value, reference) >= 0.0 else -value
                    for value in truth_normals
                ]
                preferred_normal = np.median(np.asarray(aligned), axis=0)
                preferred_norm = float(np.linalg.norm(preferred_normal))
                if preferred_norm > 1.0e-6:
                    preferred_normal /= preferred_norm
                    if np.dot(normal, preferred_normal) < 0.0:
                        normal = -normal
            elif (previous_normal := self._shelf_front_normals.get(shelf)) is not None:
                if np.dot(normal, previous_normal) < 0.0:
                    normal = -normal
            elif base_xy is not None and np.dot(base_xy - center, normal) < 0.0:
                normal = -normal
            self._shelf_front_normals[shelf] = normal.copy()

            residuals = np.abs(
                (inlier_points[:, 0] - center[0]) * direction[1]
                - (inlier_points[:, 1] - center[1]) * direction[0]
            )
            marker_ids = sorted(markers[index]["id"] for index in inliers)
            models[shelf] = {
                "shelf": shelf,
                "quality": "fitted",
                "marker_ids": marker_ids,
                "center": center,
                "tangent": direction,
                "normal": normal,
                "fit_residual_m": float(np.mean(residuals)),
            }
        return models

    def publish_generated_waypoints(
        self,
        slots: list[dict[str, Any]],
        marker_estimates: list[dict[str, Any]],
    ) -> None:
        models = self.fit_shelf_models(marker_estimates)
        stamp = self.get_clock().now().to_msg()
        shelf_pose_array = PoseArray()
        shelf_pose_array.header.frame_id = "odom"
        shelf_pose_array.header.stamp = stamp
        target_pose_array = PoseArray()
        target_pose_array.header = shelf_pose_array.header

        shelf_payload: list[dict[str, Any]] = []
        for shelf, model in sorted(models.items()):
            center = model["center"]
            normal = model["normal"]
            position = center + normal * self.scan_stand_off
            yaw = math.atan2(center[1] - position[1], center[0] - position[0])
            shelf_pose_array.poses.append(yaw_pose(position[0], position[1], yaw))
            shelf_payload.append(
                {
                    "shelf": shelf,
                    "quality": model["quality"],
                    "marker_ids": model["marker_ids"],
                    "center": [float(value) for value in center],
                    "tangent": [float(value) for value in model["tangent"]],
                    "front_normal": [float(value) for value in normal],
                    "fit_residual_m": model["fit_residual_m"],
                    "scan_pose": {
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "yaw": yaw,
                    },
                }
            )

        marker_by_id = {marker["id"]: marker for marker in marker_estimates}
        target_payload: list[dict[str, Any]] = []
        for slot in sorted(slots, key=lambda item: item["aruco_id"]):
            if not slot["stable"] or slot["kind"] not in self.requested_kinds:
                continue
            model = models.get(slot["shelf"])
            marker = marker_by_id.get(slot["aruco_id"])
            if (
                model is None
                or model["quality"] != "fitted"
                or marker is None
                or not marker["stable"]
            ):
                continue
            marker_xy = np.asarray(marker["world"][:2], dtype=np.float64)
            normal = model["normal"]
            position = marker_xy + normal * self.approach_stand_off
            yaw = math.atan2(
                marker_xy[1] - position[1], marker_xy[0] - position[0]
            )
            target_pose_array.poses.append(yaw_pose(position[0], position[1], yaw))
            target_payload.append(
                {
                    "aruco_id": slot["aruco_id"],
                    "shelf": slot["shelf"],
                    "level": slot["level"],
                    "column": slot["column"],
                    "kind": slot["kind"],
                    "votes": slot["votes"],
                    "vote_ratio": slot["vote_ratio"],
                    "approach_pose": {
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "yaw": yaw,
                    },
                }
            )

        self.shelf_pose_pub.publish(shelf_pose_array)
        self.target_pose_pub.publish(target_pose_array)
        self.waypoint_pub.publish(
            String(
                data=json.dumps(
                    {
                        "frame_id": "odom",
                        "scan_stand_off": self.scan_stand_off,
                        "approach_stand_off": self.approach_stand_off,
                        "shelves": shelf_payload,
                        "requested_targets": target_payload,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )

    def publish_inventory_map(self) -> None:
        measured_marker_estimates = self.measured_marker_estimates()
        marker_estimates = self.marker_estimates()
        marker_by_id = {marker["id"]: marker for marker in marker_estimates}
        validated_marker_ids = set(marker_by_id)
        slots: list[dict[str, Any]] = []
        for marker_id, votes in sorted(self._slot_votes.items()):
            if not votes or marker_id not in validated_marker_ids:
                continue
            kind_counts = Counter(vote["kind"] for vote in votes)
            confidence_sums: dict[str, float] = defaultdict(float)
            for vote in votes:
                confidence_sums[vote["kind"]] += vote["confidence"]
            selected_kind = max(
                kind_counts,
                key=lambda kind: (confidence_sums[kind], kind_counts[kind]),
            )
            selected_votes = [vote for vote in votes if vote["kind"] == selected_kind]
            vote_ratio = len(selected_votes) / len(votes)
            product_world = np.median(
                np.asarray([vote["product_world"] for vote in selected_votes]), axis=0
            )
            marker_world = marker_by_id[marker_id]["world"]
            slots.append(
                {
                    "aruco_id": marker_id,
                    **slot_from_aruco(marker_id),
                    "kind": selected_kind,
                    "stable": len(selected_votes) >= self.min_votes
                    and vote_ratio >= VOTE_RATIO_MIN,
                    "votes": len(selected_votes),
                    "total_votes": len(votes),
                    "vote_ratio": vote_ratio,
                    "mean_confidence": float(
                        np.mean([vote["confidence"] for vote in selected_votes])
                    ),
                    "product_world": [float(value) for value in product_world],
                    "marker_world": [float(value) for value in marker_world],
                    "last_seen": max(vote["seen_at"] for vote in selected_votes),
                }
            )

        payload = {
            "frame_id": "odom",
            "run_prefix": self._inventory_run_prefix or "",
            "requested_kinds": dict(self.requested_kinds),
            "mapped_slot_count": sum(slot["stable"] for slot in slots),
            "removed_marker_ids": sorted(self._removed_marker_ids),
            "removed_slots": [
                {"shelf": shelf, "level": level, "column": column}
                for shelf, level, column in sorted(
                    self._removed_inventory_slots
                )
            ],
            "slots": slots,
        }
        self.inventory_pub.publish(
            String(
                data=json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                )
            )
        )
        marker_payload = {
            "frame_id": "odom",
            "candidate_marker_count": sum(
                bool(samples) for samples in self._marker_tracks.values()
            ),
            "measured_marker_count": sum(
                marker["stable"] for marker in measured_marker_estimates
            ),
            "inferred_marker_count": sum(
                marker["stable"]
                and marker.get("position_source") == "grid_inferred"
                for marker in marker_estimates
            ),
            "fixed_truth_marker_count": len(marker_estimates),
            "mapped_marker_count": sum(
                marker["stable"] for marker in marker_estimates
            ),
            "rejected_marker_ids": sorted(
                set(self._marker_tracks)
                - {marker["id"] for marker in measured_marker_estimates}
            ),
            "grid_model": {
                "ready": False,
                "disabled": True,
                "reason": "fixed_server_mjcf_truth",
            },
            "markers": marker_estimates,
        }
        self.marker_map_pub.publish(
            String(
                data=json.dumps(
                    marker_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
        self.publish_generated_waypoints(slots, marker_estimates)
        now = self.get_clock().now().nanoseconds * 1.0e-9
        if now - self._last_summary_log >= 5.0:  # 库存摘要日志每 5 s 一条
            self.get_logger().info(
                f"INVENTORY_SCAN mapped={payload['mapped_slot_count']}/45 "
                f"stable_markers={marker_payload['mapped_marker_count']}/45 "
                f"fixed_truth={marker_payload['fixed_truth_marker_count']} "
                f"camera_observed={marker_payload['measured_marker_count']} "
                f"requested={dict(self.requested_kinds)}"
            )
            self._last_summary_log = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--scan-stand-off", type=float, default=0.85)
    parser.add_argument("--approach-stand-off", type=float, default=0.70)
    parser.add_argument("--aruco-truth", type=Path, default=DEFAULT_ARUCO_TRUTH)
    parser.add_argument("--no-result-image", action="store_true")
    return parser.parse_args(remove_ros_args()[1:])


def main() -> None:
    args = parse_args()
    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence must be in (0, 1]")
    if args.min_votes < 1:
        raise SystemExit("--min-votes must be at least 1")
    if not 0.55 <= args.scan_stand_off <= 1.50:
        raise SystemExit("--scan-stand-off must be between 0.55 and 1.50 m")
    if not 0.50 <= args.approach_stand_off <= 1.20:
        raise SystemExit("--approach-stand-off must be between 0.50 and 1.20 m")
    rclpy.init()
    node = ProductDetectNode(
        weights=args.weights,
        confidence=args.confidence,
        device=args.device,
        publish_result_image=not args.no_result_image,
        min_votes=args.min_votes,
        scan_stand_off=args.scan_stand_off,
        approach_stand_off=args.approach_stand_off,
        aruco_truth=args.aruco_truth,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
