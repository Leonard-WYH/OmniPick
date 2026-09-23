"""Geometry and command helpers for shelf product clearing missions.

The right-arm joint motion is deliberately a fixed template.  Chassis pose and
the vertical slide place a detected bottle at that template's work point.
Keeping these calculations ROS-independent makes them easy to check without a
running simulator.

中文说明：本文件只保存 E 柜、桌面、货位和抓取几何关系，并提供纯计算函数；
它不创建 ROS 定时器，也没有 sleep 或动作等待。修改这里会改变目标位置而非
状态机时序。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Iterable

import numpy as np


DEFAULT_TARGET_KIND = "kele"


@dataclass(frozen=True)
class BottleGeometry:
    radius_m: float
    half_height_m: float


# Intrinsic collision geometry from the Server MJCF.  Keeping this next to the
# fixed arm template lets the same proven motion grasp each package around its
# physical centre instead of reusing the shorter/narrower cola dimensions.
BOTTLE_GEOMETRY = {
    "kele": BottleGeometry(radius_m=0.0265, half_height_m=0.0725),
    "maidong": BottleGeometry(radius_m=0.0325, half_height_m=0.1050),
    # Cylindrical crisp tube: identical collision dimensions to maidong.
    "shupian": BottleGeometry(radius_m=0.0325, half_height_m=0.1050),
    # Tapered noodle cup.  The visible front surface around the grasp height is
    # about 4 cm ahead of its centre (bottom/top radii are 3.25/4.75 cm).
    "heweidao": BottleGeometry(radius_m=0.0400, half_height_m=0.0525),
    # Sandwich pack is a box-like mesh with 5 cm front half-depth.
    "sanmingzhi": BottleGeometry(radius_m=0.0500, half_height_m=0.0494),
    # Server collision geometry: an 80 mm-tall gum cylinder.
    "kouxiangtang": BottleGeometry(radius_m=0.0245, half_height_m=0.0400),
    # The fruit collision bodies are spheres; their radius is also their
    # vertical half-height above the shelf/table surface.
    "pingguo": BottleGeometry(radius_m=0.0350, half_height_m=0.0350),
    "chengzi": BottleGeometry(radius_m=0.0370, half_height_m=0.0370),
    # Tissue pack collision box: size="0.0860 0.0425 0.0440" in the Server
    # MJCF.  ``radius_m`` is the front half-depth used by the common RGB-D
    # surface-to-centre conversion; the much larger lateral half-width is
    # represented separately below for the dedicated two-hand clamp.
    "zhijin": BottleGeometry(radius_m=0.0425, half_height_m=0.0440),
}
SUPPORTED_PRODUCT_KINDS = tuple(BOTTLE_GEOMETRY)


def mission_name_for_product(product_kind: str) -> str:
    kind = str(product_kind).strip().lower()
    if kind not in BOTTLE_GEOMETRY:
        raise ValueError(f"unsupported E-shelf product kind: {product_kind}")
    return f"clear_e_{kind}"


def bottle_geometry(product_kind: str) -> BottleGeometry:
    kind = str(product_kind).strip().lower()
    try:
        return BOTTLE_GEOMETRY[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported E-shelf product kind: {product_kind}") from exc


MISSION_NAME = mission_name_for_product(DEFAULT_TARGET_KIND)

# Fixed Server MJCF geometry.  All five cabinets share the same depth, level
# heights and 3x3 slot layout; only their X coordinates and first ArUco ID
# differ.  Keep the historical E_* names below as compatibility aliases for
# the already validated fixed-E regression modes.
SHELF_NAMES = ("A", "B", "C", "D", "E")
SHELF_COLUMNS_M = {
    "A": (-1.955, -1.735, -1.515),
    "B": (-1.070, -0.850, -0.630),
    "C": (-0.185, 0.035, 0.255),
    "D": (0.700, 0.920, 1.140),
    "E": (1.585, 1.805, 2.025),
}
SHELF_FIRST_ARUCO_ID = {
    shelf: index * 9 for index, shelf in enumerate(SHELF_NAMES)
}
SHELF_PRODUCT_CENTER_Y_M = 3.243
SHELF_Y_MIN_M = 3.00
SHELF_Y_MAX_M = 3.42
SHELF_LEVEL_Z_M = (0.499, 0.851, 1.189)
# The validated E viewpoint is 10.5 cm left of the middle column.  Translating
# that same relative pose to every cabinet preserves the arm/camera geometry.
# A is beside the left wall, so keep its observation/grasp handoff point 10 cm
# farther from the shelf. Product depth and final insertion remain unchanged.
SHELF_SCAN_Y_M = 2.071
A_SHELF_SCAN_Y_NEGATIVE_SHIFT_M = 0.10
SHELF_SCAN_POSES = {
    shelf: (
        float(columns[1] - 0.105),
        SHELF_SCAN_Y_M
        - (A_SHELF_SCAN_Y_NEGATIVE_SHIFT_M if shelf == "A" else 0.0),
        math.pi / 2.0,
    )
    for shelf, columns in SHELF_COLUMNS_M.items()
}
# 后续抓取边缘列商品时，让底盘观察点先随商品所在列横移。这样视觉伺服
# 接管前，夹爪与商品已经大致处于同一条前进线上，不必在近柜阶段承担过大
# 的横向纠偏。左列保持 18 cm；右列按最新实测加大到 25 cm，补偿右侧商品
# 仍会向左抓偏的问题。首件 E 柜直达流程也在识别到具体货位后使用同一
# 列偏移，使高速直控目标与实际待抓商品对齐。
LEFT_COLUMN_OBSERVATION_X_OFFSET_M = 0.18
RIGHT_COLUMN_OBSERVATION_X_OFFSET_M = 0.25

# Fixed Server MJCF geometry for the legacy E-only API.
E_SHELF_X_MIN_M = 1.35
E_SHELF_X_MAX_M = 2.26
E_SHELF_Y_MIN_M = 3.00
E_SHELF_Y_MAX_M = 3.42
E_SHELF_COLUMNS_M = SHELF_COLUMNS_M["E"]
# All Server products are spawned on the same shelf-depth plane.  The ArUco
# tiles sit at y=3.168 m and the product centres at y=3.243 m.  Broad box
# packages use this known plane because close RGB-D can sample the gripper or
# shelf behind the box face by several centimetres.
E_SHELF_PRODUCT_CENTER_Y_M = SHELF_PRODUCT_CENTER_Y_M
# Server shelf surfaces, ordered exactly like the public ArUco numbering:
# L1 is the bottom row, L2 the middle row and L3 the top row.  Every randomized
# bottle keeps its intrinsic half-height, so its physical
# centre is the corresponding surface plus its product-specific half-height.
E_SHELF_LEVEL_Z_M = SHELF_LEVEL_Z_M
E_SHELF_FIRST_ARUCO_ID = SHELF_FIRST_ARUCO_ID["E"]

# A broad E-shelf viewpoint that the user validated in RViz.
E_SCAN_POSE = SHELF_SCAN_POSES["E"]

# Verified grasp heading and the canonical hand work point in the footprint
# frame.  Products are no longer converted into separate Nav2 pick waypoints;
# the visual controller drives the deployed hand from E_SCAN_POSE instead.
GRASP_YAW = math.pi / 2.0 - math.radians(11.0)
HAND_WORKPOINT_XY_M = np.array([0.55587420, -0.01551934], dtype=float)

# This arm template has identity wrist orientation and reaches
# [0.5559, -0.0155, 1.330-slide] in base_footprint.  Moving only the slide
# therefore covers all three shelf levels while the six arm joints stay fixed.
FIXED_GRASP_ARM = np.array(
    [0.07689, -1.55043, 0.51224, -1.51465, -1.82468, -0.04031],
    dtype=float,
)
HAND_Z_PLUS_SLIDE_M = 1.330
DEPLOY_BELOW_PRODUCT_CENTER_M = 0.010
# 双臂夹持纸巾时，夹爪碰撞几何会比末端中心向下多伸出一段。L1 实测
# 两只手向前伸入时会卡到下层隔板，因此单独把末端中心抬高 1.5 cm；
# L3 保留已经验证的 2 cm 上提量，L2 继续按包装中心夹取。
TISSUE_L1_GRASP_ABOVE_CENTER_M = 0.015
TISSUE_L3_GRASP_ABOVE_CENTER_M = 0.020
# The collision mesh of each open finger extends about 30 mm below the endpoint.
# Gum and fruit centres are only 35-40 mm above the shelf, so the normal
# centre-minus-10-mm grasp makes a finger scrape or penetrate the board once
# slide tracking error is included.  Gum and apple use centre-plus-10 mm.  The
# 74 mm orange leaves only 3 mm per side in the 80 mm open gripper.  Keep
# its endpoint just 5 mm above the sphere centre: this is 15 mm lower than the
# former upper-half grasp and gives the fingers a much more secure central
# overlap, while their lowest collision point still retains about 12 mm of
# nominal clearance over the shelf board.
RAISED_GRASP_PRODUCT_KINDS = frozenset(
    {"kouxiangtang", "pingguo", "chengzi"}
)
RAISED_GRASP_ABOVE_CENTER_M = 0.010
CHENGZI_GRASP_ABOVE_CENTER_M = 0.005
SLIDE_MIN_M = -0.04
SLIDE_MAX_M = 0.87
LIFT_M = 0.05

# Table handoff pose used by the already validated placement motion.  Keep it
# separate from the long-distance carry pose below: its elbow reaches y=-0.423
# in base_footprint, which is acceptable while parked at the table but can hit
# a corridor wall during Nav2 travel.  FK at slide=0.06 is
# [0.480, -0.150, 1.200] with identity orientation.
FIXED_CARRY_ARM = np.array(
    [
        -0.4968561444,
        -0.8344755163,
        0.0243308696,
        -1.9158412778,
        -1.6545810617,
        0.3360557562,
    ],
    dtype=float,
)
CARRY_HAND_WORKPOINT_XY_M = np.array([0.480, -0.150], dtype=float)
CARRY_HAND_Z_PLUS_SLIDE_M = 1.260

# Compact navigation carry pose.  The gripper stays upright and holds the
# bottle in front of the chest.  The former x=0.46 m pose left wrist joint 5
# torque-saturated about 0.13 rad short while carrying a bottle.  Moving the
# hand 8 cm forward and 2 cm toward the centre unloads that wrist while keeping
# every right-arm joint origin inside y=[-0.199, 0.060] instead of the table
# handoff pose's y=-0.423 m.  At the navigation slide of 0.00 m its gripper
# centre is z=0.920 m, keeping even the taller maidong bottle above the table.
# The state machine unfolds back to
# FIXED_CARRY_ARM only after the robot has stopped at the table, so the proven
# drop geometry is unchanged.
COMPACT_CARRY_ARM = np.array(
    [
        -0.9944979272,
        -2.1984834909,
        1.5851295741,
        -2.3460090315,
        -1.5174142614,
        0.7568381373,
    ],
    dtype=float,
)
COMPACT_CARRY_HAND_WORKPOINT_XY_M = np.array([0.540, 0.060], dtype=float)
COMPACT_CARRY_HAND_Z_PLUS_SLIDE_M = 0.920
COMPACT_CARRY_WRIST_YAW_RAD = 0.200

# Costmap footprints used for the long transport leg.  The compact pose keeps
# all arm joint origins within the chassis width; this small extra margin also
# covers the held bottle and link thickness without returning to the former
# over-wide elbow projection.  Shelf work uses the exact chassis footprint;
# the direct table goal keeps the conservative carry envelope, then restores
# the exact chassis footprint only after Nav2 has stopped and before arm motion.
NORMAL_FOOTPRINT = "[[0.219, 0.200], [0.219, -0.200], [-0.202, -0.200], [-0.202, 0.200]]"
CARRY_FOOTPRINT = "[[0.245, 0.215], [0.245, -0.215], [-0.225, -0.215], [-0.225, 0.215]]"
# Tissue transport intentionally uses chassis-only collision planning.  Keep
# this public constant as an alias for compatibility, but do not expand Nav2's
# footprint for a wide package or the two deployed arms: beside the E partition an
# expanded polygon invalidates every MPPI trajectory before motion can begin.
TISSUE_CARRY_FOOTPRINT = NORMAL_FOOTPRINT
TISSUE_HALF_WIDTH_M = 0.0860

# Delivery table truth from the Server MJCF.
TABLE_CENTER_X_M = -1.940
TABLE_CENTER_Y_M = -3.410
# Physical tabletop bounds from the Server MJCF.  Export these so special
# wide-product placement can clamp requested offsets before any part of the
# package crosses an edge.
TABLE_Y_MIN_M = -3.630
TABLE_Y_MAX_M = -3.190
TABLE_APPROACH_Y_M = -2.800
TABLE_YAW = -math.pi / 2.0
# 所有放置航点统一向世界 X 负方向平移 15 cm。这是底盘航点偏移，
# 单臂与双臂商品的放置位姿都由 table_approach_pose() 继承同一个改动。
TABLE_APPROACH_X_NEGATIVE_SHIFT_M = 0.150
TABLE_TOP_Z_M = 0.767
KELE_HALF_HEIGHT_M = bottle_geometry("kele").half_height_m
TABLE_DROP_PRELOAD_M = 0.010
# Nominal sphere-bottom clearance, not a contact measurement. Centimetres of
# clearance caused a visible free fall. The orange runtime path now calibrates
# its held centre and controls measured hand height before releasing, rather
# than compensating FK/tracking errors by raising this margin. Other products
# retain their existing 10 mm preload.
CHENGZI_TABLE_RELEASE_CLEARANCE_M = 0.003
TABLE_DROP_GRIPPER_Z_M = (
    TABLE_TOP_Z_M + KELE_HALF_HEIGHT_M - TABLE_DROP_PRELOAD_M
)
TABLE_DROP_SLIDE_M = CARRY_HAND_Z_PLUS_SLIDE_M - TABLE_DROP_GRIPPER_Z_M
# The robot-side row uses a 0.20 m table insertion.  The inner row uses a
# separate 0.30 m template because the released bottle was observed to settle
# shallower than its nominal kinematic point even when Nav2 reached the base
# pose.  Both rows are four centimetres deeper than the preceding layout.
# Each tuple includes small IK fallbacks that preserve the same row.
TABLE_PLACE_FORWARD_DISTANCES_M = (0.20, 0.19, 0.18, 0.17)
TABLE_INNER_PLACE_FORWARD_DISTANCES_M = (0.30, 0.29, 0.28, 0.27)
# The noodle cup tends to settle slightly toward the robot as its wide rim
# leaves the gripper.  Give only this package another 1.5 cm of arm insertion;
# the other products retain the five already tuned table locations.
HEWEIDAO_TABLE_EXTRA_FORWARD_M = 0.015
# Five explicit tabletop slots arranged as marked in the user's overhead view:
# slots 1/2/3 run left-to-right on the inner row, while slots 4/5 sit directly
# in front of slots 1/2 on the robot-side row.  The whole pattern stays on the
# middle/east side of the table to keep the extended right elbow away from the
# west wall.  The table spans x=[-2.42, -1.46] m and y=[-3.63, -3.19] m.
TABLE_SLOT_2_POSITIVE_X_SHIFT_M = 0.100
TABLE_SLOT_3_POSITIVE_X_SHIFT_M = 0.100
TABLE_SLOT_5_POSITIVE_X_SHIFT_M = 0.150
TABLE_DROP_X_OFFSETS_M = (
    -0.06,
    0.05 + TABLE_SLOT_2_POSITIVE_X_SHIFT_M,
    0.30 + TABLE_SLOT_3_POSITIVE_X_SHIFT_M,
    -0.06,
    0.05 + TABLE_SLOT_5_POSITIVE_X_SHIFT_M,
)
TABLE_INNER_ROW_Y_OFFSET_M = (
    TABLE_APPROACH_Y_M
    - TABLE_CENTER_Y_M
    - CARRY_HAND_WORKPOINT_XY_M[0]
    - TABLE_INNER_PLACE_FORWARD_DISTANCES_M[0]
)
TABLE_ROBOT_ROW_Y_OFFSET_M = (
    TABLE_APPROACH_Y_M
    - TABLE_CENTER_Y_M
    - CARRY_HAND_WORKPOINT_XY_M[0]
    - TABLE_PLACE_FORWARD_DISTANCES_M[0]
)
# 只把 1/2 号名义放置点沿世界 -Y 移动 15 cm；3/4/5 保持不变。
# 该名义点会越过部分商品的安全落桌边界，因此单手商品在实际发 Nav2
# 目标前仍由 motion_control._table_approach_pose() 按商品半径限幅。
# 这样紧凑商品能尽量靠里，橙子、苹果等较大商品也不会再次掉出桌面。
TABLE_FIRST_TWO_NEGATIVE_Y_SHIFT_M = 0.150
TABLE_DROP_Y_OFFSETS_M = (
    TABLE_INNER_ROW_Y_OFFSET_M - TABLE_FIRST_TWO_NEGATIVE_Y_SHIFT_M,
    TABLE_INNER_ROW_Y_OFFSET_M - TABLE_FIRST_TWO_NEGATIVE_Y_SHIFT_M,
    TABLE_INNER_ROW_Y_OFFSET_M,
    TABLE_ROBOT_ROW_Y_OFFSET_M,
    TABLE_ROBOT_ROW_Y_OFFSET_M,
)


@dataclass(frozen=True)
class MissionCommand:
    command: str
    mission: str = MISSION_NAME
    command_id: str = ""


def parse_mission_command(
    text: str, mission_name: str = MISSION_NAME
) -> MissionCommand:
    """Parse either a short command or the JSON command used in documentation."""

    mission_name = str(mission_name).strip()
    if mission_name not in {"clear_e_mixed", "clear_random"}:
        mission_prefix = "clear_e_"
        product_kind = (
            mission_name[len(mission_prefix) :]
            if mission_name.startswith(mission_prefix)
            else mission_name
        )
        mission_name = mission_name_for_product(product_kind)
    stripped = text.strip()
    if stripped in {mission_name, "start"}:
        return MissionCommand(command="start", mission=mission_name)
    if stripped == "stop":
        return MissionCommand(command="stop", mission=mission_name)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid mission command JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("mission command must be a JSON object")
    command = str(payload.get("command", "")).strip().lower()
    mission = str(payload.get("mission", mission_name)).strip()
    command_id = str(payload.get("command_id", "")).strip()
    if command not in {"start", "stop"}:
        raise ValueError("command must be start or stop")
    if command == "start" and mission != mission_name:
        raise ValueError(f"unsupported mission: {mission}")
    return MissionCommand(command=command, mission=mission, command_id=command_id)


def e_shelf_contains(point: Iterable[float]) -> bool:
    x, y, z = (float(value) for value in point)
    return (
        E_SHELF_X_MIN_M <= x <= E_SHELF_X_MAX_M
        and E_SHELF_Y_MIN_M <= y <= E_SHELF_Y_MAX_M
        and 0.40 <= z <= 1.45
    )


def shelf_for_point(point: Iterable[float]) -> str | None:
    """Return the nearest A--E cabinet for a world-frame shelf point."""

    x, y, z = (float(value) for value in point)
    if not (
        SHELF_Y_MIN_M <= y <= SHELF_Y_MAX_M
        and 0.40 <= z <= 1.45
    ):
        return None
    shelf, distance = min(
        (
            (name, abs(x - float(columns[1])))
            for name, columns in SHELF_COLUMNS_M.items()
        ),
        key=lambda item: item[1],
    )
    # Adjacent cabinet centres are 0.885 m apart and each cabinet is about
    # 0.91 m wide.  A 0.46 m half-width covers edge products without accepting
    # detections elsewhere in the room.
    return shelf if distance <= 0.46 else None


def shelf_contains(point: Iterable[float], shelf: str | None = None) -> bool:
    """Test whether a point lies on any cabinet, or on one named cabinet."""

    matched = shelf_for_point(point)
    if shelf is None:
        return matched is not None
    return matched == str(shelf).strip().upper()


def shelf_scan_pose(shelf: str) -> tuple[float, float, float]:
    """Return the common validated observation pose translated to a cabinet."""

    name = str(shelf).strip().upper()
    try:
        return SHELF_SCAN_POSES[name]
    except KeyError as exc:
        raise ValueError(f"invalid shelf: {shelf}") from exc


def shelf_target_observation_pose(
    shelf: str,
    marker_id: int | None,
    *,
    shift_edge_columns: bool = True,
) -> tuple[float, float, float]:
    """Return a shelf observation pose laterally aligned to one known slot.

    ArUco IDs are ordered by row, with C1/C2/C3 repeated for every row.  The
    centre column therefore keeps the calibrated shelf pose, while C1 and C3
    move it by the same amount in opposite world-X directions.  Unknown or
    out-of-shelf IDs deliberately fall back to the fixed pose.
    """

    name = str(shelf).strip().upper()
    pose = shelf_scan_pose(name)
    if not shift_edge_columns or marker_id is None:
        return pose
    try:
        slot_index = int(marker_id) - SHELF_FIRST_ARUCO_ID[name]
    except (TypeError, ValueError):
        return pose
    if not 0 <= slot_index < 9:
        return pose
    column_index = slot_index % 3
    x_shift = (
        -LEFT_COLUMN_OBSERVATION_X_OFFSET_M,
        0.0,
        RIGHT_COLUMN_OBSERVATION_X_OFFSET_M,
    )[column_index]
    return (float(pose[0] + x_shift), pose[1], pose[2])


def grasp_height_offset_for_product(
    product_kind: str,
    shelf_level: str | None = None,
) -> float:
    """Return endpoint height relative to the selected product centre."""

    kind = str(product_kind).strip().lower()
    if kind not in BOTTLE_GEOMETRY:
        raise ValueError(f"unsupported E-shelf product kind: {product_kind}")
    # The two closed grippers contact the left/right faces of the wide package;
    # centring their pads vertically leaves shelf clearance below and equal
    # support above.  The lower finger collision geometry needs extra board
    # clearance on L1 and L3; L2 retains the proven centre grasp.  The
    # single-hand template offsets below do not apply.
    if kind == "zhijin":
        level = str(shelf_level).strip().upper()
        if level == "L1":
            return TISSUE_L1_GRASP_ABOVE_CENTER_M
        if level == "L3":
            return TISSUE_L3_GRASP_ABOVE_CENTER_M
        return 0.0
    if kind == "chengzi":
        return CHENGZI_GRASP_ABOVE_CENTER_M
    if kind in RAISED_GRASP_PRODUCT_KINDS:
        return RAISED_GRASP_ABOVE_CENTER_M
    return -DEPLOY_BELOW_PRODUCT_CENTER_M


def grasp_slide_for_product_z(
    product_center_z: float,
    product_kind: str = DEFAULT_TARGET_KIND,
    shelf_level: str | None = None,
) -> float:
    """Align product height with the fixed arm template using only the slide."""

    product_center_z = float(product_center_z)
    if not math.isfinite(product_center_z):
        raise ValueError("product height must be finite")
    deploy_z = (
        product_center_z
        + grasp_height_offset_for_product(product_kind, shelf_level)
    )
    return float(
        np.clip(
            HAND_Z_PLUS_SLIDE_M - deploy_z,
            SLIDE_MIN_M,
            SLIDE_MAX_M,
        )
    )


def lifted_slide(grasp_slide: float) -> float:
    return max(SLIDE_MIN_M, float(grasp_slide) - LIFT_M)


def e_product_center_z(level: str, product_kind: str = DEFAULT_TARGET_KIND) -> float:
    """Return the Server-truth product centre height in an E-shelf row."""

    try:
        level_index = ("L1", "L2", "L3").index(str(level))
    except ValueError as exc:
        raise ValueError(f"invalid E-shelf level: {level}") from exc
    return float(
        E_SHELF_LEVEL_Z_M[level_index]
        + bottle_geometry(product_kind).half_height_m
    )


def shelf_product_center_z(
    level: str, product_kind: str = DEFAULT_TARGET_KIND
) -> float:
    """Cabinet-independent alias for the common shelf level geometry."""

    return e_product_center_z(level, product_kind)


def e_kele_center_z(level: str) -> float:
    """Backward-compatible cola centre-height helper."""

    return e_product_center_z(level, "kele")


def table_drop_slide_for_product(product_kind: str) -> float:
    """Return the product-specific slide for safe table release height."""

    kind = str(product_kind).strip().lower()
    bottom_clearance = (
        CHENGZI_TABLE_RELEASE_CLEARANCE_M
        if kind == "chengzi"
        else -TABLE_DROP_PRELOAD_M
    )
    # Orange is intentionally grasped above its centre.  Its centre therefore
    # hangs below the gripper at delivery; compensate that offset here or the
    # sphere starts below the apparent clearance and can tunnel through the
    # tabletop.  Preserve the validated placement geometry for every other
    # product.
    grasp_offset = (
        grasp_height_offset_for_product(kind) if kind == "chengzi" else 0.0
    )
    drop_gripper_z = (
        TABLE_TOP_Z_M
        + bottle_geometry(kind).half_height_m
        + bottom_clearance
        + grasp_offset
    )
    return float(CARRY_HAND_Z_PLUS_SLIDE_M - drop_gripper_z)


def nearest_e_slot(
    product_center_world: Iterable[float],
    product_kind: str = DEFAULT_TARGET_KIND,
) -> tuple[int, str, str]:
    """Infer an E shelf slot when its ArUco is not decoded in the image."""

    point = np.asarray(tuple(product_center_world), dtype=float)
    column_index = int(np.argmin(np.abs(np.asarray(E_SHELF_COLUMNS_M) - point[0])))
    expected_centers = np.asarray(
        [e_product_center_z(level, product_kind) for level in ("L1", "L2", "L3")]
    )
    level_index = int(np.argmin(np.abs(expected_centers - point[2])))
    marker_id = E_SHELF_FIRST_ARUCO_ID + level_index * 3 + column_index
    return marker_id, f"L{level_index + 1}", f"C{column_index + 1}"


def nearest_shelf_slot(
    product_center_world: Iterable[float],
    product_kind: str = DEFAULT_TARGET_KIND,
    shelf: str | None = None,
) -> tuple[int, str, str, str]:
    """Infer the nearest A--E slot and return id, shelf, level and column."""

    point = np.asarray(tuple(product_center_world), dtype=float)
    name = (
        shelf_for_point(point)
        if shelf is None
        else str(shelf).strip().upper()
    )
    if name not in SHELF_COLUMNS_M:
        raise ValueError(f"point is not on a known shelf: {point.tolist()}")
    columns = np.asarray(SHELF_COLUMNS_M[name], dtype=float)
    column_index = int(np.argmin(np.abs(columns - point[0])))
    expected_centers = np.asarray(
        [
            shelf_product_center_z(level, product_kind)
            for level in ("L1", "L2", "L3")
        ]
    )
    level_index = int(np.argmin(np.abs(expected_centers - point[2])))
    marker_id = SHELF_FIRST_ARUCO_ID[name] + level_index * 3 + column_index
    return (
        marker_id,
        name,
        f"L{level_index + 1}",
        f"C{column_index + 1}",
    )


def table_place_forward_distances(
    drop_index: int, product_kind: str | None = None
) -> tuple[float, ...]:
    """Return the slot- and product-specific arm insertion candidates."""

    index = max(0, min(int(drop_index), len(TABLE_DROP_X_OFFSETS_M) - 1))
    if index < 3:
        distances = TABLE_INNER_PLACE_FORWARD_DISTANCES_M
    else:
        distances = TABLE_PLACE_FORWARD_DISTANCES_M
    if str(product_kind).strip().lower() == "heweidao":
        return tuple(
            distance + HEWEIDAO_TABLE_EXTRA_FORWARD_M
            for distance in distances
        )
    return distances


def table_approach_pose(drop_index: int) -> tuple[float, float, float]:
    """Choose direct table-drop poses so repeated bottles do not overlap.

    Inbound transport navigates directly to this pose in one Nav2 goal.  The
    post-release retreat is a short, heading-preserving base motion rather than
    a second navigation waypoint.
    """

    index = max(0, min(int(drop_index), len(TABLE_DROP_X_OFFSETS_M) - 1))
    desired_drop_x = TABLE_CENTER_X_M + TABLE_DROP_X_OFFSETS_M[index]
    desired_drop_y = TABLE_CENTER_Y_M + TABLE_DROP_Y_OFFSETS_M[index]
    # At -pi/2 yaw the carry hand's lateral offset contributes directly to X.
    base_x = (
        desired_drop_x
        - CARRY_HAND_WORKPOINT_XY_M[1]
        - TABLE_APPROACH_X_NEGATIVE_SHIFT_M
    )
    # The carry hand starts CARRY_HAND_WORKPOINT_XY_M[0] in front of the base,
    # then PLACE_ADVANCE adds the first (nominal) forward distance.  Solving
    # base Y from the requested bottle-centre Y keeps both rows geometrically
    # explicit instead of relying on a second chassis creep near the table.
    base_y = (
        desired_drop_y
        + CARRY_HAND_WORKPOINT_XY_M[0]
        + table_place_forward_distances(index)[0]
    )
    return float(base_x), float(base_y), TABLE_YAW


def cluster_points(
    points: Iterable[Iterable[float]], radius_m: float = 0.12
) -> list[np.ndarray]:
    """Greedily cluster repeated YOLO world points from a stationary scan."""

    clusters: list[list[np.ndarray]] = []
    for raw in points:
        point = np.asarray(tuple(raw), dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        best_index = None
        best_distance = math.inf
        for index, cluster in enumerate(clusters):
            center = np.median(np.asarray(cluster), axis=0)
            distance = float(np.linalg.norm(point - center))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance <= radius_m:
            clusters[best_index].append(point)
        else:
            clusters.append([point])
    clusters.sort(key=len, reverse=True)
    return [np.asarray(cluster) for cluster in clusters]
