"""Tuning constants for the autonomous shelf-picking cycle.

This module is intentionally data-only.  Keeping calibrated values here makes
product-specific tuning reviewable without entering the ROS state machine.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .baseline_grasp_controller import INIT_ARM_L, INIT_ARM_R
from .nav2_manipulation_client import (
    LEFT_ARM_STOW_WAYPOINTS,
    LEFT_ARM_TRANSPORT,
)
from .navigation.sorting_geometry import HAND_WORKPOINT_XY_M

COMMAND_TOPIC = "/supermarket_sorting/mission_command"
STATUS_TOPIC = "/supermarket_sorting/mission_status"
INVENTORY_REMOVE_TOPIC = "/inventory/remove"
SEARCH_STAGE_SETTLE_SEC = 0.60  # 等待观察姿态和图像稳定后才累计货架目标（秒）
SEARCH_MIN_CLUSTER_SAMPLES = 3
# One YOLO message contributes only one point per visible bottle.  Observe
# several completed inference frames so a single-frame window cannot be
# mistaken for an empty shelf.
SEARCH_MIN_DETECTION_FRAMES = 6
SEARCH_OBSERVATION_TIMEOUT_SEC = 10.0  # 一次观察阶段最多等待视觉结果 10 s
OBSERVATION_HEAD_PITCH = 0.0
# 第一次放置完成后，离开桌边先把头部朝向 B 柜中部，以一幅较宽的视野
# 持续补全 A/B/C 库存；进入目标柜观察点 1 m 范围后再改为动态盯住目标柜
# 整柜中心。这个距离只切换相机观察对象，不接管或减速 Nav2。
POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M = 1.0
SHELF_OVERVIEW_CENTER_Z_M = 0.90
# Safe fallback used only before target geometry is available.  Once a shelf
# product is selected, yaw and pitch are recomputed continuously from the
# remembered/live target and measured torso height.
PICK_TRACK_HEAD_PITCH = -0.35
PICK_TRACK_HEAD_YAW_LIMIT_RAD = 0.50
PICK_TRACK_HEAD_PITCH_MIN_RAD = -0.55
PICK_TRACK_HEAD_PITCH_MAX_RAD = 0.20
# Approximate head-camera optical geometry in base_footprint.  At zero head
# pitch the camera looks down by about 0.33 rad.  Its position changes mildly
# with pitch; these linear terms fit the bundled mmk2_head_fk model over the
# tracking range and are sufficient to keep the remembered product near image
# centre while both torso height and shelf distance change.
HEAD_CAMERA_FIXED_ELEVATION_RAD = -0.330
HEAD_CAMERA_FORWARD_AT_ZERO_PITCH_M = 0.283
HEAD_CAMERA_FORWARD_PER_PITCH_M = -0.169
HEAD_CAMERA_LATERAL_M = 0.036
HEAD_CAMERA_Z_PLUS_SLIDE_AT_ZERO_PITCH_M = 1.577
HEAD_CAMERA_Z_PLUS_SLIDE_PER_PITCH_M = 0.106
PICK_TRACK_MIN_CAMERA_RANGE_M = 0.08
HEAD_PITCH_TOL_RAD = 0.04
TARGET_MEMORY_MATCH_RADIUS_M = 0.14
REACQUIRE_MAX_TARGET_DISTANCE_M = 0.14
LIVE_TARGET_MAX_AGE_SEC = 1.0  # 检测时间戳距现在不超过 1 s 才算实时视觉
# The target identity and shelf slot are already locked before deployment, so
# live tracking does not need the five-frame confirmation used by an unknown
# scene.  Two recent frames reject a one-off detection while allowing the
# first part of the approach to start using the camera within one inference
# cycle instead of travelling several seconds on slot memory.
LIVE_TRACK_MIN_SAMPLES = 2
# Use live vision to refine the lateral grasp line, but keep the known shelf
# depth fixed.  A short low-pass plus per-inference step limit prevents a box
# edge or an entering black finger from producing a sudden steering kick.
LIVE_LATERAL_FILTER_ALPHA = 0.55
LIVE_LATERAL_MAX_STEP_M = 0.015
LIVE_LATERAL_MAX_SLOT_OFFSET_M = 0.075
# 第一段视觉伺服直接比较同一相机帧中的商品框中心与实测夹爪末端投影。
# 这组量只修正底盘横向/偏航；货架层高仍由已验证的升降轴抓取高度控制。
IMAGE_SERVO_MAX_AGE_SEC = 1.0
IMAGE_SERVO_PIXEL_DEADBAND_PX = 4.0
# 苹果的球体中心在画面中很明确，4 px 通用死区仍会留下可见左偏。
# 单独收紧到 2 px，让第一段视觉伺服继续把夹爪中心向苹果中心收敛；
# 其他商品保持原死区，避免窄包装检测框抖动导致左右摆动。
PINGGUO_IMAGE_SERVO_PIXEL_DEADBAND_PX = 2.0
IMAGE_SERVO_FILTER_ALPHA = 0.55
IMAGE_SERVO_ANGULAR_KP = 2.20
IMAGE_SERVO_MAX_ANGULAR_CORRECTION_RADPS = 0.18
# While pixels are available, final grasp yaw is only a small feed-forward;
# it must never overpower the observed product-to-gripper image error.
IMAGE_SERVO_GRASP_HEADING_KP = 0.25
IMAGE_SERVO_MAX_HEADING_CORRECTION_RADPS = 0.05
IMAGE_SERVO_SLOWDOWN_START_PX = 24.0
IMAGE_SERVO_SLOWDOWN_FULL_PX = 120.0
IMAGE_SERVO_MIN_LINEAR_SCALE = 0.35
# 商品已经接近夹爪、但图像中仍有明显横向偏差时，不要仅因为进入固定的
# 末端距离就过早冻结视觉。先把像素差继续收敛到 6 px；只有进入最后 4.5 cm
# 的遮挡高风险区才无条件锁住最后一条可信视觉线，交给末端几何控制。
IMAGE_SERVO_TERMINAL_LOCK_TOL_PX = 6.0
IMAGE_SERVO_HARD_FREEZE_REMAINING_M = 0.045
# 抓取闭环不再因为等待时间或短时无进展直接把整单置为 failed。实时视觉
# 丢失时继续使用已锁定的货位/最后视觉线；连这两者都无效时保持停车等待
# 视觉恢复。货架距离、超深、最大行程等几何安全边界仍然有效。
FINE_APPROACH_TIME_ABORTS_ENABLED = False
FINE_VISION_LOSS_TIMEOUT_SEC = 3.0  # 仅保留为诊断阈值，不再触发任务失败
LIVE_TRACK_LOG_INTERVAL_SEC = 0.75  # 只限制跟踪日志频率，不影响控制速度
# Nav2 can remain ACTIVE indefinitely while making millimetre corrections or
# repeating recovery behaviours at an otherwise usable pose.  Accept a wider
# safe neighbourhood only after odometry says the chassis has remained still
# and aligned for three continuous seconds.
NAV_NEAR_GOAL_SETTLE_SEC = 1.0  # Nav2 在可接受邻域内连续静止 1.0 s 后可接管
NAV_NEAR_GOAL_YAW_TOL_RAD = 0.15
E_SCAN_NEAR_GOAL_POSITION_TOL_M = 0.15
# 桌边最外侧放置点靠近代价区，Nav2 可能在剩余 15--20 cm 时
# 因无法继续向桌边推进而反复恢复。位置进入 20 cm 且独立的
# 0.10 rad 航向门槛也合格后，直接停止 Nav2 并进入放置。
TABLE_NEAR_GOAL_POSITION_TOL_M = 0.20
# 进入桌边末端区后，Nav2 仍决定 linear.x，客户端只把 angular.z
# 替换为一个固定转速。这样保留局部规划器对前后位置和障碍的处理，
# 又不会在最后几度上把角速度慢慢衰减到几乎不动。固定 0.35 rad/s
# 不超过双臂夹持负载导航的已验证上限，因此所有商品共用一个值。
TABLE_NAV_YAW_ASSIST_START_DISTANCE_M = 0.40
TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS = 0.35
TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD = 0.10
# The fixed observation pose is about 0.65 m of hand travel from the requested
# insertion line.  The old 0.48 m ceiling was sized for a separate Nav2 pick
# waypoint and would reject every direct approach.  Keep the independent
# observation-lock and shelf-distance guards, but budget enough travel for the
# observation-to-shelf curve.  The time values below remain diagnostics only
# while ``FINE_APPROACH_TIME_ABORTS_ENABLED`` is false.
FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS = 0.008
FINE_APPROACH_TIMEOUT_MARGIN_SEC = 5.0  # 动态估算的近柜行驶时间再留 5 s 余量
FINE_APPROACH_MIN_TIMEOUT_SEC = 20.0  # 近柜阶段超时下限（不是固定等待）
FINE_APPROACH_MAX_TIMEOUT_SEC = 120.0  # 近柜阶段超时上限（不是固定等待）
FINE_APPROACH_STALL_TIMEOUT_SEC = 3.0  # 仅用于扩大安全近距接管包络，不再失败
FINE_APPROACH_PROGRESS_EPS_M = 0.003
# The fine creep starts wherever Nav2 can safely settle near the shelf.  A
# fixed travel cap is brittle: the normal 0.218 m hand/object work-point gap
# plus the permitted coarse Nav2 error can already consume about 0.34 m.
# Derive a per-attempt budget from the measured end-effector shortfall, retain
# a small steering/localisation margin, and keep an independent hard ceiling.
# A direct C1/C3 curve is longer than its pure +Y hand shortfall.  Budget the
# measured shortfall plus 15 cm of steering path; the independent 0.85 m hard
# ceiling and target-forward shelf guard still bound every attempt.
FINE_APPROACH_TRAVEL_MARGIN_M = 0.15
FINE_APPROACH_MIN_TRAVEL_LIMIT_M = 0.30
FINE_APPROACH_ABSOLUTE_MAX_TRAVEL_M = 0.85
FINE_APPROACH_STOP_TOL_M = 0.008
# The Baseline publishes an acceleration-limited command, so a zero command
# does not stop the chassis in the same control tick.  Predict the remaining
# coast distance from the *current ramped speed* and start the stop early.
# The short reaction term covers one/two control cycles and the small margin
# covers odometry quantisation; neither is a replacement for the shelf guard.
FINE_APPROACH_BRAKE_REACTION_SEC = 0.040  # 仅用于预测制动滑行，不会 sleep 40 ms
FINE_APPROACH_BRAKE_MARGIN_M = 0.002
# When lateral error is still present, preserve enough longitudinal room for
# the observation-to-slot curve to finish.  This prevents an edge-column
# target from reaching/passing grasp depth while the hand is still far left or
# right of the product.  The reserve shrinks continuously as alignment
# improves, so normal motion remains "advance while steering".
FINE_APPROACH_ALIGNMENT_RESERVE_BASE_M = 0.060
FINE_APPROACH_ALIGNMENT_RESERVE_GAIN = 0.60
FINE_APPROACH_ALIGNMENT_RESERVE_MAX_M = 0.200
PRECISION_ALIGNMENT_RESERVE_BASE_M = 0.100
PRECISION_ALIGNMENT_RESERVE_GAIN = 0.75
PRECISION_ALIGNMENT_RESERVE_MAX_M = 0.240
FINE_APPROACH_ALIGNMENT_RESERVE_RAMP_M = 0.030
# A unicycle pose controller needs a little translation to remove endpoint X
# error; commanding exactly zero at the reserve boundary can create a static
# equilibrium.  Keep a very slow coupled crawl while steering, bounded by the
# explicit 20 mm depth-overshoot guard below.
FINE_APPROACH_ALIGNMENT_CRAWL_SPEED_MPS = 0.020
# If an external collision/slip still carries the hand well past the requested
# insertion plane, fail with an explicit overshoot reason instead of either
# closing too deep or sitting at zero speed until the generic stall timer.
FINE_APPROACH_DEPTH_OVERSHOOT_LIMIT_M = 0.020
# A final few millimetres can be lost to wheel/odom quantisation after a long
# creep.  If the gripper is already inside this conservative centimetre-scale
# envelope, stop the chassis and accept only after three continuous seconds of
# real stillness.  This is deliberately narrower than the product widths and
# never bypasses the shelf-depth guard.
FINE_APPROACH_NEAR_STOP_TOL_M = 0.015
FINE_APPROACH_NEAR_SETTLE_SEC = 0.4  # 安全近距锁定后原地连续停稳 0.4 s 再夹紧
# Do not stop a normally progressing grasp this far out: bottles still benefit
# from reaching the tighter envelope above.  If the deployed gripper has made
# no measurable progress for the full stall interval, however, a 30 mm
# longitudinal shortfall still leaves it at least 5 mm beyond the product
# centre because the fixed template deliberately asks for 35 mm of insertion.
# Accept that contact-limited pose after the same real-stillness dwell instead
# of reporting fine_approach_stalled while the object is already between the
# fingers.
FINE_APPROACH_STALLED_NEAR_STOP_TOL_M = 0.030
# At the 0.08 m/s fine-creep speed and 0.8 m/s^2 command deceleration, the
# ideal stopping distance is about 4 mm.  Enter the same recovery envelope a
# little before the measured travel cap so command ramp-down does not turn a
# valid contact-limited grasp into a sub-millimetre travel-limit failure.
FINE_APPROACH_TRAVEL_BRAKE_MARGIN_M = 0.006
# Even when vision is occluded, this local-frame guard prevents the chassis
# continuing toward the shelf if remembered target/odometry geometry is bad.
FINE_APPROACH_MIN_TARGET_FORWARD_M = 0.49
# The edge rows are most sensitive to arm occlusion and small base-pose error.
# Lock the first reacquired longitudinal plane instead of chasing later RGB-D
# drift, add a conservative level-specific insertion, and use a slower final
# lateral-alignment zone.  L2 retains its proven geometry and creep behaviour.
GRASP_INSERTION_BEYOND_CENTER_M = 0.035
# 可乐、脉动的窄圆柱需要让两指根部再包住一些，避免只用指尖擦到瓶身。
# 10 mm 实测稍深，圆柱通用补偿回调为 5 mm；可乐和脉动再分别使用
# 下面的毫米级微调，避免再次恢复到此前明显过深的状态。
CYLINDER_DEEP_GRASP_KINDS = frozenset({"kele", "maidong"})
CYLINDER_EXTRA_INSERTION_M = 0.005
# 可乐本轮虽完成抓取，但指腹包覆仍略浅；只给可乐再增加 3 mm，避免把
# 已经回调过深度的脉动一起推进。
KELE_ADDITIONAL_INSERTION_M = 0.003
# 脉动本轮已基本进入两指之间，但闭合位置仍略浅。只单独增加 2 mm；
# 加上层级补偿后 L1/L2/L3 总插入量分别为 52/42/62 mm。
MAIDONG_ADDITIONAL_INSERTION_M = 0.002
# The broad, flat packages do not self-centre between the fingers like a
# bottle.  Give both of them another centimetre of insertion at every shelf
# level; the existing L1/L3 compensation is still added below.  The resulting
# shared middle-row baseline is 45 mm before the product-specific sandwich
# reduction below; all variants remain inside the independent shelf-distance
# and absolute-travel guards.
BROAD_PACKAGE_EXTRA_INSERTION_M = 0.010
BROAD_DEEP_GRASP_KINDS = frozenset({"heweidao", "sanmingzhi"})
# 三明治本轮已经夹到，但端部插入略深。只回收 3 mm，保留宽包装
# 通用补偿和 L1/L3 层级补偿，不影响合味道或其他商品。
SANMINGZHI_INSERTION_REDUCTION_M = 0.003
# 苹果本轮已经进入两指之间，但前向包覆略深，容易让指尖继续挤压球体或
# 蹭到货架。只单独回收 2 mm，保留苹果原有的高精度横向闭环和层级补偿。
PINGGUO_INSERTION_REDUCTION_M = 0.002
# 合味道杯沿较宽，通用宽包装的 10 mm 补偿仍会让指腹停在边缘。
# 单独再深 5 mm，不改动三明治和其他商品已验证的抓取深度。
HEWEIDAO_ADDITIONAL_INSERTION_M = 0.005
# 口香糖盒比通用商品再多包住 5 mm，避免停在盒子前缘时
# 闭合只擦到外壁。该补偿不影响可乐/脉动或宽包装的独立深度。
KOUXIANGTANG_EXTRA_INSERTION_M = 0.005
# 薯片筒直径约 65 mm，画面中即使末端横差只剩约 4 mm，原有深度仍可能
# 只让指尖搭在筒身前缘。独立补偿由 10 mm 增至 12 mm，让两指再深入
# 越过圆柱中心并在闭合时自定心；不改变可乐、脉动等其他商品。
SHUPIAN_EXTRA_INSERTION_M = 0.012
# The symmetric two-hand clamp benefits from entering just beyond the box centre
# before squeezing.  Five millimetres is enough to deepen the contact without
# materially reducing rear-board or elbow clearance.
TISSUE_EXTRA_INSERTION_M = 0.005
L1_GRASP_EXTRA_INSERTION_M = 0.010
L3_GRASP_EXTRA_INSERTION_M = 0.020
L1_FINE_APPROACH_MIN_TARGET_FORWARD_M = 0.48
L3_FINE_APPROACH_MIN_TARGET_FORWARD_M = 0.47
EDGE_FINE_APPROACH_MAX_ANGULAR = 0.35
EDGE_GRASP_LATERAL_TOL_M = 0.015
# 薯片筒与脉动具有相同的 65 mm 圆柱碰撞外形，夹爪闭合时可以沿圆弧
# 自定心。本轮 B/L3/C3 已到抓取深度（仅余 6.2 mm），却因为 15.35 mm
# 横向误差比通用 15 mm 阈值多 0.35 mm 而完全没有进入夹紧状态。
# 只给薯片留 3 mm 的测量/接触余量；货架前向防撞、深度超调以及停稳检查
# 仍照常生效，避免为了跨过数值边界继续把底盘顶向货架。
SHUPIAN_GRASP_LATERAL_TOL_M = 0.018
# 上、下层的罐/瓶体从斜角观察时，RGB-D 可见表面中心会比实际货位
# 中心稳定地偏左约 1--2 cm。这两类商品的实际摆放位由 ArUco 货位
# 确定，因此边层抓取用货位 X 作为横向抓取线，视觉仍用来确认目标
# 可见和跟踪头部。中层保留已验证的实时视觉横向修正。
EDGE_ROW_CYLINDER_SLOT_LATERAL_KINDS = frozenset({"kele", "maidong"})
# 10 mm 门槛容易让已经到达抓取深度的圆柱在数值临界点持续微调；
# 小幅放宽至 12 mm，减少脉动/可乐到位后的无效等待，同时仍明显严于
# 普通边层商品的 15 mm 容差。
EDGE_ROW_CYLINDER_GRASP_LATERAL_TOL_M = 0.012
# 实机画面显示末端按货位中心收敛后，两指开口仍系统性落在圆柱左侧。
# 在上、下层把最终夹爪参考点沿世界 +X（画面右侧）平移 15 mm；商品
# 像素本身仍用于第一段实时转向，不用这个常量替代视觉闭环。
EDGE_ROW_CYLINDER_GRASP_X_OFFSET_M = 0.015
# 多类单手商品都出现夹爪整体偏左，因此为所有右手抓取统一加入 5 mm
# 的世界 +X（画面右侧）标定；双臂夹持中心保持原标定。可乐上/下层
# 继续叠加原有 15 mm 货位补偿。第一段仍以实时像素误差为主，这个常量
# 只校正夹爪遮挡商品、视觉冻结后的系统偏差。
RIGHT_HAND_GRASP_X_OFFSET_M = 0.005
# 乐事薯片仍有约数毫米的稳定左偏。只在实时像素闭环进入末端遮挡区后，
# 将薯片最终夹爪线再沿世界 +X（画面右侧）移 5 mm；前段仍然直接使用
# “商品中心—夹爪中心”的同步像素误差，其他商品完全不受影响。
SHUPIAN_ADDITIONAL_GRASP_X_OFFSET_M = 0.005
# 脉动在圆柱货位补偿和右手通用标定后仍有轻微左偏。末端夹爪线再沿
# 世界 +X（画面右侧）移动 5 mm，不改变可乐及其他商品的横向标定。
MAIDONG_ADDITIONAL_GRASP_X_OFFSET_M = 0.005
# 苹果在通用 5 mm 右手标定后仍稳定落在球心左侧。末端遮挡后再沿世界
# +X（画面右侧）补 3 mm，使两指围住球体中部，且不改变前向抓取深度。
PINGGUO_ADDITIONAL_GRASP_X_OFFSET_M = 0.003
FINE_APPROACH_LATERAL_PROGRESS_EPS_M = 0.002
# The visual approach normally starts at the observation point; selected
# random-task shelf legs may now hand over shortly before it.  Cruise faster
# while there is open space, and retain a non-zero forward speed while a broad
# or nearly gripper-width product is still being centred.
DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS = 0.20 #0.16
DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS = 0.080 #0.070
# Use one continuous distance schedule for every product: retain the efficient
# cruise speed in open space, then smoothly blend down to the alignment speed
# before the hand reaches the shelf.  This replaces the previous late,
# product-specific speed step that could leave too little distance to correct
# an initially curved C1/C3 approach.
DIRECT_FINE_APPROACH_TAPER_START_M = 0.42
DIRECT_FINE_APPROACH_TAPER_END_M = 0.08
# Translation and steering now begin together.  The heading tolerances remain
# diagnostic values and document the normal observation-pose error envelope;
# they no longer gate forward motion for several seconds.
DIRECT_FINE_APPROACH_MAX_ANGULAR = 0.34
FINE_APPROACH_INITIAL_HEADING_TOL_RAD = 0.10
# The first target starts from the surveyed observation pose.  After a table
# round trip the same Nav2 goal can finish with a small but repeatable heading
# residual.  Retain both values in diagnostics, while the continuous curve
# now corrects that residual during translation instead of waiting in place.
SUBSEQUENT_FINE_APPROACH_INITIAL_HEADING_TOL_RAD = 0.06
# 除首件外，B/C 柜在距观察点约 0.25 m 时展开抓取姿态并把底盘
# 交给两段视觉闭环，给横向纠偏留下更长距离。A/D/E 柜同样在
# 0.25 m 内并行展开姿态，近航点后只要位置和朝向大致正确就取消
# Nav2，立即用视觉伺服向前抓取，不再等偏航角与航点完全一致。
# B/C 滚动交接后不再追踪观察航点；A/D/E 则使用宽松的近航点交接，
# 两种路径都避免回到“停车—摆姿—前进”的串行老流程。首件直达 E 柜在
# 距观察点 0.5 m 时才展开抓取姿态并交给视觉闭环，避免过早伸臂。
RANDOM_PICK_PREDEPLOY_DISTANCE_M = 0.10
BC_RANDOM_PICK_PREDEPLOY_DISTANCE_M = 0.25
RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD = 0.85
RANDOM_ROLLING_HANDOFF_DISTANCE_M = 0.10
BC_RANDOM_ROLLING_HANDOFF_DISTANCE_M = 0.25
RANDOM_ROLLING_HANDOFF_YAW_TOL_RAD = 0.85
# 保留这个距离常量供非随机旧流程诊断。
STATIONARY_PICK_PREDEPLOY_DISTANCE_M = 0.25
# A/D/E 后续抓取不再用 Nav2 GoalChecker 的精确到位门槛。距航点
# 10 cm 且偏航误差不超过 0.20 rad（约 11.5 度）时，若抓取姿态已就绪，
# 就立即交给视觉前进，不等 Nav2 继续低速磨到航点偏航角。
ADE_NEAR_HANDOFF_SHELVES = frozenset({"A", "D", "E"})
ADE_OBSERVATION_HANDOFF_POSITION_TOL_M = 0.10
ADE_OBSERVATION_HANDOFF_YAW_TOL_RAD = 0.20
# 该值只限制交给视觉控制器的初始速度状态，不再作为是否允许交接的门槛；
# 否则 Nav2 速度较高时会错过交接窗口并在航点停下。
RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS = 0.80
# 提前进入近柜闭环时，按交接距离扩充单次行驶预算；货架前向距离、
# 深度超调与目标丢失保护仍保持不变。
RANDOM_ROLLING_HANDOFF_EXTRA_TRAVEL_M = 2.10
# 随机任务首件去 E 柜不再创建 Nav2 目标，而由里程计姿态闭环直接发布
# /cmd_vel。首段按当前任务提速到 2.00 m/s，不受 Nav2 局部代价或观察点末端
# 行为减速；进入首件专用的 0.5 m 范围后才交给商品视觉抓取闭环。
FIRST_E_DIRECT_CRUISE_SPEED_MPS = 2.00
FIRST_PICK_PREDEPLOY_DISTANCE_M = 0.50
FIRST_PICK_ROLLING_HANDOFF_DISTANCE_M = 0.50
# 0.5 m 处交接后，抓取闭环要覆盖“当前位置到观察点”以及观察点到货架的
# 全部路程；该值只扩大首件（E 直控或 D 的 Nav2 交接）的行程预算，
# 货架前向距离和超深保护仍有效。
FIRST_PICK_HANDOFF_EXTRA_TRAVEL_M = 3.20
FIRST_E_DIRECT_POSITION_KP = 1.20
FIRST_E_DIRECT_MIN_LINEAR_MPS = 0.08
FIRST_E_DIRECT_HEADING_KP = 2.20
FIRST_E_DIRECT_MAX_ANGULAR_RADPS = 1.20
FIRST_E_DIRECT_POSITION_TOL_M = 0.10
FIRST_E_DIRECT_YAW_TOL_RAD = 0.08
# 视觉抓取控制允许继承首段 2.00 m/s 的实测速度，再由 0.8 m/s^2 的现有
# 斜坡连续减到抓取巡航速度。后续各柜的交接不再因 Nav2 当前速度被拒绝。
PICK_FINE_HANDOFF_MAX_LINEAR_MPS = max(
    FIRST_E_DIRECT_CRUISE_SPEED_MPS,
    RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS,
)
FINE_APPROACH_NEAR_ANGULAR_ZONE_M = 0.18
FINE_APPROACH_NEAR_LATERAL_ZONE_M = 0.06
FINE_APPROACH_MIN_NEAR_ANGULAR_RADPS = 0.10
FINE_APPROACH_ANGULAR_ACCEL_RADPS2 = 0.90
# 开阔段即使初始航向误差较大也保持小幅向前弧线运动，避免出现先原地调姿、
# 再继续走的停顿；真正到达抓取深度后仍会按制动锁定流程停车夹取。
FINE_APPROACH_MIN_FORWARD_ALIGNMENT_SCALE = 0.25
# The goal-bearing angle becomes numerically ill-conditioned when the base is
# almost at its terminal pose but the long deployed hand still has lateral
# error.  In that final zone, control the measured gripper endpoint directly
# through the base/hand Jacobian instead of chasing a rapidly rotating bearing.
FINE_APPROACH_TERMINAL_EE_CONTROL_ZONE_M = 0.22
FINE_APPROACH_TERMINAL_EE_KP = 2.00
# This unicycle pose curve supplies the fallback whenever the current image no
# longer contains the selected target.  With a fresh synchronized box centre,
# the product-to-gripper pixel servo has steering priority instead.  Remaining
# pose distance still scales forward speed in both cases.
DIRECT_POSE_GUIDANCE_DISTANCE_KP = 1.50
DIRECT_POSE_GUIDANCE_ALPHA_KP = 2.50
DIRECT_POSE_GUIDANCE_BETA_KP = -0.80
# The terminal base-pose law alone can trace a wide arc from the common
# observation point, especially for C1/C3.  Feed the measured gripper/product
# X error into the same steering command from the very first moving tick.  It
# keeps the hand converging toward the product throughout the approach while
# the angular acceleration/velocity limits below prevent a violent correction.
DIRECT_POSE_GUIDANCE_LATERAL_KP = 1.50
# The observation pose supplies a multi-frame, shelf-validated world point.
# Keep that point as the primary geometric reference for the complete direct
# approach.  Live detections refine it when available, but an expected arm or
# gripper occlusion must not stop the chassis halfway to the slot.  The shelf
# forward-distance, travel-budget and timeout guards below remain mandatory.
# A 65 mm-wide sandwich pack has only about 7.5 mm clearance per side in the
# fully open 80 mm parallel gripper.  Unlike a bottle it cannot self-centre by
# sliding along curved fingers, so it uses the slower coupled curve and the
# tight final endpoint tolerance below.
SANMINGZHI_ALIGN_ENTER_TOL_M = 0.006
SANMINGZHI_ALIGN_EXIT_TOL_M = 0.008
# Keep precision products tied to the common alignment-speed setting.  This
# avoids a hidden per-product 0.08 m/s ceiling after the operator raises the
# common approach speed.
SANMINGZHI_FINE_APPROACH_SPEED_MPS = DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS
SANMINGZHI_ALIGNMENT_STALL_TIMEOUT_SEC = 8.0  # 三明治允许 8 s 精细纠偏无进展
SANMINGZHI_FINE_APPROACH_MAX_ANGULAR = 0.28
# 三明治只比右夹爪的有效开口窄约 15 mm，不能在末段为追固定
# 底盘偏航而抛弃已对准的图像中心。保留实时像素伺服到剩余 10 cm，
# 然后锁住切换瞬间的航向，只做小量末端位置修正。
SANMINGZHI_TERMINAL_EE_CONTROL_ZONE_M = 0.10
SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD = math.radians(4.0)
SANMINGZHI_TERMINAL_HEADING_HOLD_KP = 0.80
SANMINGZHI_TERMINAL_HEADING_MAX_CORRECTION_RADPS = 0.08
# 苹果为圆形，手指已越过球心后可以靠对称闭合完成最后自定心。
# 仍优先追踪原来的 3 mm 精对准；只有已进入这个保守的深度/横向
# 夹持包络时才允许停稳后夹紧，避免爪子已经包住苹果却一直全开等超时。
PINGGUO_INSERTED_GRASP_DEPTH_TOL_M = 0.020
PINGGUO_INSERTED_GRASP_LATERAL_TOL_M = 0.009
# 苹果继续使用实时“物品中心—夹爪中心”像素差直到剩余约 11 cm，
# 比原来的 14 cm 晚 3 cm 再交给末端几何控制，避免一开始已经对准却在
# 后段重新向左漂；最后 4.5 cm 的强制冻结仍负责遮挡和柜体保护。
PINGGUO_TERMINAL_EE_CONTROL_ZONE_M = 0.11
# 可乐瓶身较窄，通用 22 cm 处冻结视觉容易把很小的图像偏差保留到最终
# 抓取线。将实时“商品中心—夹爪中心”像素纠偏保持到 12 cm，再交给
# 遮挡后的末端几何控制；末段另叠加上方实测得到的 5 mm 右移补偿。
KELE_TERMINAL_EE_CONTROL_ZONE_M = 0.12
# 脉动之前沿用 22 cm 通用区间，会过早退出实时像素闭环，然后用
# 0.02 m/s 左右的末端联动速度慢爬约 10 cm。与同类窄圆柱可乐一样，
# 保持视觉动态对准到剩余 12 cm 再转入末端几何控制，减少深处等待。
MAIDONG_TERMINAL_EE_CONTROL_ZONE_M = 0.12
# 薯片与可乐出现相同的“前段已对准、末段又偏左”现象，因此也把实时
# 商品中心—夹爪中心像素闭环保留到剩余 12 cm，再冻结视觉几何。
SHUPIAN_TERMINAL_EE_CONTROL_ZONE_M = 0.12
# These products retain their known ArUco slot as an identity, shelf-depth and
# safety anchor.  Their lateral grasp line is nevertheless refined from live
# vision during the open first part of the approach, then frozen before the
# hand begins to occlude the target.  Gum keeps moving while steering because
# stopping to align first made its long hand lever sweep sideways and then
# contact-limit halfway through the insertion; only the two nearly
# gripper-width fruit require pre-alignment.
PRECISION_SLOT_ANCHORED_KINDS = frozenset(
    {
        "sanmingzhi",
        "heweidao",
        "kouxiangtang",
        "pingguo",
        "chengzi",
        "zhijin",
    }
)
COUPLED_ALIGNMENT_KINDS = frozenset({"kouxiangtang"})
COUPLED_ALIGNMENT_FINAL_TOL_M = 0.015
KOUXIANGTANG_FINE_APPROACH_STALL_TIMEOUT_SEC = 4.0  # 口香糖卡滞判定上限
KOUXIANGTANG_FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS = 0.006
# 双臂夹持商品需要两只手同时压住侧面，少量横向误差不会像圆瓶一样自定心。
# 让该配置在末段降到 0.05 m/s，并把相机像素伺服保留到距离
# 抓取深度约 10 cm 时才交给末端三维控制，避免在 22 cm 处过早锁定带偏记忆线。
TISSUE_FINE_APPROACH_ALIGN_SPEED_MPS = 0.050
TISSUE_TERMINAL_EE_CONTROL_ZONE_M = 0.10
# 宽包装正面会在双手中心到达理想插入线之前先发生接触。只要剩余深度和
# 横向偏差已进入双手可包络范围，就直接对称夹紧，利用双臂完成最后自定心；
# 不再倒车重进，避免接触后反复退出导致掉落或额外耗时。
TISSUE_CONTACT_GRASP_DEPTH_TOL_M = 0.090
TISSUE_CONTACT_GRASP_LATERAL_TOL_M = 0.025
# 口香糖首次随机柜位抓取时，在通用的 22 cm 位置锁定视觉过早，
# 后半段虽然将记忆点误差收敛为零，真实盒心仍可能留有小偏差。
# 保留像素闭环到 12 cm，再交给遮挡后的末端几何控制。
KOUXIANGTANG_TERMINAL_EE_CONTROL_ZONE_M = 0.12
PRECISION_GRASP_ALIGNMENT = {
    "sanmingzhi": (
        SANMINGZHI_ALIGN_ENTER_TOL_M,
        SANMINGZHI_ALIGN_EXIT_TOL_M,
        SANMINGZHI_FINE_APPROACH_SPEED_MPS,
    ),
    "pingguo": (
        0.0030,
        0.0050,
        DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS,
    ),
    "chengzi": (
        0.0025,
        0.0045,
        DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS,
    ),
    # The two side pads leave comfortable clearance during the approach, but
    # the midpoint must be centred before they synchronously squeeze the box.
    "zhijin": (
        0.0040,
        0.0060,
        TISSUE_FINE_APPROACH_ALIGN_SPEED_MPS,
    ),
}
GRASP_CLOSE_DWELL_SEC = 0.30  # 普通商品夹爪到位后保持夹紧 0.3 s
L3_GRASP_CLOSE_DWELL_SEC = 0.40  # 高层商品夹紧后多等到 0.4 s 再抬升
# A round orange is almost as wide as the open gripper and has no flat face to
# resist inertial motion.  Command more closing preload and let it settle in
# the fingers before lifting.  Other products keep their proven close command.
CHENGZI_GRIP_CLOSE_COMMAND = 0.02
CHENGZI_GRASP_CLOSE_DWELL_SEC = 0.80  # 橙子夹紧预载稳定 0.8 s 再抬升
CHENGZI_LIFT_JOINT_SLEW = 0.65
CHENGZI_CARRY_STOW_JOINT_SLEW = 0.70
# At the lower orange grasp height, wait for tighter torso tracking before the
# chassis advances.  This preserves finger-to-board clearance even if the
# common 25 mm slide tolerance would otherwise accept a visibly low pose.
CHENGZI_DEPLOY_SLIDE_TOL_M = 0.008
# Wide/box packages can hold the fingers apart even while the close command is
# active, so measured gripper position alone cannot prove that release motion
# has actually reached the actuator.  Ramp the open command all the way out and
# let the package settle; the following torso lift/straight retreat then
# separates both fingers from the released product without an arm-space sweep.
BROAD_RELEASE_KINDS = frozenset({"heweidao", "sanmingzhi"})
BROAD_RELEASE_COMMAND_OPEN_MIN = 0.98
BROAD_RELEASE_SETTLE_SEC = 0.30  # 三明治等宽包装完全张开并落稳的时间
# After release, every product except the cup noodle first receives a
# torso-only vertical separation while the hand pose stays fixed.  This avoids
# dragging bottles, fruit, flat packages and the symmetric two-hand clamp toward
# the table edge.  Heweidao keeps its proven direct retreat because its wide
# rim needs the horizontal separation and does not tolerate an extra lift.
POST_RELEASE_LIFT_EXEMPT_KINDS = frozenset({"heweidao"})
POST_RELEASE_CLEARANCE_LIFT_M = 0.100
POST_RELEASE_SLIDE_TOL_M = 0.012
POST_RELEASE_CLEARANCE_SETTLE_SEC = 0.00  # 松爪后身体抬高到位再保持 0.00 s
# Keep the proven placement geometry unchanged, but soften the three loaded
# motions that can impart the most momentum to a loosely held package.  The
# table-specific stop dwell is shorter than the general navigation stop dwell,
# while still requiring odometry to report a stationary chassis.
TABLE_STOP_SETTLE_SEC = 0.30  # 到桌边后底盘连续静止 0.30 s 才伸手
# 单臂商品到桌后的展开、前伸和升降使用这组提速值。反馈到位门槛
# 和落稳时延保持不变，因此只是缩短实际运动过程，不会提前松爪。
PLACE_UNFOLD_GENTLE_JOINT_SLEW = 1.25
PLACE_ADVANCE_GENTLE_JOINT_SLEW = 0.42
PLACE_LOWER_GENTLE_JOINT_SLEW = 0.44
PLACE_RELEASE_JOINT_SLEW = 0.65
# 前伸手臂的反馈一到位就连续切换到身体下降，取消两个动作之间
# 额外的固定停顿。放到桌面和松爪后的稳定时间仍保留。
PLACE_ADVANCE_TO_LOWER_SETTLE_SEC = 0.0
# 橙子和苹果是滚动物，三段负载动作单独恢复到本轮提速之前的速度。
# 其他单臂商品仍使用上面的提速值；松爪后的抬身不再携物，可继续快速。
CHENGZI_PLACE_UNFOLD_JOINT_SLEW = 0.60
CHENGZI_PLACE_ADVANCE_JOINT_SLEW = 0.24
CHENGZI_PLACE_LOWER_JOINT_SLEW = 0.24
PINGGUO_PLACE_UNFOLD_JOINT_SLEW = 1.00
PINGGUO_PLACE_ADVANCE_JOINT_SLEW = 0.34
PINGGUO_PLACE_LOWER_JOINT_SLEW = 0.36
PLACE_UNFOLD_JOINT_SLEW_OVERRIDES = {
    "chengzi": CHENGZI_PLACE_UNFOLD_JOINT_SLEW,
    "pingguo": PINGGUO_PLACE_UNFOLD_JOINT_SLEW,
}
PLACE_ADVANCE_JOINT_SLEW_OVERRIDES = {
    "chengzi": CHENGZI_PLACE_ADVANCE_JOINT_SLEW,
    "pingguo": PINGGUO_PLACE_ADVANCE_JOINT_SLEW,
}
PLACE_LOWER_JOINT_SLEW_OVERRIDES = {
    "chengzi": CHENGZI_PLACE_LOWER_JOINT_SLEW,
    "pingguo": PINGGUO_PLACE_LOWER_JOINT_SLEW,
}
# 橙子在确认已经落到桌面后，只提高“张开夹爪”这一个自由度的速度。
# 下放速度、双向高度门槛和静止反馈均保持原值，避免再次半空松爪或穿桌。
# _update_chengzi_table_lower 已经要求高度、停止、新鲜反馈和手臂同时到位，
# 因此无需再叠加普通商品的 0.30 s 下放停稳等待。
CHENGZI_PLACE_LOWER_SETTLE_SEC = 0.0
CHENGZI_PLACE_RELEASE_JOINT_SLEW = 2.00
PLACE_RELEASE_JOINT_SLEW_OVERRIDES = {
    "chengzi": CHENGZI_PLACE_RELEASE_JOINT_SLEW,
}
# Orange placement uses the measured hand-to-fruit offset at grasp, then a
# two-sided height gate. Never treat "above the target" as ready to release.
CHENGZI_PLACE_EE_Z_TOL_M = 0.003
CHENGZI_PLACE_FINAL_LOWER_SLEW = 0.04  # slide command <= 12 mm/s
CHENGZI_PLACE_SLOW_DISTANCE_M = 0.100
CHENGZI_PLACE_STOP_SPEED_MPS = 0.003
CHENGZI_PLACE_COMMAND_TOL_M = 0.0005
CHENGZI_PLACE_CORRECTION_LIMIT_M = 0.020
CHENGZI_PLACE_LOWER_TIMEOUT_SEC = 30.0  # 橙子低速下降最多允许 30 s
RETURN_STOW_FAST_JOINT_SLEW = 1.60
SEARCH_RESTORE_FAST_JOINT_SLEW = 1.60
PICK_DEPLOY_FAST_JOINT_SLEW = 1.50
# 单手商品进入抓取模板时，先沿反向安全路点把低位左臂支起。左臂到位
# 之后才允许升降轴下降，并将这段关节速率从 1.50 降到 1.20，避免身体
# 突然下落使左手擦地；右臂和视觉底盘控制仍可并行运行。
PICK_DEPLOY_BODY_LOWER_JOINT_SLEW = 1.20
FAST_ARM_STAGE_DWELL_SEC = 0.12  # 快速收/展臂各路点到位后的短防抖保持
RETURN_STOW_CONCURRENT_TIMEOUT_SEC = 30.0  # 返程并行收臂的总超时上限
# 总计时不从收到任务或发出 Nav2 目标开始，而从里程计确认底盘第一次真实
# 移动开始。这样 Nav2 激活、规划和任务消息等待不会被误算为机器人运动时间。
MISSION_TIMER_LINEAR_START_MPS = 0.005
MISSION_TIMER_ANGULAR_START_RADPS = 0.010
# 每轮正常完成时覆盖为最新结果。run_full_test.sh 将项目目录挂载到
# Client 容器，所以文件会直接保存在宿主机工程根目录，不输出到终端。
MISSION_TOTAL_TIME_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "mission_total_time.txt"
)
# Raise the whole compact carry pose by 6 cm.  With the previous 0.06 m
# command the maidong bottom was about 0.755 m, slightly below the 0.767 m
# tabletop.  At 0.00 m the gripper centre is 0.920 m and the maidong bottom is
# about 0.815 m, leaving roughly 4.8 cm of table-edge clearance while retaining
# exactly the same compact six-joint arm shape.
TRANSPORT_SLIDE_M = 0.00
SLIDE_TOL_M = 0.025
ARM_SETTLE_SEC = 0.20  # 通用机械臂/升降反馈到位后的稳定保持
MANIPULATION_TIMEOUT_SEC = 20.0  # 通用单个操作阶段的故障超时
PICK_TRANSPORT_TOGETHER_TIMEOUT_SEC = 30.0  # 抓后双臂与身体并行收拢上限
# A held bottle can leave wrist joint 5 resting against its effort limit even
# though the arm is already in a compact, collision-safe transport shape.  Do
# not fail the mission on that harmless residual alone: require the other five
# joints to be close, check the measured gripper pose is still in the compact
# chest envelope, and hold that condition continuously for three seconds.
COMPACT_CARRY_SAFE_SETTLE_SEC = 1.5  # 负载下安全收拢条件需连续成立 1.5 s
COMPACT_CARRY_NON_WRIST_TOL_RAD = 0.060
COMPACT_CARRY_WRIST_TOL_RAD = 0.160
COMPACT_CARRY_MAX_TILT_RAD = math.radians(10.0)
COMPACT_CARRY_EE_X_RANGE_M = (0.47, 0.62)
COMPACT_CARRY_EE_Y_RANGE_M = (-0.02, 0.16)
# Even at the lowest accepted wrist-load residual, a maidong bottom remains
# above the 0.767 m tabletop instead of being allowed to skim its edge.
COMPACT_CARRY_EE_Z_RANGE_M = (0.89, 0.97)
COMPACT_CARRY_LINK_Y_RANGE_M = (-0.215, 0.150)
CONFIRMED_SLOT_DISTANCE_M = 0.18
GRASP_RETRY_LIMIT = 3
COSTMAP_PARAMETER_TIMEOUT_SEC = 5.0  # 切换足迹/速度限制等 Nav2 参数最多等 5 s
STOP_HANDOFF_TIMEOUT_MARGIN_SEC = 4.0  # 底盘减速停稳/控制权交接额外超时余量
TABLE_RETREAT_DISTANCE_M = 0.25
TABLE_RETREAT_TIMEOUT_SEC = 10.0  # 放置后本地直线退离桌边最多 10 s
# 商品已经松开并完成抬身净空后，直接向底盘发布固定倒车速度，不让 Nav2
# 规划倒车，也不再受通用 0.8 m/s² 指令斜坡拖慢。普通商品使用底盘直控
# 上限；双臂商品此时虽已松开，双臂仍展开，因此保留稍低但比旧值更快的速度。
TABLE_RETREAT_SPEED_MPS = 0.45
TISSUE_TABLE_RETREAT_SPEED_MPS = 0.36
# Do not return all the way to the observation pose with a loaded, fully
# extended arm.  This line is still far enough from the shelf for the checked
# compact-carry transition, while removing roughly 0.4 m from every cycle.
PICK_RETREAT_CLEAR_Y_M = 2.48
PICK_RETREAT_SPEED_MPS = 0.26
# Folding the right arm from the table-clear pose sweeps the open fingers
# toward the tabletop.  The fixed 0.25 m retreat can finish too close when the
# chassis drifts toward the table during placement.  MuJoCo contact sweeps show
# y=-2.60 m is the contact boundary for all five slots; retain another 5 cm of
# margin before allowing the fold.
TABLE_RETURN_FOLD_MIN_Y_M = -2.55

# The 172 mm-wide package cannot fit inside one 80 mm gripper.  Both
# grippers are kept closed and used as padded side contacts instead: approach
# with a generous gap, then move both endpoints inward symmetrically.  Keep the
# successful horizontal clamp unchanged all the way to the table.  A former
# 90-degree loaded roll made the left arm lag behind the right arm at its
# torque/pose limit and destroyed the balanced side pressure.
TISSUE_KIND = "zhijin"
# Keep both clamp arms close to straight while they hold the box.  At the
# ordinary 0.555 m hand workpoint the only legal dual-arm IK branch pushes the
# two elbows out to roughly +/-0.42 m.  Moving the hand centre forward lets the
# base stop farther from the shelf.  The former 0.87 m centre still left the
# clamp elbows at roughly +/-0.282 m and touched the E-side wall while turning.
# The 0.92 m centre still left the deploy/clamp envelopes at about
# +/-0.233/0.219 m.  At 0.93 m they shrink to about +/-0.211/0.196 m while
# retaining useful IK margin before the clamp branch disappears near 0.938 m,
# without changing the package contact spacing.
TISSUE_HAND_CENTER_X_M = 0.930
TISSUE_TABLE_BASE_Y_OFFSET_M = (
    TISSUE_HAND_CENTER_X_M - float(HAND_WORKPOINT_XY_M[0])
)
# The normal single-hand table template advances another 0.30 m after Nav2,
# whereas the rigid two-hand pose cannot add that arm-only insertion.
# The operator requested another 30 cm toward negative world Y on top of the
# existing 10 cm inset.  The pose helper clamps this request against the known
# table edge using the package half-depth and a physical edge margin, so the
# package is moved as far inward as possible without hanging off the table.
TISSUE_TABLE_BASE_INSET_M = 0.100
TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M = 0.300
TISSUE_TABLE_INSET_M = (
    TISSUE_TABLE_BASE_INSET_M
    + TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M
)
TISSUE_TABLE_EDGE_MARGIN_M = 0.015
# 单手商品的 nominal drop point 必须让完整碰撞体留在桌面内。该余量由
# _table_approach_pose() 结合商品半径和实际前伸距离换算成底盘 Y 下限。
# 它主要防止橙子/苹果在桌边虽然高度正确，却因水平落点越界而直接下落。
SINGLE_HAND_TABLE_EDGE_MARGIN_M = 0.010
# Preserve the same post-pick hand clearance from the shelf.  Because the
# straighter arms make the base stop earlier, the old absolute retreat line
# would otherwise already be behind the grasping base and cause no withdrawal.
TISSUE_PICK_RETREAT_Y_M = (
    PICK_RETREAT_CLEAR_Y_M - TISSUE_TABLE_BASE_Y_OFFSET_M
)
TISSUE_DEPLOY_HALF_SEPARATION_M = 0.132
# The closed-hand contact faces leave only about 1 mm preload per side at
# 0.115 m.  The former 0.111 m target still slipped during the long loaded
# route. Move each complete hand another 2 mm inward, giving roughly 7 mm of
# nominal preload per side before lateral alignment error. IK permits much
# smaller spacing, but stronger geometric overlap risks ejecting the rigid box
# or preventing the loaded arms from satisfying their target tolerance.
TISSUE_CLAMP_HALF_SEPARATION_M = 0.109
TISSUE_RELEASE_HALF_SEPARATION_M = TISSUE_DEPLOY_HALF_SEPARATION_M
TISSUE_GRASP_YAW_RAD = math.pi / 2.0
TISSUE_ARM_TOL_RAD = 0.055
TISSUE_STAGE_SETTLE_SEC = 0.12  # 双臂每个中间路点的短防抖保持
TISSUE_CLAMP_SETTLE_SEC = 1.2  # 双臂夹紧商品后保持受力 1.2 s 再抬升
TISSUE_PLACE_LOWER_SETTLE_SEC = 0.20  # 商品落桌反馈到位后保持 0.20 s
# 落桌反馈或双臂关节反馈受接触力影响时，不能阻断后续松爪动作。持续下放
# 至少 4 s 后强制进入松爪阶段，期间下放目标与夹紧目标始终保持发布。
TISSUE_PLACE_LOWER_FORCE_RELEASE_SEC = 4.0
# Once the table supports the box, an extra 4 mm total centre separation over
# the active clamp command removes the side preload. Requiring the full
# 0.264 m deploy separation made the contact-loaded arms creep for 45 seconds
# even though the box had already been released on the table.
TISSUE_RELEASE_CLEARANCE_M = 0.004
TISSUE_RELEASE_MIN_SEPARATION_M = (
    2.0 * TISSUE_CLAMP_HALF_SEPARATION_M
    + TISSUE_RELEASE_CLEARANCE_M
)
TISSUE_MOTION_TIMEOUT_SEC = 45.0  # 双臂负载动作的阶段超时上限
# Tissue is held by side friction instead of an enclosing jaw.  Use dedicated
# low-jerk rates for closing, initial lift and loaded orientation changes.
TISSUE_CLAMP_JOINT_SLEW = 0.38
TISSUE_INITIAL_LIFT_JOINT_SLEW = 0.40
TISSUE_LOADED_JOINT_SLEW = 0.40
# 商品已经由桌面承托后，双臂可以更快地向两侧卸力。使用末端实际
# 间距判断商品已经释放，不要求接触状态下的冗余关节完全收敛。
TISSUE_RELEASE_JOINT_SLEW = 0.75
TISSUE_RELEASE_SETTLE_SEC = 0.10  # 双手实际分开到安全间距后再保持 0.10 s
# 纸巾落桌后保持两只夹爪闭合，只用左右臂向外分离卸载；释放条件只检查
# 双手末端间距。若接触或 IK 反馈仍不收敛，最多等 3 s 仍进入竖直抬升，
# 避免反馈门槛阻断必须执行的抬升与回退收尾动作。
TISSUE_RELEASE_FORCE_LIFT_SEC = 3.0
# 抬升阶段继续发布闭爪、分臂和竖直抬升目标；即使反馈门槛没有收敛，3 s
# 后也必须开始直线后退，避免已经落桌的商品把流程卡死在桌边。
TISSUE_RELEASE_LIFT_FORCE_RETREAT_SEC = 3.0
# 商品松爪前尽量由桌面承托。实测下放反馈通常比目标再低约 6 mm，保留
# 8 mm 名义余量后，实际自由落差约 2 mm，既减少掉落冲击又避免穿桌。
TISSUE_TABLE_BOTTOM_CLEARANCE_M = 0.008
# The package is held by side friction at the end of long arms.  Limit the
# loaded Nav2 leg to 0.045 m/s.  With the dedicated 0.35 rad/s steering cap,
# this keeps the commanded turn radius small enough for the chassis to follow
# the right-side corridor path instead of cutting left.
TISSUE_NAV_SPEED_LIMIT_MPS = 0.045
# Nav2's SpeedLimit message constrains translation but this Humble MPPI build
# does not include wz in that limit.  Apply an explicit dynamic angular cap as
# well.  The large shelf-exit reversal is completed separately at the very
# gentle pre-turn rate below.  Once already facing outward, give MPPI enough
# steering authority to follow the global path while its linear speed remains
# limited to 0.045 m/s.  The cabinet-specific shelf-exit pre-turn remains a
# separate fixed-rate action and is intentionally not changed here.
TISSUE_NAV_MAX_ANGULAR_RADPS = 0.35
# A friction-clamped package must not enter ordinary navigation while it
# still needs the initial shelf-exit turn.  The required clear direction is
# cabinet-specific: positive yaw is a left turn and negative yaw is a right
# turn.  Nav2 only takes ownership after this fixed-direction motion has
# reached its requested relative angle and the chassis has stopped completely.
TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF = {
    "A": math.radians(-90.0),
    "B": math.radians(-180.0),
    "C": math.radians(150.0),
    "D": math.radians(90.0),
    "E": math.radians(70.0),
}
TISSUE_PRETURN_MAX_ANGULAR_RADPS = 0.35
TISSUE_PRETURN_YAW_TOL_RAD = math.radians(5.0)
# Simulation contact/load can make the measured yaw rate lower than the fixed
# command, so leave enough deadline margin for the full measured motion.
TISSUE_PRETURN_TIMEOUT_SEC = 60.0  # 双臂负载按来源柜定向、定速预转的总超时
TISSUE_PRETURN_STOP_TIMEOUT_SEC = 5.0  # 预转结束后最多等底盘停稳 5 s
# 必须和 nav2_mppi_controller.yaml 的普通 wz_max 一致；否则双臂负载任务
# 解除 0.35 rad/s 限速后会把控制器错误恢复成旧的过大角速度。
NORMAL_MPPI_MAX_ANGULAR_RADPS = 1.8
# Recovery Spin and BackUp behaviors bypass MPPI's FollowPath velocity cap.
# Use a loaded-clamp BT containing no autonomous recovery motion so every base
# movement while the friction-clamped box is loaded stays under the explicit
# FollowPath angular limit above.
TISSUE_NAV_BEHAVIOR_TREE = str(
    Path(__file__).resolve().parents[2]
    / "config"
    / "navigate_to_pose_tissue_safe.xml"
)
# Avoid unnecessary vertical acceleration while the box is held only by side
# friction.  Raise a low-shelf package just enough to clear the delivery table;
# L2/L3 packages already satisfy this after the normal 5 cm shelf lift.
TISSUE_TRANSPORT_BOTTOM_CLEARANCE_M = 0.080

# Mirror the collision-checked left-arm fold about the base x-z plane.  The
# resulting transport endpoints are [0.136, +/-0.186, 0.444] at slide=0.06:
# both hands are low and close to the chassis for the return through the
# divider.  At the table the right arm first returns through INIT_ARM_R, then
# follows these waypoints outward/down; at shelf E that route is reversed.
ARM_MIRROR_SIGNS = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
RIGHT_ARM_TRANSPORT = LEFT_ARM_TRANSPORT * ARM_MIRROR_SIGNS
RIGHT_ARM_STOW_WAYPOINTS = LEFT_ARM_STOW_WAYPOINTS * ARM_MIRROR_SIGNS
RIGHT_ARM_DOWN_WAYPOINTS = np.vstack(
    [np.asarray(INIT_ARM_R, dtype=float), RIGHT_ARM_STOW_WAYPOINTS]
)
# Leaving one shelf observation point for another is different from leaving
# the delivery table: at a shelf both arms are in their open waiting poses.
# Fold the left arm through the same collision-checked route instead of
# commanding it straight to the low transport endpoint in one interpolation.
LEFT_ARM_DOWN_WAYPOINTS = np.vstack(
    [np.asarray(INIT_ARM_L, dtype=float), LEFT_ARM_STOW_WAYPOINTS]
)
RIGHT_ARM_SCAN_WAYPOINTS = np.vstack(
    [
        RIGHT_ARM_STOW_WAYPOINTS[1],
        RIGHT_ARM_STOW_WAYPOINTS[0],
        np.asarray(INIT_ARM_R, dtype=float),
    ]
)
LEFT_ARM_SCAN_WAYPOINTS = np.vstack(
    [
        LEFT_ARM_STOW_WAYPOINTS[1],
        LEFT_ARM_STOW_WAYPOINTS[0],
        np.asarray(INIT_ARM_L, dtype=float),
    ]
)
SEARCH_ARMS_RESTORE_WAYPOINTS = tuple(
    zip(LEFT_ARM_SCAN_WAYPOINTS, RIGHT_ARM_SCAN_WAYPOINTS)
)

__all__ = [
    'COMMAND_TOPIC',
    'STATUS_TOPIC',
    'INVENTORY_REMOVE_TOPIC',
    'SEARCH_STAGE_SETTLE_SEC',
    'SEARCH_MIN_CLUSTER_SAMPLES',
    'SEARCH_MIN_DETECTION_FRAMES',
    'SEARCH_OBSERVATION_TIMEOUT_SEC',
    'OBSERVATION_HEAD_PITCH',
    'POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M',
    'SHELF_OVERVIEW_CENTER_Z_M',
    'PICK_TRACK_HEAD_PITCH',
    'PICK_TRACK_HEAD_YAW_LIMIT_RAD',
    'PICK_TRACK_HEAD_PITCH_MIN_RAD',
    'PICK_TRACK_HEAD_PITCH_MAX_RAD',
    'HEAD_CAMERA_FIXED_ELEVATION_RAD',
    'HEAD_CAMERA_FORWARD_AT_ZERO_PITCH_M',
    'HEAD_CAMERA_FORWARD_PER_PITCH_M',
    'HEAD_CAMERA_LATERAL_M',
    'HEAD_CAMERA_Z_PLUS_SLIDE_AT_ZERO_PITCH_M',
    'HEAD_CAMERA_Z_PLUS_SLIDE_PER_PITCH_M',
    'PICK_TRACK_MIN_CAMERA_RANGE_M',
    'HEAD_PITCH_TOL_RAD',
    'TARGET_MEMORY_MATCH_RADIUS_M',
    'REACQUIRE_MAX_TARGET_DISTANCE_M',
    'LIVE_TARGET_MAX_AGE_SEC',
    'LIVE_TRACK_MIN_SAMPLES',
    'LIVE_LATERAL_FILTER_ALPHA',
    'LIVE_LATERAL_MAX_STEP_M',
    'LIVE_LATERAL_MAX_SLOT_OFFSET_M',
    'IMAGE_SERVO_MAX_AGE_SEC',
    'IMAGE_SERVO_PIXEL_DEADBAND_PX',
    'PINGGUO_IMAGE_SERVO_PIXEL_DEADBAND_PX',
    'IMAGE_SERVO_FILTER_ALPHA',
    'IMAGE_SERVO_ANGULAR_KP',
    'IMAGE_SERVO_MAX_ANGULAR_CORRECTION_RADPS',
    'IMAGE_SERVO_GRASP_HEADING_KP',
    'IMAGE_SERVO_MAX_HEADING_CORRECTION_RADPS',
    'IMAGE_SERVO_SLOWDOWN_START_PX',
    'IMAGE_SERVO_SLOWDOWN_FULL_PX',
    'IMAGE_SERVO_MIN_LINEAR_SCALE',
    'IMAGE_SERVO_TERMINAL_LOCK_TOL_PX',
    'IMAGE_SERVO_HARD_FREEZE_REMAINING_M',
    'FINE_APPROACH_TIME_ABORTS_ENABLED',
    'FINE_VISION_LOSS_TIMEOUT_SEC',
    'LIVE_TRACK_LOG_INTERVAL_SEC',
    'NAV_NEAR_GOAL_SETTLE_SEC',
    'NAV_NEAR_GOAL_YAW_TOL_RAD',
    'E_SCAN_NEAR_GOAL_POSITION_TOL_M',
    'TABLE_NEAR_GOAL_POSITION_TOL_M',
    'TABLE_NAV_YAW_ASSIST_START_DISTANCE_M',
    'TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS',
    'TABLE_NAV_FAST_ACCEPT_YAW_TOL_RAD',
    'FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS',
    'FINE_APPROACH_TIMEOUT_MARGIN_SEC',
    'FINE_APPROACH_MIN_TIMEOUT_SEC',
    'FINE_APPROACH_MAX_TIMEOUT_SEC',
    'FINE_APPROACH_STALL_TIMEOUT_SEC',
    'FINE_APPROACH_PROGRESS_EPS_M',
    'FINE_APPROACH_TRAVEL_MARGIN_M',
    'FINE_APPROACH_MIN_TRAVEL_LIMIT_M',
    'FINE_APPROACH_ABSOLUTE_MAX_TRAVEL_M',
    'FINE_APPROACH_STOP_TOL_M',
    'FINE_APPROACH_BRAKE_REACTION_SEC',
    'FINE_APPROACH_BRAKE_MARGIN_M',
    'FINE_APPROACH_ALIGNMENT_RESERVE_BASE_M',
    'FINE_APPROACH_ALIGNMENT_RESERVE_GAIN',
    'FINE_APPROACH_ALIGNMENT_RESERVE_MAX_M',
    'PRECISION_ALIGNMENT_RESERVE_BASE_M',
    'PRECISION_ALIGNMENT_RESERVE_GAIN',
    'PRECISION_ALIGNMENT_RESERVE_MAX_M',
    'FINE_APPROACH_ALIGNMENT_RESERVE_RAMP_M',
    'FINE_APPROACH_ALIGNMENT_CRAWL_SPEED_MPS',
    'FINE_APPROACH_DEPTH_OVERSHOOT_LIMIT_M',
    'FINE_APPROACH_NEAR_STOP_TOL_M',
    'FINE_APPROACH_NEAR_SETTLE_SEC',
    'FINE_APPROACH_STALLED_NEAR_STOP_TOL_M',
    'FINE_APPROACH_TRAVEL_BRAKE_MARGIN_M',
    'FINE_APPROACH_MIN_TARGET_FORWARD_M',
    'GRASP_INSERTION_BEYOND_CENTER_M',
    'CYLINDER_DEEP_GRASP_KINDS',
    'CYLINDER_EXTRA_INSERTION_M',
    'KELE_ADDITIONAL_INSERTION_M',
    'MAIDONG_ADDITIONAL_INSERTION_M',
    'BROAD_PACKAGE_EXTRA_INSERTION_M',
    'BROAD_DEEP_GRASP_KINDS',
    'SANMINGZHI_INSERTION_REDUCTION_M',
    'PINGGUO_INSERTION_REDUCTION_M',
    'HEWEIDAO_ADDITIONAL_INSERTION_M',
    'KOUXIANGTANG_EXTRA_INSERTION_M',
    'SHUPIAN_EXTRA_INSERTION_M',
    'TISSUE_EXTRA_INSERTION_M',
    'L1_GRASP_EXTRA_INSERTION_M',
    'L3_GRASP_EXTRA_INSERTION_M',
    'L1_FINE_APPROACH_MIN_TARGET_FORWARD_M',
    'L3_FINE_APPROACH_MIN_TARGET_FORWARD_M',
    'EDGE_FINE_APPROACH_MAX_ANGULAR',
    'EDGE_GRASP_LATERAL_TOL_M',
    'SHUPIAN_GRASP_LATERAL_TOL_M',
    'EDGE_ROW_CYLINDER_SLOT_LATERAL_KINDS',
    'EDGE_ROW_CYLINDER_GRASP_LATERAL_TOL_M',
    'EDGE_ROW_CYLINDER_GRASP_X_OFFSET_M',
    'RIGHT_HAND_GRASP_X_OFFSET_M',
    'SHUPIAN_ADDITIONAL_GRASP_X_OFFSET_M',
    'MAIDONG_ADDITIONAL_GRASP_X_OFFSET_M',
    'PINGGUO_ADDITIONAL_GRASP_X_OFFSET_M',
    'FINE_APPROACH_LATERAL_PROGRESS_EPS_M',
    'DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS',
    'DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS',
    'DIRECT_FINE_APPROACH_TAPER_START_M',
    'DIRECT_FINE_APPROACH_TAPER_END_M',
    'DIRECT_FINE_APPROACH_MAX_ANGULAR',
    'FINE_APPROACH_INITIAL_HEADING_TOL_RAD',
    'SUBSEQUENT_FINE_APPROACH_INITIAL_HEADING_TOL_RAD',
    'RANDOM_PICK_PREDEPLOY_DISTANCE_M',
    'BC_RANDOM_PICK_PREDEPLOY_DISTANCE_M',
    'RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD',
    'RANDOM_ROLLING_HANDOFF_DISTANCE_M',
    'BC_RANDOM_ROLLING_HANDOFF_DISTANCE_M',
    'RANDOM_ROLLING_HANDOFF_YAW_TOL_RAD',
    'STATIONARY_PICK_PREDEPLOY_DISTANCE_M',
    'ADE_NEAR_HANDOFF_SHELVES',
    'ADE_OBSERVATION_HANDOFF_POSITION_TOL_M',
    'ADE_OBSERVATION_HANDOFF_YAW_TOL_RAD',
    'RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS',
    'RANDOM_ROLLING_HANDOFF_EXTRA_TRAVEL_M',
    'FIRST_E_DIRECT_CRUISE_SPEED_MPS',
    'FIRST_PICK_PREDEPLOY_DISTANCE_M',
    'FIRST_PICK_ROLLING_HANDOFF_DISTANCE_M',
    'FIRST_PICK_HANDOFF_EXTRA_TRAVEL_M',
    'FIRST_E_DIRECT_POSITION_KP',
    'FIRST_E_DIRECT_MIN_LINEAR_MPS',
    'FIRST_E_DIRECT_HEADING_KP',
    'FIRST_E_DIRECT_MAX_ANGULAR_RADPS',
    'FIRST_E_DIRECT_POSITION_TOL_M',
    'FIRST_E_DIRECT_YAW_TOL_RAD',
    'PICK_FINE_HANDOFF_MAX_LINEAR_MPS',
    'FINE_APPROACH_NEAR_ANGULAR_ZONE_M',
    'FINE_APPROACH_NEAR_LATERAL_ZONE_M',
    'FINE_APPROACH_MIN_NEAR_ANGULAR_RADPS',
    'FINE_APPROACH_ANGULAR_ACCEL_RADPS2',
    'FINE_APPROACH_MIN_FORWARD_ALIGNMENT_SCALE',
    'FINE_APPROACH_TERMINAL_EE_CONTROL_ZONE_M',
    'FINE_APPROACH_TERMINAL_EE_KP',
    'DIRECT_POSE_GUIDANCE_DISTANCE_KP',
    'DIRECT_POSE_GUIDANCE_ALPHA_KP',
    'DIRECT_POSE_GUIDANCE_BETA_KP',
    'DIRECT_POSE_GUIDANCE_LATERAL_KP',
    'SANMINGZHI_ALIGN_ENTER_TOL_M',
    'SANMINGZHI_ALIGN_EXIT_TOL_M',
    'SANMINGZHI_FINE_APPROACH_SPEED_MPS',
    'SANMINGZHI_ALIGNMENT_STALL_TIMEOUT_SEC',
    'SANMINGZHI_FINE_APPROACH_MAX_ANGULAR',
    'SANMINGZHI_TERMINAL_EE_CONTROL_ZONE_M',
    'SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD',
    'SANMINGZHI_TERMINAL_HEADING_HOLD_KP',
    'SANMINGZHI_TERMINAL_HEADING_MAX_CORRECTION_RADPS',
    'PINGGUO_INSERTED_GRASP_DEPTH_TOL_M',
    'PINGGUO_INSERTED_GRASP_LATERAL_TOL_M',
    'PINGGUO_TERMINAL_EE_CONTROL_ZONE_M',
    'KELE_TERMINAL_EE_CONTROL_ZONE_M',
    'MAIDONG_TERMINAL_EE_CONTROL_ZONE_M',
    'SHUPIAN_TERMINAL_EE_CONTROL_ZONE_M',
    'PRECISION_SLOT_ANCHORED_KINDS',
    'COUPLED_ALIGNMENT_KINDS',
    'COUPLED_ALIGNMENT_FINAL_TOL_M',
    'KOUXIANGTANG_FINE_APPROACH_STALL_TIMEOUT_SEC',
    'KOUXIANGTANG_FINE_APPROACH_MIN_EFFECTIVE_SPEED_MPS',
    'TISSUE_FINE_APPROACH_ALIGN_SPEED_MPS',
    'TISSUE_TERMINAL_EE_CONTROL_ZONE_M',
    'TISSUE_CONTACT_GRASP_DEPTH_TOL_M',
    'TISSUE_CONTACT_GRASP_LATERAL_TOL_M',
    'KOUXIANGTANG_TERMINAL_EE_CONTROL_ZONE_M',
    'PRECISION_GRASP_ALIGNMENT',
    'GRASP_CLOSE_DWELL_SEC',
    'L3_GRASP_CLOSE_DWELL_SEC',
    'CHENGZI_GRIP_CLOSE_COMMAND',
    'CHENGZI_GRASP_CLOSE_DWELL_SEC',
    'CHENGZI_LIFT_JOINT_SLEW',
    'CHENGZI_CARRY_STOW_JOINT_SLEW',
    'CHENGZI_DEPLOY_SLIDE_TOL_M',
    'BROAD_RELEASE_KINDS',
    'BROAD_RELEASE_COMMAND_OPEN_MIN',
    'BROAD_RELEASE_SETTLE_SEC',
    'POST_RELEASE_LIFT_EXEMPT_KINDS',
    'POST_RELEASE_CLEARANCE_LIFT_M',
    'POST_RELEASE_SLIDE_TOL_M',
    'POST_RELEASE_CLEARANCE_SETTLE_SEC',
    'TABLE_STOP_SETTLE_SEC',
    'PLACE_UNFOLD_GENTLE_JOINT_SLEW',
    'PLACE_ADVANCE_GENTLE_JOINT_SLEW',
    'PLACE_LOWER_GENTLE_JOINT_SLEW',
    'PLACE_RELEASE_JOINT_SLEW',
    'PLACE_ADVANCE_TO_LOWER_SETTLE_SEC',
    'CHENGZI_PLACE_UNFOLD_JOINT_SLEW',
    'CHENGZI_PLACE_ADVANCE_JOINT_SLEW',
    'CHENGZI_PLACE_LOWER_JOINT_SLEW',
    'PINGGUO_PLACE_UNFOLD_JOINT_SLEW',
    'PINGGUO_PLACE_ADVANCE_JOINT_SLEW',
    'PINGGUO_PLACE_LOWER_JOINT_SLEW',
    'PLACE_UNFOLD_JOINT_SLEW_OVERRIDES',
    'PLACE_ADVANCE_JOINT_SLEW_OVERRIDES',
    'PLACE_LOWER_JOINT_SLEW_OVERRIDES',
    'CHENGZI_PLACE_LOWER_SETTLE_SEC',
    'CHENGZI_PLACE_RELEASE_JOINT_SLEW',
    'PLACE_RELEASE_JOINT_SLEW_OVERRIDES',
    'CHENGZI_PLACE_EE_Z_TOL_M',
    'CHENGZI_PLACE_FINAL_LOWER_SLEW',
    'CHENGZI_PLACE_SLOW_DISTANCE_M',
    'CHENGZI_PLACE_STOP_SPEED_MPS',
    'CHENGZI_PLACE_COMMAND_TOL_M',
    'CHENGZI_PLACE_CORRECTION_LIMIT_M',
    'CHENGZI_PLACE_LOWER_TIMEOUT_SEC',
    'RETURN_STOW_FAST_JOINT_SLEW',
    'SEARCH_RESTORE_FAST_JOINT_SLEW',
    'PICK_DEPLOY_FAST_JOINT_SLEW',
    'PICK_DEPLOY_BODY_LOWER_JOINT_SLEW',
    'FAST_ARM_STAGE_DWELL_SEC',
    'RETURN_STOW_CONCURRENT_TIMEOUT_SEC',
    'MISSION_TIMER_LINEAR_START_MPS',
    'MISSION_TIMER_ANGULAR_START_RADPS',
    'MISSION_TOTAL_TIME_OUTPUT_PATH',
    'TRANSPORT_SLIDE_M',
    'SLIDE_TOL_M',
    'ARM_SETTLE_SEC',
    'MANIPULATION_TIMEOUT_SEC',
    'PICK_TRANSPORT_TOGETHER_TIMEOUT_SEC',
    'COMPACT_CARRY_SAFE_SETTLE_SEC',
    'COMPACT_CARRY_NON_WRIST_TOL_RAD',
    'COMPACT_CARRY_WRIST_TOL_RAD',
    'COMPACT_CARRY_MAX_TILT_RAD',
    'COMPACT_CARRY_EE_X_RANGE_M',
    'COMPACT_CARRY_EE_Y_RANGE_M',
    'COMPACT_CARRY_EE_Z_RANGE_M',
    'COMPACT_CARRY_LINK_Y_RANGE_M',
    'CONFIRMED_SLOT_DISTANCE_M',
    'GRASP_RETRY_LIMIT',
    'COSTMAP_PARAMETER_TIMEOUT_SEC',
    'STOP_HANDOFF_TIMEOUT_MARGIN_SEC',
    'TABLE_RETREAT_DISTANCE_M',
    'TABLE_RETREAT_TIMEOUT_SEC',
    'TABLE_RETREAT_SPEED_MPS',
    'TISSUE_TABLE_RETREAT_SPEED_MPS',
    'PICK_RETREAT_CLEAR_Y_M',
    'PICK_RETREAT_SPEED_MPS',
    'TABLE_RETURN_FOLD_MIN_Y_M',
    'TISSUE_KIND',
    'TISSUE_HAND_CENTER_X_M',
    'TISSUE_TABLE_BASE_Y_OFFSET_M',
    'TISSUE_TABLE_BASE_INSET_M',
    'TISSUE_TABLE_NEGATIVE_Y_SHIFT_REQUEST_M',
    'TISSUE_TABLE_INSET_M',
    'TISSUE_TABLE_EDGE_MARGIN_M',
    'SINGLE_HAND_TABLE_EDGE_MARGIN_M',
    'TISSUE_PICK_RETREAT_Y_M',
    'TISSUE_DEPLOY_HALF_SEPARATION_M',
    'TISSUE_CLAMP_HALF_SEPARATION_M',
    'TISSUE_RELEASE_HALF_SEPARATION_M',
    'TISSUE_GRASP_YAW_RAD',
    'TISSUE_ARM_TOL_RAD',
    'TISSUE_STAGE_SETTLE_SEC',
    'TISSUE_CLAMP_SETTLE_SEC',
    'TISSUE_PLACE_LOWER_SETTLE_SEC',
    'TISSUE_PLACE_LOWER_FORCE_RELEASE_SEC',
    'TISSUE_RELEASE_CLEARANCE_M',
    'TISSUE_RELEASE_MIN_SEPARATION_M',
    'TISSUE_MOTION_TIMEOUT_SEC',
    'TISSUE_CLAMP_JOINT_SLEW',
    'TISSUE_INITIAL_LIFT_JOINT_SLEW',
    'TISSUE_LOADED_JOINT_SLEW',
    'TISSUE_RELEASE_JOINT_SLEW',
    'TISSUE_RELEASE_SETTLE_SEC',
    'TISSUE_RELEASE_FORCE_LIFT_SEC',
    'TISSUE_RELEASE_LIFT_FORCE_RETREAT_SEC',
    'TISSUE_TABLE_BOTTOM_CLEARANCE_M',
    'TISSUE_NAV_SPEED_LIMIT_MPS',
    'TISSUE_NAV_MAX_ANGULAR_RADPS',
    'TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF',
    'TISSUE_PRETURN_MAX_ANGULAR_RADPS',
    'TISSUE_PRETURN_YAW_TOL_RAD',
    'TISSUE_PRETURN_TIMEOUT_SEC',
    'TISSUE_PRETURN_STOP_TIMEOUT_SEC',
    'NORMAL_MPPI_MAX_ANGULAR_RADPS',
    'TISSUE_NAV_BEHAVIOR_TREE',
    'TISSUE_TRANSPORT_BOTTOM_CLEARANCE_M',
    'ARM_MIRROR_SIGNS',
    'RIGHT_ARM_TRANSPORT',
    'RIGHT_ARM_STOW_WAYPOINTS',
    'RIGHT_ARM_DOWN_WAYPOINTS',
    'LEFT_ARM_DOWN_WAYPOINTS',
    'RIGHT_ARM_SCAN_WAYPOINTS',
    'LEFT_ARM_SCAN_WAYPOINTS',
    'SEARCH_ARMS_RESTORE_WAYPOINTS',
]
