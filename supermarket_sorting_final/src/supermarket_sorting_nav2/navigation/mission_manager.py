"""Mission states and validated waypoint loading.

The task client owns the manipulation details.  This module deliberately keeps
the high-level mission states and all field-measured poses independent from the
verified Baseline implementation.

中文说明：集中定义任务状态、航点格式和状态进入时间。``elapsed`` 只是读取
当前状态已经持续多久，供上层非阻塞状态机判断，不会在本模块产生等待。
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
from pathlib import Path
from typing import Dict

import yaml


class MissionState(Enum):
    WAIT_TASK = auto()
    NAV_TO_PICK = auto()
    WAIT_PICK_NAV_DONE = auto()
    STOP_NAV = auto()
    PERCEPTION = auto()
    FINE_APPROACH = auto()
    GRASP = auto()
    CHECK_GRASP = auto()
    FINE_RETREAT = auto()
    STOP_BASE_DRIVE = auto()
    STOW_LEFT_ARM = auto()
    NAV_TO_LEFT = auto()
    WAIT_LEFT_NAV_DONE = auto()
    ARRIVED = auto()
    PLACE_ADVANCE = auto()
    PLACE_LOWER = auto()
    PLACE_RELEASE = auto()
    PLACE_RETRACT = auto()
    PLACE_LIFT_CLEAR = auto()
    PLACE_MOVE_ASIDE = auto()
    RESTORE_RIGHT_ARM_DEPLOYED = auto()
    DONE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class Waypoint:
    name: str
    frame_id: str
    x: float
    y: float
    yaw: float
    needs_tuning: bool = False


def load_waypoints(path: str, require_tuned: bool = True) -> Dict[str, Waypoint]:
    """Load and validate the two odom-frame mission waypoints.

    Placeholder coordinates are legal YAML, but are rejected before the ROS
    node starts when ``needs_tuning`` is true.  This prevents an accidental
    navigation command to a guessed pose.
    """

    waypoint_path = Path(path).expanduser().resolve()
    with waypoint_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    if not isinstance(raw, dict):
        raise ValueError(f"waypoint file must contain a mapping: {waypoint_path}")

    result: Dict[str, Waypoint] = {}
    for name in ("pick_approach", "left_goal"):
        data = raw.get(name)
        if not isinstance(data, dict):
            raise ValueError(f"missing waypoint mapping: {name}")
        missing = {key for key in ("frame_id", "x", "y", "yaw") if key not in data}
        if missing:
            raise ValueError(f"waypoint {name} is missing: {', '.join(sorted(missing))}")
        if data["frame_id"] != "odom":
            raise ValueError(f"waypoint {name} must use frame_id=odom")
        needs_tuning = data.get("needs_tuning", False)
        if not isinstance(needs_tuning, bool):
            raise ValueError(f"waypoint {name} needs_tuning must be true or false")
        try:
            waypoint = Waypoint(
                name=name,
                frame_id="odom",
                x=float(data["x"]),
                y=float(data["y"]),
                yaw=float(data["yaw"]),
                needs_tuning=needs_tuning,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"waypoint {name} has a non-numeric pose") from exc
        if not all(math.isfinite(value) for value in (waypoint.x, waypoint.y, waypoint.yaw)):
            raise ValueError(f"waypoint {name} pose must contain only finite numbers")
        if require_tuned and waypoint.needs_tuning:
            raise ValueError(
                f"waypoint {name} is marked NEEDS_TUNING in {waypoint_path}; "
                "measure it, update the numbers, then set needs_tuning: false"
            )
        result[name] = waypoint
    return result


class MissionStateMachine:
    """Small transition helper with entry detection and consistent logging."""

    def __init__(self, node, initial_state: MissionState):
        self._node = node
        self.state = initial_state
        self.entered_at = self._now()
        self._entry_pending = True
        self._node.get_logger().info(f"MISSION_STATE {self.state.name}")

    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    @property
    def elapsed(self) -> float:
        return self._now() - self.entered_at

    def consume_entry(self) -> bool:
        if not self._entry_pending:
            return False
        self._entry_pending = False
        return True

    def transition(self, new_state: MissionState, reason: str = "") -> None:
        if new_state == self.state:
            return
        suffix = f" reason={reason}" if reason else ""
        self._node.get_logger().info(
            f"MISSION_TRANSITION {self.state.name}->{new_state.name}{suffix}"
        )
        self.state = new_state
        self.entered_at = self._now()
        self._entry_pending = True
