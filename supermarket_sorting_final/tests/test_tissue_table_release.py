"""纸巾落桌分臂释放、抬升与回退衔接的回归测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from supermarket_sorting_nav2 import autonomous_sorting_mission as cycle
from supermarket_sorting_nav2.navigation.mission_manager import MissionStateMachine


class TissueReleaseHarness(cycle.EProductCycleTissueMixin):
    tick = cycle.EProductCycleClient.tick
    _reset_place_settle = cycle.Nav2TaskClient._reset_place_settle
    _place_target_settled = cycle.Nav2TaskClient._place_target_settled

    def __init__(self):
        self.time = 0.0
        self.logger = Mock()
        self.target_kind = "zhijin"
        self.base_xy = np.zeros(2, dtype=float)
        self.base_yaw = 0.0
        self.target_geometry = SimpleNamespace(half_height_m=0.13)
        self.jpos = {
            "left_arm_eef_gripper_joint": 0.40,
            "right_arm_eef_gripper_joint": 0.40,
        }
        self.tc = np.zeros(19, dtype=float)
        self.action = np.zeros(19, dtype=float)
        self.tc[11] = self.tc[18] = cycle.GRIP_CLOSE
        self.action[11] = self.action[18] = cycle.GRIP_CLOSE
        self.joint_slew = 0.0
        self._shutdown_started = False
        self._pick_stow_active = False
        self._return_stow_active = False
        self._pick_deploy_left_restore_active = False
        self._return_nav_footprint_restore_active = False
        self.base_control_enabled = False
        self.manipulation_enabled = False
        self._last_state_log = self.time
        self._place_ready_since = None
        self._release_clearance_slide_m = None
        self._tissue_table_arms = (
            np.full(6, 1.0, dtype=float),
            np.full(6, -1.0, dtype=float),
        )
        self._tissue_release_arms = None
        self._tissue_carry_center_z_m = 0.90
        self._tissue_transport_slide_m = 0.40
        self._table_drop_slide_m = 0.50
        self.slide_meas = 0.40
        self._release_solution = (
            np.full(6, 2.0, dtype=float),
            np.full(6, -2.0, dtype=float),
        )
        self._solve_tissue_arm_targets = Mock(
            return_value=self._release_solution
        )
        self._both_arms_at_target = Mock(return_value=True)
        self._mark_delivery_attempt = Mock()
        self._timed_out = Mock(return_value=False)
        self._publish_outputs = Mock()
        self.set_twist = Mock()
        self._fail = Mock()
        self._measured_separation = (
            cycle.TISSUE_RELEASE_MIN_SEPARATION_M - 0.010
        )
        self.mission = MissionStateMachine(
            self, cycle.CycleState.PLACE_RELEASE
        )

    def now(self):
        return self.time

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self.time * 1e9))
        )

    def get_logger(self):
        return self.logger

    def _tissue_hand_poses(self):
        left = np.eye(4, dtype=float)
        right = np.eye(4, dtype=float)
        left[1, 3] = 0.5 * self._measured_separation
        right[1, 3] = -0.5 * self._measured_separation
        return left, right


class TissueTableReleaseTests(unittest.TestCase):
    def test_closed_grippers_and_outward_arm_motion_start_together(self):
        robot = TissueReleaseHarness()

        robot.tick()

        np.testing.assert_allclose(robot.tc[5:11], robot._release_solution[0])
        np.testing.assert_allclose(robot.tc[12:18], robot._release_solution[1])
        self.assertEqual(robot.tc[11], cycle.GRIP_CLOSE)
        self.assertEqual(robot.tc[18], cycle.GRIP_CLOSE)

    def test_release_readiness_depends_only_on_measured_arm_separation(self):
        robot = TissueReleaseHarness()
        robot._measured_separation = cycle.TISSUE_RELEASE_MIN_SEPARATION_M

        self.assertTrue(
            robot._tissue_release_actuators_ready(
                robot._measured_separation
            )
        )

    def test_release_fallback_enters_mandatory_lift_instead_of_failed(self):
        robot = TissueReleaseHarness()
        robot.tick()  # consume entry and issue simultaneous release targets
        robot.time = cycle.TISSUE_RELEASE_FORCE_LIFT_SEC + 0.01

        robot.tick()

        self.assertEqual(
            robot.mission.state, cycle.CycleState.PLACE_RELEASE_LIFT
        )
        robot._mark_delivery_attempt.assert_called_once_with()
        robot._fail.assert_not_called()

    def test_release_ik_failure_holds_table_pose_and_keeps_grippers_closed(self):
        robot = TissueReleaseHarness()
        robot._solve_tissue_arm_targets.return_value = None

        robot.tick()

        np.testing.assert_allclose(
            robot._tissue_release_arms[0], robot._tissue_table_arms[0]
        )
        np.testing.assert_allclose(
            robot._tissue_release_arms[1], robot._tissue_table_arms[1]
        )
        self.assertEqual(robot.tc[11], cycle.GRIP_CLOSE)
        self.assertEqual(robot.tc[18], cycle.GRIP_CLOSE)
        robot._fail.assert_not_called()

    def test_lower_feedback_failure_still_enters_release(self):
        robot = TissueReleaseHarness()
        robot._both_arms_at_target.return_value = False
        robot.slide_meas = 0.0
        robot.mission = MissionStateMachine(
            robot, cycle.CycleState.PLACE_LOWER
        )
        robot.time = cycle.TISSUE_PLACE_LOWER_FORCE_RELEASE_SEC + 0.01
        robot._last_state_log = robot.time

        robot.tick()

        self.assertEqual(robot.mission.state, cycle.CycleState.PLACE_RELEASE)
        robot._fail.assert_not_called()

    def test_clearance_lift_continues_to_direct_table_retreat(self):
        robot = TissueReleaseHarness()
        robot._tissue_release_arms = robot._release_solution
        robot.mission = MissionStateMachine(
            robot, cycle.CycleState.PLACE_RELEASE_LIFT
        )

        robot.tick()

        self.assertEqual(robot.mission.state, cycle.CycleState.RETREAT_TABLE)
        robot._fail.assert_not_called()

    def test_lift_feedback_failure_still_enters_retreat(self):
        robot = TissueReleaseHarness()
        robot._tissue_release_arms = robot._release_solution
        robot.slide_meas = 0.0
        robot.mission = MissionStateMachine(
            robot, cycle.CycleState.PLACE_RELEASE_LIFT
        )
        robot.time = cycle.TISSUE_RELEASE_LIFT_FORCE_RETREAT_SEC + 0.01
        robot._last_state_log = robot.time

        robot.tick()

        self.assertEqual(robot.mission.state, cycle.CycleState.RETREAT_TABLE)
        robot._fail.assert_not_called()

    def test_retreat_starts_even_if_footprint_service_is_unavailable(self):
        robot = TissueReleaseHarness()
        robot._begin_footprint_change = Mock(return_value=False)
        robot._poll_footprint_change = Mock(return_value=True)
        robot._enable_baseline_base = Mock()
        robot.mission = MissionStateMachine(
            robot, cycle.CycleState.RETREAT_TABLE
        )

        robot.tick()

        self.assertEqual(robot.mission.state, cycle.CycleState.RETREAT_TABLE)
        robot._enable_baseline_base.assert_called_once_with()
        robot.set_twist.assert_called_with(
            -cycle.TISSUE_TABLE_RETREAT_SPEED_MPS, 0.0
        )
        robot._fail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
