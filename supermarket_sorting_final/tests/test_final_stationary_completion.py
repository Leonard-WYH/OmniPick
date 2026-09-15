"""最后一件商品在桌边退出后原地结束的回归测试。"""

from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from supermarket_sorting_nav2 import autonomous_sorting_mission as cycle
from supermarket_sorting_nav2.navigation.mission_manager import MissionStateMachine


class FinalCompletionHarness:
    tick = cycle.EProductCycleClient.tick
    _pending_delivery_is_final = (
        cycle.EProductCycleClient._pending_delivery_is_final
    )
    _freeze_mission_total_time = (
        cycle.EProductCycleClient._freeze_mission_total_time
    )
    _confirm_final_delivery_at_table = (
        cycle.EProductCycleClient._confirm_final_delivery_at_table
    )
    _start_navigation_after_table_retreat = (
        cycle.EProductCycleClient._start_navigation_after_table_retreat
    )

    def __init__(self, state):
        self.time = 50.0
        self.logger = Mock()
        self._shutdown_started = False
        self.base_xy = np.zeros(2)
        self.jpos = {}
        self._pick_stow_active = False
        self._return_stow_active = False
        self._return_arms_stowed = False
        self._return_nav_footprint_restore_active = False
        self._table_exit_reached = True
        self._table_retreat_start_xy = None
        self._table_retreat_heading = 0.0
        self._uses_tissue_two_hand_grasp = False
        self._tissue_release_arms = None
        self.target_kind = "kele"
        self.base_yaw = 0.0
        self.base_control_enabled = False
        self.manipulation_enabled = False
        self.cur_lin = 0.0
        self.cur_ang = 0.0
        self.des_lin = 0.0
        self.des_ang = 0.0
        self.tc = np.zeros(19, dtype=float)
        self._mission_motion_started_at = 10.0
        self._mission_motion_finished_at = None
        self._mission_total_time_sec = None
        self._mission_target_count = 5
        self._picked_count = 4
        self._target_queue = deque([41])
        self._known_targets = {41: {"kind": "zhijin"}}
        self._pending_verification = {
            "aruco_id": 41,
            "grasp_attempt": 1,
        }
        self._event_prefix = "E_ZHIJIN"
        self._last_state_log = float("inf")
        self.set_twist = Mock()
        self._hold_direct_retreat_release_pose = Mock()
        self._enable_baseline_base = Mock()
        self._odom_stopped = Mock(return_value=True)
        self._poll_footprint_change = Mock(return_value=True)
        self._release_baseline_base = Mock()
        self._begin_footprint_change = Mock(return_value=True)
        self._start_concurrent_return_stow = Mock()
        self._send_goal = Mock()
        self._publish_outputs = Mock()
        self._timed_out = Mock()
        self._fail = Mock()
        self.nav2 = SimpleNamespace(cancel=Mock(), stop_robot=Mock())
        self.mission = MissionStateMachine(self, state)

    def now(self):
        return self.time

    def get_clock(self):
        return SimpleNamespace(
            now=lambda: SimpleNamespace(nanoseconds=int(self.time * 1e9))
        )

    def get_logger(self):
        return self.logger


class FinalStationaryCompletionTests(unittest.TestCase):
    def test_last_retreat_stops_timer_and_does_not_send_return_goal(self):
        robot = FinalCompletionHarness(cycle.CycleState.STOP_TABLE_RETREAT)
        output_file = Mock()

        with patch("builtins.print") as terminal_print:
            with patch.object(
                cycle, "MISSION_TOTAL_TIME_OUTPUT_PATH", output_file
            ):
                robot.tick()

        self.assertEqual(
            robot.mission.state,
            cycle.CycleState.FINAL_STATIONARY_RESTORE,
        )
        self.assertEqual(robot._mission_total_time_sec, 40.0)
        output_file.write_text.assert_called_once_with(
            "[任务计时完成] MISSION_TOTAL_TIME "
            "seconds=40.000 duration=00:40.000\n",
            encoding="utf-8",
        )
        terminal_print.assert_not_called()
        robot.logger.warning.assert_not_called()
        robot._send_goal.assert_not_called()
        robot._start_concurrent_return_stow.assert_called_once_with(
            navigation_active=False
        )

    def test_timer_waits_until_table_exit_position_is_reached(self):
        robot = FinalCompletionHarness(cycle.CycleState.STOP_TABLE_RETREAT)
        robot._table_exit_reached = False
        output_file = Mock()

        with patch.object(
            cycle, "MISSION_TOTAL_TIME_OUTPUT_PATH", output_file
        ):
            robot.tick()

        self.assertEqual(
            robot.mission.state,
            cycle.CycleState.STOP_TABLE_RETREAT,
        )
        self.assertIsNone(robot._mission_motion_finished_at)
        self.assertIsNone(robot._mission_total_time_sec)
        output_file.write_text.assert_not_called()
        robot._start_concurrent_return_stow.assert_not_called()

    def test_stationary_restore_completes_last_delivery(self):
        robot = FinalCompletionHarness(
            cycle.CycleState.FINAL_STATIONARY_RESTORE
        )
        robot._return_arms_stowed = True

        robot.tick()

        self.assertEqual(robot.mission.state, cycle.CycleState.DONE)
        self.assertEqual(robot._picked_count, 5)
        self.assertFalse(robot._target_queue)
        self.assertIsNone(robot._pending_verification)
        robot._send_goal.assert_not_called()

    def test_nonfinal_delivery_keeps_return_navigation(self):
        robot = FinalCompletionHarness(cycle.CycleState.STOP_TABLE_RETREAT)
        robot._mission_target_count = 5
        robot._picked_count = 2
        robot._target_queue = deque([41, 42, 43])
        robot._known_targets.update(
            {42: {"kind": "kele"}, 43: {"kind": "maidong"}}
        )

        def send_goal(_pose, _name, wait_state):
            robot.mission.transition(wait_state)

        robot._send_goal.side_effect = send_goal

        robot.tick()

        robot._send_goal.assert_called_once()
        self.assertEqual(
            robot.mission.state, cycle.CycleState.WAIT_E_SCAN_NAV
        )
        self.assertIsNone(robot._mission_motion_finished_at)

    def test_nonfinal_tissue_reasserts_normal_nav_limits_before_next_goal(self):
        robot = FinalCompletionHarness(cycle.CycleState.STOP_TABLE_RETREAT)
        robot._mission_target_count = 5
        robot._picked_count = 2
        robot._target_queue = deque([41, 42, 43])
        robot._known_targets.update(
            {42: {"kind": "kele"}, 43: {"kind": "maidong"}}
        )
        robot._uses_tissue_two_hand_grasp = True
        robot.target_kind = "zhijin"
        robot._pending_verification["kind"] = "zhijin"
        robot._controller_angular_limit_current = (
            cycle.NORMAL_MPPI_MAX_ANGULAR_RADPS
        )
        robot._set_navigation_speed_limit = Mock()
        robot._poll_controller_angular_limit = Mock(return_value=True)
        robot._begin_controller_angular_limit = Mock(return_value=True)

        def send_goal(_pose, _name, wait_state):
            robot.mission.transition(wait_state)

        robot._send_goal.side_effect = send_goal

        robot.tick()

        robot._set_navigation_speed_limit.assert_called_once_with(0.0)
        robot._begin_controller_angular_limit.assert_not_called()
        robot._send_goal.assert_called_once()
        self.assertEqual(
            robot.mission.state, cycle.CycleState.WAIT_E_SCAN_NAV
        )

    def test_nonfinal_retreat_hands_to_nav2_without_stop_state(self):
        robot = FinalCompletionHarness(cycle.CycleState.RETREAT_TABLE)
        robot._mission_target_count = 5
        robot._picked_count = 2
        robot._target_queue = deque([41, 42, 43])
        robot._known_targets.update(
            {42: {"kind": "kele"}, 43: {"kind": "maidong"}}
        )
        robot._table_exit_reached = False
        robot._table_retreat_start_xy = np.zeros(2, dtype=float)
        robot.base_xy = np.array(
            [cycle.TABLE_RETREAT_DISTANCE_M, 0.0], dtype=float
        )
        robot.mission.consume_entry()

        def send_goal(_pose, _name, wait_state):
            robot.mission.transition(wait_state)

        robot._send_goal.side_effect = send_goal

        robot.tick()

        self.assertEqual(
            robot.mission.state, cycle.CycleState.WAIT_E_SCAN_NAV
        )
        self.assertTrue(robot._table_exit_reached)
        self.assertEqual(robot.cur_lin, 0.0)
        self.assertEqual(robot.tc[0], 0.0)
        robot._release_baseline_base.assert_called_once_with()
        robot._send_goal.assert_called_once()

    def test_table_retreat_starts_at_fixed_direct_speed_without_ramp(self):
        robot = FinalCompletionHarness(cycle.CycleState.RETREAT_TABLE)
        robot._table_exit_reached = False
        robot.base_xy = np.zeros(2, dtype=float)

        robot.tick()

        self.assertEqual(robot.mission.state, cycle.CycleState.RETREAT_TABLE)
        self.assertEqual(robot.cur_lin, -cycle.TABLE_RETREAT_SPEED_MPS)
        self.assertEqual(robot.tc[0], -cycle.TABLE_RETREAT_SPEED_MPS)
        robot.set_twist.assert_called_with(
            -cycle.TABLE_RETREAT_SPEED_MPS, 0.0
        )

    def test_compact_arms_restore_normal_footprint_during_navigation(self):
        robot = SimpleNamespace(
            _return_stow_active=True,
            _return_stow_both_arms=False,
            _return_stow_navigation_active=True,
            _return_stow_started_at=0.0,
            _return_arm_stage=len(cycle.RIGHT_ARM_DOWN_WAYPOINTS) - 1,
            _return_arm_stage_ready_since=0.0,
            _return_arms_stowed=False,
            _return_nav_footprint_restore_active=False,
            _footprint_mode="carry",
            larm_meas=np.asarray(cycle.LEFT_ARM_TRANSPORT, dtype=float),
            slide_meas=cycle.TRANSPORT_SLIDE_M,
            tc=np.zeros(19, dtype=float),
            now=Mock(return_value=10.0),
            _right_arm_at_target=Mock(return_value=True),
            _begin_footprint_change=Mock(return_value=True),
            _poll_footprint_change=Mock(return_value=True),
            _fail=Mock(),
            get_logger=Mock(
                return_value=SimpleNamespace(info=Mock(), warning=Mock())
            ),
        )

        cycle.EProductCycleClient._update_concurrent_return_stow(robot)

        self.assertTrue(robot._return_arms_stowed)
        self.assertFalse(robot._return_stow_active)
        self.assertTrue(robot._return_nav_footprint_restore_active)
        robot._begin_footprint_change.assert_called_once_with("normal")

        cycle.EProductCycleClient._update_return_navigation_footprint(robot)

        self.assertFalse(robot._return_nav_footprint_restore_active)
        robot._fail.assert_not_called()


if __name__ == "__main__":
    unittest.main()
