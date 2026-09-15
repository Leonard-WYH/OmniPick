"""载物轮廓阻塞时的桌面导航降级重试回归测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from supermarket_sorting_nav2 import autonomous_sorting_mission as cycle
from supermarket_sorting_nav2.navigation.nav2_manager import NavResult


class DeliveryNavigationRetryTests(unittest.TestCase):
    def make_robot(self, retry_used=False):
        nav2 = SimpleNamespace(
            wait_result=Mock(return_value=NavResult.FAILED),
            stop_robot=Mock(),
        )
        mission = SimpleNamespace(transition=Mock())
        robot = SimpleNamespace(
            nav2=nav2,
            mission=mission,
            _footprint_mode="carry",
            _delivery_nav_chassis_retry_used=retry_used,
            _nav_near_goal_since=None,
            _nav_terminal_hold_commanded=False,
            _navigation_pose_error=Mock(return_value=(1.2, 0.1)),
            _odom_stopped=Mock(return_value=True),
            get_logger=Mock(
                return_value=SimpleNamespace(
                    info=Mock(), warning=Mock(), error=Mock()
                )
            ),
            _fail=Mock(),
        )
        return robot

    def test_failed_carry_navigation_retries_once_with_chassis_footprint(self):
        robot = self.make_robot()

        cycle.EProductCycleClient._wait_navigation(
            robot,
            cycle.CycleState.STOP_TABLE,
            "delivery_table_drop",
            (-1.85, -2.8, -1.57),
            0.12,
        )

        self.assertTrue(robot._delivery_nav_chassis_retry_used)
        robot.nav2.stop_robot.assert_called_once_with()
        robot.mission.transition.assert_called_once_with(
            cycle.CycleState.RETRY_TABLE_CHASSIS_FOOTPRINT,
            "retry_delivery_with_chassis_footprint",
        )
        robot._fail.assert_not_called()

    def test_second_failure_is_terminal(self):
        robot = self.make_robot(retry_used=True)

        cycle.EProductCycleClient._wait_navigation(
            robot,
            cycle.CycleState.STOP_TABLE,
            "delivery_table_drop",
            (-1.85, -2.8, -1.57),
            0.12,
        )

        robot._fail.assert_called_once_with("delivery_table_drop:FAILED")
        robot.mission.transition.assert_not_called()


class DeliveryTerminalYawAssistTests(unittest.TestCase):
    def test_far_from_table_leaves_complete_twist_to_nav2(self):
        self.assertIsNone(
            cycle._table_nav_angular_override(0.41, 0.5)
        )
        self.assertAlmostEqual(
            cycle.TABLE_NAV_YAW_ASSIST_START_DISTANCE_M, 0.40
        )

    def test_terminal_turn_uses_one_fixed_magnitude(self):
        positive = cycle._table_nav_angular_override(0.40, 0.5)
        negative = cycle._table_nav_angular_override(0.40, -0.5)
        self.assertAlmostEqual(
            positive, cycle.TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS
        )
        self.assertAlmostEqual(
            negative, -cycle.TABLE_NAV_FIXED_ANGULAR_SPEED_RADPS
        )

    def test_aligned_terminal_zone_holds_yaw_and_fast_accepts(self):
        self.assertAlmostEqual(
            cycle.TABLE_NEAR_GOAL_POSITION_TOL_M, 0.20
        )
        self.assertEqual(
            cycle._table_nav_angular_override(0.10, 0.05), 0.0
        )
        self.assertTrue(
            cycle._table_nav_pose_fast_acceptable(0.10, 0.05)
        )
        self.assertTrue(
            cycle._table_nav_pose_fast_acceptable(0.13, 0.05)
        )
        self.assertTrue(
            cycle._table_nav_pose_fast_acceptable(0.16, 0.05)
        )
        self.assertFalse(
            cycle._table_nav_pose_fast_acceptable(0.21, 0.05)
        )
        self.assertFalse(
            cycle._table_nav_pose_fast_acceptable(0.10, 0.11)
        )


if __name__ == "__main__":
    unittest.main()
