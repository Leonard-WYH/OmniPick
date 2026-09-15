"""NavigateToPose 原生目标预占机制的回归测试。"""

from types import SimpleNamespace
import math
import unittest
from unittest.mock import Mock

from geometry_msgs.msg import PoseStamped

from supermarket_sorting_nav2.navigation.nav2_manager import (
    Nav2Manager,
    NavResult,
)


class Nav2GoalPreemptionTests(unittest.TestCase):
    def make_manager(self):
        manager = object.__new__(Nav2Manager)
        manager._node = SimpleNamespace(get_logger=Mock(return_value=Mock()))
        manager._client = Mock()
        manager._client.server_is_ready.return_value = True
        manager._client.send_goal_async.return_value = Mock()
        manager._cmd_vel_pub = Mock()
        manager._goal_debug_pub = Mock()
        manager._goal_handle = Mock()
        manager._send_future = None
        manager._result_future = None
        manager._cancel_future = None
        manager._cancel_requested = False
        manager._result = NavResult.ACTIVE
        manager._goal_name = "old_navigation_goal"
        manager._goal_generation = 1
        manager._make_pose = Mock(return_value=PoseStamped())
        return manager

    def test_active_goal_can_be_replaced_without_cancel(self):
        manager = self.make_manager()

        self.assertTrue(
            manager.go_to_pose(
                -1.85,
                -2.80,
                -math.pi / 2.0,
                name="delivery_table_drop",
                behavior_tree="tissue.xml",
                replace_active=True,
            )
        )

        self.assertEqual(manager.result, NavResult.STARTING)
        self.assertEqual(manager._goal_generation, 2)
        manager._goal_handle.cancel_goal_async.assert_not_called()
        manager._client.send_goal_async.assert_called_once()

    def test_old_result_cannot_overwrite_replacement(self):
        manager = self.make_manager()
        manager._goal_generation = 2
        manager._result = NavResult.STARTING
        stale_future = Mock()

        manager._result_cb(
            stale_future,
            generation=1,
            goal_name="old_navigation_goal",
        )

        stale_future.result.assert_not_called()
        self.assertEqual(manager.result, NavResult.STARTING)

if __name__ == "__main__":
    unittest.main()
