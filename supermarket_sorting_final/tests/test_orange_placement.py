"""Orange placement regressions. Run in the ROS client image; no ROS node,
simulator connection or actuator command is created by these tests.

中文说明：用可控的 0.02 s 虚拟时钟验证橙子下降与松爪门槛，不会连接仿真器，
也不会真的等待墙钟时间或发送机器人指令。

source /opt/ros/humble/setup.bash
PYTHONPATH=src:$PYTHONPATH python3 -m unittest discover -s tests -v
"""

from collections import deque
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from supermarket_sorting_nav2 import autonomous_sorting_mission as cycle
from supermarket_sorting_nav2.baseline_grasp_controller import TaskOneClient
from supermarket_sorting_nav2.kinematics.mmk2_kdl import MMK2Kdl
from supermarket_sorting_nav2.navigation.mission_manager import MissionStateMachine


class PlacementHarness:
    """Execute production lowering/release states with deterministic feedback."""

    _capture_chengzi_grasp_geometry = cycle.EProductCycleClient._capture_chengzi_grasp_geometry
    _chengzi_estimated_bottom_z = cycle.EProductCycleClient._chengzi_estimated_bottom_z
    _calibrate_chengzi_table_drop_slide = cycle.EProductCycleClient._calibrate_chengzi_table_drop_slide
    _update_chengzi_table_lower = cycle.EProductCycleClient._update_chengzi_table_lower
    _log_chengzi_release_height = cycle.EProductCycleClient._log_chengzi_release_height
    _released_product_kind = cycle.EProductCycleClient._released_product_kind
    _post_release_lift_required = cycle.EProductCycleClient._post_release_lift_required
    _hold_direct_retreat_release_pose = cycle.EProductCycleClient._hold_direct_retreat_release_pose
    _timed_out = cycle.EProductCycleClient._timed_out
    _reset_place_settle = cycle.Nav2TaskClient._reset_place_settle
    _place_target_settled = cycle.Nav2TaskClient._place_target_settled
    smooth_step = TaskOneClient.smooth_step
    tick = cycle.EProductCycleClient.tick

    def __init__(self):
        self.time = 0.0
        self.dt = 0.02  # 测试状态机的虚拟步长；循环只推进变量，不会 sleep
        self.logger = Mock()
        self.target_kind = "chengzi"
        self.target_geometry = cycle.bottle_geometry("chengzi")
        self.tc = np.zeros(19)
        self.tc[18] = cycle.CHENGZI_GRIP_CLOSE_COMMAND
        self.action = self.tc.copy()
        self.tc_prev = self.tc.copy()
        self.joint_move_ratio = np.zeros(19)
        self.joint_slew = cycle.CHENGZI_PLACE_LOWER_JOINT_SLEW
        self.slide_meas = 0.006  # measured load bias seen in the real logs
        self.jpos = {}
        self.jvel = {"slide_joint": 0.0}
        self.pose = np.eye(4)
        self.pose[2, 3] = 1.10
        self._chengzi_grasp_center_in_hand = np.array([-0.035, 0., -0.005])
        self._chengzi_place_initial_slide = None
        self._chengzi_place_heights = deque(maxlen=30)
        self._chengzi_place_feedback = {}
        self._place_advanced_arm = np.zeros(6)
        self._joint_feedback_received_at = self.time
        self._place_ready_since = None
        self._release_clearance_slide_m = None
        self._right_arm_at_target = Mock(return_value=True)
        self._right_gripper_is_open = True  # sphere props jaw open while held
        self._uses_tissue_two_hand_grasp = False
        self.base_xy = np.zeros(2)
        self._pick_stow_active = self._return_stow_active = False
        self.base_control_enabled = False
        self.manipulation_enabled = True
        self._publish_outputs = Mock()
        self._last_state_log = float("inf")
        self._mark_delivery_attempt = Mock()
        self.failure = None
        self.mission = MissionStateMachine(self, cycle.CycleState.PLACE_LOWER)

    def now(self):
        return self.time

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.time * 1e9)))

    def get_logger(self):
        return self.logger

    def ee_footprint_pose(self):
        return self.pose.copy()

    def world_to_footprint(self, point):
        return np.asarray(point)

    def _fail(self, reason):
        self.failure = reason
        self.mission.transition(cycle.CycleState.FAILED, reason)

    def feedback_at_gap(self, gap, duration=0.5, fresh=True):
        """用默认 0.5 s 虚拟反馈窗口检查下降门槛，不产生真实延时。"""
        self.pose[2, 3] = (
            cycle.TABLE_TOP_Z_M + self.target_geometry.half_height_m
            + gap - self._chengzi_grasp_center_in_hand[2]
        )
        ready = False
        for _ in range(round(duration / self.dt)):
            self.time += self.dt
            if fresh:
                self._joint_feedback_received_at = self.time
            ready = self._update_chengzi_table_lower()
        return ready


class OrangePlacementTests(unittest.TestCase):
    def setUp(self):
        self.robot = PlacementHarness()
        self.assertTrue(self.robot._calibrate_chengzi_table_drop_slide())

    def test_old_six_centimetre_air_release_is_not_ready(self):
        r = self.robot
        r.action[2] = r._table_drop_slide_m
        self.assertFalse(r.feedback_at_gap(0.060))

    def test_not_ready_while_command_still_descending(self):
        r = self.robot
        r.action[2] = r._table_drop_slide_m - 0.014  # final log before release
        self.assertFalse(r.feedback_at_gap(cycle.CHENGZI_TABLE_RELEASE_CLEARANCE_M))

    def test_both_sides_of_height_gate(self):
        r = self.robot
        for gap in (-0.010, 0.020):
            r.action[2] = r._table_drop_slide_m
            self.assertFalse(r.feedback_at_gap(gap))

    def test_velocity_and_fresh_feedback_required(self):
        r = self.robot
        r.action[2] = r._table_drop_slide_m
        r.jvel["slide_joint"] = 0.025
        self.assertFalse(r.feedback_at_gap(0.003))
        r.jvel["slide_joint"] = 0.0
        self.assertFalse(r.feedback_at_gap(0.003, fresh=False))
        self.assertEqual(r.joint_slew, 0.0)
        self.assertTrue(r.feedback_at_gap(0.003))

    def test_arm_motion_blocks_release(self):
        r = self.robot
        r.action[2] = r._table_drop_slide_m
        r._right_arm_at_target.return_value = False
        self.assertFalse(r.feedback_at_gap(0.003))

    def test_calibration_uses_command_and_measured_pose_not_nominal_fk(self):
        r = self.robot
        bottom = r._chengzi_estimated_bottom_z()
        expected = r.action[2] + bottom - cycle.TABLE_TOP_Z_M - 0.003
        self.assertAlmostEqual(r._table_drop_slide_m, expected)
        self.assertNotAlmostEqual(r._table_drop_slide_m, expected + r.slide_meas)

    def test_grasp_offset_follows_wrist_rotation(self):
        r = self.robot
        r.OBJECT_WORLD = r.pose[:3, 3] + np.array([-0.035, 0.002, -0.001])
        self.assertTrue(r._capture_chengzi_grasp_geometry())
        r.pose[:3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
        self.assertAlmostEqual(
            r._chengzi_estimated_bottom_z(),
            r.pose[2, 3] + 0.035 - r.target_geometry.half_height_m,
        )

    def test_actual_arm_fk_at_all_three_shelf_levels(self):
        kdl = MMK2Kdl()
        for level in ("L1", "L2", "L3"):
            with self.subTest(level=level):
                r = PlacementHarness()
                centre_z = cycle.e_product_center_z(level, "chengzi")
                # Include the measured 6 mm torso bias at pickup and delivery.
                pick_slide = cycle.grasp_slide_for_product_z(centre_z, "chengzi")
                _, r.pose = kdl.forward_kinematics(
                    np.concatenate([[pick_slide + .006], cycle.FIXED_GRASP_ARM]),
                    index="right",
                )
                r.OBJECT_WORLD = r.pose[:3, 3] + [-0.035, 0., 0.]
                r.OBJECT_WORLD[2] = centre_z
                self.assertTrue(r._capture_chengzi_grasp_geometry())
                r._place_advanced_arm = cycle.FIXED_CARRY_ARM
                _, r.pose = kdl.forward_kinematics(
                    np.concatenate([[.006], cycle.FIXED_CARRY_ARM]), index="right",
                )
                self.assertTrue(r._calibrate_chengzi_table_drop_slide())
                _, r.pose = kdl.forward_kinematics(
                    np.concatenate([[r._table_drop_slide_m + .006], cycle.FIXED_CARRY_ARM]),
                    index="right",
                )
                self.assertAlmostEqual(
                    r._chengzi_estimated_bottom_z() - cycle.TABLE_TOP_Z_M,
                    cycle.CHENGZI_TABLE_RELEASE_CLEARANCE_M, places=5,
                )

    def test_edge_shelf_tissue_grasps_are_raised(self):
        """L1/L3 paper packs clear their boards while L2 stays centred."""

        slides = {
            level: cycle.grasp_slide_for_product_z(
                cycle.e_product_center_z(level, "zhijin"),
                "zhijin",
                level,
            )
            for level in ("L1", "L2", "L3")
        }
        legacy_l3_slide = cycle.grasp_slide_for_product_z(
            cycle.e_product_center_z("L3", "zhijin"), "zhijin"
        )

        self.assertAlmostEqual(
            legacy_l3_slide - slides["L3"],
            cycle.TISSUE_L3_GRASP_ABOVE_CENTER_M,
        )
        self.assertAlmostEqual(
            cycle.grasp_height_offset_for_product("zhijin", "L1"),
            cycle.TISSUE_L1_GRASP_ABOVE_CENTER_M,
        )
        self.assertAlmostEqual(
            cycle.grasp_height_offset_for_product("zhijin", "L2"), 0.0
        )
        self.assertAlmostEqual(
            cycle.grasp_height_offset_for_product("zhijin", "L3"),
            cycle.TISSUE_L3_GRASP_ABOVE_CENTER_M,
        )
        self.assertAlmostEqual(
            slides["L2"] - slides["L1"],
            (
                cycle.e_product_center_z("L1", "zhijin")
                - cycle.e_product_center_z("L2", "zhijin")
                + cycle.TISSUE_L1_GRASP_ABOVE_CENTER_M
            ),
        )

    def test_missing_or_invalid_grasp_geometry_cannot_calibrate(self):
        r = self.robot
        r._chengzi_grasp_center_in_hand = None
        self.assertFalse(r._calibrate_chengzi_table_drop_slide())
        r.OBJECT_WORLD = np.full(3, np.nan)
        self.assertFalse(r._capture_chengzi_grasp_geometry())

    def test_near_table_lowering_is_slow(self):
        r = self.robot
        r.feedback_at_gap(0.2)
        far = r.joint_slew
        r.feedback_at_gap(0.015)
        self.assertLess(r.joint_slew, far)
        self.assertLessEqual(0.3 * r.joint_slew, 0.012 + 1e-9)

    def test_full_lower_release_lift_sequence_with_servo_bias(self):
        r = self.robot
        start_slide, start_z = r.slide_meas, r.pose[2, 3]
        release_at = None
        for _ in range(2000):
            old_slide = r.slide_meas
            # Simple lagged actuator, not a contact/dynamics simulation.
            r.slide_meas += (r.action[2] + 0.006 - r.slide_meas) * 0.2
            r.jvel["slide_joint"] = (r.slide_meas - old_slide) / r.dt
            r.pose[2, 3] = start_z - (r.slide_meas - start_slide)
            r.time += r.dt
            r._joint_feedback_received_at = r.time
            r.tick()
            if r.mission.state == cycle.CycleState.FAILED:
                self.fail(r.failure)
            if r.tc[18] == cycle.GRIP_OPEN and release_at is None:
                release_at = r.time
                self.assertLessEqual(abs(r.jvel["slide_joint"]), cycle.CHENGZI_PLACE_STOP_SPEED_MPS)
                gap = r._chengzi_estimated_bottom_z() - cycle.TABLE_TOP_Z_M
                self.assertGreaterEqual(gap, 0.)
                self.assertLessEqual(gap, 0.006)
                self.assertTrue(r._chengzi_place_feedback["command_done"])
            if r.mission.state == cycle.CycleState.PLACE_RELEASE_LIFT:
                break
        self.assertIsNotNone(release_at)
        self.assertEqual(r.mission.state, cycle.CycleState.PLACE_RELEASE_LIFT)
        # Faster opening removes roughly half a second after the fruit is
        # already safely at table height, while keeping the release settle.
        self.assertGreater(
            r.time - release_at, cycle.PLACE_RELEASE_SETTLE_SEC + 0.45
        )
        self.assertLess(
            r.time - release_at, cycle.PLACE_RELEASE_SETTLE_SEC + 0.65
        )
        r._mark_delivery_attempt.assert_called_once()

    def test_blocked_lowering_times_out_without_opening(self):
        r = self.robot
        for _ in range(1600):
            r.time += r.dt
            r._joint_feedback_received_at = r.time
            r.tick()  # fixed measured pose: actuator cannot lower
            if r.mission.state == cycle.CycleState.FAILED:
                break
        self.assertEqual(r.failure, "place_lower_timeout")
        self.assertEqual(r.tc[18], cycle.CHENGZI_GRIP_CLOSE_COMMAND)
        self.assertEqual(r.tc[2], r.action[2])
        r._mark_delivery_attempt.assert_not_called()

    def test_nonfinite_height_pauses_without_nan_commands(self):
        r = self.robot
        r.pose[2, 3] = np.nan
        self.assertFalse(r._update_chengzi_table_lower())
        self.assertEqual(r.joint_slew, 0.)
        self.assertTrue(np.isfinite(r.tc).all())

    def test_release_entry_rechecks_height_before_open_command(self):
        r = self.robot
        r.action[2] = r._table_drop_slide_m
        self.assertTrue(r.feedback_at_gap(0.003))
        r.mission.transition(cycle.CycleState.PLACE_RELEASE)
        r.pose[2, 3] += 0.030  # feedback changed after previous state's decision
        r.tick()
        self.assertEqual(r.failure, "chengzi_release_height_guard")
        self.assertEqual(r.tc[18], cycle.CHENGZI_GRIP_CLOSE_COMMAND)

    def test_settle_timer_restarts_when_motion_returns(self):
        r = self.robot
        self.assertFalse(r._place_target_settled(True, 0.65, "test"))
        r.time += 0.6
        self.assertFalse(r._place_target_settled(False, 0.65, "test"))
        r.time += 0.1
        self.assertFalse(r._place_target_settled(True, 0.65, "test"))
        r.time += 0.66
        self.assertTrue(r._place_target_settled(True, 0.65, "test"))

    def test_fruit_loaded_place_speeds_keep_pre_acceleration_values(self):
        self.assertEqual(
            cycle.PLACE_UNFOLD_JOINT_SLEW_OVERRIDES["chengzi"], 0.60
        )
        self.assertEqual(
            cycle.PLACE_ADVANCE_JOINT_SLEW_OVERRIDES["chengzi"], 0.24
        )
        self.assertEqual(
            cycle.PLACE_LOWER_JOINT_SLEW_OVERRIDES["chengzi"], 0.24
        )
        self.assertEqual(
            cycle.PLACE_RELEASE_JOINT_SLEW_OVERRIDES["chengzi"], 2.00
        )
        self.assertEqual(cycle.CHENGZI_PLACE_LOWER_SETTLE_SEC, 0.0)
        self.assertEqual(
            cycle.PLACE_UNFOLD_JOINT_SLEW_OVERRIDES["pingguo"], 1.00
        )
        self.assertEqual(
            cycle.PLACE_ADVANCE_JOINT_SLEW_OVERRIDES["pingguo"], 0.34
        )
        self.assertEqual(
            cycle.PLACE_LOWER_JOINT_SLEW_OVERRIDES["pingguo"], 0.36
        )
        for kind in ("kele", "maidong", "zhijin", "shupian", "heweidao"):
            with self.subTest(accelerated_kind=kind):
                self.assertEqual(
                    cycle.PLACE_UNFOLD_JOINT_SLEW_OVERRIDES.get(
                        kind, cycle.PLACE_UNFOLD_GENTLE_JOINT_SLEW
                    ),
                    1.25,
                )
                self.assertEqual(
                    cycle.PLACE_ADVANCE_JOINT_SLEW_OVERRIDES.get(
                        kind, cycle.PLACE_ADVANCE_GENTLE_JOINT_SLEW
                    ),
                    0.42,
                )
                self.assertEqual(
                    cycle.PLACE_LOWER_JOINT_SLEW_OVERRIDES.get(
                        kind, cycle.PLACE_LOWER_GENTLE_JOINT_SLEW
                    ),
                    0.44,
                )

    def test_tissue_loaded_nav_limits_are_independent_of_preturn(self):
        self.assertEqual(cycle.TISSUE_NAV_SPEED_LIMIT_MPS, 0.045)
        self.assertEqual(cycle.TISSUE_NAV_MAX_ANGULAR_RADPS, 0.35)
        self.assertEqual(cycle.TISSUE_PRETURN_MAX_ANGULAR_RADPS, 0.35)
        self.assertEqual(cycle.NORMAL_MPPI_MAX_ANGULAR_RADPS, 1.8)

    def test_tissue_delivery_has_no_intermediate_waypoint(self):
        self.assertFalse(hasattr(cycle, "TISSUE_TABLE_VIA_POSE"))
        self.assertFalse(
            hasattr(cycle.CycleState, "NAV_TISSUE_TABLE_VIA")
        )
        self.assertFalse(
            hasattr(cycle.CycleState, "WAIT_TISSUE_TABLE_VIA")
        )

    def test_tissue_uses_tighter_later_visual_handoff(self):
        tissue_profile = cycle._precision_grasp_profile("zhijin")
        self.assertEqual(tissue_profile, (0.004, 0.006, 0.050))
        self.assertEqual(
            cycle._fine_terminal_ee_control_zone("zhijin"), 0.10
        )
        self.assertEqual(
            cycle._fine_terminal_ee_control_zone("pingguo"),
            0.11,
        )
        self.assertEqual(
            cycle._fine_terminal_ee_control_zone("kouxiangtang"), 0.12
        )
        # Open-space cruise remains common; only the loaded clamp's terminal speed
        # is reduced for the symmetric two-hand centring operation.
        self.assertAlmostEqual(
            cycle._fine_approach_scheduled_speed(0.50, "zhijin"),
            cycle.DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS,
        )
        self.assertAlmostEqual(
            cycle._fine_approach_scheduled_speed(0.0, "zhijin"), 0.050
        )
        self.assertAlmostEqual(
            cycle._fine_approach_scheduled_speed(0.0, "pingguo"),
            cycle.DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS,
        )

    def test_sandwich_preserves_visual_heading_and_apple_has_envelope(self):
        sandwich = object.__new__(cycle.EProductCycleClient)
        sandwich.target_kind = "sanmingzhi"
        self.assertAlmostEqual(
            sandwich._target_grasp_yaw(), cycle.GRASP_YAW
        )
        self.assertAlmostEqual(
            cycle._fine_terminal_ee_control_zone("sanmingzhi"), 0.10
        )
        self.assertAlmostEqual(
            cycle.SANMINGZHI_TERMINAL_HEADING_HOLD_TOL_RAD,
            math.radians(4.0),
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("heweidao", "L2"),
            0.050,
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("sanmingzhi", "L2"),
            0.042,
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("sanmingzhi", "L3"),
            0.062,
        )
        # 苹果只比通用深度浅 2 mm，L1/L3 层级补偿仍然照常叠加。
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("pingguo", "L1"), 0.043
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("pingguo", "L2"), 0.033
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("pingguo", "L3"), 0.053
        )
        self.assertEqual(cycle.PLACE_ADVANCE_TO_LOWER_SETTLE_SEC, 0.0)
        # 口香糖只在通用深度上增加 5 mm，层级补偿仍独立叠加。
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("kouxiangtang", "L2"),
            0.040,
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("kouxiangtang", "L3"),
            0.060,
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("shupian", "L1"), 0.057
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("shupian", "L2"), 0.047
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("shupian", "L3"), 0.067
        )
        right_error, right_correction = (
            cycle._sanmingzhi_terminal_heading_hold(1.55, 1.40)
        )
        left_error, left_correction = (
            cycle._sanmingzhi_terminal_heading_hold(1.40, 1.55)
        )
        self.assertGreater(right_error, 0.0)
        self.assertEqual(
            right_correction,
            cycle.SANMINGZHI_TERMINAL_HEADING_MAX_CORRECTION_RADPS,
        )
        self.assertLess(left_error, 0.0)
        self.assertEqual(
            left_correction,
            -cycle.SANMINGZHI_TERMINAL_HEADING_MAX_CORRECTION_RADPS,
        )

        self.assertTrue(
            cycle._pingguo_inserted_grasp_envelope(
                "pingguo", 0.0128, 0.0074
            )
        )
        self.assertFalse(
            cycle._pingguo_inserted_grasp_envelope(
                "pingguo", 0.021, 0.0074
            )
        )
        self.assertFalse(
            cycle._pingguo_inserted_grasp_envelope(
                "pingguo", 0.0128, 0.010
            )
        )
        self.assertTrue(
            cycle._tissue_contact_grasp_envelope(
                "zhijin", 0.0806, 0.024
            )
        )
        self.assertFalse(
            cycle._tissue_contact_grasp_envelope(
                "zhijin", 0.091, 0.024
            )
        )
        self.assertFalse(
            cycle._tissue_contact_grasp_envelope(
                "zhijin", 0.0806, 0.026
            )
        )
        # A 15 mm offset is still safely enclosed by the two hands.  Contact
        # now proceeds directly to the clamp without a reverse/re-entry pass.
        self.assertTrue(
            cycle._tissue_contact_grasp_envelope(
                "zhijin",
                0.089,
                0.015,
            )
        )

    def test_edge_row_cylinders_use_surveyed_slot_lateral_reference(self):
        for kind in ("kele", "maidong"):
            for level in ("L1", "L3"):
                with self.subTest(kind=kind, level=level):
                    self.assertTrue(
                        cycle._edge_row_cylinder_slot_lateral_locked(
                            kind, level
                        )
                    )
                    self.assertEqual(
                        cycle._fine_lateral_tolerance(kind, level), 0.012
                    )
                    self.assertAlmostEqual(
                        cycle._grasp_lateral_target_x(kind, level, 2.025),
                        2.050 if kind == "maidong" else 2.045,
                    )
            self.assertFalse(
                cycle._edge_row_cylinder_slot_lateral_locked(kind, "L2")
            )
            self.assertEqual(
                cycle._fine_lateral_tolerance(kind, "L2"), 0.015
            )
            self.assertAlmostEqual(
                cycle._grasp_lateral_target_x(kind, "L2", 1.805),
                1.815 if kind == "maidong" else 1.810,
            )

        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("maidong", "L1"), 0.052
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("kele", "L3"), 0.063
        )

        for level in ("L1", "L2", "L3"):
            with self.subTest(shupian_level=level):
                self.assertEqual(
                    cycle._fine_lateral_tolerance("shupian", level), 0.018
                )
                self.assertAlmostEqual(
                    cycle._grasp_lateral_target_x(
                        "shupian", level, 1.805
                    ),
                    1.815,
                )

        for kind in ("chengzi", "heweidao", "sanmingzhi", "zhijin"):
            with self.subTest(unaffected_kind=kind):
                self.assertFalse(
                    cycle._edge_row_cylinder_slot_lateral_locked(kind, "L1")
                )

        for kind in (
            "chengzi",
            "heweidao",
            "sanmingzhi",
            "kouxiangtang",
        ):
            with self.subTest(common_right_offset_kind=kind):
                self.assertAlmostEqual(
                    cycle._grasp_lateral_target_x(kind, "L2", 1.805),
                    1.810,
                )
        self.assertAlmostEqual(
            cycle._grasp_lateral_target_x("pingguo", "L2", 1.805),
            1.813,
        )
        self.assertAlmostEqual(
            cycle._grasp_lateral_target_x("zhijin", "L2", 1.805),
            1.805,
        )

    def test_image_servo_metadata_and_turn_direction(self):
        pixel = cycle._detection_image_center(
            "maidong:0|pixel=742.500,391.250"
        )
        np.testing.assert_allclose(pixel, [742.5, 391.25])
        self.assertIsNone(cycle._detection_image_center("maidong:0"))
        self.assertIsNone(
            cycle._detection_image_center("maidong:0|pixel=bad,391")
        )

        right_error, right_turn = cycle._image_servo_correction(
            700.0, 620.0, 600.0
        )
        left_error, left_turn = cycle._image_servo_correction(
            540.0, 620.0, 600.0
        )
        self.assertEqual(right_error, 80.0)
        self.assertLess(right_turn, 0.0)
        self.assertEqual(left_error, -80.0)
        self.assertGreater(left_turn, 0.0)
        _, deadband_turn = cycle._image_servo_correction(
            622.0, 620.0, 600.0
        )
        self.assertEqual(deadband_turn, 0.0)
        _, apple_deadband_turn = cycle._image_servo_correction(
            623.0,
            620.0,
            600.0,
            cycle.PINGGUO_IMAGE_SERVO_PIXEL_DEADBAND_PX,
        )
        self.assertLess(apple_deadband_turn, 0.0)
        # Even the worst opposite final-yaw feed-forward cannot reverse a
        # strong current-frame pixel correction.
        self.assertLess(
            cycle._image_servo_angular_command(right_turn, 10.0), 0.0
        )
        self.assertGreater(
            cycle._image_servo_angular_command(left_turn, -10.0), 0.0
        )

        self.assertTrue(
            cycle._fine_visual_alignment_pending(10.0, -10.0, 9.9)
        )
        self.assertFalse(
            cycle._fine_visual_alignment_pending(10.0, -5.0, 9.9)
        )
        self.assertFalse(
            cycle._fine_visual_alignment_pending(
                10.0,
                -10.0,
                10.0 - cycle.IMAGE_SERVO_MAX_AGE_SEC - 0.1,
            )
        )

    def test_image_servo_uses_fresh_paired_pixels_then_falls_back(self):
        robot = SimpleNamespace(
            time=10.0,
            _current_target={"surface_world": np.array([2.0, 3.2, 0.9])},
            _camera_k=np.array(
                [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]]
            ),
            _image_reacquire_points=deque(
                [
                    (
                        9.9,
                        np.array([2.01, 3.2, 0.9]),
                        np.array([760.0, 350.0]),
                        np.array([620.0, 500.0]),
                    )
                ]
            ),
            _fine_pixel_servo_active=False,
            _fine_product_pixel=None,
            _fine_gripper_pixel=None,
            _fine_pixel_error_px=None,
            _fine_pixel_error_raw_px=None,
            _fine_pixel_correction_radps=None,
            _fine_pixel_linear_scale=1.0,
            _fine_pixel_observation_at=None,
            _fine_pixel_filter_observation_at=None,
        )
        robot.now = lambda: robot.time
        correction = cycle.EProductCycleClient._update_fine_pixel_servo(robot)
        self.assertTrue(robot._fine_pixel_servo_active)
        self.assertLess(correction, 0.0)
        self.assertEqual(robot._fine_pixel_error_px, 140.0)
        self.assertEqual(
            robot._fine_pixel_linear_scale,
            cycle.IMAGE_SERVO_MIN_LINEAR_SCALE,
        )

        robot.time += cycle.IMAGE_SERVO_MAX_AGE_SEC + 0.1
        self.assertIsNone(
            cycle.EProductCycleClient._update_fine_pixel_servo(robot)
        )
        self.assertFalse(robot._fine_pixel_servo_active)
        self.assertEqual(robot._fine_pixel_linear_scale, 1.0)

    def test_gripper_projection_uses_measured_head_camera_pose(self):
        robot = SimpleNamespace(
            _camera_k=np.array(
                [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]]
            ),
            _base_position_xyz=np.array([1.0, 2.0, 0.0]),
            _base_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            jpos={"head_yaw_joint": 0.15, "head_pitch_joint": -0.25},
            action=np.zeros(19),
            slide_meas=0.3,
            _head_camera_fk=cycle.MMK2FK(),
        )
        fk = robot._head_camera_fk
        fk.set_base_pose(
            robot._base_position_xyz, robot._base_quaternion_wxyz
        )
        fk.set_slide_joint(robot.slide_meas)
        fk.set_head_joints([0.15, -0.25])
        camera_position, camera_quaternion_wxyz = fk.get_head_camera_pose()
        camera_rotation = cycle.Rotation.from_quat(
            [
                camera_quaternion_wxyz[1],
                camera_quaternion_wxyz[2],
                camera_quaternion_wxyz[3],
                camera_quaternion_wxyz[0],
            ]
        ).as_matrix()
        optical_axis_point = camera_position + camera_rotation @ np.array(
            [0.0, 0.0, 1.0]
        )
        projected = cycle.EProductCycleClient._project_world_to_head_image(
            robot, optical_axis_point
        )
        np.testing.assert_allclose(projected, [640.0, 360.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
