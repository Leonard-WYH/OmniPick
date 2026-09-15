import json
import math
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from supermarket_sorting_nav2 import autonomous_sorting_mission as cycle
from supermarket_sorting_nav2.sorting_config import (
    TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF,
)
from supermarket_sorting_nav2.tissue_handling import (
    _tissue_preturn_signed_angle_rad,
)
from supermarket_sorting_nav2.navigation.sorting_geometry import (
    CARRY_HAND_WORKPOINT_XY_M,
    LEFT_COLUMN_OBSERVATION_X_OFFSET_M,
    RIGHT_COLUMN_OBSERVATION_X_OFFSET_M,
    SHELF_COLUMNS_M,
    SHELF_FIRST_ARUCO_ID,
    SHELF_NAMES,
    TABLE_APPROACH_X_NEGATIVE_SHIFT_M,
    TABLE_APPROACH_Y_M,
    TABLE_CENTER_X_M,
    TABLE_DROP_X_OFFSETS_M,
    TABLE_FIRST_TWO_NEGATIVE_Y_SHIFT_M,
    TABLE_SLOT_2_POSITIVE_X_SHIFT_M,
    TABLE_SLOT_3_POSITIVE_X_SHIFT_M,
    TABLE_SLOT_5_POSITIVE_X_SHIFT_M,
    TABLE_Y_MIN_M,
    bottle_geometry,
    nearest_shelf_slot,
    parse_mission_command,
    shelf_contains,
    shelf_scan_pose,
    shelf_target_observation_pose,
    table_approach_pose,
)
from supermarket_sorting_nav2.perception.product_detector import (
    ProductDetectNode,
)


class RandomShelfGeometryTests(unittest.TestCase):
    def test_tissue_preturn_signed_angle_is_shelf_specific(self):
        expected_degrees = {
            "E": 70.0,
            "A": -90.0,
            "B": -180.0,
            "C": 150.0,
            "D": 90.0,
        }

        self.assertEqual(
            set(TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF),
            set(expected_degrees),
        )
        for shelf, degrees in expected_degrees.items():
            with self.subTest(shelf=shelf):
                configured = TISSUE_PRETURN_SIGNED_ANGLE_RAD_BY_SHELF[shelf]
                resolved = _tissue_preturn_signed_angle_rad(shelf)
                self.assertAlmostEqual(math.degrees(configured), degrees)
                self.assertAlmostEqual(math.degrees(resolved), degrees)

    def test_all_five_cabinets_map_to_their_nine_slot_ids(self):
        for shelf_index, shelf in enumerate(SHELF_NAMES):
            for level_index, z in enumerate((0.5484, 0.9235, 1.226)):
                for column_index, x in enumerate(SHELF_COLUMNS_M[shelf]):
                    marker, found_shelf, level, column = nearest_shelf_slot(
                        (x, 3.243, z), "kele", shelf
                    )
                    self.assertEqual(marker, shelf_index * 9 + level_index * 3 + column_index)
                    self.assertEqual(found_shelf, shelf)
                    self.assertEqual(level, f"L{level_index + 1}")
                    self.assertEqual(column, f"C{column_index + 1}")
                    self.assertTrue(shelf_contains((x, 3.243, z), shelf))

    def test_scan_pose_translates_the_validated_e_geometry(self):
        for shelf in SHELF_NAMES:
            pose = shelf_scan_pose(shelf)
            self.assertAlmostEqual(pose[0], SHELF_COLUMNS_M[shelf][1] - 0.105)
            self.assertAlmostEqual(
                pose[1], 1.971 if shelf == "A" else 2.071
            )
            self.assertAlmostEqual(pose[2], math.pi / 2.0)

    def test_known_edge_columns_shift_later_observation_pose(self):
        for shelf in SHELF_NAMES:
            fixed = shelf_scan_pose(shelf)
            first_id = SHELF_FIRST_ARUCO_ID[shelf]
            left = shelf_target_observation_pose(shelf, first_id)
            middle = shelf_target_observation_pose(shelf, first_id + 1)
            right = shelf_target_observation_pose(shelf, first_id + 2)

            self.assertAlmostEqual(
                left[0], fixed[0] - LEFT_COLUMN_OBSERVATION_X_OFFSET_M
            )
            self.assertEqual(middle, fixed)
            self.assertAlmostEqual(
                right[0], fixed[0] + RIGHT_COLUMN_OBSERVATION_X_OFFSET_M
            )
            self.assertEqual(left[1:], fixed[1:])
            self.assertEqual(right[1:], fixed[1:])

    def test_unknown_slot_and_first_e_override_keep_fixed_pose(self):
        fixed = shelf_scan_pose("E")
        self.assertEqual(shelf_target_observation_pose("E", None), fixed)
        self.assertEqual(
            shelf_target_observation_pose(
                "E",
                SHELF_FIRST_ARUCO_ID["E"] + 2,
                shift_edge_columns=False,
            ),
            fixed,
        )
        self.assertEqual(
            shelf_target_observation_pose("E", SHELF_FIRST_ARUCO_ID["D"]),
            fixed,
        )

    def test_clear_random_command_is_supported(self):
        command = parse_mission_command("clear_random", "clear_random")
        self.assertEqual(command.command, "start")
        self.assertEqual(command.mission, "clear_random")

    def test_all_table_waypoints_include_negative_x_shift(self):
        for slot_index, offset in enumerate(TABLE_DROP_X_OFFSETS_M):
            pose = table_approach_pose(slot_index)
            expected_x = (
                TABLE_CENTER_X_M
                + offset
                - CARRY_HAND_WORKPOINT_XY_M[1]
                - TABLE_APPROACH_X_NEGATIVE_SHIFT_M
            )
            self.assertAlmostEqual(pose[0], expected_x)

    def test_selected_table_waypoints_move_positive_x(self):
        self.assertAlmostEqual(
            TABLE_DROP_X_OFFSETS_M[1],
            0.05 + TABLE_SLOT_2_POSITIVE_X_SHIFT_M,
        )
        self.assertAlmostEqual(
            TABLE_DROP_X_OFFSETS_M[2],
            0.30 + TABLE_SLOT_3_POSITIVE_X_SHIFT_M,
        )
        self.assertAlmostEqual(
            TABLE_DROP_X_OFFSETS_M[4],
            0.05 + TABLE_SLOT_5_POSITIVE_X_SHIFT_M,
        )

    def test_first_two_table_waypoints_shift_toward_negative_y(self):
        for slot_index in (0, 1):
            self.assertAlmostEqual(
                table_approach_pose(slot_index)[1],
                TABLE_APPROACH_Y_M - TABLE_FIRST_TWO_NEGATIVE_Y_SHIFT_M,
            )
        for slot_index in (2, 3, 4):
            self.assertAlmostEqual(
                table_approach_pose(slot_index)[1], TABLE_APPROACH_Y_M
            )

    def test_orange_shifted_slots_are_clamped_inside_physical_table_edge(self):
        orange = bottle_geometry("chengzi")
        for slot_index in range(3):
            robot = object.__new__(cycle.EProductCycleClient)
            robot.target_kind = "chengzi"
            robot.target_geometry = orange
            robot._table_drop_index = lambda _fallback, value=slot_index: value
            pose = robot._table_approach_pose(slot_index)
            nominal_center_y = (
                pose[1]
                - CARRY_HAND_WORKPOINT_XY_M[0]
                - cycle.table_place_forward_distances(
                    slot_index, "chengzi"
                )[0]
            )
            self.assertGreaterEqual(
                nominal_center_y - orange.radius_m,
                TABLE_Y_MIN_M + cycle.SINGLE_HAND_TABLE_EDGE_MARGIN_M,
            )

    def test_runtime_table_pose_clamps_deep_product_inside_edge(self):
        robot = object.__new__(cycle.EProductCycleClient)
        robot.target_kind = "heweidao"
        robot.target_geometry = bottle_geometry("heweidao")
        robot._table_drop_index = lambda _fallback: 0

        pose = robot._table_approach_pose(0)
        forward_m = cycle.table_place_forward_distances(0, "heweidao")[0]
        nominal_center_y = (
            pose[1] - CARRY_HAND_WORKPOINT_XY_M[0] - forward_m
        )

        self.assertGreater(pose[1], table_approach_pose(0)[1])
        self.assertGreaterEqual(
            nominal_center_y - robot.target_geometry.radius_m,
            TABLE_Y_MIN_M + cycle.SINGLE_HAND_TABLE_EDGE_MARGIN_M,
        )


class RandomShelfPriorityTests(unittest.TestCase):
    def setUp(self):
        self.robot = object.__new__(cycle.EProductCycleClient)
        self.robot.base_xy = np.array((0.0, 0.0), dtype=float)
        self.robot._picked_count = 0
        self.robot._random_remaining_kinds = []

    @staticmethod
    def candidate(shelf, marker):
        return {
            "shelf": shelf,
            "aruco_id": marker,
            "product_world": np.array(
                (SHELF_COLUMNS_M[shelf][1], 3.243, 0.9), dtype=float
            ),
            "mean_confidence": 0.9,
        }

    def test_first_item_prefers_e_then_d_then_c(self):
        self.robot._picked_count = 0
        candidates = [
            self.candidate("C", 18),
            self.candidate("D", 27),
            self.candidate("E", 36),
        ]
        selected = min(candidates, key=self.robot._random_candidate_key)
        self.assertEqual(selected["shelf"], "E")

    def test_task_tissue_filter_forces_tissue_first_at_initial_handoff(self):
        self.robot._forced_first_target_kind = "zhijin"
        candidates = [
            {**self.candidate("E", 39), "kind": "chengzi"},
            {**self.candidate("E", 40), "kind": "zhijin"},
            {**self.candidate("E", 41), "kind": "kouxiangtang"},
        ]

        selected = min(candidates, key=self.robot._random_candidate_key)

        self.assertEqual(selected["kind"], "zhijin")
        self.assertEqual(selected["aruco_id"], 40)

    def test_forced_first_kind_waits_when_only_other_products_are_visible(self):
        self.robot._forced_first_target_kind = "zhijin"
        candidates = [
            {**self.candidate("E", 39), "kind": "chengzi"},
            {**self.candidate("E", 41), "kind": "kouxiangtang"},
        ]

        filtered = self.robot._first_pick_kind_candidates(candidates)

        self.assertEqual(filtered, [])

    def test_after_first_item_prefers_b_then_a_or_c(self):
        self.robot._picked_count = 1
        candidates = [
            self.candidate("E", 36),
            self.candidate("C", 18),
            self.candidate("A", 0),
            self.candidate("B", 9),
        ]
        selected = min(candidates, key=self.robot._random_candidate_key)
        self.assertEqual(selected["shelf"], "B")

    def test_scan_fallback_order_changes_after_first_delivery(self):
        self.robot._picked_count = 0
        self.assertEqual(self.robot._random_scan_order(), ("E", "D", "C", "B", "A"))
        self.robot._picked_count = 1
        self.assertEqual(self.robot._random_scan_order(), ("B", "A", "C", "D", "E"))

    def test_later_items_keep_common_scan_fallback_order(self):
        self.robot._picked_count = 4
        self.robot._random_remaining_kinds = ["zhijin"]
        self.robot._latest_inventory = {"slots": []}
        self.robot.base_xy = np.asarray(shelf_scan_pose("E")[:2], dtype=float)

        self.assertEqual(
            self.robot._random_scan_order(),
            ("B", "A", "C", "D", "E"),
        )

    def test_pick_handoff_policy_matches_shelf_and_mission_phase(self):
        self.robot._random_shelf_mode = True

        self.robot._picked_count = 0
        for shelf in ("E", "D"):
            self.robot._active_shelf = shelf
            self.assertEqual(
                self.robot._random_pick_handoff_policy(), "rolling"
            )
        self.robot._active_shelf = "C"
        self.assertEqual(
            self.robot._random_pick_handoff_policy(), "traditional"
        )

        self.robot._picked_count = 1
        for shelf in ("A", "D", "E"):
            self.robot._active_shelf = shelf
            self.assertEqual(
                self.robot._random_pick_handoff_policy(),
                "stationary_predeploy",
            )
        for shelf in ("B", "C"):
            self.robot._active_shelf = shelf
            self.assertEqual(
                self.robot._random_pick_handoff_policy(), "rolling"
            )

    def test_common_grasp_speeds_receive_small_increase(self):
        self.assertAlmostEqual(
            cycle.DIRECT_FINE_APPROACH_CRUISE_SPEED_MPS, 0.20
        )
        self.assertAlmostEqual(
            cycle.DIRECT_FINE_APPROACH_ALIGN_SPEED_MPS, 0.080
        )
        self.assertFalse(cycle.FINE_APPROACH_TIME_ABORTS_ENABLED)

    def test_reacquire_with_empty_live_buffer_falls_back_without_exception(self):
        self.robot._current_target = {
            "aruco_id": 40,
            "surface_world": np.array((1.80, 3.20, 0.90), dtype=float),
        }
        self.robot._reacquire_points = deque()
        self.robot.now = Mock(return_value=10.0)

        self.assertFalse(self.robot._reacquire_target(tracking=True))

    def test_random_pick_starts_before_nav2_terminal_yaw_adjustment(self):
        self.assertAlmostEqual(cycle.RANDOM_PICK_PREDEPLOY_DISTANCE_M, 0.10)
        self.assertAlmostEqual(cycle.RANDOM_ROLLING_HANDOFF_DISTANCE_M, 0.10)
        self.assertAlmostEqual(
            cycle.BC_RANDOM_PICK_PREDEPLOY_DISTANCE_M, 0.25
        )
        self.assertAlmostEqual(
            cycle.BC_RANDOM_ROLLING_HANDOFF_DISTANCE_M, 0.25
        )
        self.assertAlmostEqual(
            cycle.FIRST_PICK_PREDEPLOY_DISTANCE_M, 0.5
        )
        self.assertAlmostEqual(
            cycle.FIRST_PICK_ROLLING_HANDOFF_DISTANCE_M, 0.5
        )
        self.assertAlmostEqual(
            cycle._random_pick_predeploy_distance(0, "rolling", "E"), 0.5
        )
        for shelf in ("B", "C"):
            self.assertAlmostEqual(
                cycle._random_pick_predeploy_distance(1, "rolling", shelf),
                0.25,
            )
            self.assertAlmostEqual(
                cycle._random_rolling_handoff_distance(1, shelf), 0.25
            )
        for shelf in ("A", "D", "E"):
            self.assertAlmostEqual(
                cycle._random_pick_predeploy_distance(
                    1, "stationary_predeploy", shelf
                ),
                0.25,
            )
        self.assertAlmostEqual(
            cycle._random_pick_predeploy_distance(1, "rolling", "D"),
            0.10,
        )
        self.assertAlmostEqual(
            cycle._random_rolling_handoff_distance(1, "D"), 0.10
        )

    def test_ade_handoff_accepts_near_alignment_and_requires_pick_pose(self):
        for shelf in ("A", "D", "E"):
            self.assertTrue(
                cycle._stationary_predeploy_handoff_ready(
                    "stationary_predeploy", shelf, 0.10, 0.20, True
                )
            )
            self.assertFalse(
                cycle._stationary_predeploy_handoff_ready(
                    "stationary_predeploy", shelf, 0.101, 0.20, True
                )
            )
            self.assertFalse(
                cycle._stationary_predeploy_handoff_ready(
                    "stationary_predeploy", shelf, 0.10, 0.201, True
                )
            )
            self.assertFalse(
                cycle._stationary_predeploy_handoff_ready(
                    "stationary_predeploy", shelf, 0.10, 0.20, False
                )
            )
        for shelf in ("B", "C"):
            self.assertFalse(
                cycle._stationary_predeploy_handoff_ready(
                    "rolling", shelf, 0.0, 0.0, True
                )
            )
        self.assertAlmostEqual(cycle.STATIONARY_PICK_PREDEPLOY_DISTANCE_M, 0.25)
        self.assertEqual(
            cycle.RANDOM_PICK_PREDEPLOY_YAW_TOL_RAD,
            cycle.RANDOM_ROLLING_HANDOFF_YAW_TOL_RAD,
        )
        self.assertAlmostEqual(
            cycle.RANDOM_ROLLING_HANDOFF_MAX_LINEAR_MPS, 0.8
        )
        self.assertAlmostEqual(
            cycle.RANDOM_ROLLING_HANDOFF_EXTRA_TRAVEL_M, 2.1
        )

    def test_rolling_handoff_never_bypasses_large_yaw_error(self):
        # This is the exact B->C failure shape from the 20260909 run: the
        # chassis was within 0.249 m but still about 95 degrees sideways.
        self.assertFalse(
            cycle._rolling_pick_handoff_ready(
                "rolling", 0.249, 0.25, 1.664, True
            )
        )
        # A previously successful B approach at 0.598 rad remains eligible.
        self.assertTrue(
            cycle._rolling_pick_handoff_ready(
                "rolling", 0.243, 0.25, 0.598, True
            )
        )
        self.assertFalse(
            cycle._rolling_pick_handoff_ready(
                "rolling", 0.243, 0.25, 0.598, False
            )
        )

    def test_post_first_camera_scans_abc_then_selected_cabinet(self):
        far = cycle.POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M + 0.01
        near = cycle.POST_FIRST_ABC_TRANSIT_SCAN_DISTANCE_M
        self.assertEqual(
            cycle._random_scan_focus_shelf("A", 1, True, far),
            ("B", "abc_transit_overview"),
        )
        self.assertEqual(
            cycle._random_scan_focus_shelf("A", 1, True, near),
            ("A", "target_shelf_overview"),
        )
        self.assertEqual(
            cycle._random_scan_focus_shelf("E", 0, True, far),
            ("E", "target_shelf_overview"),
        )

    def test_first_e_direct_command_runs_at_full_speed_on_straight_leg(self):
        self.assertAlmostEqual(
            cycle.FIRST_E_DIRECT_CRUISE_SPEED_MPS, 2.0
        )
        self.assertAlmostEqual(
            cycle.PICK_FINE_HANDOFF_MAX_LINEAR_MPS, 2.0
        )
        pose = shelf_scan_pose("E")
        base_xy = np.array((pose[0], pose[1] - 4.0), dtype=float)

        linear, angular, distance, yaw_error, arrived = (
            cycle._first_e_direct_velocity_command(
                base_xy, math.pi / 2.0, pose
            )
        )

        self.assertAlmostEqual(
            linear, cycle.FIRST_E_DIRECT_CRUISE_SPEED_MPS
        )
        self.assertAlmostEqual(angular, 0.0)
        self.assertAlmostEqual(distance, 4.0)
        self.assertAlmostEqual(yaw_error, 0.0)
        self.assertFalse(arrived)

    def test_first_e_direct_command_keeps_moving_during_large_turn(self):
        pose = (0.0, 4.0, math.pi / 2.0)
        linear, angular, _, _, arrived = (
            cycle._first_e_direct_velocity_command(
                np.array((0.0, 0.0), dtype=float), 0.0, pose
            )
        )

        self.assertGreater(linear, 0.0)
        self.assertGreater(angular, 0.0)
        self.assertFalse(arrived)

    def test_first_e_direct_drive_is_only_used_once_before_first_pick(self):
        self.robot._random_shelf_mode = True
        self.robot._picked_count = 0
        self.robot._first_e_direct_used = False
        self.assertTrue(self.robot._should_use_first_e_direct_drive("E"))
        self.assertFalse(self.robot._should_use_first_e_direct_drive("D"))

        self.robot._first_e_direct_used = True
        self.assertFalse(self.robot._should_use_first_e_direct_drive("E"))

        self.robot._first_e_direct_used = False
        self.robot._picked_count = 1
        self.assertFalse(self.robot._should_use_first_e_direct_drive("E"))

    def test_narrow_cylinders_keep_live_alignment_and_grasp_deeper(self):
        self.assertAlmostEqual(
            cycle._fine_terminal_ee_control_zone("kele"), 0.12
        )
        self.assertAlmostEqual(
            cycle._fine_terminal_ee_control_zone("maidong"), 0.12
        )
        self.assertAlmostEqual(
            cycle._fine_terminal_ee_control_zone("shupian"), 0.12
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("kele", "L2"),
            cycle.GRASP_INSERTION_BEYOND_CENTER_M
            + cycle.CYLINDER_EXTRA_INSERTION_M
            + cycle.KELE_ADDITIONAL_INSERTION_M,
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("maidong", "L2"),
            cycle.GRASP_INSERTION_BEYOND_CENTER_M
            + cycle.CYLINDER_EXTRA_INSERTION_M
            + cycle.MAIDONG_ADDITIONAL_INSERTION_M,
        )
        self.assertAlmostEqual(
            cycle._grasp_insertion_for_product("shupian", "L2"),
            cycle.GRASP_INSERTION_BEYOND_CENTER_M
            + cycle.SHUPIAN_EXTRA_INSERTION_M,
        )

    def test_rolling_handoff_does_not_publish_a_stop_command(self):
        self.robot.nav2 = Mock()
        self.robot.nav2.is_active = False
        self.robot.get_logger = Mock(return_value=Mock())
        self.robot.des_lin = self.robot.cur_lin = 0.0
        self.robot.des_ang = self.robot.cur_ang = 0.0
        self.robot.tc = np.zeros(19, dtype=float)
        self.robot.base_control_enabled = False

        self.robot._enable_pick_fine_base(0.60)

        self.robot.nav2.stop_robot.assert_not_called()
        self.assertAlmostEqual(self.robot.cur_lin, 0.60)
        self.assertAlmostEqual(self.robot.tc[0], 0.60)
        self.assertTrue(self.robot.base_control_enabled)

    def test_rolling_pick_readiness_uses_body_relative_endpoint(self):
        self.robot.target_kind = "kele"
        self.robot._target_slide = 0.42
        self.robot.state_t0 = 0.0
        self.robot.now = Mock(return_value=10.0)
        self.robot.tc = np.zeros(19, dtype=float)
        self.robot.tc[2] = self.robot._target_slide
        self.robot.tc[5:11] = cycle.INIT_ARM_L
        self.robot._pick_deploy_left_restore_active = False
        self.robot.jpos = {"slide_joint": self.robot._target_slide}
        # Deliberately exceed the strict joint threshold: this models the
        # harmless residual seen in the simulator while the endpoint itself
        # is already at the fixed grasp work point.
        self.robot.jpos["right_arm_joint1"] = (
            cycle.DEPLOY_JOINT_TOL + 0.01
        )
        endpoint = np.eye(4, dtype=float)
        endpoint[:3, 3] = np.array(
            [
                cycle.HAND_WORKPOINT_XY_M[0],
                cycle.HAND_WORKPOINT_XY_M[1],
                cycle.HAND_Z_PLUS_SLIDE_M - self.robot._target_slide,
            ]
        )
        self.robot.ee_footprint_pose = Mock(return_value=endpoint)

        self.assertTrue(self.robot._pick_template_ready())

    def test_pick_torso_waits_until_low_left_arm_is_supported(self):
        self.robot.target_kind = "kele"
        self.robot._target_slide = 0.42
        self.robot.tc = np.zeros(19, dtype=float)
        self.robot.action = np.zeros(19, dtype=float)
        self.robot.action[2] = 0.08
        self.robot.jpos = {
            f"left_arm_joint{i + 1}": float(value)
            for i, value in enumerate(cycle.LEFT_ARM_TRANSPORT)
        }
        self.robot.get_logger = Mock(return_value=Mock())

        self.robot._start_pick_left_support_restore()

        self.assertTrue(self.robot._pick_deploy_left_restore_active)
        self.assertAlmostEqual(self.robot.tc[2], 0.08)
        self.assertNotAlmostEqual(self.robot.tc[2], self.robot._target_slide)

    def test_pick_torso_lowers_slowly_when_left_arm_is_already_supported(self):
        self.robot.target_kind = "kele"
        self.robot._target_slide = 0.42
        self.robot.tc = np.zeros(19, dtype=float)
        self.robot.action = np.zeros(19, dtype=float)
        self.robot.jpos = {
            f"left_arm_joint{i + 1}": float(value)
            for i, value in enumerate(cycle.INIT_ARM_L)
        }

        self.robot._start_pick_left_support_restore()

        self.assertFalse(self.robot._pick_deploy_left_restore_active)
        self.assertAlmostEqual(self.robot.tc[2], self.robot._target_slide)
        self.assertAlmostEqual(
            self.robot.joint_slew,
            cycle.PICK_DEPLOY_BODY_LOWER_JOINT_SLEW,
        )
        self.assertLess(
            cycle.PICK_DEPLOY_BODY_LOWER_JOINT_SLEW,
            cycle.PICK_DEPLOY_FAST_JOINT_SLEW,
        )

    def test_current_shelf_grasp_rejects_inventory_memory(self):
        self.robot._active_shelf = "E"
        self.robot._random_remaining_kinds = ["sanmingzhi"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_failed_marker_ids = set()
        self.robot._random_planned_marker_id = 37
        remembered = {
            **self.candidate("E", 37),
            "kind": "sanmingzhi",
            "match_source": "inventory_memory",
        }
        live = {
            **self.candidate("E", 36),
            "kind": "sanmingzhi",
            "match_source": "shelf_inferred",
        }

        selected = self.robot._select_live_candidate_at_active_shelf(
            [remembered, live]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["aruco_id"], 36)
        self.assertNotEqual(selected["match_source"], "inventory_memory")

    def test_single_association_conflict_does_not_delete_latched_inventory(self):
        self.robot._random_shelf_mode = True
        self.robot._random_remaining_kinds = ["shupian"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_failed_marker_ids = set()
        self.robot._random_rejected_inventory_pairs = {(10, "shupian")}
        self.robot._random_empty_shelf_kinds = set()
        self.robot.base_xy = None
        self.robot._latest_inventory = {
            "slots": [
                {
                    "stable": True,
                    "kind": "shupian",
                    "shelf": "B",
                    "aruco_id": 10,
                    "product_world": [-0.955, 3.243, 0.55],
                    "level": "L1",
                    "column": "C2",
                    "votes": 12,
                }
            ]
        }

        candidates = self.robot._random_inventory_candidates()
        self.assertEqual([item["aruco_id"] for item in candidates], [10])

    def test_fresh_live_product_restores_rejected_inventory_pair(self):
        self.robot._random_remaining_kinds = ["shupian"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_rejected_inventory_pairs = {(10, "shupian")}
        self.robot._random_empty_shelf_kinds = {("B", "shupian")}
        self.robot._random_inventory_candidates = Mock(return_value=[])
        live = {
            **self.candidate("B", 10),
            "kind": "shupian",
            "match_source": "shelf_inferred",
        }

        candidates = self.robot._random_candidates_with_live([live])

        self.assertEqual([item["aruco_id"] for item in candidates], [10])
        self.assertNotIn(
            (10, "shupian"), self.robot._random_rejected_inventory_pairs
        )
        self.assertNotIn(
            ("B", "shupian"), self.robot._random_empty_shelf_kinds
        )

    def test_removed_physical_slot_cannot_reenter_under_another_marker_id(self):
        self.robot._random_shelf_mode = True
        self.robot._random_remaining_kinds = ["shupian"]
        self.robot._random_completed_marker_ids = {10}
        self.robot._random_failed_marker_ids = set()
        self.robot._random_removed_inventory_slots = {("B", "L1", "C2")}
        self.robot._random_rejected_inventory_pairs = set()
        self.robot._random_empty_shelf_kinds = set()
        self.robot.base_xy = None
        self.robot._latest_inventory = {
            "slots": [
                {
                    "stable": True,
                    "kind": "shupian",
                    "shelf": "B",
                    # Deliberately use another ID for the already emptied
                    # physical slot; the slot tombstone must still reject it.
                    "aruco_id": 11,
                    "product_world": [-0.955, 3.243, 0.55],
                    "level": "L1",
                    "column": "C2",
                    "votes": 12,
                }
            ]
        }
        live = {
            **self.candidate("B", 11),
            "kind": "shupian",
            "level": "L1",
            "column": "C2",
            "match_source": "shelf_inferred",
        }

        self.assertEqual(self.robot._random_inventory_candidates(), [])
        self.assertEqual(self.robot._random_candidates_with_live([live]), [])

    def test_completed_target_is_purged_and_published_to_perception(self):
        self.robot._random_completed_marker_ids = {10}
        self.robot._random_removed_inventory_slots = set()
        self.robot._latest_inventory = {
            "mapped_slot_count": 2,
            "slots": [
                {
                    "stable": True,
                    "aruco_id": 10,
                    "shelf": "B",
                    "level": "L1",
                    "column": "C2",
                },
                {
                    "stable": True,
                    "aruco_id": 28,
                    "shelf": "D",
                    "level": "L1",
                    "column": "C2",
                },
            ],
        }
        self.robot._random_last_live_candidates = [
            {
                "aruco_id": 10,
                "shelf": "B",
                "level": "L1",
                "column": "C2",
            }
        ]
        self.robot._persistent_inventory_slots = {
            ("B", "L1", "C2"): {
                "stable": True,
                "aruco_id": 10,
                "shelf": "B",
                "level": "L1",
                "column": "C2",
            },
            ("D", "L1", "C2"): {
                "stable": True,
                "aruco_id": 28,
                "shelf": "D",
                "level": "L1",
                "column": "C2",
            },
        }
        self.robot.task_payload = {"run_prefix": "visual-memory-test"}
        self.robot._inventory_remove_pub = Mock()
        self.robot.get_logger = Mock(return_value=Mock())

        self.robot._remove_random_target_from_visual_inventory(
            {
                "aruco_id": 10,
                "shelf": "B",
                "level": "L1",
                "column": "C2",
            }
        )

        self.assertEqual(
            [slot["aruco_id"] for slot in self.robot._latest_inventory["slots"]],
            [28],
        )
        self.assertEqual(self.robot._latest_inventory["mapped_slot_count"], 1)
        self.assertEqual(self.robot._random_last_live_candidates, [])
        self.assertNotIn(
            ("B", "L1", "C2"), self.robot._persistent_inventory_slots
        )
        self.assertIn(
            ("D", "L1", "C2"), self.robot._persistent_inventory_slots
        )
        message = self.robot._inventory_remove_pub.publish.call_args.args[0]
        payload = json.loads(message.data)
        self.assertEqual(payload["removed_marker_ids"], [10])
        self.assertEqual(
            payload["removed_slots"],
            [{"shelf": "B", "level": "L1", "column": "C2"}],
        )

    def test_complete_shelf_observation_persists_missing_requested_kind(self):
        self.robot._active_shelf = "B"
        self.robot._search_detection_frames = cycle.SEARCH_MIN_DETECTION_FRAMES
        self.robot._random_empty_shelf_kinds = set()
        self.robot.get_logger = Mock(return_value=Mock())
        live = {
            **self.candidate("B", 9),
            "kind": "pingguo",
            "match_source": "shelf_inferred",
        }

        self.robot._record_random_shelf_kind_visibility(
            [live], {"pingguo", "shupian"}, reason="test"
        )

        self.assertNotIn(
            ("B", "pingguo"), self.robot._random_empty_shelf_kinds
        )
        self.assertIn(
            ("B", "shupian"), self.robot._random_empty_shelf_kinds
        )

    def test_current_view_absence_does_not_override_b_inventory_priority(self):
        self.robot._random_shelf_mode = True
        self.robot._picked_count = 1
        self.robot._random_remaining_kinds = ["shupian"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_failed_marker_ids = set()
        self.robot._random_rejected_inventory_pairs = set()
        self.robot._random_empty_shelf_kinds = {("B", "shupian")}
        self.robot._random_unreachable_shelves = set()
        self.robot._random_planned_marker_id = None
        self.robot.base_xy = None
        self.robot._latest_inventory = {
            "slots": [
                {
                    "stable": True,
                    "kind": "shupian",
                    "shelf": shelf,
                    "aruco_id": marker_id,
                    "product_world": [
                        SHELF_COLUMNS_M[shelf][1], 3.243, 0.55
                    ],
                    "level": "L1",
                    "column": "C2",
                    "votes": 12,
                }
                for shelf, marker_id in (("B", 10), ("D", 28))
            ]
        }

        selected = self.robot._select_random_candidate()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["shelf"], "B")
        self.assertEqual(selected["aruco_id"], 10)

    def test_all_requested_classes_remain_schedulable_together(self):
        self.robot._random_shelf_mode = True
        self.robot._random_remaining_kinds = ["zhijin", "shupian"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_removed_inventory_slots = set()
        self.robot._random_failed_marker_ids = set()
        self.robot._random_rejected_inventory_pairs = set()
        self.robot._random_empty_shelf_kinds = set()
        self.robot._inventory_conflicted_slots = set()
        self.robot.base_xy = None
        self.robot._latest_inventory = {
            "slots": [
                {
                    "stable": True,
                    "kind": "zhijin",
                    "shelf": "E",
                    "aruco_id": 36,
                    "product_world": [SHELF_COLUMNS_M["E"][0], 3.243, 0.55],
                    "level": "L1",
                    "column": "C1",
                    "votes": 12,
                },
                {
                    "stable": True,
                    "kind": "shupian",
                    "shelf": "B",
                    "aruco_id": 10,
                    "product_world": [SHELF_COLUMNS_M["B"][1], 3.243, 0.55],
                    "level": "L1",
                    "column": "C2",
                    "votes": 12,
                },
            ]
        }
        live_tissue = {
            **self.candidate("D", 27),
            "kind": "zhijin",
            "level": "L1",
            "column": "C1",
            "match_source": "shelf_inferred",
        }
        live_ordinary = {
            **self.candidate("B", 11),
            "kind": "shupian",
            "level": "L1",
            "column": "C3",
            "match_source": "shelf_inferred",
        }

        memory = self.robot._random_inventory_candidates()
        merged = self.robot._random_candidates_with_live(
            [live_tissue, live_ordinary]
        )

        self.assertEqual({item["aruco_id"] for item in memory}, {10, 36})
        self.assertEqual(
            {item["aruco_id"] for item in merged}, {10, 11, 27, 36}
        )
        self.assertEqual(
            {item["kind"] for item in merged}, {"shupian", "zhijin"}
        )

    def test_all_classes_use_common_post_delivery_cabinet_priority(self):
        self.robot._random_shelf_mode = True
        self.robot._picked_count = 4
        self.robot._random_remaining_kinds = ["zhijin"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_removed_inventory_slots = set()
        self.robot._random_failed_marker_ids = set()
        self.robot._random_rejected_inventory_pairs = set()
        self.robot._random_empty_shelf_kinds = set()
        self.robot._random_unreachable_shelves = set()
        self.robot._random_planned_marker_id = None
        self.robot._random_inventory_candidates = Mock(return_value=[])
        self.robot.base_xy = np.asarray(table_approach_pose(2)[:2], dtype=float)
        shorter_traversable_route = {
            **self.candidate("B", 10),
            "kind": "zhijin",
            "level": "L1",
            "column": "C2",
            "match_source": "shelf_inferred",
            "mean_confidence": 0.0,
        }
        misleading_straight_line_route = {
            **self.candidate("A", 3),
            "kind": "zhijin",
            "level": "L2",
            "column": "C1",
            "match_source": "shelf_inferred",
            "mean_confidence": 1.0,
        }

        selected = self.robot._select_random_candidate(
            [misleading_straight_line_route, shorter_traversable_route]
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["aruco_id"], 10)
        self.assertEqual(selected["shelf"], "B")

    def test_tissue_release_keeps_jaws_closed_and_separates_arms(self):
        robot = object.__new__(cycle.EProductCycleClient)
        clock = {"now": 0.0}
        robot.get_clock = lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(
                nanoseconds=int(clock["now"] * 1e9)
            )
        )
        robot.get_logger = Mock(return_value=Mock())
        robot.base_xy = np.zeros(2, dtype=float)
        robot.jpos = {
            "left_arm_eef_gripper_joint": cycle.GRIP_CLOSE,
            "right_arm_eef_gripper_joint": cycle.GRIP_CLOSE,
        }
        robot.tc = np.zeros(19, dtype=float)
        robot.action = np.zeros(19, dtype=float)
        robot.target_kind = "zhijin"
        robot._pick_stow_active = False
        robot._return_stow_active = False
        robot.base_control_enabled = False
        robot.manipulation_enabled = False
        robot.max_lin = 1.0
        robot.max_ang = 1.0
        robot._last_state_log = math.inf
        robot._place_ready_since = None
        robot._tissue_table_arms = (
            np.full(6, 0.1, dtype=float),
            np.full(6, 0.2, dtype=float),
        )
        robot._tissue_release_arms = (
            np.full(6, 0.3, dtype=float),
            np.full(6, 0.4, dtype=float),
        )
        robot._tissue_hand_poses = Mock(
            return_value=(np.eye(4), np.eye(4))
        )
        robot._publish_outputs = Mock()
        robot._mark_delivery_attempt = Mock()
        robot.mission = cycle.MissionStateMachine(
            robot, cycle.CycleState.PLACE_RELEASE
        )
        # Skip entry geometry construction; this isolates the command order
        # after a valid release pose has already been solved.
        robot.mission.consume_entry()

        robot.tick()

        # Keep both finger mechanisms closed and unload the package solely by
        # moving the complete hands outward.
        self.assertTrue(np.allclose(robot.tc[5:11], 0.3))
        self.assertTrue(np.allclose(robot.tc[12:18], 0.4))
        self.assertEqual(robot.tc[11], cycle.GRIP_CLOSE)
        self.assertEqual(robot.tc[18], cycle.GRIP_CLOSE)

        # Once measured hand clearance is present, release can settle without
        # waiting for either closed finger mechanism to move.
        left_pose = np.eye(4)
        right_pose = np.eye(4)
        half_clearance = 0.5 * (
            cycle.TISSUE_RELEASE_MIN_SEPARATION_M + 0.01
        )
        left_pose[1, 3] = half_clearance
        right_pose[1, 3] = -half_clearance
        robot._tissue_hand_poses.return_value = (left_pose, right_pose)
        clock["now"] += 0.02
        robot.tick()

        self.assertTrue(np.allclose(robot.tc[5:11], 0.3))
        self.assertTrue(np.allclose(robot.tc[12:18], 0.4))

        clock["now"] += cycle.TISSUE_RELEASE_SETTLE_SEC + 0.02
        robot.tick()
        self.assertEqual(
            robot.mission.state, cycle.CycleState.PLACE_RELEASE_LIFT
        )

        # The lift state must keep both grippers closed and must not reapply a
        # finger-feedback gate that could block the mandatory retreat.
        robot._table_drop_slide_m = 0.50
        robot.jpos["slide_joint"] = 0.50
        robot.tick()
        robot.jpos["slide_joint"] = robot._release_clearance_slide_m
        clock["now"] += 0.02
        robot.tick()
        if robot.mission.state == cycle.CycleState.PLACE_RELEASE_LIFT:
            clock["now"] += cycle.POST_RELEASE_CLEARANCE_SETTLE_SEC + 0.02
            robot.tick()
        self.assertEqual(
            robot.mission.state, cycle.CycleState.RETREAT_TABLE
        )

    def test_random_drop_slots_reserve_third_position_for_first_tissue(self):
        kinds = ["zhijin", "kele", "pingguo", "shupian", "maidong"]
        self.robot._mission_kind_sequence = list(kinds)
        self.robot._random_remaining_kinds = list(kinds)

        slots = []
        for picked_count, kind in enumerate(kinds):
            self.robot._picked_count = picked_count
            slots.append(self.robot._random_drop_slot_for_kind(kind))
            self.robot._random_remaining_kinds.remove(kind)

        self.assertEqual(slots, [2, 0, 1, 3, 4])

    def test_random_drop_slots_stay_sequential_without_tissue(self):
        kinds = ["kele", "pingguo", "shupian", "maidong", "chengzi"]
        self.robot._mission_kind_sequence = list(kinds)
        self.robot._random_remaining_kinds = list(kinds)

        slots = []
        for picked_count, kind in enumerate(kinds):
            self.robot._picked_count = picked_count
            slots.append(self.robot._random_drop_slot_for_kind(kind))
            self.robot._random_remaining_kinds.remove(kind)

        self.assertEqual(slots, [0, 1, 2, 3, 4])

    def test_fixed_queue_uses_positions_1245_around_tissue_position_3(self):
        self.assertEqual(
            self.robot._table_drop_slots_for_sequence(
                ["kele", "zhijin", "pingguo", "shupian", "maidong"]
            ),
            [0, 2, 1, 3, 4],
        )

    def test_repeated_tissue_uses_only_one_reserved_third_position(self):
        self.assertEqual(
            self.robot._table_drop_slots_for_sequence(
                ["zhijin", "kele", "zhijin", "shupian", "zhijin"]
            ),
            [2, 0, 1, 3, 4],
        )

    def test_unlocked_rolling_route_collects_product_vision_while_moving(self):
        self.robot.mission = SimpleNamespace(
            state=cycle.CycleState.WAIT_E_SCAN_NAV
        )
        self.robot._random_shelf_mode = True
        self.robot._nav_pick_handoff_mode = "rolling"
        self.robot.target_locked = False
        self.robot._active_shelf = "B"
        self.robot._allowed_target_kinds = {"kouxiangtang"}
        self.robot._search_points = deque(maxlen=20)
        self.robot._search_detection_frames = 0
        self.robot._search_last_frame_at = None
        self.robot.now = Mock(return_value=12.5)
        position = SimpleNamespace(
            x=SHELF_COLUMNS_M["B"][1], y=3.243, z=0.9
        )
        result = SimpleNamespace(
            hypothesis=SimpleNamespace(class_id="kouxiangtang"),
            pose=SimpleNamespace(
                pose=SimpleNamespace(position=position)
            ),
        )
        message = SimpleNamespace(
            detections=[SimpleNamespace(results=[result], id="")]
        )

        self.robot.product_cb(message)

        self.assertEqual(self.robot._search_detection_frames, 1)
        self.assertEqual(self.robot._search_last_frame_at, 12.5)
        self.assertEqual(len(self.robot._search_points), 1)
        self.assertEqual(self.robot._search_points[0][0], "kouxiangtang")

    def test_rolling_pick_can_use_en_route_live_candidate_without_inventory(self):
        self.robot._random_shelf_mode = True
        self.robot._active_shelf = "B"
        self.robot._picked_count = 1
        self.robot._random_remaining_kinds = ["kouxiangtang"]
        self.robot._random_completed_marker_ids = set()
        self.robot._random_failed_marker_ids = set()
        self.robot._random_planned_marker_id = None
        self.robot._random_inventory_candidates = Mock(return_value=[])
        live = {
            **self.candidate("B", 10),
            "kind": "kouxiangtang",
            "match_source": "shelf_inferred",
        }
        self.robot._search_candidates = Mock(return_value=[live])

        selected = self.robot._active_shelf_pick_candidate()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["aruco_id"], 10)
        self.robot._search_candidates.assert_called_once_with()

    def test_heweidao_pending_release_never_requests_torso_lift(self):
        self.robot.target_kind = "pingguo"
        self.robot._current_target = None
        self.robot._pending_verification = {"kind": "heweidao"}
        self.robot._table_drop_slide_m = 0.37
        self.robot.tc = np.zeros(19, dtype=float)
        self.robot.joint_slew = 0.0

        self.assertFalse(self.robot._post_release_lift_required())
        self.robot._hold_direct_retreat_release_pose()
        self.assertAlmostEqual(self.robot.tc[2], 0.37)
        self.assertAlmostEqual(self.robot.tc[18], cycle.GRIP_OPEN)
        self.assertAlmostEqual(
            self.robot.joint_slew, cycle.PLACE_RELEASE_JOINT_SLEW
        )

    def test_observed_shelf_memory_is_excluded_from_reschedule(self):
        self.robot._random_unreachable_shelves = set()
        self.robot._random_planned_marker_id = None
        self.robot._random_candidates_with_live = lambda _live: [
            self.candidate("E", 36),
            self.candidate("D", 27),
        ]

        selected = self.robot._select_random_candidate(
            excluded_shelves={"E"}
        )

        self.assertEqual(selected["shelf"], "D")

    def test_first_post_delivery_route_uses_concrete_abc_memory_target(self):
        self.robot._random_post_first_abc_scan_pending = True
        self.robot._random_unreachable_shelves = set()
        selected = self.candidate("B", 11)
        self.robot._select_random_candidate = lambda: selected
        self.robot.get_logger = Mock(return_value=Mock())
        routed = []
        self.robot._route_to_random_shelf = (
            lambda shelf, **kwargs: routed.append((shelf, kwargs)) or True
        )

        self.assertTrue(
            self.robot._schedule_random_target(start_return_stow=True)
        )
        self.assertEqual(routed[0][0], "B")
        self.assertEqual(routed[0][1]["marker_id"], 11)
        self.assertTrue(routed[0][1]["start_return_stow"])
        self.assertFalse(self.robot._random_post_first_abc_scan_pending)

    def test_first_post_delivery_without_memory_keeps_b_observation_fallback(self):
        self.robot._random_post_first_abc_scan_pending = True
        self.robot._random_unreachable_shelves = set()
        self.robot._select_random_candidate = lambda: None
        routed = []
        self.robot._route_to_random_shelf = (
            lambda shelf, **kwargs: routed.append((shelf, kwargs)) or True
        )

        self.assertTrue(
            self.robot._schedule_random_target(start_return_stow=True)
        )
        self.assertEqual(routed[0][0], "B")
        self.assertNotIn("marker_id", routed[0][1])

    def test_observation_point_uses_latched_target_when_current_frame_is_empty(self):
        self.robot._random_post_first_abc_scan_pending = False
        self.robot._active_shelf = "B"
        self.robot._random_planned_marker_id = 10
        self.robot._random_planned_kind = "shupian"
        remembered = {
            **self.candidate("B", 10),
            "kind": "shupian",
            "match_source": "inventory_memory",
        }
        self.robot._select_live_candidate_at_active_shelf = Mock(
            return_value=None
        )
        self.robot._random_candidates_with_live = Mock(
            return_value=[remembered]
        )
        self.robot._activate_random_candidate = Mock()
        self.robot.get_logger = Mock(return_value=Mock())

        self.assertTrue(
            self.robot._schedule_random_target(
                live_candidates=[], already_at_scan=True
            )
        )
        self.robot._activate_random_candidate.assert_called_once_with(
            remembered
        )

    def test_rolling_pick_preempts_unfinished_return_stow(self):
        self.robot._nav_pick_prepared = False
        self.robot._nav_pick_handoff_mode = "rolling"
        self.robot._first_e_direct_active = False
        self.robot._return_stow_active = True
        self.robot._return_stow_navigation_active = True
        self.robot._return_stow_started_at = 1.0
        self.robot._return_arm_stage_ready_since = 1.0
        self.robot._return_arms_stowed = True
        self.robot._active_shelf = "B"
        self.robot.target_kind = "sanmingzhi"
        selected = self.candidate("B", 11)
        selected["kind"] = "sanmingzhi"
        self.robot._active_shelf_pick_candidate = lambda: selected
        self.robot._activate_random_candidate = Mock()
        self.robot._command_pick_template = Mock(return_value=True)
        self.robot.get_logger = Mock(return_value=Mock())

        self.assertTrue(
            self.robot._prepare_random_pick_during_navigation()
        )
        self.assertFalse(self.robot._return_stow_active)
        self.assertFalse(self.robot._return_stow_navigation_active)
        self.assertTrue(self.robot._nav_pick_prepared)


class ProductInventoryRemovalTests(unittest.TestCase):
    @staticmethod
    def fixed_marker(
        marker_id: int,
        x: float,
        *,
        level: str = "L1",
        column: str = "C1",
        z: float = 0.5,
    ):
        return {
            "id": marker_id,
            "shelf": "B",
            "level": level,
            "column": column,
            "world": [x, 3.168, z],
        }

    def test_fixed_geometry_associates_depth_products_without_aruco(self):
        markers = [
            self.fixed_marker(9, -1.07, column="C1"),
            self.fixed_marker(10, -0.85, column="C2"),
        ]
        products = [
            {"class": "kele", "world": [-1.065, 3.22, 0.573]},
            {"class": "maidong", "world": [-0.855, 3.21, 0.606]},
        ]

        matches = ProductDetectNode.associate_products_to_fixed_slots(
            products, markers
        )

        self.assertEqual(
            {(product_index, markers[marker_index]["id"])
             for product_index, marker_index in matches},
            {(0, 9), (1, 10)},
        )

    def test_fixed_geometry_keeps_live_matches_and_removed_slots_reserved(self):
        markers = [
            self.fixed_marker(9, -1.07, column="C1"),
            self.fixed_marker(10, -0.85, column="C2"),
            self.fixed_marker(11, -0.63, column="C3"),
        ]
        products = [
            {"class": "kele", "world": [-1.07, 3.243, 0.5725]},
            {"class": "kele", "world": [-0.85, 3.243, 0.5725]},
            {"class": "kele", "world": [-0.63, 3.243, 0.5725]},
        ]

        matches = ProductDetectNode.associate_products_to_fixed_slots(
            products,
            markers,
            used_product_indices={0},
            used_marker_ids={9},
            removed_marker_ids={10},
            removed_inventory_slots={("B", "L1", "C3")},
        )

        self.assertEqual(matches, [])

    def test_fixed_geometry_is_unique_and_rejects_non_shelf_detection(self):
        markers = [self.fixed_marker(9, -1.07)]
        products = [
            {"class": "kele", "world": [-1.070, 3.243, 0.5725]},
            {"class": "kele", "world": [-1.075, 3.243, 0.5725]},
            {"class": "kele", "world": [-1.070, 2.80, 0.5725]},
        ]

        matches = ProductDetectNode.associate_products_to_fixed_slots(
            products, markers
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0], (0, 0))

    def test_perception_removal_deletes_votes_and_blocks_repopulation(self):
        detector = SimpleNamespace(
            _removed_marker_ids=set(),
            _removed_inventory_slots=set(),
            _inventory_run_prefix="visual-memory-test",
            _slot_votes={10: deque(({"kind": "shupian"},)), 28: deque()},
            get_logger=Mock(return_value=Mock()),
        )
        message = SimpleNamespace(
            data=(
                '{"run_prefix":"visual-memory-test",'
                '"removed_marker_ids":[10],'
                '"removed_slots":[{"shelf":"B","level":"L1",'
                '"column":"C2"}]}'
            )
        )

        ProductDetectNode.inventory_remove_cb(detector, message)

        self.assertEqual(detector._removed_marker_ids, {10})
        self.assertEqual(
            detector._removed_inventory_slots, {("B", "L1", "C2")}
        )
        self.assertNotIn(10, detector._slot_votes)
        self.assertIn(28, detector._slot_votes)

    def test_client_inventory_ledger_survives_empty_camera_updates(self):
        robot = object.__new__(cycle.EProductCycleClient)
        robot._random_shelf_mode = True
        robot.task_payload = {"run_prefix": "ledger-test"}
        robot._inventory_ledger_run_prefix = "ledger-test"
        robot._persistent_inventory_slots = {}
        robot._inventory_conflict_log_keys = set()
        robot._inventory_conflicted_slots = set()
        robot._random_completed_marker_ids = set()
        robot._random_removed_inventory_slots = set()
        robot._latest_inventory = {}
        robot.get_logger = Mock(return_value=Mock())
        slot = {
            "stable": True,
            "aruco_id": 10,
            "shelf": "B",
            "level": "L1",
            "column": "C2",
            "kind": "shupian",
            "product_world": [-0.85, 3.243, 0.604],
            "votes": 3,
        }

        robot.inventory_cb(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "run_prefix": "ledger-test",
                        "slots": [slot],
                    }
                )
            )
        )
        robot.inventory_cb(
            SimpleNamespace(
                data=json.dumps(
                    {"run_prefix": "ledger-test", "slots": []}
                )
            )
        )

        self.assertEqual(robot._latest_inventory["mapped_slot_count"], 1)
        self.assertEqual(robot._latest_inventory["slots"][0]["aruco_id"], 10)
        self.assertTrue(robot._latest_inventory["persistent_ledger"])

    def test_conflicting_stable_class_is_retained_but_not_blindly_routed(self):
        robot = object.__new__(cycle.EProductCycleClient)
        robot._random_shelf_mode = True
        robot.task_payload = {"run_prefix": "conflict-test"}
        robot._inventory_ledger_run_prefix = "conflict-test"
        robot._persistent_inventory_slots = {}
        robot._inventory_conflict_log_keys = set()
        robot._inventory_conflicted_slots = set()
        robot._random_completed_marker_ids = set()
        robot._random_removed_inventory_slots = set()
        robot._random_failed_marker_ids = set()
        robot._random_remaining_kinds = ["sanmingzhi"]
        robot._latest_inventory = {}
        robot.base_xy = None
        robot.get_logger = Mock(return_value=Mock())
        base_slot = {
            "stable": True,
            "aruco_id": 17,
            "shelf": "B",
            "level": "L3",
            "column": "C3",
            "product_world": [-0.63, 3.243, 1.24],
            "votes": 6,
        }

        robot.inventory_cb(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "run_prefix": "conflict-test",
                        "slots": [{**base_slot, "kind": "sanmingzhi"}],
                    }
                )
            )
        )
        robot.inventory_cb(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "run_prefix": "conflict-test",
                        "slots": [{**base_slot, "kind": "pingguo"}],
                    }
                )
            )
        )

        # Memory is not deleted by a contradictory observation, but the
        # disputed B slot cannot win cabinet priority and cause an empty trip.
        self.assertEqual(
            robot._latest_inventory["slots"][0]["kind"], "sanmingzhi"
        )
        self.assertIn(("B", "L3", "C3"), robot._inventory_conflicted_slots)
        self.assertEqual(robot._random_inventory_candidates(), [])

        # Seeing the latched class again makes the stored candidate usable.
        robot.inventory_cb(
            SimpleNamespace(
                data=json.dumps(
                    {
                        "run_prefix": "conflict-test",
                        "slots": [{**base_slot, "kind": "sanmingzhi"}],
                    }
                )
            )
        )
        self.assertNotIn(
            ("B", "L3", "C3"), robot._inventory_conflicted_slots
        )
        self.assertEqual(
            [item["aruco_id"] for item in robot._random_inventory_candidates()],
            [17],
        )


class RandomTaskContractTests(unittest.TestCase):
    @staticmethod
    def random_robot():
        robot = object.__new__(cycle.EProductCycleClient)
        robot._random_shelf_mode = True
        robot._allowed_target_kinds = tuple(cycle.SUPPORTED_PRODUCT_KINDS)
        robot._mission_event_prefix = "RANDOM"
        robot._event_prefix = "RANDOM"
        robot.task_received = False
        robot.task_payload = None
        robot._required_target_count = None
        robot._mission_kind_sequence = []
        robot._random_remaining_kinds = []
        robot._configured_forced_first_target_kind = ""
        robot._forced_first_target_kind = ""
        robot.get_logger = Mock(return_value=Mock())
        return robot

    @staticmethod
    def task_message(kinds):
        payload = {"targets": [{"kind": kind} for kind in kinds]}
        return payload, SimpleNamespace(data=json.dumps(payload))

    def test_random_task_accepts_all_classes_and_repeated_kinds(self):
        robot = self.random_robot()
        kinds = ["kele", "pingguo", "zhijin", "kele", "zhijin"]
        payload, message = self.task_message(kinds)

        robot.task_cb(message)

        self.assertTrue(robot.task_received)
        self.assertEqual(robot.task_payload, payload)
        self.assertEqual(robot._required_target_count, 5)
        expected = ["zhijin", "kele", "pingguo", "kele", "zhijin"]
        self.assertEqual(robot._mission_kind_sequence, expected)
        self.assertEqual(robot._random_remaining_kinds, expected)
        self.assertEqual(robot._forced_first_target_kind, "zhijin")

    def test_random_task_without_tissue_preserves_published_order(self):
        robot = self.random_robot()
        kinds = ["kele", "pingguo", "kele", "maidong", "chengzi"]
        payload, message = self.task_message(kinds)

        robot.task_cb(message)

        self.assertTrue(robot.task_received)
        self.assertEqual(robot.task_payload, payload)
        self.assertEqual(robot._mission_kind_sequence, kinds)
        self.assertEqual(robot._random_remaining_kinds, kinds)
        self.assertEqual(robot._forced_first_target_kind, "")

    def test_random_task_rejects_unsupported_kind(self):
        robot = self.random_robot()
        _payload, message = self.task_message(
            ["kele", "maidong", "unknown", "pingguo", "zhijin"]
        )

        robot.task_cb(message)

        self.assertFalse(robot.task_received)
        self.assertIsNone(robot.task_payload)
        self.assertIsNone(robot._required_target_count)
        self.assertEqual(robot._mission_kind_sequence, [])
        self.assertEqual(robot._random_remaining_kinds, [])
        rejection = robot.get_logger.return_value.error.call_args.args[0]
        self.assertIn("RANDOM_TASK_REJECT", rejection)

    def test_random_task_rejects_non_five_item_payload(self):
        robot = self.random_robot()
        _payload, message = self.task_message(
            ["kele", "maidong", "pingguo", "chengzi"]
        )

        robot.task_cb(message)

        self.assertFalse(robot.task_received)
        rejection = robot.get_logger.return_value.error.call_args.args[0]
        self.assertIn("RANDOM_TASK_REJECT", rejection)


if __name__ == "__main__":
    unittest.main()
