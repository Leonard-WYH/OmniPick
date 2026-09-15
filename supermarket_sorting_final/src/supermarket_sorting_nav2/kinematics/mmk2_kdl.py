"""MMK2 躯干与双臂的正/逆运动学封装。

中文维护说明：组合升降轴、左右臂基座变换和 ArmKdl 求解器，为抓取模板、
碰撞包络和末端位置校验提供统一坐标计算；不含 ROS 等待或动作时延。
"""

from __future__ import annotations

import numpy as np
#from arm_control_py.basic_part import _cos, _sin
from .arm_kdl import ArmKdl
from dataclasses import dataclass


@dataclass
class SpineKdl:
    dx: float
    dz: float
    joint_limits: np.array

    def get_transformation_matrix(self, q: float) -> np.array:
        """Get the transformation matrix of the spine"""
        # fmt: off
        return np.array([[0, 1,  0,     self.dx],
                         [1, 0,  0,           0],
                         [0, 0, -1, self.dz - q],
                         [0, 0,  0,           1]])
        # fmt: on


@dataclass
class Spine2ArmKdl:
    dx: float
    dy: float
    dz: float

    def get_transformation_matrix(self, index: str = "left") -> np.array:
        """Get the transformation matrix of the spine to arm"""
        coef = 1 if index == "left" else -1
        # fmt: off
        return np.array([[-coef * np.sqrt(2)/2,        0, coef * np.sqrt(2)/2, coef * self.dx],
                         [        np.sqrt(2)/2,        0,        np.sqrt(2)/2,        self.dy],
                         [                   0, coef * 1,                   0,        self.dz],
                         [                   0,        0,                   0,              1]])
        # fmt: on


class MMK2Kdl:
    """Kinematics and dynamics of mmk2"""

    def __init__(self, punish: list[float] = None, iteration: int = 100):
        self.left_arm = ArmKdl()
        self.right_arm = ArmKdl()
        self.spine = SpineKdl(dx=0.033942, dz=1.406, joint_limits=[-0.04, 0.87])
        self.spine2arm = Spine2ArmKdl(dx=0.10704, dy=0.02283, dz=0.09475)
        self.punish = (
            punish
            if punish is not None
            else [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.iteration = iteration

    def _cal_punish(self, q1: np.array, q2: np.array) -> float:
        """Calculate the punishment of the joint angles"""
        assert len(q1) == len(q2), "q1 and q2 should have the same length"
        assert len(self.punish) == len(q1), "punish should have the same length as q1"
        return np.sum(self.punish * np.abs(q1 - q2))

    def forward_kinematics(
        self, q: np.array, index: str = None
    ) -> tuple[np.array, np.array]:
        """
        Get the forward kinematics of mmk2

        Args:
            q: joint angles
            index: index of the arm, "left" or "right", if None, return both arms
        Returns:
            T_left: transformation matrix of the left arm
            T_right: transformation matrix of the right arm
        """

        if index is None:
            assert len(q) == 13, "q should be a 13-element array"
        else:
            assert len(q) == 7, "q should be a 7-element array"
            assert index in ["left", "right"], "index should be 'left' or 'right'"
        T_spine = self.spine.get_transformation_matrix(q[0])
        T_spine2arm_left = self.spine2arm.get_transformation_matrix("left")
        T_spine2arm_right = self.spine2arm.get_transformation_matrix("right")

        if index is None:
            T_left_arm = self.left_arm.forward_kinematics(q[1:7])
            T_right_arm = self.right_arm.forward_kinematics(q[7:13])

            T_left = T_spine @ T_spine2arm_left @ T_left_arm
            T_right = T_spine @ T_spine2arm_right @ T_right_arm
            return T_left, T_right

        if index == "left":
            T_left_arm = self.left_arm.forward_kinematics(q[1:7])
            T_left = T_spine @ T_spine2arm_left @ T_left_arm
            return T_left, None

        if index == "right":
            T_right_arm = self.right_arm.forward_kinematics(q[1:7])
            T_right = T_spine @ T_spine2arm_right @ T_right_arm
            return None, T_right

    def _inverse_kinematics(
        self,
        T_left: np.array,
        T_right: np.array,
        target_height: float,
        ref_arm_pos: np.array = None,
    ) -> np.array:
        """Get the inverse kinematics of mmk2"""

        assert T_left is not None or T_right is not None

        index = (
            "both"
            if (T_left is not None and T_right is not None)
            else ("left" if T_left is not None else "right")
        )


        if T_left is not None:
            assert T_left.shape == (4, 4), "T_left should be a 4x4 matrix"
        if T_right is not None:
            assert T_right.shape == (4, 4), "T_right should be a 4x4 matrix"

        if ref_arm_pos is not None:
            if index == "both":
                assert (
                    len(ref_arm_pos) == 12
                ), "ref_arm_pos should be a 12-element array"
            else:
                assert (
                    len(ref_arm_pos) == 6 or len(ref_arm_pos) == 12
                ), "ref_arm_pos should be a 6-element array or a 12-element array"

        T_spine = self.spine.get_transformation_matrix(target_height)

        if index == "both":
            T_spine2arm_left = self.spine2arm.get_transformation_matrix("left")
            T_spine2arm_right = self.spine2arm.get_transformation_matrix("right")
            T_left_arm = np.linalg.inv(T_spine @ T_spine2arm_left) @ T_left
            T_right_arm = np.linalg.inv(T_spine @ T_spine2arm_right) @ T_right
            left_arm_joints = self.left_arm.inverse_kinematics(
                T_left_arm, ref_arm_pos[:6]
            )
            right_arm_joints = self.right_arm.inverse_kinematics(
                T_right_arm, ref_arm_pos[6:]
            )
            if left_arm_joints is None or right_arm_joints is None:
                return None
            res = []
            for joints in left_arm_joints:
                for joints2 in right_arm_joints:
                    res.append(
                        np.array(
                            [
                                target_height,
                                joints[0],
                                joints[1],
                                joints[2],
                                joints[3],
                                joints[4],
                                joints[5],
                                joints2[0],
                                joints2[1],
                                joints2[2],
                                joints2[3],
                                joints2[4],
                                joints2[5],
                            ]
                        )
                    )
            return res

        if index == "left":
            T_spine2arm = self.spine2arm.get_transformation_matrix("left")
            T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T_left
            if ref_arm_pos is not None and len(ref_arm_pos) == 12:
                ref_arm_pos = ref_arm_pos[:6]
            arm_joints = self.left_arm.inverse_kinematics(T_arm, ref_arm_pos)
            if arm_joints is None:
                return None
            res = []
            for joints in arm_joints:
                res.append(
                    np.array(
                        [
                            target_height,
                            joints[0],
                            joints[1],
                            joints[2],
                            joints[3],
                            joints[4],
                            joints[5],
                        ]
                    )
                )
            return res

        if index == "right":
            T_spine2arm = self.spine2arm.get_transformation_matrix("right")
            T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T_right
            if ref_arm_pos is not None and len(ref_arm_pos) == 12:
                ref_arm_pos = ref_arm_pos[6:]
            arm_joints = self.right_arm.inverse_kinematics(T_arm, ref_arm_pos)
            if arm_joints is None:
                return None
            res = []
            for joints in arm_joints:
                res.append(
                    np.array(
                        [
                            target_height,
                            joints[0],
                            joints[1],
                            joints[2],
                            joints[3],
                            joints[4],
                            joints[5],
                        ]
                    )
                )
            return res

    def inverse_kinematics(
        self,
        T_left: np.array = None,
        T_right: np.array = None,
        ref_pos: np.array = None,
        target_height: float = None,
    ) -> np.array:
        """
        Get the inverse kinematics of mmk2, T_left and T_right should not be None at the same time, ref_pos should not be None if target_height is None

        Args:
            T_left: transformation matrix of the left arm
            T_right: transformation matrix of the right arm
            ref_pos: reference joint angles
            target_height: target height of the spine

        Returns:
            res: joint angles of the arm
        """

        assert T_left is not None or T_right is not None

        index = (
            "both"
            if (T_left is not None and T_right is not None)
            else ("left" if T_left is not None else "right")
        )

        if T_left is not None:
            assert T_left.shape == (4, 4), "T_left should be a 4x4 matrix"
        if T_right is not None:
            assert T_right.shape == (4, 4), "T_right should be a 4x4 matrix"

        assert (
            ref_pos is not None or target_height is not None
        ), "ref_pos and target_height should not be None at the same time"

        if ref_pos is not None:
            if index == "both":
                assert len(ref_pos) == 13, "ref_pos should be a 13-element array"
            else:
                assert (
                    len(ref_pos) == 13 or len(ref_pos) == 7
                ), "ref_pos should be a 7-element array or a 13-element array"

        res: np.array

        if target_height != None:
            ref_arm_pos = None
            if ref_pos is not None:
                ref_arm_pos = ref_pos[1:]
            return self._inverse_kinematics(T_left, T_right, target_height, ref_arm_pos)

        sum_punish: float = np.inf
        target_height = ref_pos[0]

        ref_arm_pos = ref_pos[1:]
        res = self._inverse_kinematics(T_left, T_right, target_height, ref_arm_pos)
        if res is not None and len(res) > 0:
            sum_punish = self._cal_punish(ref_pos, res[0])

        for i in range(self.iteration):
            target_height = np.random.uniform(
                self.spine.joint_limits[0], self.spine.joint_limits[1]
            )
            ref_arm_pos = None
            ref_arm_pos = ref_pos[1:]
            temp_res = self._inverse_kinematics(
                T_left, T_right, target_height, ref_arm_pos
            )
            if temp_res is None or len(temp_res) == 0:
                continue
            temp_punish = self._cal_punish(ref_pos, temp_res[0])
            if temp_punish < sum_punish:
                sum_punish = temp_punish
                res = temp_res

        return res


def main():
    """Example usage of the MMK2Kdl class."""
    mmk2 = MMK2Kdl()
    # q = np.array(
    #     [
    #         0.571,
    #         0.838,
    #         0.009,
    #         0.923,
    #         -1.675,
    #         -1.604,
    #         -0.009,
    #         -2.325,
    #         -1.868,
    #         3.077,
    #         1.696,
    #         1.437,
    #         2.596,
    #     ]
    # )
    q = np.array(
        [
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    # q = np.array([0.0, 0, 0, 0, -1.51, -0.766, 1.57, 0, 0, 0, 1.51, 0.766, -1.57])
    T_left, T_right = mmk2.forward_kinematics(q)
    print("Left Arm Transformation Matrix:\n", T_left)
    print("Right Arm Transformation Matrix:\n", T_right)

    joints_left = mmk2.inverse_kinematics(T_left, None, target_height=q[0], ref_pos=q)
    print("Left Arm Joints:\n", joints_left)

    joints_right = mmk2.inverse_kinematics(None, T_right, target_height=q[0])
    print("Right Arm Joints:\n", joints_right)

    joints = mmk2.inverse_kinematics(T_left, T_right, ref_pos=q)
    print("Joints with ref_pos:\n", joints)

    T_left, T_right = mmk2.forward_kinematics(joints[0])

    print("Left Arm Transformation Matrix from Inverse Kinematics:\n", T_left)
    print("Right Arm Transformation Matrix from Inverse Kinematics:\n", T_right)

def test():
    mmk2 = MMK2Kdl()

    T_right = np.eye(4)
    T_right[:3, :3] = np.eye(3)
    T_right[:3, 3] = [0.306, -0.21, 1.071]
    ref_pose = np.zeros(7)
    ref_pose[0] = 0.0
    ref_pose[1:] = [ -0. ,   -0.166,  0.032,  0. ,   -1.571 , -2.223]

    joints_right = mmk2.inverse_kinematics(None, T_right, target_height=ref_pose[0], ref_pos=ref_pose)
    print("Right Arm Joints:\n", joints_right)

if __name__ == "__main__":
    main()
