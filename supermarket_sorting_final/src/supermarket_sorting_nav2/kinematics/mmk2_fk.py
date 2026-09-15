"""基于简化 MuJoCo 模型计算 MMK2 头部相机在世界坐标系中的姿态。

视觉节点把实时底盘、升降轴和头部关节反馈写入模型，再读取 headeye site；
本模块没有控制循环和人为时延。
"""

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mmk2_head_fk.xml"


class MMK2FK:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.dirty = True

    def _qpos_address(self, joint_name):
        return int(self.model.joint(joint_name).qposadr[0])

    def set_base_pose(self, position, orientation):
        address = self._qpos_address("base_joint")
        self.data.qpos[address:address + 3] = position
        self.data.qpos[address + 3:address + 7] = orientation
        self.dirty = True

    def set_slide_joint(self, joint_angle):
        self.data.qpos[self._qpos_address("slide_joint")] = joint_angle
        self.dirty = True

    def set_head_joints(self, joint_angles):
        if len(joint_angles) != 2:
            raise ValueError("head joints must contain yaw and pitch")
        self.data.qpos[self._qpos_address("head_yaw_joint")] = joint_angles[0]
        self.data.qpos[self._qpos_address("head_pitch_joint")] = joint_angles[1]
        self.dirty = True

    def get_head_camera_pose(self):
        if self.dirty:
            mujoco.mj_forward(self.model, self.data)
            self.dirty = False

        site = self.data.site("headeye")
        position = site.xpos.copy()
        quaternion = Rotation.from_matrix(site.xmat.reshape(3, 3)).as_quat()
        return position, quaternion[[3, 0, 1, 2]]
