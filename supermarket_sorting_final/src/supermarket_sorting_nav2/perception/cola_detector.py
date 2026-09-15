#!/usr/bin/env python3
"""
kele perception node for the Supermarket Sorting task.

Mirrors the reference material_detection_client/yolo_detect.py architecture
but is specialised for the kele bottle and outputs poses in the WORLD frame
(the client consumes world-frame targets directly via arm_to()).

Pipeline
--------
  /head_camera/color/image_raw            (RGB,  bgr8 / rgb8)
  /head_camera/aligned_depth_to_color/... (depth, mono16 in mm)
  /head_camera/color/camera_info          (K)
  /joint_states + /odom                   (drive MMK2FK -> camera-in-world)
        |
        v  YOLO detector -> bbox centre (u,v)
        v  pixel2cam: deproject (u,v,depth) with K  -> camera-frame point
        v  T_cam_world @ p_cam (MMK2FK headeye site) -> WORLD point
        |
        v  publish /kele/detections (vision_msgs/Detection3DArray, world frame)
           publish /kele/result_image (debug overlay)

The camera->world transform uses the baseline's robot-only MMK2FK model,
fed with the live base pose (odom) + slide/head joints (joint_states).  The
'headeye' site already carries the OpenGL->OpenCV optical-frame flip, so the
deprojected point maps to world with NO extra axis swap (validated to
0.0 mm round-trip error).

中文说明：可乐专用 RGB-D 检测节点，图像到达即推理并发布，不设置固定观察
延时；目标累计和停稳判断由任务客户端负责。
"""

import argparse
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..kinematics.mmk2_fk import MMK2FK
from .yolo_backend import YoloBackend

DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "products.pt"
TARGET_CLASS = "kele"


class KeleDetectNode(Node):
    def __init__(self, weights=DEFAULT_WEIGHTS, pub_res_img=True, device="auto", confidence=0.65):
        super().__init__("kele_detect")
        self.bridge = CvBridge()
        self.pub_res_img = pub_res_img

        # camera intrinsics (from camera_info)
        self.K = None
        self._depth_msg = None

        # live robot state for the camera->world transform
        self.fk = MMK2FK()
        self.base_pos = None        # [x, y, z]
        self.base_quat = None       # [w, x, y, z]
        self.slide = 0.0
        self.head = [0.0, 0.0]

        self.detector = YoloBackend(weights, confidence=confidence, device=device)
        self.get_logger().info(f"kele_detect up; weights={weights}")

        # subscriptions
        self.create_subscription(CameraInfo, "/head_camera/color/camera_info",
                                 self.camera_info_cb, 10)
        self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, 10)
        self.create_subscription(Image, "/head_camera/color/image_raw",
                                 self.rgb_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, 10)

        # publishers
        self.det_pub = self.create_publisher(Detection3DArray, "/kele/detections", 10)
        self.img_pub = self.create_publisher(Image, "/kele/result_image", 5)

    # ---- state callbacks ----
    def camera_info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def depth_cb(self, msg: Image):
        self._depth_msg = msg

    def js_cb(self, msg: JointState):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    # ---- camera->world transform from live state ----
    def camera_world_tmat(self):
        """4x4 camera(optical)->world built from odom + slide/head via MMK2FK."""
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        pos, quat = self.fk.get_head_camera_pose()   # quat wxyz, world
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def pixel_to_cam(self, u, v, depth_m):
        """Deproject a pixel + metric depth to a camera-optical-frame point."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        return np.array([x, y, depth_m])

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        """Median depth (m) over a patch, ignoring zero (invalid) pixels."""
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) * 1e-3 if len(valid) else 0.0

    # ---- main RGB callback ----
    def rgb_cb(self, msg: Image):
        if self.K is None or self._depth_msg is None:
            return
        T_cam_world = self.camera_world_tmat()
        if T_cam_world is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)  # mono16, mm

        dets = self.detector.detect(rgb)

        out = []
        vis = rgb.copy() if self.pub_res_img else rgb
        for d in dets:
            # 兼容历史单可乐流程：仓库统一使用九类 products.pt，但该节点只向
            # /kele/detections 发布可乐，不能把其他类别混入旧控制器。
            if d["class"] != TARGET_CLASS:
                continue
            u, v = int(d["x"]), int(d["y"])
            depth_m = self.patch_depth_m(depth, u, v)
            if depth_m <= 0.0:
                continue
            p_cam = self.pixel_to_cam(u, v, depth_m)
            p_world = (T_cam_world @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]

            rec = {"class": d["class"], "conf": d.get("conf", 0.0), "world": p_world}
            out.append(rec)

            if self.pub_res_img:
                w, h = int(d["w"]), int(d["h"])
                cv2.rectangle(vis, (u - w // 2, v - h // 2),
                              (u + w // 2, v + h // 2), (0, 255, 0), 2)
                cv2.putText(vis, f"{d['class']} ({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f})",
                            (u - 60, v - h // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        self.publish_detections(out, msg.header.stamp)
        if self.pub_res_img:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

    def publish_detections(self, recs, stamp):
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        for r in recs:
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(r["class"])
            hyp.hypothesis.score = float(r["conf"])
            hyp.pose.pose.position.x = float(r["world"][0])
            hyp.pose.pose.position.y = float(r["world"][1])
            hyp.pose.pose.position.z = float(r["world"][2])
            det.results.append(hyp)
            msg.detections.append(det)
        self.det_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser(description="kele perception node")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="YOLO checkpoint path")
    parser.add_argument("--confidence", type=float, default=0.65,
                        help="minimum detection confidence")
    parser.add_argument("--no-result-image", action="store_true",
                        help="disable /kele/result_image publishing")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="YOLO inference device (default: auto)")
    args = parser.parse_args()

    rclpy.init()
    node = KeleDetectNode(weights=args.weights,
                          pub_res_img=not args.no_result_image,
                          device=args.device,
                          confidence=args.confidence)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
