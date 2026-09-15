#!/usr/bin/env python3
"""
9-class product perception node for the Supermarket Sorting task.

Generalised from kele_detect.py to detect all 9 product classes.

Pipeline
--------
  /head_camera/color/image_raw                 (RGB,  bgr8 / rgb8)
  /head_camera/aligned_depth_to_color/...      (depth, mono16 in mm)
  /head_camera/color/camera_info               (K)
  /joint_states + /odom                        (drive MMK2FK -> camera-in-world)
        |
        v  2-D detector backend (Blob / GT / YOLO)  -> bbox centre (u,v) + class name
        v  pixel2cam: deproject (u,v,depth) with K  -> camera-frame point
        v  T_cam_world @ p_cam (MMK2FK headeye site) -> WORLD point
        |
        v  publish /product/detections (vision_msgs/Detection3DArray, world frame)
           publish /product/result_image (debug overlay)

class_id string in each Detection3D result carries the product kind name
(e.g. "kele", "pingguo", ...) so the client can filter by task target.
"""

import json
import os
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
import rclpy.qos
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       qos_profile_sensor_data)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

from discoverse.robots.mmk2.mmk2_fk import MMK2FK

from backends import GtProjectionBackend, BlobBackend, YoloBackend
from depth_roi import sanmingzhi_front_face

LAYOUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "retail_competition_layout.json")

ARUCO_MARKER_SIZE = 0.03   # metres — must match aruco_detect.py
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
DEFAULT_CKPT = os.path.join(_CKPT_DIR, "products.pt")


class ProductDetectNode(Node):
    def __init__(self, backend="blob", pub_res_img=True, device="auto", ckpt=None):
        super().__init__("product_detect")
        self.bridge = CvBridge()
        self.pub_res_img = pub_res_img

        self.K = None
        self._depth_msg = None

        # Detection enabled flag — toggled by /product/detect_enable.
        # Camera subscriptions are ALWAYS active; we skip processing when disabled
        # to avoid GPU/CPU overhead. This avoids unreliable dynamic sub creation.
        self._detection_enabled = True
        self._roi_request = None

        self.fk = MMK2FK()
        self.base_pos = None
        self.base_quat = None
        self.slide = 0.0
        self.head = [0.0, 0.0]

        ckpt_path = ckpt or DEFAULT_CKPT
        self.backend_name = backend
        if backend == "gt":
            self.detector = GtProjectionBackend(LAYOUT_JSON)
        elif backend == "yolo":
            self.detector = YoloBackend(ckpt_path, device=device)
        else:
            self.detector = BlobBackend()
        self.get_logger().info(
            f"product_detect up; backend={backend}, ckpt={ckpt_path if backend == 'yolo' else 'n/a'}")

        # Match the publisher's RELIABLE QoS to ensure the DDS connection stays
        # active even when the subscriber's executor is busy (e.g. YOLO inference).
        # BEST_EFFORT here caused the subscription to fire only once: CycloneDDS
        # drops subsequent frames when the single-threaded spin is not yet running.
        _img_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.create_subscription(CameraInfo, "/head_camera/color/camera_info",
                                 self.camera_info_cb, 10)
        # Camera subscriptions are permanently active.  Processing is gated by
        # self._detection_enabled so we skip inference during navigation phases.
        self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw",
                                 self.depth_cb, _img_qos)
        self.create_subscription(Image, "/head_camera/color/image_raw",
                                 self.rgb_cb, _img_qos)

        # Use TRANSIENT_LOCAL to receive the latest enable/disable state immediately.
        _enable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Bool, "/product/detect_enable",
                                 self._detect_enable_cb, _enable_qos)
        self.create_subscription(String, "/product/roi_request",
                                 self._roi_request_cb, _enable_qos)
        self.create_subscription(JointState, "/joint_states", self.js_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self.odom_cb, qos_profile_sensor_data)

        self.det_pub = self.create_publisher(Detection3DArray, "/product/detections", 10)
        self.img_pub = self.create_publisher(Image, "/product/result_image", 5)
        self.aruco_pub = self.create_publisher(String, "/aruco/world_detections", 10)

        # Build aruco_id → object_kind mapping from the layout JSON
        self._aruco_id_to_kind: dict = {}
        self._aruco_id_to_layout: dict = {}
        self._slot_to_layout: dict = {}
        try:
            with open(LAYOUT_JSON) as f:
                layout = json.load(f)
            for slot in layout:
                self._aruco_id_to_kind[slot["aruco_id"]] = slot["object_kind"]
                self._aruco_id_to_layout[slot["aruco_id"]] = slot
                self._slot_to_layout[(slot["shelf"], slot["level"],
                                      slot["column"])] = slot
        except Exception as exc:
            self.get_logger().warn(f"Could not load layout for ArUco kind map: {exc}")

        # Initialise ArUco detector (OpenCV ≥ 4.7 API)
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    def _detect_enable_cb(self, msg: Bool):
        if msg.data != self._detection_enabled:
            self._detection_enabled = msg.data
            state = "ENABLED" if msg.data else "DISABLED"
            self.get_logger().info(f"camera detection {state}")

    def _roi_request_cb(self, msg: String):
        try:
            request = json.loads(msg.data)
        except Exception:
            request = None
        if not isinstance(request, dict) or not request.get("enabled"):
            self._roi_request = None
            return
        slot = request.get("slot")
        if (request.get("kind") != "sanmingzhi"
                or not isinstance(slot, list) or len(slot) != 3):
            self._roi_request = None
            return
        stage = str(request.get("stage", "moved"))
        expected_y = request.get("expected_center_y")
        try:
            expected_y = float(expected_y) if expected_y is not None else None
        except (TypeError, ValueError):
            expected_y = None
        self._roi_request = (
            "sanmingzhi", tuple(str(v) for v in slot), stage, expected_y)

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

    def camera_world_tmat(self):
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        self.fk.set_left_arm_joints([0.0] * 6)
        self.fk.set_right_arm_joints([0.0] * 6)
        pos, quat = self.fk.get_head_camera_pose()
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def pixel_to_cam(self, u, v, depth_m):
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        return np.array([(u - cx) * depth_m / fx,
                         (v - cy) * depth_m / fy,
                         depth_m])

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        if len(valid) == 0:
            return 0.0
        # Use 15th-percentile rather than median: selects the closest (front-face)
        # surface rather than a mid-depth point that may be the shelf back wall.
        # This prevents objects being detected "behind" shallow foreground items.
        return float(np.percentile(valid, 15)) * 1e-3

    def rgb_cb(self, msg: Image):
        if not self._detection_enabled:
            return
        if self.K is None or self._depth_msg is None:
            return
        T_cw = self.camera_world_tmat()
        if T_cw is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)

        dets = self.detector.detect(rgb, depth, self.K, T_cw)

        out = []
        vis = rgb.copy() if self.pub_res_img else rgb
        for d in dets:
            u, v = int(d["x"]), int(d["y"])
            depth_m = self.patch_depth_m(depth, u, v)
            if depth_m <= 0.0:
                continue
            p_cam = self.pixel_to_cam(u, v, depth_m)
            p_world = (T_cw @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
            out.append({"class": d["class"], "conf": d.get("conf", 0.0), "world": p_world})
            if self.pub_res_img:
                w2, h2 = int(d["w"]) // 2, int(d["h"]) // 2
                cv2.rectangle(vis, (u - w2, v - h2), (u + w2, v + h2), (0, 255, 0), 2)
                cv2.putText(vis,
                            f"{d['class']} ({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f})",
                            (u - 60, v - h2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # ---- ArUco detection: publish world-frame marker positions ----
        aruco_records = self._detect_aruco(
            rgb, T_cw, vis if self.pub_res_img else None)
        if not any(item["class"] == "sanmingzhi" for item in out):
            roi_slot = None
            if (self._roi_request is not None
                    and self._roi_request[0] == "sanmingzhi"):
                roi_slot = self._slot_to_layout.get(self._roi_request[1])
            if roi_slot is not None:
                expected_y = self._roi_request[3]
                # A moved observation must be north of the pre-push centre
                # plane.  Allowing 45 mm south admitted the same stationary
                # shelf surface before and after the cage transfer.
                roi_min_y = (float(expected_y) + 0.010
                             if expected_y is not None else
                             float(roi_slot["world_position"][1]) - 0.045
                             if self._roi_request[2] == "baseline" else None)
                front = sanmingzhi_front_face(
                    depth, self.K, T_cw, roi_slot["world_position"],
                    min_world_y=roi_min_y)
                if front is not None:
                    out.append({"class": "sanmingzhi", "conf": 0.36,
                                "world": front})
                    now_s = self.get_clock().now().nanoseconds * 1e-9
                    if now_s - getattr(self, '_roi_log_t', 0.0) >= 0.5:
                        self._roi_log_t = now_s
                        self.get_logger().info(
                            "[SAN_DEPTH_ROI] "
                            f"slot={self._roi_request[1]} "
                            f"stage={self._roi_request[2]} "
                            f"front={np.round(front, 3)}")
            for marker in aruco_records:
                if any(item["class"] == "sanmingzhi" for item in out):
                    break
                if marker.get("kind") != "sanmingzhi":
                    continue
                slot = self._aruco_id_to_layout.get(marker["id"])
                if slot is None:
                    continue
                front = sanmingzhi_front_face(
                    depth, self.K, T_cw, slot["world_position"])
                if front is None:
                    continue
                out.append({"class": "sanmingzhi", "conf": 0.36,
                            "world": front})
                self.get_logger().info(
                    "[SAN_DEPTH_ROI] "
                    f"marker={marker['id']} front={np.round(front, 3)}")
                break

        self.publish_detections(out, msg.header.stamp)

        if self.pub_res_img:
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

    def _detect_aruco(self, rgb, T_cw, vis):
        """Detect ArUco markers and publish world positions on /aruco/world_detections."""
        if self.K is None:
            return []
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        half = ARUCO_MARKER_SIZE * 0.5
        obj_pts = np.array([
            [-half, half, 0.0], [half, half, 0.0],
            [half, -half, 0.0], [-half, -half, 0.0],
        ], dtype=np.float32)
        dist = np.zeros((4, 1), dtype=np.float32)

        records = []
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in {m for m in range(45)}:
                continue
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, corners[i].reshape(4, 2).astype(np.float32),
                self.K.astype(np.float32), dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            p_cam = tvec.flatten()
            p_world = (T_cw @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
            kind = self._aruco_id_to_kind.get(int(marker_id), f"id_{marker_id}")
            records.append({"id": int(marker_id), "kind": kind,
                            "world": p_world.tolist()})
            if vis is not None:
                ctr = corners[i].reshape(4, 2).mean(axis=0).astype(int)
                cv2.putText(vis, f"aruco:{marker_id}",
                            (ctr[0] - 20, ctr[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)
        if records:
            self.aruco_pub.publish(String(data=json.dumps(records, separators=(",", ":"))))
        return records

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
    parser = argparse.ArgumentParser(description="9-class product perception node")
    parser.add_argument("--backend", default="blob",
                        choices=["blob", "gt", "yolo"],
                        help="2-D detector backend (default: blob)")
    parser.add_argument("--ckpt", default=None,
                        help="YOLO checkpoint path (default: checkpoints/products.pt)")
    parser.add_argument("--no-result-image", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    rclpy.init()
    node = ProductDetectNode(backend=args.backend,
                              pub_res_img=not args.no_result_image,
                              device=args.device,
                              ckpt=args.ckpt)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
