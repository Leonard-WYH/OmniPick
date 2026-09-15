#!/usr/bin/env python3
"""Generate a high-quality nine-class YOLO dataset from the retail GS scene.

This script is designed for the ``supermarket_sorting:server`` image. It uses
the image's complete Discoverse scene and assets while the host perception
directory is bind-mounted over the image's perception directory.

Labels do not come from approximate, hand-written 3-D cuboids. A second GS
render attaches a deterministic feature code to every product instance. Static
scenery and the robot retain zero-valued features, so they still occlude
products correctly. The decoded visible instance mask is cleaned and converted
to a tight bounding box.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np


TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_DIR.parents[1]
ASSETS_DIR = TASK_DIR / "models"
os.environ.setdefault("DISCOVERSE_ASSETS_DIR", str(ASSETS_DIR))
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(REPO_ROOT))

from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg  # noqa: E402


LAYOUT_JSON = TASK_DIR / "retail_competition_layout.json"
SOURCE_XML = TASK_DIR / "mjcf" / "retail_competition.xml"
RUNTIME_XML = Path("/tmp/retail_competition_dataset_runtime.xml")

HEAD_CAM_ID = 0
FOVY_DEG = 45.29
IMG_W, IMG_H = 640, 480

# This order is the public contract between labels, data.yaml and products.pt.
CLASS_NAMES = [
    "sanmingzhi",   # 0 - sandwich
    "heweidao",     # 1 - soy-sauce bottle
    "shupian",      # 2 - chips bag
    "zhijin",       # 3 - tissue box
    "maidong",      # 4 - Maidong drink
    "kouxiangtang", # 5 - chewing gum
    "pingguo",      # 6 - apple
    "chengzi",      # 7 - orange
    "kele",         # 8 - cola bottle
]
CLASS_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

CLASS_COLOURS_BGR = [
    (60, 180, 255),
    (70, 220, 70),
    (255, 170, 50),
    (220, 90, 220),
    (50, 220, 220),
    (240, 100, 80),
    (80, 80, 255),
    (40, 140, 255),
    (255, 80, 140),
]

SHELF_SURFACE_Z = {"L1": 0.499, "L2": 0.851, "L3": 1.189}

# The low support threshold captures fuzzy Gaussian edges; the stronger seed
# rejects weak feature mixtures at occlusion boundaries.
DEFAULT_FEATURE_DIM = 12
DEFAULT_SUPPORT_NORM = 0.012
DEFAULT_SEED_NORM = 0.055
DEFAULT_SUPPORT_COS = 0.70
DEFAULT_SEED_COS = 0.86
DEFAULT_SUPPORT_MARGIN = 0.035
DEFAULT_SEED_MARGIN = 0.10
DEFAULT_MIN_MASK_AREA = 45
DEFAULT_MIN_BOX_SIDE = 10
DEFAULT_MAX_BOX_SIDE = 360
DEFAULT_MIN_MASK_FILL = 0.45


# ---------------------------------------------------------------------------
# Scene construction (same resources and renderer as the server image)
# ---------------------------------------------------------------------------
def _local_robot_gs_model_dict():
    out = {}
    for name, path in MMK2Cfg.gs_model_dict.items():
        if path.startswith("mobile_chassis/mmk2/"):
            out[name] = path.replace("mobile_chassis/mmk2/", "mmk2/")
        elif path.startswith("manipulator/airbot_play/"):
            out[name] = path.replace("manipulator/airbot_play/", "airbot_play/")
        else:
            out[name] = path
    return out


def _write_runtime_xml():
    text = SOURCE_XML.read_text().replace("__REPO_ROOT__", str(TASK_DIR))
    RUNTIME_XML.write_text(text)
    return str(RUNTIME_XML)


def _resolve_background_ply():
    fitted = ASSETS_DIR / "3dgs" / "shentoon" / "retail_background_fit.ply"
    if fitted.exists():
        return "shentoon/retail_background_fit.ply"
    return "shentoon/dummy_background.ply"


def validate_scene_files():
    missing = [path for path in (LAYOUT_JSON, SOURCE_XML, ASSETS_DIR) if not path.exists()]
    if missing:
        formatted = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Run this training bundle inside the full server image. "
            f"Missing scene resources:\n  {formatted}"
        )


def build_sim():
    """Build the headless MMK2 + Gaussian-Splatting simulator."""
    validate_scene_files()
    cfg = MMK2Cfg()
    cfg.mjcf_file_path = _write_runtime_xml()
    cfg.use_gaussian_renderer = True
    cfg.enable_render = True
    cfg.headless = True

    layout = json.loads(LAYOUT_JSON.read_text())
    cfg.obj_list = [slot["body"] for slot in layout]
    cfg.gs_model_dict = _local_robot_gs_model_dict()
    cfg.gs_model_dict["background"] = _resolve_background_ply()
    for slot in layout:
        cfg.gs_model_dict[slot["body"]] = slot["gs_ply"]

    cfg.obs_rgb_cam_id = [HEAD_CAM_ID]
    cfg.obs_depth_cam_id = [HEAD_CAM_ID]
    cfg.lidar_s2_sim = False
    cfg.render_set = {"fps": 24, "width": IMG_W, "height": IMG_H}

    sim = MMK2Base(cfg)
    sim.reset()
    return sim


# ---------------------------------------------------------------------------
# Product and camera randomisation
# ---------------------------------------------------------------------------
def randomise_product_layout(sim, original_layout, rng, xy_jitter=0.022, yaw_max=0.55):
    """Shuffle products between slots and add physically plausible XY/yaw jitter."""
    import mujoco

    permutation = rng.permutation(len(original_layout))
    updated = []
    round_objects = {"heweidao", "maidong", "pingguo", "chengzi", "kele"}

    for source_index, source in enumerate(original_layout):
        destination = original_layout[int(permutation[source_index])]
        source_surface = SHELF_SURFACE_Z.get(source.get("level", "L1"), 0.499)
        half_height = float(source["world_position"][2]) - source_surface
        destination_surface = SHELF_SURFACE_Z.get(destination.get("level", "L1"), 0.499)

        x = float(destination["world_position"][0] + rng.uniform(-xy_jitter, xy_jitter))
        y = float(destination["world_position"][1] + rng.uniform(-xy_jitter, xy_jitter))
        z = float(destination_surface + half_height)
        if source["object_kind"] in round_objects:
            yaw = float(rng.uniform(-math.pi, math.pi))
        else:
            yaw = float(rng.uniform(-yaw_max, yaw_max))

        body_name = source["body"]
        if body_name in sim.free_body_qpos_ids:
            joint_id = sim.free_body_qpos_ids[body_name]
            qpos_id = int(sim.mj_model.jnt_qposadr[joint_id])
            sim.mj_data.qpos[qpos_id:qpos_id + 3] = [x, y, z]
            sim.mj_data.qpos[qpos_id + 3:qpos_id + 7] = [
                math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)
            ]

        item = dict(source)
        item["world_position"] = [x, y, z]
        item["destination_shelf"] = destination.get("shelf")
        item["destination_level"] = destination.get("level")
        item["destination_column"] = destination.get("column")
        item["yaw"] = yaw
        updated.append(item)

    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    return updated


def set_robot_pose(sim, pose):
    """Write base, lift and head joints directly into qpos."""
    import mujoco

    qpos = sim.mj_data.qpos
    yaw = pose["yaw"]
    qpos[0:3] = [pose["base_x"], pose["base_y"], 0.0]
    qpos[3:7] = [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]
    qpos[9] = pose["slide"]
    qpos[10] = pose["head_yaw"]
    qpos[11] = pose["head_pitch"]
    sim.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim.mj_model, sim.mj_data)


def sample_pose(rng, mode):
    """Sample camera poses with broad distance, angle and height coverage."""
    if mode == "baseline":
        base_x = float(np.clip(rng.normal(0.91, 0.28), 0.20, 1.55))
        base_y = float(np.clip(rng.normal(2.46, 0.25), 1.95, 2.88))
        target_x = float(np.clip(rng.normal(0.98, 0.34), 0.15, 1.70))
        target_y = 3.243
        yaw = math.atan2(target_y - base_y, target_x - base_x)
        yaw += float(rng.uniform(-0.09, 0.09))
        slide = float(rng.uniform(0.02, 0.28))
        head_pitch = float(rng.uniform(-0.82, -0.24))
        head_yaw = float(rng.uniform(-0.10, 0.10))
        focus_level = "mixed"
    else:
        target_x = float(rng.uniform(-2.02, 2.02))
        target_y = 3.243
        if mode == "wide":
            distance = float(rng.uniform(0.55, 1.55))
            lateral = float(rng.uniform(-0.65, 0.65))
        else:  # diverse deliberately balances near, middle and far views
            bucket = int(rng.integers(0, 3))
            distance_ranges = ((0.48, 0.78), (0.78, 1.18), (1.18, 1.65))
            distance = float(rng.uniform(*distance_ranges[bucket]))
            lateral = float(rng.uniform(-0.78, 0.78))

        base_x = float(np.clip(target_x + lateral, -2.55, 2.55))
        base_y = float(target_y - distance)
        yaw = math.atan2(target_y - base_y, target_x - base_x)
        yaw += float(rng.uniform(-0.08, 0.08))

        level_index = int(rng.integers(0, 3))
        focus_level = ("L1", "L2", "L3")[level_index]
        slide_ranges = ((0.00, 0.15), (0.04, 0.27), (0.12, 0.36))
        pitch_ranges = ((-0.92, -0.52), (-0.70, -0.30), (-0.48, -0.10))
        slide = float(rng.uniform(*slide_ranges[level_index]))
        head_pitch = float(rng.uniform(*pitch_ranges[level_index]))
        if rng.random() < 0.18:
            slide = float(rng.uniform(0.0, 0.36))
            head_pitch = float(rng.uniform(-0.92, -0.08))
            focus_level = "mixed"
        head_yaw = float(rng.uniform(-0.16, 0.16))

    return {
        "base_x": base_x,
        "base_y": base_y,
        "yaw": float(yaw),
        "slide": slide,
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
        "target_x": target_x,
        "target_y": target_y,
        "focus_level": focus_level,
    }


# ---------------------------------------------------------------------------
# Exact visible-instance feature rendering
# ---------------------------------------------------------------------------
def _make_instance_codes(count, dimension, seed=92821):
    """Create well-separated deterministic unit vectors for instance IDs."""
    rng = np.random.default_rng(seed)
    candidates = rng.normal(size=(4096, dimension)).astype(np.float32)
    candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
    chosen = [candidates[0]]
    available = candidates[1:]
    while len(chosen) < count:
        current = np.stack(chosen)
        score = np.max(np.abs(available @ current.T), axis=1)
        index = int(np.argmin(score))
        chosen.append(available[index])
        available = np.delete(available, index, axis=0)
    return np.stack(chosen).astype(np.float32)


class InstanceFeatureRenderer:
    """Render an occlusion-aware feature image identifying every product body."""

    def __init__(self, sim, layout, feature_dim=DEFAULT_FEATURE_DIM):
        import torch

        self.sim = sim
        self.codes_np = _make_instance_codes(len(layout), feature_dim)
        renderer = sim.gs_renderer
        gaussian_count = int(renderer.gaussians.xyz.shape[0])
        self.features = torch.zeros(
            (gaussian_count, feature_dim),
            dtype=torch.float32,
            device=renderer.gaussians.device,
        )
        missing = []
        for instance_id, slot in enumerate(layout):
            body = slot["body"]
            start = renderer.gaussian_start_indices.get(body)
            end = renderer.gaussian_end_indices.get(body)
            if start is None or end is None:
                missing.append(body)
                continue
            code = torch.as_tensor(
                self.codes_np[instance_id], dtype=torch.float32, device=self.features.device
            )
            self.features[start:end] = code
        if missing:
            raise RuntimeError(f"Product GS ranges missing from renderer: {missing}")
        self.codes = torch.as_tensor(
            self.codes_np, dtype=torch.float32, device=self.features.device
        )

    def _camera_matrices(self):
        import torch

        sim = self.sim
        device = self.features.device
        camera_position = torch.as_tensor(
            sim.mj_data.cam_xpos[[HEAD_CAM_ID]], dtype=torch.float32, device=device
        )
        camera_rotation = torch.as_tensor(
            sim.mj_data.cam_xmat[[HEAD_CAM_ID]].reshape(1, 3, 3),
            dtype=torch.float32,
            device=device,
        )
        world_from_camera = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        world_from_camera[:, :3, :3] = camera_rotation
        world_from_camera[:, :3, 3] = camera_position
        world_from_camera[:, :3, 1] *= -1
        world_from_camera[:, :3, 2] *= -1
        view = torch.linalg.inv(world_from_camera)

        fovy = math.radians(float(sim.mj_model.cam_fovy[HEAD_CAM_ID]))
        focal = IMG_H / (2.0 * math.tan(fovy * 0.5))
        intrinsics = torch.zeros((1, 3, 3), dtype=torch.float32, device=device)
        intrinsics[0, 0, 0] = focal
        intrinsics[0, 1, 1] = focal
        intrinsics[0, 0, 2] = IMG_W * 0.5
        intrinsics[0, 1, 2] = IMG_H * 0.5
        intrinsics[0, 2, 2] = 1.0
        return view, intrinsics

    def render_decoded(self):
        """Return best instance, cosine score, score margin and feature norm."""
        import torch
        from gsplat.rendering import rasterization

        renderer = self.sim.gs_renderer
        view, intrinsics = self._camera_matrices()
        rendered, _, _ = rasterization(
            means=renderer.gaussians.xyz,
            quats=renderer.gaussians.rot,
            scales=renderer.gaussians.scale,
            opacities=renderer.gaussians.opacity,
            colors=self.features,
            viewmats=view,
            Ks=intrinsics,
            width=IMG_W,
            height=IMG_H,
            sh_degree=None,
            packed=False,
            render_mode="RGB",
            channel_chunk=16,
        )
        feature_image = rendered[0]
        magnitude = torch.linalg.vector_norm(feature_image, dim=-1)
        unit = feature_image / magnitude.clamp_min(1e-8).unsqueeze(-1)
        scores = unit @ self.codes.T
        top_values, top_indices = torch.topk(scores, k=2, dim=-1)
        best = top_indices[..., 0]
        cosine = top_values[..., 0]
        margin = top_values[..., 0] - top_values[..., 1]
        return (
            best.to(torch.int16).cpu().numpy(),
            cosine.to(torch.float16).cpu().numpy().astype(np.float32),
            margin.to(torch.float16).cpu().numpy().astype(np.float32),
            magnitude.to(torch.float16).cpu().numpy().astype(np.float32),
        )


def _component_for_instance(instance_id, best, cosine, margin, magnitude, args):
    support = (
        (best == instance_id)
        & (magnitude >= args.support_norm)
        & (cosine >= args.support_cos)
        & (margin >= args.support_margin)
    )
    seed = (
        (best == instance_id)
        & (magnitude >= args.seed_norm)
        & (cosine >= args.seed_cos)
        & (margin >= args.seed_margin)
    )
    if not np.any(seed):
        return None

    support_u8 = support.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support_u8 = cv2.morphologyEx(support_u8, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support_u8, connectivity=8)
    if count <= 1:
        return None

    best_component = None
    best_rank = None
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < args.min_mask_area:
            continue
        component = labels == component_id
        strong_pixels = int(np.count_nonzero(component & seed))
        if strong_pixels == 0:
            continue
        rank = (strong_pixels, area)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_component = component
    return best_component


def labels_from_instance_render(decoded, layout, args):
    """Convert decoded visible masks into tight YOLO-ready boxes."""
    best, cosine, margin, magnitude = decoded
    labels = []
    instance_masks = {}
    for instance_id, slot in enumerate(layout):
        kind = slot.get("object_kind", "")
        if kind not in CLASS_ID:
            continue
        component = _component_for_instance(
            instance_id, best, cosine, margin, magnitude, args
        )
        if component is None:
            continue
        ys, xs = np.where(component)
        x0 = max(0, int(xs.min()) - args.box_padding)
        y0 = max(0, int(ys.min()) - args.box_padding)
        x1 = min(IMG_W, int(xs.max()) + 1 + args.box_padding)
        y1 = min(IMG_H, int(ys.max()) + 1 + args.box_padding)
        width, height = x1 - x0, y1 - y0
        area = int(len(xs))
        mask_fill = float(area / max(1, width * height))
        if not (
            args.min_box_side <= width <= args.max_box_side
            and args.min_box_side <= height <= args.max_box_side
        ):
            continue
        if mask_fill < args.min_mask_fill:
            continue
        labels.append(
            {
                "class_id": CLASS_ID[kind],
                "class_name": kind,
                "instance_id": instance_id,
                "body": slot["body"],
                "box": [x0, y0, x1, y1],
                "mask_area": area,
                "mask_fill": mask_fill,
                "median_cosine": float(np.median(cosine[component])),
                "median_margin": float(np.median(margin[component])),
            }
        )
        instance_masks[instance_id] = component
    return labels, instance_masks


# ---------------------------------------------------------------------------
# Appearance augmentation (no geometric change, so boxes remain exact)
# ---------------------------------------------------------------------------
def _smooth_background(rng, height, width):
    base = rng.integers(35, 221, size=3).astype(np.float32)
    delta = rng.uniform(-45.0, 45.0, size=3).astype(np.float32)
    if rng.random() < 0.5:
        weight = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :, None]
    else:
        weight = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None, None]
    image = base[None, None, :] + weight * delta[None, None, :]
    image = np.broadcast_to(image, (height, width, 3)).copy()
    low_frequency = rng.normal(
        0.0, 8.0, size=(max(2, height // 40), max(2, width // 40), 3)
    )
    low_frequency = cv2.resize(
        low_frequency.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    return np.clip(image + low_frequency, 0, 255).astype(np.uint8)


def _appearance_jitter(rng, image):
    work = image.astype(np.float32) / 255.0
    gamma = float(rng.uniform(0.82, 1.20))
    work = np.power(np.clip(work, 0.0, 1.0), gamma)
    gains = rng.uniform(0.90, 1.10, size=(1, 1, 3)).astype(np.float32)
    work *= gains
    work = work * float(rng.uniform(0.90, 1.10)) + float(rng.uniform(-0.035, 0.035))
    if rng.random() < 0.30:
        work = cv2.GaussianBlur(work, (0, 0), float(rng.uniform(0.25, 0.80)))
    if rng.random() < 0.35:
        noise = rng.normal(0.0, rng.uniform(0.003, 0.012), size=work.shape)
        work += noise.astype(np.float32)
    return np.clip(work * 255.0, 0, 255).astype(np.uint8)


def _void_mask(image_bgr):
    value = image_bgr.max(axis=2)
    mask = (value <= 10).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def appearance_variants(rng, image_bgr, count):
    variants = []
    void = _void_mask(image_bgr)
    void3 = (void > 0)[:, :, None]
    height, width = image_bgr.shape[:2]
    for _ in range(count):
        background = _smooth_background(rng, height, width)
        composite = np.where(void3, background, image_bgr)
        variants.append(_appearance_jitter(rng, composite))
    return variants


# ---------------------------------------------------------------------------
# Dataset writing and visual QA
# ---------------------------------------------------------------------------
def labels_to_yolo(labels):
    lines = []
    for label in labels:
        x0, y0, x1, y1 = label["box"]
        center_x = (x0 + x1) * 0.5 / IMG_W
        center_y = (y0 + y1) * 0.5 / IMG_H
        width = (x1 - x0) / IMG_W
        height = (y1 - y0) / IMG_H
        lines.append(
            f"{label['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
        )
    return "\n".join(lines)


def save_sample(out_dir, split, name, image_bgr, labels):
    image_path = out_dir / "images" / split / f"{name}.jpg"
    label_path = out_dir / "labels" / split / f"{name}.txt"
    if not cv2.imwrite(str(image_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Failed to write {image_path}")
    label_path.write_text(labels_to_yolo(labels))


def draw_debug_overlay(image_bgr, labels, instance_masks):
    overlay = image_bgr.copy()
    for label in labels:
        colour = CLASS_COLOURS_BGR[label["class_id"]]
        mask = instance_masks[label["instance_id"]]
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.72
            + np.asarray(colour, dtype=np.float32) * 0.28
        ).astype(np.uint8)
    result = cv2.addWeighted(image_bgr, 0.35, overlay, 0.65, 0.0)
    for label in labels:
        x0, y0, x1, y1 = label["box"]
        colour = CLASS_COLOURS_BGR[label["class_id"]]
        cv2.rectangle(result, (x0, y0), (x1 - 1, y1 - 1), colour, 2)
        caption = f"{label['class_name']} {label['mask_fill']:.2f}"
        text_y = max(13, y0 - 4)
        cv2.putText(
            result, caption, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            result, caption, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, colour, 1, cv2.LINE_AA,
        )
    return result


def create_contact_sheets(debug_dir, columns=4, rows=3, max_sheets=8):
    images = sorted(debug_dir.glob("frame_*.jpg"))
    if not images:
        return []
    per_sheet = columns * rows
    maximum = min(len(images), per_sheet * max_sheets)
    indices = np.linspace(0, len(images) - 1, maximum, dtype=int)
    chosen = [images[index] for index in indices]
    outputs = []
    for sheet_index, start in enumerate(range(0, len(chosen), per_sheet)):
        batch = chosen[start:start + per_sheet]
        canvas = np.full((rows * 240, columns * 320, 3), 245, dtype=np.uint8)
        for cell, path in enumerate(batch):
            image = cv2.imread(str(path))
            if image is None:
                continue
            thumb = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            row, column = divmod(cell, columns)
            canvas[row * 240:(row + 1) * 240, column * 320:(column + 1) * 320] = thumb
        output = debug_dir / f"contact_sheet_{sheet_index:02d}.jpg"
        cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        outputs.append(output)
    return outputs


def prepare_output(out_dir, overwrite, debug_enabled):
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output is not empty: {out_dir}. Use --overwrite or choose another --out."
            )
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if debug_enabled:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="generate a tight-mask nine-class product YOLO dataset"
    )
    parser.add_argument("--frames", type=int, default=1600, help="accepted base renders")
    parser.add_argument("--variants", type=int, default=1, help="appearance variants per render")
    parser.add_argument(
        "--pose-mode", default="diverse", choices=["diverse", "wide", "baseline"],
        help="diverse balances shelf level, distance and oblique viewing angle",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out", default=str(TASK_DIR / "perception" / "dataset"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--layout-shuffle-every", type=int, default=8)
    parser.add_argument("--xy-jitter", type=float, default=0.022)
    parser.add_argument("--product-yaw-max", type=float, default=0.55)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--support-norm", type=float, default=DEFAULT_SUPPORT_NORM)
    parser.add_argument("--seed-norm", type=float, default=DEFAULT_SEED_NORM)
    parser.add_argument("--support-cos", type=float, default=DEFAULT_SUPPORT_COS)
    parser.add_argument("--seed-cos", type=float, default=DEFAULT_SEED_COS)
    parser.add_argument("--support-margin", type=float, default=DEFAULT_SUPPORT_MARGIN)
    parser.add_argument("--seed-margin", type=float, default=DEFAULT_SEED_MARGIN)
    parser.add_argument("--min-mask-area", type=int, default=DEFAULT_MIN_MASK_AREA)
    parser.add_argument("--min-box-side", type=int, default=DEFAULT_MIN_BOX_SIDE)
    parser.add_argument("--max-box-side", type=int, default=DEFAULT_MAX_BOX_SIDE)
    parser.add_argument("--min-mask-fill", type=float, default=DEFAULT_MIN_MASK_FILL)
    parser.add_argument("--min-labels-per-frame", type=int, default=4)
    parser.add_argument("--box-padding", type=int, default=1)
    parser.add_argument("--max-attempt-factor", type=int, default=8)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--debug-max", type=int, default=240)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frames <= 0 or args.variants < 0:
        raise ValueError("--frames must be positive and --variants cannot be negative")
    if not 0.0 < args.val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1")
    if args.feature_dim < 4:
        raise ValueError("--feature-dim must be at least 4")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    prepare_output(out_dir, args.overwrite, args.debug_overlay)

    sim = build_sim()
    original_layout = json.loads(LAYOUT_JSON.read_text())
    layout = list(original_layout)
    instance_renderer = InstanceFeatureRenderer(sim, original_layout, args.feature_dim)
    print(
        f"[gen] slots={len(layout)}, classes={len(CLASS_NAMES)}, frames={args.frames}, "
        f"images/frame={1 + args.variants}, pose={args.pose_mode}, "
        f"feature_dim={args.feature_dim}, background={_resolve_background_ply()}"
    )

    accepted = 0
    attempts = 0
    image_count = 0
    base_box_count = 0
    rejected_empty = 0
    split_images = Counter()
    split_saved_boxes = Counter()
    class_saved_boxes = Counter()
    max_attempts = args.frames * args.max_attempt_factor
    debug_stride = max(1, int(math.ceil(args.frames / max(1, args.debug_max))))
    manifest_path = out_dir / "manifest.jsonl"

    with manifest_path.open("w") as manifest:
        while accepted < args.frames and attempts < max_attempts:
            attempts += 1
            if attempts == 1 or (
                args.layout_shuffle_every > 0
                and (attempts - 1) % args.layout_shuffle_every == 0
            ):
                layout = randomise_product_layout(
                    sim, original_layout, rng,
                    xy_jitter=args.xy_jitter, yaw_max=args.product_yaw_max,
                )

            pose = sample_pose(rng, args.pose_mode)
            set_robot_pose(sim, pose)
            sim.render()
            rgb = sim.img_rgb_obs_s[HEAD_CAM_ID]
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            decoded = instance_renderer.render_decoded()
            labels, masks = labels_from_instance_render(decoded, layout, args)
            if len(labels) < args.min_labels_per_frame:
                rejected_empty += 1
                continue

            split = "val" if rng.random() < args.val_frac else "train"
            base_name = f"f{accepted:06d}"
            variants = [rgb_bgr] + appearance_variants(rng, rgb_bgr, args.variants)
            for variant_index, image in enumerate(variants):
                save_sample(out_dir, split, f"{base_name}_v{variant_index}", image, labels)
                image_count += 1
                split_images[split] += 1
                split_saved_boxes[split] += len(labels)

            for label in labels:
                class_saved_boxes[label["class_name"]] += 1 + args.variants
            base_box_count += len(labels)

            manifest.write(
                json.dumps(
                    {"frame": base_name, "split": split, "pose": pose, "labels": labels},
                    ensure_ascii=False,
                ) + "\n"
            )

            if args.debug_overlay and accepted % debug_stride == 0:
                debug = draw_debug_overlay(rgb_bgr, labels, masks)
                cv2.imwrite(
                    str(out_dir / "debug" / f"frame_{accepted:06d}.jpg"), debug,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )

            accepted += 1
            if accepted % 25 == 0 or accepted == args.frames:
                print(
                    f"[gen] accepted={accepted}/{args.frames}, attempts={attempts}, "
                    f"images={image_count}, base_boxes={base_box_count}"
                )

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    contact_sheets = create_contact_sheets(out_dir / "debug") if args.debug_overlay else []
    stats = {
        "seed": args.seed,
        "requested_base_frames": args.frames,
        "accepted_base_frames": accepted,
        "attempts": attempts,
        "rejected_empty": rejected_empty,
        "variants_per_frame": args.variants,
        "total_images": image_count,
        "base_frame_boxes": base_box_count,
        "saved_label_boxes": int(sum(class_saved_boxes.values())),
        "images_by_split": dict(split_images),
        "boxes_by_split": dict(split_saved_boxes),
        "boxes_by_class": {name: int(class_saved_boxes[name]) for name in CLASS_NAMES},
        "contact_sheets": [str(path) for path in contact_sheets],
        "arguments": vars(args),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"[gen] data.yaml -> {data_yaml}")
    print(f"[gen] stats -> {out_dir / 'stats.json'}")
    for sheet in contact_sheets:
        print(f"[gen] QA contact sheet -> {sheet}")
    if accepted < args.frames:
        print(
            f"[gen] WARNING: accepted only {accepted}/{args.frames} frames before "
            f"the {max_attempts}-attempt limit"
        )


if __name__ == "__main__":
    main()
