#!/usr/bin/env python3
"""Evaluate the trained 9-class YOLO model and create visual comparisons.

The default run performs two checks:

1. Evaluate the complete validation split and save the Ultralytics plots.
2. Randomly sample validation images and save ground-truth/prediction comparisons.

Run this script through ``~/OmniPick/YOLO_training/test_yolo.sh`` so
that it uses the same Docker image and dependencies as training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


PERCEPTION_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PERCEPTION_DIR / "checkpoints" / "products.pt"
DEFAULT_DATA = PERCEPTION_DIR / "dataset" / "data.yaml"
DEFAULT_OUTPUT = PERCEPTION_DIR / "runs" / "products_9class_test"
EXPECTED_NAMES = [
    "sanmingzhi",
    "heweidao",
    "shupian",
    "zhijin",
    "maidong",
    "kouxiangtang",
    "pingguo",
    "chengzi",
    "kele",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BGR colours used consistently on ground-truth and prediction images.
CLASS_COLOURS = [
    (230, 110, 40),
    (60, 180, 75),
    (255, 210, 45),
    (80, 90, 230),
    (210, 90, 210),
    (220, 170, 55),
    (45, 190, 225),
    (150, 95, 220),
    (70, 220, 150),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate products.pt and generate GT/prediction comparison images"
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device such as 0, or cpu")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="confidence threshold for the sampled prediction images",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="sample both v0 originals and v1 appearance augmentations (default: v0 only)",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="skip full validation and only generate sampled visual comparisons",
    )
    return parser.parse_args()


def normalise_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=int)]
    return [str(name) for name in raw_names]


def load_dataset(data_path: Path, split: str) -> tuple[dict[str, Any], Path, list[Path]]:
    with data_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    if not root.exists() and (data_path.parent / "images").is_dir():
        # data.yaml contains the in-container absolute path. This fallback also
        # keeps the script usable when called directly from a compatible host.
        root = data_path.parent.resolve()

    split_value = config.get(split)
    if not split_value:
        raise ValueError(f"dataset YAML has no '{split}' entry: {data_path}")
    sources = split_value if isinstance(split_value, list) else [split_value]

    images: list[Path] = []
    for source_value in sources:
        source = Path(source_value)
        if not source.is_absolute():
            source = root / source
        if source.is_dir():
            images.extend(
                path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                candidate = Path(line.strip())
                if not candidate.is_absolute():
                    candidate = root / candidate
                if candidate.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(candidate)
        elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source)

    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f"no images found for split '{split}' under {root}")
    return config, root, images


def label_path_for(image_path: Path, dataset_root: Path) -> Path:
    try:
        parts = list(image_path.resolve().relative_to(dataset_root.resolve()).parts)
        image_index = parts.index("images")
        parts[image_index] = "labels"
        return (dataset_root / Path(*parts)).with_suffix(".txt")
    except (ValueError, OSError):
        return image_path.parent.parent.parent / "labels" / image_path.parent.name / (
            image_path.stem + ".txt"
        )


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    if not label_path.is_file():
        return labels
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            print(f"[test] warning: ignoring malformed {label_path}:{line_number}")
            continue
        class_id = int(float(fields[0]))
        cx, cy, box_width, box_height = map(float, fields[1:])
        x1 = int(round((cx - box_width / 2.0) * width))
        y1 = int(round((cy - box_height / 2.0) * height))
        x2 = int(round((cx + box_width / 2.0) * width))
        y2 = int(round((cy + box_height / 2.0) * height))
        labels.append(
            {
                "class_id": class_id,
                "xyxy": [
                    max(0, min(width - 1, x1)),
                    max(0, min(height - 1, y1)),
                    max(0, min(width - 1, x2)),
                    max(0, min(height - 1, y2)),
                ],
            }
        )
    return labels


def draw_box(
    image: np.ndarray,
    xyxy: Iterable[float],
    text: str,
    colour: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
    height, width = image.shape[:2]
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
    thickness = max(2, round(min(width, height) / 320))
    font_scale = max(0.45, min(width, height) / 1000.0)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y1 = max(0, y1 - text_height - baseline - 5)
    label_x2 = min(width - 1, x1 + text_width + 6)
    cv2.rectangle(image, (x1, label_y1), (label_x2, y1), colour, -1)
    cv2.putText(
        image,
        text,
        (x1 + 3, max(text_height + 1, y1 - baseline - 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (15, 15, 15),
        thickness,
        cv2.LINE_AA,
    )


def add_header(image: np.ndarray, text: str) -> np.ndarray:
    header_height = max(34, round(image.shape[0] * 0.065))
    output = cv2.copyMakeBorder(
        image, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28)
    )
    cv2.putText(
        output,
        text,
        (12, round(header_height * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, header_height / 52.0),
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def fit_to_tile(image: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    canvas = np.full((tile_height, tile_width, 3), 24, dtype=np.uint8)
    scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x_offset = (tile_width - resized.shape[1]) // 2
    y_offset = (tile_height - resized.shape[0]) // 2
    canvas[
        y_offset : y_offset + resized.shape[0], x_offset : x_offset + resized.shape[1]
    ] = resized
    return canvas


def create_contact_sheets(comparisons: list[Path], output_dir: Path) -> list[Path]:
    sheet_paths: list[Path] = []
    rows, columns = 3, 2
    tile_width, tile_height = 700, 310
    per_sheet = rows * columns
    for sheet_index, start in enumerate(range(0, len(comparisons), per_sheet)):
        canvas = np.full(
            (rows * tile_height, columns * tile_width, 3), 24, dtype=np.uint8
        )
        for offset, comparison_path in enumerate(comparisons[start : start + per_sheet]):
            image = cv2.imread(str(comparison_path))
            if image is None:
                continue
            tile = fit_to_tile(image, tile_width, tile_height)
            row, column = divmod(offset, columns)
            canvas[
                row * tile_height : (row + 1) * tile_height,
                column * tile_width : (column + 1) * tile_width,
            ] = tile
        sheet_path = output_dir / f"contact_sheet_{sheet_index:02d}.jpg"
        cv2.imwrite(str(sheet_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        sheet_paths.append(sheet_path)
    return sheet_paths


def scalar(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def run_validation(
    model: Any,
    args: argparse.Namespace,
    names: list[str],
    output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    print(f"[test] validating the complete '{args.split}' split ...")
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name="validation",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    box = metrics.box
    overall = {
        "precision": scalar(box.mp),
        "recall": scalar(box.mr),
        "map50": scalar(box.map50),
        "map75": scalar(box.map75),
        "map50_95": scalar(box.map),
    }

    available_class_ids = [int(class_id) for class_id in box.ap_class_index]
    per_class: list[dict[str, Any]] = []
    for class_id, name in enumerate(names):
        row: dict[str, Any] = {"class_id": class_id, "class_name": name}
        if class_id in available_class_ids:
            metric_index = available_class_ids.index(class_id)
            precision, recall, map50, map50_95 = box.class_result(metric_index)
            row.update(
                {
                    "precision": scalar(precision),
                    "recall": scalar(recall),
                    "map50": scalar(map50),
                    "map50_95": scalar(map50_95),
                }
            )
        else:
            row.update(
                {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}
            )
        per_class.append(row)
    return overall, per_class


def write_metrics(
    output_dir: Path,
    weights: Path,
    data: Path,
    split: str,
    names: list[str],
    overall: dict[str, float] | None,
    per_class: list[dict[str, Any]],
    sample_records: list[dict[str, Any]],
) -> None:
    summary = {
        "weights": str(weights.resolve()),
        "data": str(data.resolve()),
        "split": split,
        "classes": names,
        "overall": overall,
        "per_class": per_class,
        "sample_predictions": sample_records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if per_class:
        with (output_dir / "per_class_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_class[0]))
            writer.writeheader()
            writer.writerows(per_class)


def generate_comparisons(
    model: Any,
    images: list[Path],
    dataset_root: Path,
    names: list[str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    candidates = images
    if not args.all_variants:
        originals = [path for path in images if path.stem.endswith("_v0")]
        if originals:
            candidates = originals

    sample_count = min(max(0, args.samples), len(candidates))
    selected = random.Random(args.seed).sample(candidates, sample_count)
    selected.sort()
    if not selected:
        return [], []

    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[test] predicting {len(selected)} sampled images "
        f"(conf={args.conf:.3f}, iou={args.iou:.3f}) ..."
    )
    results = model.predict(
        source=[str(path) for path in selected],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    comparison_paths: list[Path] = []
    sample_records: list[dict[str, Any]] = []
    for image_path, result in zip(selected, results):
        original = result.orig_img.copy()
        height, width = original.shape[:2]
        ground_truth = read_yolo_labels(
            label_path_for(image_path, dataset_root), width, height
        )
        gt_image = original.copy()
        prediction_image = original.copy()
        gt_counts: Counter[str] = Counter()
        prediction_counts: Counter[str] = Counter()

        for label in ground_truth:
            class_id = label["class_id"]
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            gt_counts[class_name] += 1
            colour = CLASS_COLOURS[class_id % len(CLASS_COLOURS)]
            draw_box(gt_image, label["xyxy"], f"GT {class_name}", colour)

        predictions: list[dict[str, Any]] = []
        if result.boxes is not None:
            xyxy_values = result.boxes.xyxy.detach().cpu().numpy()
            class_values = result.boxes.cls.detach().cpu().numpy().astype(int)
            conf_values = result.boxes.conf.detach().cpu().numpy()
            for xyxy, class_id, confidence in zip(
                xyxy_values, class_values, conf_values
            ):
                class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
                prediction_counts[class_name] += 1
                colour = CLASS_COLOURS[class_id % len(CLASS_COLOURS)]
                draw_box(
                    prediction_image,
                    xyxy,
                    f"P {class_name} {confidence:.2f}",
                    colour,
                )
                predictions.append(
                    {
                        "class_id": int(class_id),
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "xyxy": [float(value) for value in xyxy],
                    }
                )

        gt_panel = add_header(gt_image, f"GROUND TRUTH | {image_path.name}")
        prediction_panel = add_header(
            prediction_image, f"PREDICTION | conf >= {args.conf:.2f}"
        )
        comparison = np.hstack((gt_panel, prediction_panel))
        comparison_path = comparisons_dir / image_path.name
        cv2.imwrite(
            str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        comparison_paths.append(comparison_path)
        sample_records.append(
            {
                "image": str(image_path),
                "ground_truth_count": len(ground_truth),
                "prediction_count": len(predictions),
                "ground_truth_by_class": dict(sorted(gt_counts.items())),
                "prediction_by_class": dict(sorted(prediction_counts.items())),
                "predictions": predictions,
            }
        )
    return comparison_paths, sample_records


def main() -> int:
    args = parse_args()
    args.weights = args.weights.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.weights.is_file():
        raise FileNotFoundError(f"weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset YAML not found: {args.data}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be in [0, 1]")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("--iou must be in [0, 1]")

    args.output.mkdir(parents=True, exist_ok=True)
    _, dataset_root, images = load_dataset(args.data, args.split)

    import torch
    from ultralytics import YOLO

    # Ultralytics 8.0.196 checkpoints need weights_only=False with torch>=2.6.
    original_torch_load = torch.load

    def compatible_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = compatible_torch_load
    try:
        print(f"[test] weights: {args.weights}")
        print(f"[test] data:    {args.data}")
        print(f"[test] output:  {args.output}")
        model = YOLO(str(args.weights))
        names = normalise_names(model.names)
        print(f"[test] model classes ({len(names)}): {names}")
        if names != EXPECTED_NAMES:
            raise RuntimeError(
                "unexpected model classes/order; expected "
                f"{EXPECTED_NAMES}, got {names}"
            )

        overall: dict[str, float] | None = None
        per_class: list[dict[str, Any]] = []
        if not args.skip_val:
            overall, per_class = run_validation(
                model, args, names, args.output
            )

        comparison_paths, sample_records = generate_comparisons(
            model, images, dataset_root, names, args, args.output
        )
        sheet_paths = create_contact_sheets(comparison_paths, args.output)
        write_metrics(
            args.output,
            args.weights,
            args.data,
            args.split,
            names,
            overall,
            per_class,
            sample_records,
        )
    finally:
        torch.load = original_torch_load

    if overall:
        print(
            "[test] overall: "
            f"P={overall['precision']:.4f}, R={overall['recall']:.4f}, "
            f"mAP50={overall['map50']:.4f}, mAP50-95={overall['map50_95']:.4f}"
        )
    print(f"[test] summary:       {args.output / 'summary.json'}")
    if per_class:
        print(f"[test] class metrics: {args.output / 'per_class_metrics.csv'}")
    print(f"[test] comparisons:   {args.output / 'comparisons'}")
    for sheet_path in sheet_paths:
        print(f"[test] contact sheet: {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
