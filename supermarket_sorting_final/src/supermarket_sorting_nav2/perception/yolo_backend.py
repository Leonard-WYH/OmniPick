#!/usr/bin/env python3
"""YOLO inference backend with TensorRT preference and PyTorch fallback.

中文说明：负责权重加载、设备选择和单帧推理。``products.engine`` 只能在
CUDA 上运行；若引擎初始化或首次推理失败，会自动回退到同目录的
``products.pt``。这里没有定时器或人为时延，实际处理周期由图像到达速度和
一次 YOLO 推理耗时决定。
"""

from pathlib import Path
from typing import Any

import numpy as np


class YoloBackend:
    """Load an Ultralytics PyTorch model or exported TensorRT engine."""

    def __init__(self, weights: Path, confidence: float = 0.65, device: str = "auto"):
        self.confidence = confidence
        self.model: Any = None
        self.device = device
        self.class_names: list[str] = []
        self.backend = ""
        self.active_weights = Path(weights)
        self._fallback_used = False

        requested_weights = Path(weights)
        if not requested_weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {requested_weights}")

        import torch

        self._torch = torch
        self.selected_device = self._select_device(torch, device)
        self._prediction_device: int | str = (
            0 if self.selected_device.type == "cuda" else "cpu"
        )
        self.fallback_weights = (
            requested_weights.with_suffix(".pt")
            if requested_weights.suffix.lower() == ".engine"
            else None
        )

        candidate = requested_weights
        if candidate.suffix.lower() == ".engine" and self.selected_device.type != "cuda":
            candidate = self._require_fallback(
                "TensorRT requires CUDA, but CUDA is unavailable"
            )

        try:
            self._load(candidate)
        except Exception as exc:
            if candidate.suffix.lower() != ".engine":
                raise
            fallback = self._require_fallback(
                f"TensorRT engine initialization failed: {exc}"
            )
            self._load(fallback)

    def _require_fallback(self, reason: str) -> Path:
        if self.fallback_weights is None or not self.fallback_weights.is_file():
            raise RuntimeError(
                f"{reason}; PyTorch fallback weights are unavailable: "
                f"{self.fallback_weights}"
            )
        self._fallback_used = True
        print(
            f"[YoloBackend] {reason}; falling back to {self.fallback_weights}"
        )
        return self.fallback_weights

    def _load(self, weights: Path) -> None:
        from ultralytics import YOLO

        suffix = weights.suffix.lower()
        if suffix == ".engine":
            # TensorRT engines are already placed on the CUDA device chosen at
            # export time. Calling .to() or model.eval() is invalid here.
            model = YOLO(str(weights), task="detect")
            raw_names = model.names  # Force engine deserialization now.
            backend = "tensorrt"
        elif suffix == ".pt":
            original_load = self._torch.load

            def compatible_load(*args, **kwargs):
                # products.pt contains a complete Ultralytics checkpoint.
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            self._torch.load = compatible_load
            try:
                model = YOLO(str(weights)).to(self.selected_device)
                model.model.eval()
                raw_names = model.names
            finally:
                self._torch.load = original_load
            backend = "pytorch"
        else:
            raise ValueError(
                f"unsupported YOLO weights extension {suffix!r}; use .engine or .pt"
            )

        self.model = model
        self.backend = backend
        self.active_weights = weights
        if isinstance(raw_names, dict):
            self.class_names = [
                str(raw_names[index]) for index in sorted(raw_names, key=int)
            ]
        else:
            self.class_names = [str(name) for name in raw_names]

        print(
            f"[YoloBackend] loaded {weights} with {backend} on "
            f"{self.selected_device}; classes={self.class_names}"
        )

    @staticmethod
    def _select_device(torch, requested: str):
        requested = requested.lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("YOLO device must be auto, cpu, or cuda")
        if requested == "cpu":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            if requested == "cuda":
                raise RuntimeError("CUDA was requested but is unavailable")
            print("[YoloBackend] CUDA unavailable; using CPU")
            return torch.device("cpu")

        major, minor = torch.cuda.get_device_capability(0)
        capability = major * 10 + minor
        supported = [
            int(arch[3:])
            for arch in torch.cuda.get_arch_list()
            if arch.startswith("sm_")
        ]
        compatible = any(
            arch // 10 == major and arch % 10 <= minor for arch in supported
        )
        if compatible:
            return torch.device("cuda:0")
        if requested == "cuda":
            raise RuntimeError(
                f"GPU sm_{capability} is unsupported by this PyTorch build: {supported}"
            )
        print(
            f"[YoloBackend] GPU sm_{capability} is unsupported by this PyTorch "
            f"build ({supported}); using CPU"
        )
        return torch.device("cpu")

    def _predict(self, rgb: np.ndarray):
        return self.model.predict(
            source=rgb,
            conf=self.confidence,
            device=self._prediction_device,
            verbose=False,
        )[0]

    def detect(self, rgb: np.ndarray) -> list[dict]:
        try:
            results = self._predict(rgb)
        except Exception as exc:
            if self.backend != "tensorrt" or self._fallback_used:
                raise
            fallback = self._require_fallback(
                f"TensorRT engine inference failed: {exc}"
            )
            self._load(fallback)
            results = self._predict(rgb)

        detections = []
        for box in results.boxes:
            confidence = float(box.conf.item())
            if confidence < self.confidence:
                continue
            class_id = int(box.cls.item())
            if class_id < 0 or class_id >= len(self.class_names):
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            detections.append(
                {
                    "class_id": class_id,
                    "class": self.class_names[class_id],
                    "x": (x0 + x1) // 2,
                    "y": (y0 + y1) // 2,
                    "w": x1 - x0,
                    "h": y1 - y0,
                    "xyxy": (x0, y0, x1, y1),
                    "conf": confidence,
                }
            )
        return detections
