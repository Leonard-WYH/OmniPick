#!/usr/bin/env python3
"""Fine-tune YOLOv8s on the 9-class product dataset.

Runs INSIDE the Docker container (needs ultralytics + a GPU).  Expects the
dataset produced by gen_dataset.py at perception/dataset/.  After training it
copies the best weights to perception/checkpoints/products.pt and prints the
validation mAP so the caller can sanity-check the run.

Run (inside container):
    cd examples/supermarket_sorting
    python3 perception/train_yolo.py --epochs 100
"""
import os
import argparse
import shutil
from pathlib import Path

PERCEPTION_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PERCEPTION_DIR / "dataset" / "data.yaml"
CKPT_DIR = PERCEPTION_DIR / "checkpoints"
FINAL_CKPT = CKPT_DIR / "products.pt"


def _resolve_weights(requested: str) -> str:
    """Return an explicit local checkpoint or an Ultralytics model identifier."""
    from pathlib import Path as _P
    explicit = _P(requested)
    if explicit.is_file():
        print(f"[train] using weights: {explicit}")
        return str(explicit)

    # Let Ultralytics use/download its standard pretrained model when it is not
    # already cached. Training outputs are generated artifacts and are not kept
    # as duplicate repository weights.
    if requested == "yolov8s.pt":
        # Also check common cache directories
        candidates = [
            _P.home() / ".cache/ultralytics/yolov8s.pt",
            _P("/workspace") / "yolov8s.pt",
            _P("/tmp/yolov8s.pt"),
        ]
        for c in candidates:
            if c.is_file():
                print(f"[train] found yolov8s.pt at {c}")
                return str(c)

        print("[train] yolov8s.pt is not cached; Ultralytics will download it")
        return requested

    raise FileNotFoundError(
        f"Weights '{requested}' not found. Provide a valid local checkpoint or "
        "use the default yolov8s.pt model identifier."
    )


def main():
    ap = argparse.ArgumentParser(description="fine-tune YOLOv8s for kele")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="local checkpoint or Ultralytics model identifier")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", default=-1,
                    help="batch size (-1 = ultralytics auto)")
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop patience (epochs)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--amp", default="false",
                    choices=["true", "false", "1", "0", "yes", "no", "on", "off"],
                    help="enable AMP training checks; false avoids downloading yolov8n.pt")
    ap.add_argument("--project", default=str(PERCEPTION_DIR / "runs"))
    ap.add_argument("--name", default="products_yolov8s")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    # ultralytics 8.0.196 + torch>=2.6 needs weights_only=False for old ckpts
    _orig = torch.load
    def _compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig(*a, **kw)
    torch.load = _compat

    weights = _resolve_weights(args.weights)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    metrics = None
    try:
        batch = int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch
        model = YOLO(weights)
        amp = str(args.amp).lower() in {"true", "1", "yes", "on"}
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=batch,
            patience=args.patience,
            device=args.device,
            amp=amp,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )
        # Free GPU before validation so server container doesn't cause OOM
        import torch as _torch
        _torch.cuda.empty_cache()
        metrics = model.val()
    except Exception as exc:
        print(f"[train] WARNING during val(): {exc}")
    finally:
        torch.load = _orig
        # Always copy best.pt regardless of val() outcome
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        if best.is_file():
            shutil.copy2(best, FINAL_CKPT)
            print(f"[train] copied {best} -> {FINAL_CKPT}")
        else:
            print(f"[train] WARNING: best.pt not found at {best}")

    try:
        if metrics is not None:
            print(f"[train] val mAP50-95 = {metrics.box.map:.4f}")
            print(f"[train] val mAP50    = {metrics.box.map50:.4f}")
    except Exception as e:
        print(f"[train] could not read metrics: {e}")


if __name__ == "__main__":
    main()
