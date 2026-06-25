#!/usr/bin/env python3
"""Train (or fine-tune) a local YOLO playing-card detector.

This is a thin wrapper around ultralytics so the whole thing stays local and
free — no cloud, no API, no tokens.

You need a 52-class playing-card detection dataset in YOLO format. Public
options (search for "playing cards" object-detection datasets) export a
``data.yaml`` plus train/valid/test image+label folders. Point --data at that
yaml. Class names should be card codes like AS, KH, 10D so cardvision can parse
them (see cardvision/mldetect.parse_card_code); otherwise add overrides to
config.ML_CLASS_MAP.

Examples
--------
  # Train from the small YOLO base model:
  python train_yolo.py --data path/to/data.yaml --epochs 80 --model yolov8n.pt

  # After training, copy the best weights where cardvision looks for them:
  #   runs/detect/train/weights/best.pt  ->  models/cards.pt
  python train_yolo.py --data path/to/data.yaml --export models/cards.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to dataset data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="base model to start from")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0 for GPU, 'cpu' for CPU")
    ap.add_argument("--export", default="models/cards.pt",
                    help="copy the best trained weights here when done")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, device=args.device
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists() and args.export:
        Path(args.export).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, args.export)
        print(f"Best weights -> {args.export}")
    else:
        print(f"Training done. Best weights: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
