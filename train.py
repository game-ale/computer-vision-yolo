"""YOLO model training script.

Usage examples
--------------
Train with default settings (requires a data.yaml in the project root):
    python train.py --data data.yaml --epochs 50

Resume from a checkpoint:
    python train.py --data data.yaml --weights runs/train/exp/weights/last.pt --epochs 100

Fine-tune from a pre-trained model:
    python train.py --data data.yaml --weights best.pt --epochs 30 --imgsz 640
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on a custom dataset.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to dataset YAML file (e.g. data.yaml)",
    )
    parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="Starting weights: a pre-trained model name or path to a .pt file",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size (pixels)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 = auto)")
    parser.add_argument(
        "--device",
        default="",
        help="Device to train on: '' (auto), 'cpu', '0', '0,1', …",
    )
    parser.add_argument("--project", default="runs/train", help="Output root directory")
    parser.add_argument("--name", default="exp", help="Output run name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path.resolve()}")

    model = YOLO(args.weights)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
