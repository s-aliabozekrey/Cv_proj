"""Train a YOLO detector for the bowling-pin dataset.

This module can be run directly to fine-tune a pretrained YOLO model on the
Roboflow dataset shipped with the project.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "Bowling Pin Detection.v1i.yolov12" / "data.yaml"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "yolo12n.pt"
DEFAULT_OUTPUT_WEIGHTS = PROJECT_ROOT / "models" / "yolo_weights.pt"


def train_bowling_pin_model(
    data_yaml: Path = DATASET_YAML,
    base_model: Path = DEFAULT_BASE_MODEL,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    project: Path | None = None,
    name: str = "bowling_pin_detector",
    output_weights: Path = DEFAULT_OUTPUT_WEIGHTS,
):
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")

    model = YOLO(str(base_model))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project or (PROJECT_ROOT / "runs" / "detect")),
        name=name,
        exist_ok=True,
    )

    best_weights = Path(model.trainer.best) if model.trainer and getattr(model.trainer, "best", None) else None
    if best_weights is not None and best_weights.exists():
        output_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, output_weights)

    return results, output_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the bowling-pin dataset.")
    parser.add_argument("--data", type=Path, default=DATASET_YAML, help="Path to the dataset YAML file.")
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL, help="Path to the pretrained YOLO base model.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--device", type=str, default=None, help="Training device, for example cpu, 0, or 0,1.")
    parser.add_argument("--project", type=Path, default=None, help="Ultralytics project directory.")
    parser.add_argument("--name", type=str, default="bowling_pin_detector", help="Run name for Ultralytics outputs.")
    parser.add_argument("--output-weights", type=Path, default=DEFAULT_OUTPUT_WEIGHTS, help="Where to copy the trained best.pt file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_bowling_pin_model(
        data_yaml=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        output_weights=args.output_weights,
    )


if __name__ == "__main__":
	main()
