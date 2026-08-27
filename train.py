"""Fine-tune YOLOv8 on a real RDD2022-derived dataset.

The dataset must already be prepared in Ultralytics YOLO format with train and
validation splits. No synthetic images or labels are created by this script.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

EXPECTED_CLASSES = ["Pothole", "Longitudinal Crack", "Transverse Crack"]


def load_dataset_config(path):
    """Validate and return a real Ultralytics dataset configuration."""
    import yaml

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not config.get("train") or not config.get("val"):
        raise ValueError("Dataset YAML must define both train and val splits.")
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    if names != EXPECTED_CLASSES:
        raise ValueError(f"Dataset classes must be exactly {EXPECTED_CLASSES}; got {names}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="RDD2022 YOLO-format dataset YAML")
    parser.add_argument("--base-model", default="yolov8n.yaml", help="YOLO architecture or pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device, or 'cpu'")
    parser.add_argument("--project", default="runs/road_damage")
    parser.add_argument("--name", default="rdd2022_india")
    args = parser.parse_args()

    data_path = load_dataset_config(args.data)
    model = YOLO(args.base_model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    print(f"Training complete. Use {Path(args.project) / args.name / 'weights' / 'best.pt'} for evaluation.")


if __name__ == "__main__":
    main()
