"""Evaluate a trained YOLOv8 road-damage checkpoint on the held-out split."""

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--data", required=True, help="Same dataset YAML used for training")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="RESULTS.md")
    parser.add_argument("--append", action="store_true", help="Append this evaluation to an existing Markdown report.")
    args = parser.parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    results_path = Path(args.output)
    report_prefix = "" if args.append and results_path.exists() else "# RoadSense AI Evaluation Results\n\n"
    report = report_prefix + (
        f"- Run timestamp (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"- Checkpoint: `{weights}`\n"
        f"- Dataset config: `{Path(args.data).resolve()}`\n"
        f"- Evaluated split: `{args.split}`\n"
        f"- Image size: `{args.imgsz}`\n"
        f"- Device: `{args.device}`\n"
        f"- Host platform: `{platform.platform()}`\n"
        f"- mAP@0.50: `{map50:.6f}`\n"
        f"- mAP@0.50:0.95: `{map5095:.6f}`\n\n"
        "These values were produced by this command and are not a claim about any\n"
        "other dataset, split, hardware, or training run.\n",
    )
    if args.append and results_path.exists():
        report = "\n## Evaluation Run\n\n" + report
    results_path.write_text(report, encoding="utf-8")
    print(f"Wrote {results_path}: mAP50={map50:.6f}, mAP50-95={map5095:.6f}")


if __name__ == "__main__":
    main()
