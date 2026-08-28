"""Measure YOLOv8 inference latency on the selected hardware."""

import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cpu", help="Inference device, such as 'cpu' or '0' for the first CUDA GPU.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="BENCHMARK.md")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")

    model = YOLO(str(Path(args.weights).resolve()))
    model(image, imgsz=args.imgsz, device=args.device, verbose=False)
    timings = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        model(image, imgsz=args.imgsz, device=args.device, verbose=False)
        timings.append((time.perf_counter() - start) * 1000)

    timings_array = np.array(timings)
    output_path = Path(args.output)
    output_path.write_text(
        "# RoadSense AI Inference Benchmark\n\n"
        f"- Run timestamp (UTC): {datetime.now(timezone.utc).isoformat()}\n"
        f"- Checkpoint: `{Path(args.weights).resolve()}`\n"
        f"- Input image: `{Path(args.image).resolve()}`\n"
        f"- Device: `{args.device}`\n"
        f"- Image size: `{args.imgsz}`\n"
        f"- Iterations (after one warm-up): `{args.iterations}`\n"
        f"- Host platform: `{platform.platform()}`\n"
        f"- Mean latency: `{timings_array.mean():.3f} ms`\n"
        f"- Median latency: `{np.median(timings_array):.3f} ms`\n"
        f"- P95 latency: `{np.percentile(timings_array, 95):.3f} ms`\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}: mean={timings_array.mean():.3f} ms")


if __name__ == "__main__":
    main()
