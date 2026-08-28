"""Download an RDD2022 India YOLO export from Roboflow Universe.

The API key is read from the prompt or ROBOFLOW_API_KEY. Project identifiers
are arguments because Roboflow Universe project slugs can change over time.
"""

import argparse
import getpass
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", required=True, type=int, help="Roboflow dataset version")
    parser.add_argument("--location", default="data/rdd2022", help="Export directory")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY") or getpass.getpass("Roboflow API key: ")
    if not api_key:
        parser.error("A Roboflow API key is required via ROBOFLOW_API_KEY or the prompt.")

    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise SystemExit("Install the mirror dependency first: python -m pip install roboflow") from error

    location = Path(args.location).resolve()
    location.mkdir(parents=True, exist_ok=True)
    roboflow = Roboflow(api_key=api_key)
    project = roboflow.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=str(location))
    print(f"Downloaded {dataset.name} to {location}")
    print(f"Use: python prepare_dataset.py --source-dir {location}")


if __name__ == "__main__":
    main()
