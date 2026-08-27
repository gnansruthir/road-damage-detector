import cv2
import numpy as np

from pipeline.clahe import apply_clahe
from pipeline.detector import RoadDamageDetector
from pipeline.severity import analyze_severity


def test_pipeline_handles_empty_detections(tmp_path):
    image_path = tmp_path / "blank.png"
    cv2.imwrite(str(image_path), np.zeros((300, 300, 3), dtype=np.uint8))

    image = apply_clahe(str(image_path))
    detections = RoadDamageDetector().detect(image)
    metrics = analyze_severity(detections, image)

    assert metrics["detections"] == []