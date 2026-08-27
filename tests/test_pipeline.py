import cv2
import numpy as np

from pipeline.clahe import apply_clahe
from pipeline.detector import RoadDamageDetector
from pipeline.map_generator import CivicMapGenerator
from pipeline.severity import analyze_severity


def test_pipeline_handles_empty_detections(tmp_path):
    image_path = tmp_path / "blank.png"
    cv2.imwrite(str(image_path), np.zeros((300, 300, 3), dtype=np.uint8))

    image = apply_clahe(str(image_path))
    detections = RoadDamageDetector().detect(image)
    metrics = analyze_severity(detections, image)
    severity = (
        "Critical" if metrics["critical_count"] > 0
        else "Medium" if metrics["medium_count"] > 0
        else "Small"
    )
    target = next(
        (d for d in detections if d.get("severity") == severity),
        {"class": "No defect"}
    )

    map_generator = CivicMapGenerator()
    map_generator.add_damage_point(severity, target["class"], metrics["repair_priority"])

    assert metrics["detections"] == []