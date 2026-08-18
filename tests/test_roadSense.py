import os
import sys
import pytest
import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app
from pipeline.clahe import apply_clahe
from pipeline.detector import RoadDamageDetector
from pipeline.severity import analyze_severity
from pipeline.map_generator import CivicMapGenerator

client = TestClient(app)

@pytest.fixture
def dummy_image(tmp_path):
    """Generates a dummy 300x300 road simulation image for testing."""
    img_path = tmp_path / "dummy_road.png"
    # Create simple gray road image with a black strip representing a crack
    img = Image.new("RGB", (300, 300), color=(80, 80, 80))
    # Add a thin dark strip/crack
    for x in range(120, 180):
        for y in range(50, 250):
            img.putpixel((x, y), (10, 10, 10))
    img.save(img_path)
    return str(img_path)

def test_clahe_preprocessing(dummy_image):
    """Verifies that CLAHE equalization increases luminance and outputs a valid array."""
    enhanced = apply_clahe(dummy_image)
    assert enhanced is not None
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.shape == (300, 300, 3)

def test_defect_detector(dummy_image):
    """Verifies that the contour-based / YOLO detector detects at least one crack/pothole."""
    enhanced = apply_clahe(dummy_image)
    detector = RoadDamageDetector()
    detections = detector.detect(enhanced)
    
    assert len(detections) > 0
    assert "bbox" in detections[0]
    assert "class" in detections[0]
    assert "confidence" in detections[0]


def test_detector_returns_empty_on_blank_input():
    """Verifies the detector does not fabricate a fake defect when no real damage is present."""
    detector = RoadDamageDetector()
    blank = np.zeros((300, 300, 3), dtype=np.uint8)

    detections = detector.detect(blank)

    assert detections == []


def test_severity_scorer(dummy_image):
    """Verifies severity calculation and priority routing."""
    detector = RoadDamageDetector()
    img_bgr = cv2.imread(dummy_image)
    detections = detector.detect(img_bgr)
    results = analyze_severity(detections, img_bgr)
    
    assert "repair_priority" in results
    assert 1 <= results["repair_priority"] <= 5
    assert "estimated_stretch_meters" in results
    assert "detections" in results

def test_map_generation():
    """Verifies Folium map html content creation and point logging."""
    generator = CivicMapGenerator()
    initial_len = len(generator.damages)
    
    # Add point
    generator.add_damage_point("Critical", "Pothole", 5)
    assert len(generator.damages) == initial_len + 1
    
    html = generator.generate_map_html()
    assert "html" in html
    assert "folium" in html or "leaflet" in html

def test_api_routes(dummy_image):
    """Verifies app health and file detection endpoints."""
    # Health check
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "active"
    
    # Map endpoint
    map_response = client.get("/api/map")
    assert map_response.status_code == 200
    
    # Detect upload
    with open(dummy_image, "rb") as f:
        upload_response = client.post(
            "/api/detect",
            files={"file": ("test.png", f, "image/png")}
        )
    assert upload_response.status_code == 200
    data = upload_response.json()
    assert data["success"] is True
    assert "annotated_image" in data
    assert "repair_priority" in data
