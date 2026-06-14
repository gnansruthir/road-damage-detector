import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Ensure base directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.clahe import apply_clahe
from pipeline.detector import RoadDamageDetector
from pipeline.severity import analyze_severity
from pipeline.map_generator import CivicMapGenerator

def main():
    print("=== RoadSense AI Pipeline Validation ===")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(base_dir, "samples", "sample_road.png")
    output_dir = os.path.join(base_dir, "static", "output")
    map_output_path = os.path.join(output_dir, "live_map.html")
    
    os.makedirs(os.path.join(base_dir, "samples"), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create a dummy road scan image
    print("Generating simulated road scan image...")
    # Base dark gray asphalt texture
    img = Image.new("RGB", (640, 480), color=(60, 60, 65))
    draw = ImageDraw.Draw(img)
    # Draw a yellow lane divider marker
    draw.line([320, 0, 320, 480], fill=(220, 180, 20), width=8)
    
    # Draw simulated pothole (dark irregular oval)
    draw.ellipse([150, 200, 220, 260], fill=(20, 20, 22))
    # Draw simulated crack (jagged lines)
    draw.line([400, 100, 430, 200], fill=(25, 25, 27), width=3)
    draw.line([430, 200, 410, 320], fill=(25, 25, 27), width=3)
    
    img.save(sample_path)
    print(f"Sample road saved at: {sample_path}")
    
    # 2. Run CLAHE contrast enhancement
    print("\nRunning CLAHE night-mode/shadow contrast enhancement...")
    enhanced_img = apply_clahe(sample_path)
    enhanced_path = os.path.join(output_dir, "enhanced_sample.png")
    cv2.imwrite(enhanced_path, enhanced_img)
    print(f"Enhanced image saved at: {enhanced_path}")
    
    # 3. Initialize Detector & Predict
    print("\nInitializing YOLOv8 / CV damage detector...")
    detector = RoadDamageDetector()
    detections = detector.detect(enhanced_img)
    print(f"Detected {len(detections)} defect segments.")
    for idx, d in enumerate(detections):
        print(f"  [{idx + 1}] Class: {d['class']} | Conf: {d['confidence']:.2f} | BBox: {d['bbox']}")
        
    # 4. Severity Scoring Engine
    print("\nExecuting Severity & Priority scoring engine...")
    metrics = analyze_severity(detections, enhanced_img)
    print("--- SCORING METRICS ---")
    print(f"Critical Defect Count: {metrics['critical_count']}")
    print(f"Medium Defect Count: {metrics['medium_count']}")
    print(f"Small Defect Count: {metrics['small_count']}")
    print(f"Calculated Repair Priority: {metrics['repair_priority']}/5")
    print(f"Estimated Affected Stretch: {metrics['estimated_stretch_meters']} meters")
    
    # 5. Live GPS map overlay
    print("\nUpdating Civic GPS Map & Heatmap overlays...")
    map_gen = CivicMapGenerator()
    lat, lng = map_gen.add_damage_point(
        severity="Critical" if metrics["critical_count"] > 0 else "Medium" if metrics["medium_count"] > 0 else "Small",
        class_name=detections[0]["class"],
        priority=metrics["repair_priority"]
    )
    print(f"Pushed GPS coordinate: [{lat:.6f}, {lng:.6f}]")
    map_gen.generate_map_html(map_output_path)
    print(f"Interactive Folium map updated at: {map_output_path}")
    print("\n=== Validation Completed Successfully ===")

if __name__ == "__main__":
    main()
