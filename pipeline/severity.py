import cv2
import numpy as np

def analyze_severity(detections, image_bgr):
    """
    Analyzes the severity and repair priority of detected road damages.
    
    detections: list of dictionaries returned by the detector.
    image_bgr: original OpenCV image.
    
    Returns:
        severity_results: dict containing counts, severity classification,
                          repair priority score, and stretch estimation.
    """
    h, w, _ = image_bgr.shape
    total_area = w * h
    
    critical_count = 0
    medium_count = 0
    small_count = 0
    
    # Calculate texture contrast & severity per detection
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Ensure boxes are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        area_ratio = box_area / total_area
        
        # Calculate texture roughness (standard deviation of grayscale values inside box)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            texture_roughness = float(np.std(gray_crop))
        else:
            texture_roughness = 0.0
            
        # Determine individual severity based on class, size ratio, and texture
        cls = det["class"]
        
        # Scoring logic
        severity_score = 0.0
        # Potholes are inherently more severe than cracks
        if cls == "Pothole":
            severity_score += 0.4
        else:
            severity_score += 0.2
            
        # Size factor
        severity_score += min(area_ratio * 10, 0.4) # Caps size factor at 0.4
        
        # Texture roughness factor
        severity_score += min(texture_roughness / 100, 0.2) # Caps roughness at 0.2
        
        if severity_score >= 0.6 or (cls == "Pothole" and area_ratio > 0.02):
            det["severity"] = "Critical"
            critical_count += 1
        elif severity_score >= 0.35:
            det["severity"] = "Medium"
            medium_count += 1
        else:
            det["severity"] = "Small"
            small_count += 1

    # Calculate overall Repair Priority (1 to 5)
    # 5 = Immediate danger, 1 = Low priority monitoring
    if critical_count > 0:
        if critical_count >= 3 or any(d["class"] == "Pothole" and d["severity"] == "Critical" for d in detections):
            priority = 5  # Critical Emergency
        else:
            priority = 4  # Urgent Repair
    elif medium_count > 0:
        priority = 3  # Normal Schedule
    elif small_count > 1:
        priority = 2  # Routine Maintenance
    else:
        priority = 1  # Monitor Only

    # Estimate stretch length affected (simple heuristic based on spread of boxes)
    if detections:
        xs = [d["bbox"][0] for d in detections] + [d["bbox"][2] for d in detections]
        min_x, max_x = min(xs), max(xs)
        spread_ratio = (max_x - min_x) / w
        estimated_stretch = round(0.5 + spread_ratio * 4.5, 1) # between 0.5m and 5.0m
    else:
        estimated_stretch = 0.0

    return {
        "critical_count": critical_count,
        "medium_count": medium_count,
        "small_count": small_count,
        "repair_priority": priority,
        "estimated_stretch_meters": estimated_stretch,
        "detections": detections
    }
