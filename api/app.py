import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import cv2

from pipeline.clahe import apply_clahe
from pipeline.detector import RoadDamageDetector
from pipeline.severity import analyze_severity
from pipeline.map_generator import CivicMapGenerator

# Set up directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "output")
MAP_PATH = os.path.join(BASE_DIR, "static", "output", "live_map.html")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize detector and map
detector = RoadDamageDetector()
civic_map = CivicMapGenerator()
# Write initial map file
civic_map.generate_map_html(MAP_PATH)

app = FastAPI(title="RoadSense AI - Civic Detection & Mapping Server")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static directory
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

@app.post("/api/detect")
async def detect_damage(file: UploadFile = File(...)):
    """
    Receives upload, processes image using CLAHE and YOLO/OpenCV pipelines,
    extracts severity, plots it to Folium map, and returns metrics.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    # Save input image
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
        
    try:
        # 1. Apply CLAHE Preprocessing
        enhanced_image = apply_clahe(input_path)
        
        # Save CLAHE output for comparison if needed
        clahe_path = os.path.join(OUTPUT_DIR, f"clahe_{file_id}{ext}")
        cv2.imwrite(clahe_path, enhanced_image)
        
        # 2. Run Road Damage Detection (YOLO / OpenCV)
        detections = detector.detect(enhanced_image)
        
        # 3. Analyze severity & calculate repair priority
        metrics = analyze_severity(detections, enhanced_image)
        
        # 4. Annotate image with color-coded bounding boxes matching severity
        annotated_image = enhanced_image.copy()
        for det in metrics["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            color = (82, 172, 76) # Green (BGR)
            if det["severity"] == "Critical":
                color = (85, 0, 255) # Red (BGR)
            elif det["severity"] == "Medium":
                color = (35, 107, 255) # Electric Orange (BGR)
                
            # Draw rectangle
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            # Label
            label = f"{det['class']} ({det['severity']})"
            cv2.putText(annotated_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        annotated_path = os.path.join(OUTPUT_DIR, f"annotated_{file_id}{ext}")
        cv2.imwrite(annotated_path, annotated_image)
        
        # 5. Dynamic GPS Placement and map rewrite
        target_finding = detections[0] if detections else {"class": "Pothole", "severity": "Small"}
        lat, lng = civic_map.add_damage_point(
            severity=metrics["critical_count"] > 0 and "Critical" or metrics["medium_count"] > 0 and "Medium" or "Small",
            class_name=target_finding["class"],
            priority=metrics["repair_priority"]
        )
        
        # Save updated map
        civic_map.generate_map_html(MAP_PATH)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {e}")
        
    return {
        "success": True,
        "original_image": f"/static/uploads/{file_id}{ext}",
        "clahe_image": f"/static/output/clahe_{file_id}{ext}",
        "annotated_image": f"/static/output/annotated_{file_id}{ext}",
        "detections_count": len(detections),
        "critical_count": metrics["critical_count"],
        "medium_count": metrics["medium_count"],
        "small_count": metrics["small_count"],
        "repair_priority": metrics["repair_priority"],
        "estimated_stretch": metrics["estimated_stretch_meters"],
        "gps": {"lat": lat, "lng": lng}
    }

@app.get("/api/map", response_class=HTMLResponse)
def get_map():
    """Returns the generated Folium HTML map."""
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Map generating...</h3>"

@app.get("/api/health")
def health():
    return {"status": "active", "database_points": len(civic_map.damages)}
