import os
import uuid
import shutil
import threading
import time
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import cv2
from PIL import Image, UnidentifiedImageError

from pipeline.clahe import apply_clahe
from pipeline.detector import RoadDamageDetector
from pipeline.severity import analyze_severity
from pipeline.map_generator import CivicMapGenerator

# Set up directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "output")
MAP_PATH = os.path.join(BASE_DIR, "static", "output", "live_map.html")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_GENERATED_FILES = 100
FILE_RETENTION_SECONDS = 24 * 60 * 60

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize detector and map
configured_weights = os.getenv("MODEL_WEIGHTS_PATH")
default_weights = os.path.join(BASE_DIR, "weights", "best.pt")
weights_path = configured_weights or (default_weights if os.path.exists(default_weights) else None)
detector = RoadDamageDetector(weights_path=weights_path)
civic_map = CivicMapGenerator()
map_lock = threading.Lock()
# Write initial map file
civic_map.generate_map_html(MAP_PATH)

app = FastAPI(title="RoadSense AI - Civic Detection & Mapping Server")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "CORS_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",") if origin.strip()],
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

    content_length = file.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit.")

    file_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 10 MB limit.")

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    cleanup_generated_files()

    # Save input image
    try:
        with open(input_path, "wb") as buffer:
            buffer.write(file_bytes)
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
        if metrics["critical_count"] > 0:
            map_severity = "Critical"
        elif metrics["medium_count"] > 0:
            map_severity = "Medium"
        else:
            map_severity = "Small"

        if metrics["detections"]:
            target_finding = next(
                det for det in metrics["detections"]
                if det["severity"] == map_severity
            )
            with map_lock:
                lat, lng = civic_map.add_damage_point(
                    severity=map_severity,
                    class_name=target_finding["class"],
                    priority=metrics["repair_priority"]
                )
                civic_map.generate_map_html(MAP_PATH)
        else:
            lat, lng = None, None

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
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


def cleanup_generated_files():
    """Remove expired generated files and enforce a bounded output directory."""
    now = time.time()
    candidates = []
    for directory in (UPLOAD_DIR, OUTPUT_DIR):
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if name in {".gitkeep", "live_map.html"} or not os.path.isfile(path):
                continue
            candidates.append((os.path.getmtime(path), path))

    for modified, path in candidates:
        if now - modified > FILE_RETENTION_SECONDS:
            try:
                os.remove(path)
            except OSError:
                pass

    remaining = sorted(
        ((os.path.getmtime(path), path) for _, path in candidates if os.path.exists(path)),
        reverse=True
    )
    for _, path in remaining[MAX_GENERATED_FILES:]:
        try:
            os.remove(path)
        except OSError:
            pass

@app.get("/api/map", response_class=HTMLResponse)
def get_map():
    """Returns the generated Folium HTML map."""
    with map_lock:
        if os.path.exists(MAP_PATH):
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                return f.read()
    return "<h3>Map generating...</h3>"

@app.get("/api/health")
def health():
    with map_lock:
        point_count = len(civic_map.damages)
    return {"status": "active", "database_points": point_count}
