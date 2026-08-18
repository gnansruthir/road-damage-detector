# RoadSense AI — Road Damage Severity & Civic Intelligence

RoadSense AI is an urban operations dashboard designed to monitor road network health in real time. It combines OpenCV-based crack and pothole detection with optional YOLOv8 inference support for a real fine-tuned checkpoint when one is available, plus CLAHE preprocessing, a geometric-texture severity analyzer, and an interactive Folium GPS map dashboard.

---

## Key Features
- **YOLOv8 Damage Detection**: Supports custom road-damage checkpoints when a trained model is supplied. No fine-tuned RDD2022 checkpoint is included in this repository yet.
- **CV-Heuristic Fallback**: Uses contour analysis and edge detection to identify potholes and cracks from visual structure without relying on unrelated COCO labels or fabricated confidence scores.
- **CLAHE Low-Light Fix**: An OpenCV preprocessing pipeline that normalizes contrast under night or heavy shadow conditions to improve visibility before defect detection.
- **Severity Scoring Engine**: Calculates damage scale relative to the lane width and analyzes texture roughness (using grayscale standard deviation) to grade repairs as *Small*, *Medium*, or *Critical*.
- **Repair Priority Telemetry**: Assigns a civic repair priority index from `1` (Monitor) to `5` (Immediate Emergency) and estimates the affected stretch.
- **Folium GPS Heatmap**: Renders an interactive Leaflet map using dark CartoDB tiles. Integrates a density heatmap and drops dynamic, color-coded status pins.
- **Urban Command Center UI**: Near-black theme (`#060608`, `#0D0D12`) featuring isometric grid layouts, rain effects, 3-column live demo consoles, and real-time civic statistics.

---

## Pipeline Flow

```
                      [ Road Scan Image / Frame ]
                                   │
                         ┌─────────▼─────────┐
                         │ CLAHE Enhancer    │ (OpenCV contrast correction)
                         └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ YOLOv8 Detector   │ (Pothole & crack boundaries)
                         └─────────┬─────────┘
                                   │
                         ┌─────────▼─────────┐
                         │ Scoring Engine    │ (Severity & Priority index 1-5)
                         └─────────┬─────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   [ Annotated Output Image ]                [ Folium GPS Heatmap Update ]
```

---

## Quick Start (Local Run)

### 1. Installation
Install core packages (includes YOLOv8, OpenCV, FastAPI, and Folium):
```powershell
pip install -r requirements.txt
```

### 2. Launch Console
Start the FastAPI uvicorn daemon:
```powershell
python -m uvicorn api.app:app --reload
```
Access the dashboard in your web browser: **`http://127.0.0.1:8000`**

### 3. Run Automated Tests
Execute the pytest suite (verifies CLAHE matrices, detector fallbacks, scoring logic, and api routes):
```powershell
pytest tests/
```

---

## Deployment (Render with Docker)
This project is configured with a size-optimized Dockerfile installing CPU-only PyTorch, making it compatible with cloud platforms like Render.

1. Commit and push your code to GitHub.
2. Log into **Render** and create a **Web Service**.
3. Choose **Docker** as your environment.
4. Render will compile the container and provide your live public URL!
