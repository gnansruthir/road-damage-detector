# RoadSense AI — Road Damage Severity & Civic Intelligence

RoadSense AI is an urban operations dashboard designed to monitor road network health in real time. It combines OpenCV-based crack and pothole detection with optional YOLOv8 inference support for a real fine-tuned checkpoint when one is available, plus CLAHE preprocessing, a geometric-texture severity analyzer, and an EXIF-grounded Folium civic map.

---

## Key Features
- **YOLOv8 Damage Detection**: Supports custom road-damage checkpoints when a trained model is supplied. No fine-tuned RDD2022 checkpoint or measured model result is included in this repository yet.
- **CV-Heuristic Fallback**: Uses contour analysis and edge detection to identify potholes and cracks from visual structure without relying on unrelated COCO labels or fabricated confidence scores.
- **CLAHE Low-Light Fix**: An OpenCV preprocessing pipeline that normalizes contrast under night or heavy shadow conditions to improve visibility before defect detection.
- **Severity Scoring Engine**: Calculates damage scale relative to the lane width and analyzes texture roughness (using grayscale standard deviation) to grade repairs as *Small*, *Medium*, or *Critical*.
- **Repair Priority Telemetry**: Assigns a civic repair priority index from `1` (Monitor) to `5` (Immediate Emergency) and estimates the affected stretch.
- **EXIF-Grounded Civic Map**: Renders an interactive Leaflet map using only GPS coordinates found in uploaded image EXIF metadata. Images without GPS produce no map point.
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
                         │ Detector          │ (YOLOv8 checkpoint or CV fallback)
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
On Linux or macOS:
```bash
python3 -m pip install -r requirements.txt
```

For local test tooling, install the development requirements:
```bash
python3 -m pip install -r requirements-dev.txt
```

### 2. Launch Console
Start the FastAPI uvicorn daemon:
```powershell
python -m uvicorn api.app:app --reload
```
On Linux or macOS:
```bash
python3 -m uvicorn api.app:app --reload
```
Access the dashboard in your web browser: **`http://127.0.0.1:8000`**

### 3. Run Automated Tests
Execute the pytest suite (verifies CLAHE matrices, detector fallbacks, scoring logic, and api routes):
```powershell
pytest tests/
```

The application uses the OpenCV heuristic fallback unless a compatible three-class
checkpoint is supplied through `MODEL_WEIGHTS_PATH` or placed at `weights/best.pt`.
The unauthenticated `/api/detect` endpoint is intended for demos; production deployments
should add authentication, rate limiting, and durable storage before exposing it publicly.
Map state is held in memory and rendered to one HTML file, so the included locking is
safe within a single process; multi-worker deployments should use shared durable storage.

## Training and Measurement

The repository does not contain RDD2022 data or claim a completed fine-tuning run. Prepare
a genuine RDD2022-derived Ultralytics dataset with documented `train` and `val` splits, then run:

```bash
python train.py --data path/to/rdd2022.yaml --epochs 100 --device 0
python eval.py --weights runs/road_damage/rdd2022_india/weights/best.pt --data path/to/rdd2022.yaml
python benchmark.py --weights runs/road_damage/rdd2022_india/weights/best.pt --image path/to/validation.jpg --device 0
```

To prepare the India split, run `python prepare_dataset.py`. The script writes
`rdd2022.yaml` and `rdd2022_night.yaml`; use `python prepare_dataset.py --dry-run
--limit 25` to inspect an already extracted dataset without creating output files.
The hosted-GPU workflow is also documented in `train_rdd2022.ipynb`. Append the
night evaluation to an existing report with `eval.py --append`.

`eval.py` writes measured mAP values to `RESULTS.md`, and `benchmark.py` writes measured
latency statistics to `BENCHMARK.md`. Those files should only be published after the commands
are run against the real dataset and target hardware. The map is an EXIF GPS map, not a live
location feed; photos without coordinates are intentionally omitted.

---

## Deployment (Render with Docker)
This project is configured with a size-optimized Dockerfile installing CPU-only PyTorch, making it compatible with cloud platforms like Render.

1. Commit and push your code to GitHub.
2. Log into **Render** and create a **Web Service**.
3. Choose **Docker** as your environment.
4. Render will compile the container and provide your live public URL!
