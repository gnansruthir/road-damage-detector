import os
import cv2

# RDD2022 Custom Classes
DAMAGE_CLASSES = ["Pothole", "Longitudinal Crack", "Transverse Crack"]

class RoadDamageDetector:
    def __init__(self, weights_path=None):
        self.weights_path = weights_path
        self.use_yolo = False
        self.model = None
        self.last_mode = "cv_fallback"

        try:
            from ultralytics import YOLO
            if weights_path and os.path.exists(weights_path):
                self.model = YOLO(weights_path)
                self.use_yolo = True
                print(f"Loaded custom model weights: {weights_path}")
            else:
                print("No trained checkpoint supplied. Using CV-based heuristic fallback.")
        except ImportError:
            print("Ultralytics YOLO not found or failed to load. Using CV-based heuristic fallback.")

    def detect(self, image_bgr, conf_threshold=0.25):
        """
        Runs object detection on the input BGR image.
        If a real fine-tuned checkpoint is available, it runs YOLOv8 inference.
        Otherwise, it falls back to OpenCV contours & Canny edge detection to locate
        actual cracks/potholes in the image using visual features only.
        """
        if not hasattr(image_bgr, "shape"):
            raise ValueError("Image must be a valid OpenCV image array.")
        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image.")
        h, w, _ = image_bgr.shape
        if h < 1 or w < 1:
            raise ValueError("Image must not be empty.")
        detections = []
        self.last_mode = "cv_fallback"

        if self.use_yolo and self.model:
            try:
                results = self.model(image_bgr, conf=conf_threshold, verbose=False)[0]
                boxes = results.boxes

                if len(results.names) != 3:
                    raise ValueError("Loaded model does not match this road-damage class set.")

                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    if isinstance(results.names, dict):
                        class_name = results.names.get(cls_id)
                    else:
                        class_name = results.names[cls_id] if cls_id < len(results.names) else None
                    if class_name is None:
                        continue

                    detections.append({
                        "bbox": [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])],
                        "class": class_name,
                        "confidence": conf
                    })
                if detections:
                    self.last_mode = "yolo"
            except Exception as e:
                print(f"YOLO inference error: {e}. Falling back to OpenCV detection.")
                self.use_yolo = False
                detections = []

        if not self.use_yolo or not detections:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            candidate_masks = [
                cv2.Canny(blurred, 50, 150),
                cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)[1]
            ]

            count = 0
            for mask in candidate_masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    area = cv2.contourArea(cnt)
                    area_ratio = area / float(max(1, w * h))

                    if area > 100 and area_ratio < 0.35 and 20 < cw < w * 0.9 and 20 < ch < h * 0.9:
                        ratio = cw / ch if ch > 0 else 0
                        if ratio > 2.5:
                            class_name = "Longitudinal Crack"
                        elif ratio < 0.4:
                            class_name = "Transverse Crack"
                        else:
                            class_name = "Pothole"

                        confidence = 0.20 + min(0.75, area_ratio * 40.0)

                        detections.append({
                            "bbox": [x, y, x + cw, y + ch],
                            "class": class_name,
                            "confidence": float(confidence)
                        })
                        count += 1
                        if count >= 10:
                            return detections

        if self.last_mode != "yolo":
            self.last_mode = "cv_fallback"
        return detections
