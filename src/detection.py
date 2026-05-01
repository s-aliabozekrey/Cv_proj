import os
from ultralytics import YOLO

# Load YOLO model once. Prefer local weights, otherwise fall back to a
# small pretrained weight (will be downloaded by ultralytics if needed).
WEIGHTS_PATH = "models/yolo_weights.pt"
if os.path.exists(WEIGHTS_PATH):
    model = YOLO(WEIGHTS_PATH)
else:
    print(f"Warning: weights not found at {WEIGHTS_PATH}. Falling back to 'yolo12n.pt'.")
    model = YOLO("yolo12n.pt")

def detect_objects(frame):
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        detections.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2)
        })

    return detections