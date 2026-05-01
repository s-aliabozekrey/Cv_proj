import os
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_weights() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("YOLO_WEIGHTS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            PROJECT_ROOT / "models" / "yolo_weights.pt",
            PROJECT_ROOT / "runs" / "detect" / "bowling_pin_detector" / "weights" / "best.pt",
            PROJECT_ROOT / "yolo12n.pt",
        ]
    )

    return candidates


def _resolve_weights_path() -> Path:
    for candidate in _candidate_weights():
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "yolo12n.pt"


model = YOLO(str(_resolve_weights_path()))

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