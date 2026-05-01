import argparse
import os
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "data" / "input_videos"
DEFAULT_VIDEO = "bowling_sample.mp4"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "output_videos" / "result.mp4"
DEFAULT_TRAIN_DATA = PROJECT_ROOT / "data" / "Bowling Pin Detection.v1i.yolov12" / "data.yaml"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "models" / "yolo_weights.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the bowling-pin model and run video inference.")
    parser.add_argument("--video", type=Path, default=None, help="Path to the input video. Defaults to the first .mp4 in data/input_videos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for the rendered output video.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH, help="Path to the trained YOLO weights to use for inference.")
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument("--train", dest="train", action="store_true", help="Train the bowling-pin detector before running inference.")
    train_group.add_argument("--skip-train", dest="train", action="store_false", help="Skip training and run inference with the selected weights.")
    parser.set_defaults(train=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAIN_DATA, help="Dataset YAML used when --train is set.")
    parser.add_argument("--base-model", type=Path, default=PROJECT_ROOT / "yolo12n.pt", help="Base model used when --train is set.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs when --train is set.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size when --train is set.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size when --train is set.")
    parser.add_argument("--device", type=str, default=None, help="Training device when --train is set.")
    parser.add_argument("--project", type=Path, default=None, help="Ultralytics project directory when --train is set.")
    parser.add_argument("--name", type=str, default="bowling_pin_detector", help="Run name when --train is set.")
    return parser


def resolve_video_path(explicit_video: Path | None = None) -> Path:
    if explicit_video is not None:
        return explicit_video

    candidate = VIDEO_DIR / DEFAULT_VIDEO
    if candidate.exists():
        return candidate

    alt = candidate.with_suffix(candidate.suffix + ".mp4")
    if alt.exists():
        return alt

    if VIDEO_DIR.is_dir():
        for filename in sorted(VIDEO_DIR.iterdir()):
            if filename.suffix.lower() == ".mp4":
                return filename

    return candidate


def main() -> None:
    args = build_parser().parse_args()

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    if args.train:
        from models.yolo_model import train_bowling_pin_model

        train_bowling_pin_model(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            output_weights=args.weights,
        )

    os.environ["YOLO_WEIGHTS_PATH"] = str(args.weights)

    from detection import detect_objects
    from scoring import count_fallen_pins
    from visualization import draw_results

    video_path = resolve_video_path(args.video)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        knocked_down = count_fallen_pins(detections)

        output_frame = draw_results(frame, detections, knocked_down)

        writer.write(output_frame)
        cv2.imshow("Bowling Score Prediction", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()