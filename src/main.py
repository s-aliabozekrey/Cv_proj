import os
import cv2
from detection import detect_objects
from scoring import count_fallen_pins
from visualization import draw_results

VIDEO_DIR = "data/input_videos"
DEFAULT_VIDEO = "bowling_sample.mp4"

# Choose a video: prefer DEFAULT_VIDEO, else try alternatives or pick first .mp4
candidate = os.path.join(VIDEO_DIR, DEFAULT_VIDEO)
if os.path.exists(candidate):
    VIDEO_PATH = candidate
else:
    alt = candidate + ".mp4"
    if os.path.exists(alt):
        VIDEO_PATH = alt
    else:
        VIDEO_PATH = None
        if os.path.isdir(VIDEO_DIR):
            for f in os.listdir(VIDEO_DIR):
                if f.lower().endswith(".mp4"):
                    VIDEO_PATH = os.path.join(VIDEO_DIR, f)
                    break
        if VIDEO_PATH is None:
            VIDEO_PATH = candidate  # keep original path; will error later

OUTPUT_PATH = "data/output_videos/result.mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_index = 0
    standing_pins_before = None

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