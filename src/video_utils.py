import cv2
import os


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    return cap


def get_video_properties(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count
    }


def create_video_writer(output_path, width, height, fps, codec="mp4v"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Could not create video writer: {output_path}")

    return writer


def save_frame(frame, output_dir, frame_index):
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f"frame_{frame_index:05d}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


def read_all_frames(video_path):
    cap = load_video(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def release_resources(cap=None, writer=None):
    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()