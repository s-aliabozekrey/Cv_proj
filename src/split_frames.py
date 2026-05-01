import os

from video_utils import get_video_properties, load_video, save_frame

VIDEO_DIR = "data/input_videos"
FRAMES_DIR = "data/frames"
DEFAULT_VIDEO = "bowling_sample.mp4"


def resolve_video_path():
    candidate = os.path.join(VIDEO_DIR, DEFAULT_VIDEO)
    if os.path.exists(candidate):
        return candidate

    alt = candidate + ".mp4"
    if os.path.exists(alt):
        return alt

    if os.path.isdir(VIDEO_DIR):
        for filename in sorted(os.listdir(VIDEO_DIR)):
            if filename.lower().endswith(".mp4"):
                return os.path.join(VIDEO_DIR, filename)

    raise FileNotFoundError(f"No .mp4 video found in {VIDEO_DIR}")


def split_video_into_frames(video_path, output_dir=FRAMES_DIR):
    cap = load_video(video_path)
    props = get_video_properties(cap)

    os.makedirs(output_dir, exist_ok=True)

    frame_index = 0
    saved_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        saved_paths.append(save_frame(frame, output_dir, frame_index))
        frame_index += 1

    cap.release()

    return {
        "video_path": video_path,
        "output_dir": output_dir,
        "frame_count": frame_index,
        "video_properties": props,
        "saved_paths": saved_paths,
    }


def main():
    video_path = resolve_video_path()
    result = split_video_into_frames(video_path)

    print(f"Video: {result['video_path']}")
    print(f"Frames saved to: {result['output_dir']}")
    print(f"Total frames extracted: {result['frame_count']}")


if __name__ == "__main__":
    main()
