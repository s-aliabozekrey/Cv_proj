from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_results(frame, detections, knocked_down):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        conf = det["confidence"]

        label = f"Class {cls_id}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Knocked Out Pins: {knocked_down}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return frame


def build_visualization_artifacts(output_base, frame_metrics, fallen_pins):
    output_base = Path(output_base)
    artifacts = {}

    if frame_metrics:
        times = [entry["time_sec"] for entry in frame_metrics]
        standing_values = [entry["standing"] for entry in frame_metrics]
        fallen_values = [entry["fallen"] for entry in frame_metrics]
        score_values = [entry["score"] for entry in frame_metrics]

        figure, axes = plt.subplots(figsize=(11, 5.5))
        axes.plot(times, standing_values, label="Standing", linewidth=2)
        axes.plot(times, fallen_values, label="Fallen", linewidth=2)
        axes.plot(times, score_values, label="Score", linewidth=2, linestyle="--")
        axes.set_title("Bowling Pin Status Over Time")
        axes.set_xlabel("Time (seconds)")
        axes.set_ylabel("Count")
        axes.set_ylim(bottom=0)
        axes.grid(True, alpha=0.25)
        axes.legend(loc="upper right")
        figure.tight_layout()

        trends_path = output_base.with_name(f"{output_base.name}_trends.png")
        figure.savefig(trends_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        artifacts["trends"] = trends_path

    if fallen_pins:
        figure, axes = plt.subplots(figsize=(11, 5.5))
        labels = [str(pin_id) for _, pin_id, _, _ in fallen_pins]
        orders = [fall_order for fall_order, _, _, _ in fallen_pins]
        times = [fall_time_sec if fall_time_sec is not None else 0.0 for _, _, fall_time_sec, _ in fallen_pins]

        bars = axes.bar(labels, times, color="#2c7be5")
        axes.set_title("Pin Fall Timeline")
        axes.set_xlabel("Pin ID")
        axes.set_ylabel("Fall Time (seconds)")
        axes.set_ylim(bottom=0)
        axes.grid(True, axis="y", alpha=0.25)

        for bar, fall_order in zip(bars, orders):
            axes.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"#{fall_order}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        figure.tight_layout()

        timeline_path = output_base.with_name(f"{output_base.name}_fall_order.png")
        figure.savefig(timeline_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        artifacts["fall_order"] = timeline_path

    return artifacts