from __future__ import annotations

from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - fallback for headless environments
    tk = None
    ttk = None

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - fallback if Pillow is unavailable
    Image = None
    ImageTk = None


def _load_image(path: Path, max_size: tuple[int, int]):
    if Image is None or ImageTk is None:
        return None

    image = Image.open(path)
    image.thumbnail(max_size)
    return ImageTk.PhotoImage(image)


def _make_text_block(parent, lines):
    text = tk.Text(parent, wrap="word", height=10, padx=12, pady=12)
    text.pack(fill="both", expand=True)
    text.insert("1.0", "\n".join(lines))
    text.config(state="disabled")
    return text


def _image_tab(notebook, title, path: Path):
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=title)

    if not path.exists():
        ttk.Label(frame, text=f"Missing visualization file:\n{path}").pack(
            expand=True, fill="both", padx=24, pady=24
        )
        return frame

    image = _load_image(path, (1200, 700))
    if image is None:
        ttk.Label(frame, text=f"Preview unavailable for:\n{path}").pack(
            expand=True, fill="both", padx=24, pady=24
        )
        return frame

    label = ttk.Label(frame, image=image)
    label.image = image
    label.pack(expand=True, fill="both", padx=12, pady=12)
    return frame


def show_visualization_dashboard(result):
    if tk is None or ttk is None:
        print("Tkinter is unavailable; generated files:")
        for name, path in result.get("artifacts", {}).items():
            print(f"- {name}: {path}")
        return

    root = tk.Tk()
    root.title("Bowling Pin Visualizations")
    root.geometry("1280x860")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    summary_frame = ttk.Frame(notebook)
    notebook.add(summary_frame, text="Summary")

    summary_lines = [
        "Run summary",
        f"Output video: {result.get('output_video_path')}",
        f"Timeline CSV: {result.get('csv_path')}",
        f"Frames processed: {result.get('frame_count', 0)}",
        f"Final score: {result.get('score', 0)}",
        f"Max fallen pins: {result.get('max_fallen_count', 0)}",
    ]
    _make_text_block(summary_frame, summary_lines)

    artifacts = result.get("artifacts", {})
    _image_tab(notebook, "Trend Chart", artifacts.get("trends", Path("")))
    _image_tab(notebook, "Fall Order", artifacts.get("fall_order", Path("")))

    files_frame = ttk.Frame(notebook)
    notebook.add(files_frame, text="Files")
    file_lines = ["Generated visualization files:"]
    if artifacts:
        for name, path in artifacts.items():
            file_lines.append(f"{name}: {path}")
    else:
        file_lines.append("No visualization images were generated for this run.")
    _make_text_block(files_frame, file_lines)

    root.mainloop()