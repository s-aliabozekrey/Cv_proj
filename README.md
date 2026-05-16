# Bowling Pin Analysis — Computer Vision Project

Comprehensive toolkit for detecting, tracking, and analysing bowling pins in video. This repository includes data preparation, model inference, tracking, scoring, visualization, and a lightweight dashboard for result inspection.

**Primary goals:**
- Build a reproducible pipeline for bowling-pin detection and event scoring.
- Provide visualization and a dashboard to explore detection/tracking outputs.
- Export analysis results as CSV for downstream analytics.

**Note:** This README focuses on usage and developer guidance. See the source files for implementation details.

**Project status:** Experimental research/prototype (not production hardened).

**Table of contents**
- **Project overview** — what this repo provides
- **Quick start** — run inference and view outputs
- **Repository structure** — where to find code and data
- **Usage** — commands for detection, tracking, visualization, and dashboard
- **Models & Data** — included pretrained weights and dataset layout
- **Outputs** — CSV and visualization outputs
- **Development** — testing, linting, extending
- **Contact & License**

## Requirements

The Python dependencies are listed in [requirements.txt](requirements.txt#L1). Recommended workflow is to use the included virtual environment or create a new one and install the requirements.

Example (Windows PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Run the main pipeline on a video (inference -> tracking -> visualization):

```bash
python src/main.py --input input_videos/my_video.mp4 --output output/ --weights models/yolov8n.pt
```

The script will produce detection overlays, tracking outputs, and CSV exports into the `output/` folder.

## Repository structure

- [src](src) — core codebase: detection, tracking, visualization, utilities
	- [src/main.py](src/main.py#L1) — pipeline entrypoint
	- [src/detection.py](src/detection.py#L1) — model inference utilities
	- [src/tracking.py](src/tracking.py#L1) — tracking and identity assignment
	- [src/scoring.py](src/scoring.py#L1) — bowling-pin scoring logic
	- [src/visualization.py](src/visualization.py#L1) — image & video visualizers
	- [src/dashboard.py](src/dashboard.py#L1) — dashboard server / UI integration
	- [src/vizual/visuzalis.py](src/vizual/visuzalis.py#L1) — visualization helpers
	- [src/boxesfuncs.py](src/boxesfuncs.py#L1) — bbox helpers
	- [src/video_utils.py](src/video_utils.py#L1) — frame I/O and utilities
- `models/` — pretrained weights used for inference
	- `yolov8n.pt`, `yolo12n.pt`, `best.pt` (examples)
- `data/` — dataset and Roboflow exports
- `input_videos/` — input sample videos
- `output/` — inference outputs (images, videos, CSVs)
- `requirements.txt` — Python dependencies

## Data layout

Dataset prepared under `data/` follows the common split:

- `train/`, `valid/`, `test/` — each with `images/` and `labels/`.
- `data/data.yaml` — dataset config used for training/inference.

## Models

Pretrained model files are available in `models/`. Replace or retrain as needed. Training scripts are not bundled; use standard YOLO training flows with `data/data.yaml`.

## Usage: detailed commands

- Run detection-only (saves JSON/CSV of detections):

```bash
python src/detection.py --source input_videos/test_video.mp4 --weights models/yolov8n.pt --save-csv
```

- Run full pipeline (detection -> tracker -> scoring -> visualization):

```bash
python src/main.py --input input_videos/test_video.mp4 --weights models/yolov8n.pt --output output/
```

- Start the dashboard (UI to explore frames, tracks, and scores):

```bash
python src/dashboard.py --port 8050
```

Open `http://localhost:8050` in your browser to view visualizations and CSV summaries.

## Visualization & Dashboard features

- Per-frame overlay: bounding boxes, track IDs, confidence scores, and timestamps.
- Track timelines: view each object's trajectory across frames.
- Scoring panel: aggregated bowling-pin scores per video/time-step and downloadable CSVs.
- Quick-export: save annotated video and per-frame CSV (saved to `output/`).

## Outputs

- Annotated frames/videos: saved in `output/` with predictable filenames.
- CSV exports: time-step based CSVs (`output_13_time_step.csv`, etc.) for downstream analysis.

## Development notes

- Logging: code logs to console; extend as needed for production.
- Tests: there are no unit tests bundled; add tests under `tests/` if required.
- Style: follow existing project conventions when editing code.

## Contributing

If you'd like to contribute:

1. Fork the repo and create a feature branch.
2. Run the pipeline locally and ensure outputs are reproducible.
3. Submit a PR with a clear description and any sample outputs.

## Contact

Project maintained by the original repository author. For questions or help, open an issue or contact the maintainer.

## License

This repository does not include an explicit license. Add a `LICENSE` file if you plan to open source the project.
