# REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM

A real-time computer-vision system for detecting, classifying, tracking and monitoring road-traffic density from uploaded videos, images and webcam input.

> Adapted from the upstream ATCC project. Original project and MIT licence attribution are retained in this repository.

## Features

- Vehicle detection using YOLOv8
- Vehicle classification: car, motorcycle, bus and truck
- Centroid-based vehicle tracking and duplicate-detection suppression
- Line-crossing vehicle counting
- Region-of-interest (ROI) vehicle counts and Low / Medium / High traffic-density estimation
- Traffic-signal priority analysis based on traffic conditions
- Streamlit dashboard with video, image and webcam input
- SQLite storage for vehicle counts, traffic events and density records
- NVIDIA CUDA / Apple MPS acceleration when supported by the installed PyTorch build

## Requirements

- Python 3.8 or newer
- Windows, macOS or Linux
- A webcam or video/image input (optional)
- Compatible NVIDIA GPU and CUDA-enabled PyTorch for CUDA acceleration; CPU mode is also supported

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA CUDA acceleration, install the PyTorch build compatible with your driver and CUDA runtime. See https://pytorch.org/get-started/locally/.

## Model weights

The YOLOv8 Nano file `yolov8n.pt` is downloaded automatically by Ultralytics on first run when it is not present locally. The optional fine-tuned `best.pt` model enables additional three-wheeler / auto-rickshaw detection; obtain that file from the upstream project and place it in the project root to enable this class.

## Run

```bash
python install.py
python launcher.py
```

Or start Streamlit directly:

```bash
streamlit run unified_app.py
```

The dashboard is normally available at http://localhost:8501.

## Project files

- `unified_app.py` — Streamlit application, vehicle detection/tracking, ROI density monitoring and analytics
- `launcher.py` — application launcher
- `install.py` — dependency installation and environment verification
- `test_system.py` — basic project/database verification
- `README_UNIFIED.md` — extended technical overview
- `AUTO_MODEL_SETUP.md` — optional three-wheeler model notes
- `STARTUP_GUIDE.md` — startup instructions

## Data storage

The application creates a local SQLite database named `unified_traffic_database.db` and uses these tables:

- `VehicleCounts` — line-crossing counts by vehicle class
- `TrafficEvents` — traffic-control events
- `TrafficDensity` — ROI totals, class counts, density level and ROI bounds

Generated databases, logs and runtime output are local artifacts and are not committed.

## Project attribution

Adapted from [deepakp1688-ops/ATCC](https://github.com/deepakp1688-ops/ATCC). The original MIT licence and copyright notice are preserved in `LICENSE`. See that file for applicable terms.

## Disclaimer

Traffic-density estimates and signal-priority outputs depend on the configured detection models, ROI and thresholds. They are intended for project/research demonstration and should not be used to control live public-road traffic infrastructure without independent validation and appropriate safety engineering.
