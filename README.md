# Automatic Traffic Classification & Counting (ATCC)

This project is an ATCC system for detecting, classifying, tracking and counting road vehicles from video, images or a webcam.

## Features

- Vehicle detection using YOLOv8 COCO weights
- Vehicle classes: car, motorcycle, bus and truck
- Centroid-based vehicle tracking
- Lane crossing/counting for Lane A and Lane B
- Traffic-density based signal-priority logic
- SQLite storage for vehicle counts and traffic events
- Streamlit web interface
- Video, image and webcam input

## Run

```powershell
cd <project-directory>
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run unified_app.py
```

Alternatively:

```powershell
python launcher.py
```

## Project structure

- `unified_app.py` — ATCC application
- `launcher.py` — application launcher
- `install.py` — dependency installer/checker
- `test_system.py` — ATCC verification tests
- `yolov8n.pt` — YOLOv8 model used for vehicle detection
- `notebook/atcc.ipynb` — ATCC notebook
- `unified_traffic_database.db` — ATCC SQLite database

## Database

The application stores ATCC data in:

- `VehicleCounts`
- `TrafficEvents`
