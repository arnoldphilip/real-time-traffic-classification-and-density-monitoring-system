# ATCC Auto-Rickshaw Detection

The ATCC application keeps the original `yolov8n.pt` detector for car, motorcycle, bus and truck, and additionally loads `best.pt` for South-Asian 3-wheeler detection.

Place the downloaded model at:

`ATCC-Infosys-main/best.pt`

The dedicated model's class `2` (3-wheeler) is displayed and counted as `auto`. When the two detectors overlap, the dedicated auto detection replaces the original detector's overlapping vehicle detection to avoid double-counting.

Run:

`streamlit run unified_app.py`
