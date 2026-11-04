# ATCC — Automatic Traffic Classification & Counting

## Overview

ATCC detects road vehicles, classifies them, tracks their movement across configurable counting lines, records counts, and provides traffic-signal analysis.

## Detection classes

The YOLOv8 COCO model is used for:

- Car
- Motorcycle
- Bus
- Truck

## Processing pipeline

```text
Video / Camera / Photo Input
        ↓
YOLO Vehicle Detection
        ↓
Vehicle Classification & Tracking
        ↓
Count Vehicles in Region of Interest (ROI)
        ↓
Calculate Traffic Density (Low / Medium / High)
        ↓
Streamlit Dashboard & SQLite Database
```

## Hardware & GPU Acceleration

ATCC supports GPU compute acceleration (NVIDIA CUDA / Apple MPS) for 5x–10x faster video frame rate inference.

- **Automatic Device Selection**: Auto-detects NVIDIA GPUs (`cuda:0`, `cuda:1`, etc.) or Apple Silicon (`mps`).
- **FP16 Half-Precision**: Toggling FP16 reduces VRAM consumption and increases frame processing speed.
- **PyTorch CUDA Wheel**:
  ```powershell
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

## Configuration

The model path is configured in `unified_app.py` (e.g. `yolov8n.pt`, `yolov8m.pt`, `best.pt`).

Counting lines are configured in `VehicleTracker.counting_line`.

Region of Interest (ROI) boundaries and density level thresholds (Low/Medium/High) are configurable in the Streamlit sidebar under **🏙️ Region of Interest (ROI) & Density**.

Traffic-signal thresholds are configured in `TrafficSignalManager`.

Compute device & precision controls are available under **⚡ Compute & GPU Speedup** in the Streamlit sidebar.

## Run

```powershell
python install.py
python launcher.py
```

## Dashboard & Security Architecture

The dashboard implements role-based access control (RBAC) and video stream management configured via `atcc_config.json`:

- **Admin Login**:
  - Full control over all users and feeds.
  - Can view, pause, resume, and delete any video feed.
  - Full user management (add, edit, activate, deactivate, delete users).
  - System activity audit logging and analytics.
- **User Login**:
  - User accounts can add and manage only their own live video feeds.
  - Strict ownership isolation enforced at the database layer (users can never access, pause, or delete another user's feeds).
- **Live MP4 Feeds & Playback**:
  - Feeds support continuous automatic looping and muting.
  - Three-dot hover/click menu (`⋮ Actions`) on each feed provides **Pause**, **Resume**, **Delete**, and **Loop** controls.
  - Direct real-time YOLOv8 vehicle detection & counting can be launched on any live feed.

### Default Accounts

| Role | Username | Password | User ID | Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Admin** | `admin` | `admin123` | 1 | Full system control |
| **User** | `highway_operator` | `user123` | 101 | Own videos only (e.g. Highway Camera 1) |
| **User** | `city_operator` | `user123` | 102 | Own videos only (e.g. City Road Camera) |
| **User** | `traffic_user` | `user123` | 103 | Own videos only |

## Configuration Files & Modules

| File | Purpose |
| :--- | :--- |
| `unified_app.py` | Main Streamlit dashboard, navigation, and login |
| `atcc_config.json` | Project, authentication, permissions, and dashboard configuration |
| `videos.json` | Sample video records |
| `database.py` | SQLite database operations and strict security enforcement |
| `auth.py` | Admin and user authentication with bcrypt password hashing |

## Database tables

- `Users`: User accounts, bcrypt password hashes, roles (`admin`, `user`), and account status.
- `VideoFeeds`: Registered camera feeds, assigned owner, status (`active`/`paused`), and loop configuration.
- `VehicleCounts`: Real-time vehicle line crossing events.
- `TrafficEvents`: Automated signal control and priority events.
- `TrafficDensity`: ROI vehicle density counts and level estimates.
- `ActivityLogs`: Security and administrative audit trail.
- `AccidentEvents`: Real-time detected collisions, sudden crashes, confidence %, location, and resolution status.

## Real-Time Accident Detection & Alert System

- **Multi-signal detection**: Identifies vehicle collisions, abrupt crash stops, and erratic trajectory spins directly from tracking history.
- **Danger Symbol Display**: Large blinking danger symbol (⚠️) with "ACCIDENT DETECTED!" banner and camera location.
- **Audio & Browser Notification**: Emergency siren tone (Web Audio API) and native HTML5 browser push notifications.
- **System Condition Tri-State**: `🟢 NORMAL` (no accidents), `🟡 WARNING` (suspicious activity / acknowledged), and `🔴 DANGER` (unresolved accidents).
- **Accident Alert History Panel**: Dedicated dashboard panel for viewing, acknowledging, and resolving alerts with full timestamps and confidence ratings.
- `VideoFeeds`: Live MP4 video streams, titles, owner `user_id`, statuses (`active`, `paused`), and loop flags.
- `ActivityLogs`: Security and audit logging of logins, video additions, pauses, resumes, and deletions.
- `TrafficDensity`: Stores timestamp, total vehicle count in ROI, vehicle-type counts (Car, Bus, Truck, Bike, Auto), density level (Low/Medium/High), and ROI bounds.
- `VehicleCounts`: Stores timestamped vehicle line crossing counts by class.
- `TrafficEvents`: Stores traffic-control events generated by the ATCC logic.

