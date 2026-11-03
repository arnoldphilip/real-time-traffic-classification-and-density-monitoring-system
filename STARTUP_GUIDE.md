# REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM — Startup Guide

## 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 3. Start the application

```powershell
python launcher.py
```

Alternatively:

```powershell
streamlit run unified_app.py
```

The Streamlit interface normally opens at `http://localhost:8501`.

## 4. Verify the project

```powershell
python test_system.py
```

The basic verification checks required project files and database tables. It is not an end-to-end model-accuracy or live-camera test.
