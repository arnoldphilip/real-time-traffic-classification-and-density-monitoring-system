# ATCC Startup Guide

## 1. Activate the existing virtual environment

```powershell
cd <project-directory>
.\venv\Scripts\Activate.ps1
```

## 2. Install dependencies if required

```powershell
python -m pip install -r requirements.txt
```

## 3. Start ATCC

```powershell
streamlit run unified_app.py
```

Or:

```powershell
python launcher.py
```

The Streamlit interface normally opens at `http://localhost:8501`.

## 4. Verify the project

```powershell
python test_system.py
```

The test confirms that the ATCC files and database tables are available.
