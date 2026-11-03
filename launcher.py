#!/usr/bin/env python3
"""Launcher for the REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM."""
import os
import subprocess
import sys

PROJECT_NAME = "REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM"


def main():
    print("=" * 70)
    print(PROJECT_NAME)
    print("=" * 70)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    if not os.path.exists("unified_app.py"):
        print("ERROR: unified_app.py not found")
        return False
    if not os.path.exists("yolov8n.pt"):
        print("INFO: yolov8n.pt is not present locally; Ultralytics may download it on first run.")

    try:
        import streamlit  # noqa: F401
        import cv2  # noqa: F401
        import ultralytics  # noqa: F401
        print("Required packages are available")
    except ImportError as exc:
        print(f"Missing package: {exc}")
        print("Run: python -m pip install -r requirements.txt")
        return False

    print("Starting dashboard at http://localhost:8501")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "unified_app.py",
            "--server.headless", "true",
        ], check=False)
    except KeyboardInterrupt:
        print("Application stopped")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
