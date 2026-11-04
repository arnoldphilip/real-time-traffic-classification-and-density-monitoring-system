#!/usr/bin/env python3
"""ATCC system launcher."""
import os
import subprocess
import sys


def main():
    print("=" * 55)
    print("Automatic Traffic Classification & Counting (ATCC)")
    print("=" * 55)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")

    if not os.path.exists("unified_app.py"):
        print("ERROR: unified_app.py not found")
        return False
    if not os.path.exists("yolov8n.pt"):
        print("WARNING: yolov8n.pt not found. Ultralytics may download it on first run.")

    try:
        import streamlit  # noqa: F401
        import cv2  # noqa: F401
        import ultralytics  # noqa: F401
        print("Required packages are available")
    except ImportError as exc:
        print(f"Missing package: {exc}")
        print("Run: python -m pip install -r requirements.txt")
        return False

    print("Starting ATCC at http://localhost:8501")
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
