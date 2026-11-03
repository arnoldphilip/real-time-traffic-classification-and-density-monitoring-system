#!/usr/bin/env python3
"""Install and verify the traffic classification and density monitoring application."""
import os
import shutil
import subprocess
import sys


def has_nvidia_gpu():
    """Check if NVIDIA GPU driver tool is present on host system."""
    return shutil.which("nvidia-smi") is not None


def main():
    if sys.version_info < (3, 8):
        print("Python 3.8 or newer is required.")
        return False

    print("Installing base requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if has_nvidia_gpu():
        print("NVIDIA GPU hardware detected. Install a compatible PyTorch CUDA build if required.")

    os.makedirs("json", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    from ultralytics import YOLO  # noqa: F401
    import cv2  # noqa: F401
    import streamlit  # noqa: F401
    import torch  # noqa: F401

    print("=" * 70)
    print("REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print("Run: python launcher.py  OR  streamlit run unified_app.py")
    print("=" * 70)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
