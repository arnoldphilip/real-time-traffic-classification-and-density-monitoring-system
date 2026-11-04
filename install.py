#!/usr/bin/env python3
"""Install and verify the ATCC application with GPU support."""
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
        print("NVIDIA GPU hardware detected! Installing PyTorch CUDA package for GPU speedup...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])

    os.makedirs("json", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    from ultralytics import YOLO  # noqa: F401
    import cv2  # noqa: F401
    import streamlit  # noqa: F401
    import torch  # noqa: F401

    print("=" * 55)
    print("ATCC installation verified successfully.")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Active GPU: {torch.cuda.get_device_name(0)}")
    print("Run: python launcher.py  OR  streamlit run unified_app.py")
    print("=" * 55)
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

