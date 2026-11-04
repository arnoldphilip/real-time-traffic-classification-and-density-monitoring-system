"""ATCC Local Engine Package.

Provides local video processing, YOLO inference, accident detection,
and structured telemetry synchronization with the central ATCC server.
"""
from .model_manager import (
    load_model_safely,
    verify_model_file,
    get_available_local_models,
    MODELS_DIR,
)
from .video_processor import LocalVideoProcessor
from .accident_processor import LocalAccidentDetector
from .communication import ServerClient, get_hardware_info
from .local_processor import run_local_agent

__all__ = [
    "load_model_safely",
    "verify_model_file",
    "get_available_local_models",
    "MODELS_DIR",
    "LocalVideoProcessor",
    "LocalAccidentDetector",
    "ServerClient",
    "get_hardware_info",
    "run_local_agent",
]

