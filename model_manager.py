"""Model Manager for ATCC Local Engine and Central Server.

Handles model path resolution, offline verification, safe local-first loading,
and graceful fallback without unhandled GitHub connection crashes.
"""
import os
import sys
import time
import urllib.request
import urllib.error
import logging
from pathlib import Path

logger = logging.getLogger("atcc.model_manager")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [ModelMgr] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Standard YOLOv8 model registry with expected approximate sizes in MB
MODEL_REGISTRY = {
    "yolov8n.pt": {
        "label": "YOLOv8 Nano (Lightweight, Fast)",
        "filename": "yolov8n.pt",
        "approx_size_mb": 6.2,
        "min_size_bytes": 4 * 1024 * 1024,
        "recommended_for": "CPU / Low-spec machines",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
    },
    "yolov8s.pt": {
        "label": "YOLOv8 Small (Balanced)",
        "filename": "yolov8s.pt",
        "approx_size_mb": 21.5,
        "min_size_bytes": 15 * 1024 * 1024,
        "recommended_for": "Mid-tier CPU / Entry GPU",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
    },
    "yolov8m.pt": {
        "label": "YOLOv8 Medium (High Accuracy - Recommended)",
        "filename": "yolov8m.pt",
        "approx_size_mb": 49.7,
        "min_size_bytes": 35 * 1024 * 1024,
        "recommended_for": "Standard Dedicated GPU",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt",
    },
    "yolov8l.pt": {
        "label": "YOLOv8 Large (Very High Precision)",
        "filename": "yolov8l.pt",
        "approx_size_mb": 83.7,
        "min_size_bytes": 60 * 1024 * 1024,
        "recommended_for": "High-end GPU",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l.pt",
    },
    "yolov8x.pt": {
        "label": "YOLOv8 Extra Large (Maximum Precision)",
        "filename": "yolov8x.pt",
        "approx_size_mb": 130.5,
        "min_size_bytes": 90 * 1024 * 1024,
        "recommended_for": "High-end Server / Workstation GPU",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x.pt",
    },
    "best.pt": {
        "label": "Custom Auto-Rickshaw (3-Wheeler Detector)",
        "filename": "best.pt",
        "approx_size_mb": 5.1,
        "min_size_bytes": 3 * 1024 * 1024,
        "recommended_for": "South-Asian 3-Wheeler Detection",
        "url": None,
    },
}

# Cache for loaded YOLO models to avoid re-instantiation
_LOADED_MODELS_CACHE = {}


def find_model_path(model_filename: str) -> Path | None:
    """Check whether a model exists locally in models/ or project root.

    Returns the absolute Path if present and non-empty, otherwise None.
    """
    cleaned_name = Path(model_filename).name

    candidates = [
        MODELS_DIR / cleaned_name,
        BASE_DIR / cleaned_name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            # Check minimum size (at least 100 KB to avoid zero-byte stub files)
            if candidate.stat().st_size > 100 * 1024:
                return candidate.resolve()

    return None


def verify_model_file(filepath: Path | str) -> tuple[bool, str]:
    """Verify that a model file is complete and not corrupted."""
    path = Path(filepath)
    if not path.exists():
        return False, f"File does not exist: {path}"

    size = path.stat().st_size
    filename = path.name

    if filename in MODEL_REGISTRY:
        min_bytes = MODEL_REGISTRY[filename]["min_size_bytes"]
        if size < min_bytes:
            return False, (
                f"File '{filename}' appears truncated or incomplete "
                f"({size / (1024 * 1024):.2f} MB, expected at least {min_bytes / (1024 * 1024):.2f} MB)."
            )

    try:
        # Quick check with torch to ensure checkpoint is parseable
        import torch
        # Load weights without instantiating full model graph for speed
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True, "Valid model file."
    except Exception as exc:
        return False, f"Model verification failed: {exc}"


def get_available_local_models() -> dict[str, dict]:
    """Return dictionary of models and their availability status on this machine."""
    status = {}
    for filename, meta in MODEL_REGISTRY.items():
        found_path = find_model_path(filename)
        is_available = found_path is not None
        status[filename] = {
            **meta,
            "available": is_available,
            "local_path": str(found_path) if found_path else None,
            "local_size_mb": round(found_path.stat().st_size / (1024 * 1024), 2) if found_path else 0,
        }
    return status


def get_best_available_model(preferred_model: str = "yolov8x.pt") -> tuple[str, Path | None]:
    """Find the best available local model, falling back gracefully if preferred is missing."""
    # If preferred exists, return it
    found = find_model_path(preferred_model)
    if found:
        return preferred_model, found

    # Fallback priority hierarchy: x -> m -> s -> n
    priority_order = ["yolov8x.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "best.pt"]

    for model_name in priority_order:
        found = find_model_path(model_name)
        if found:
            return model_name, found

    return "", None


def download_model_with_retry(
    model_filename: str,
    max_retries: int = 3,
    timeout: int = 30,
    progress_callback=None,
) -> tuple[bool, str]:
    """Download missing model from official Ultralytics release with retries and timeout.

    Does NOT overwrite existing valid files. Uses a temporary .tmp file.
    """
    if model_filename not in MODEL_REGISTRY:
        return False, f"Unknown model '{model_filename}'."

    meta = MODEL_REGISTRY[model_filename]
    url = meta.get("url")
    if not url:
        return False, (
            f"No public download URL for custom model '{model_filename}'. "
            f"Please place it manually in '{MODELS_DIR}/{model_filename}'."
        )

    # Check if already present
    existing_path = find_model_path(model_filename)
    if existing_path:
        is_valid, reason = verify_model_file(existing_path)
        if is_valid:
            return True, f"Model '{model_filename}' is already available locally at {existing_path}."

    target_path = MODELS_DIR / model_filename
    temp_path = MODELS_DIR / f"{model_filename}.tmp"

    logger.info(f"Downloading {model_filename} from {url}...")

    last_error = None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ATCC-ModelManager/2.0",
        "Accept": "*/*",
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_retries} for {model_filename}...")
            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 256  # 256 KB chunks

                with open(temp_path, "wb") as f_out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)

            # Verify downloaded file
            is_valid, msg = verify_model_file(temp_path)
            if not is_valid:
                raise ValueError(f"Downloaded file verification failed: {msg}")

            # Atomic rename
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)
            logger.info(f"Successfully downloaded {model_filename} to {target_path}")
            return True, f"Successfully downloaded and verified '{model_filename}'."

        except Exception as exc:
            last_error = exc
            logger.warning(f"Attempt {attempt} failed: {exc}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(2 * attempt)

    manual_instructions = get_manual_placement_guide(model_filename)
    return False, (
        f"Download failed after {max_retries} attempts: {last_error}\n\n"
        f"{manual_instructions}"
    )


def get_manual_placement_guide(model_filename: str) -> str:
    """Return clear instructions for manual model placement without internet."""
    url = MODEL_REGISTRY.get(model_filename, {}).get("url", "https://github.com/ultralytics/assets/releases")
    return (
        f"📋 MANUAL MODEL PLACEMENT INSTRUCTIONS:\n"
        f"1. Download '{model_filename}' from:\n"
        f"   {url}\n"
        f"2. Copy the downloaded file directly to the ATCC models directory:\n"
        f"   {MODELS_DIR}\\{model_filename}\n"
        f"3. The application will detect and load it immediately without needing internet access."
    )


def load_model_safely(
    model_name_or_path: str,
    device: str = "cpu",
    allow_fallback: bool = True,
) -> tuple[object | None, str, dict]:
    """Safely load a YOLO model with local-first checking and fallback.

    Parameters
    ----------
    model_name_or_path : str
        Model filename (e.g. 'yolov8x.pt') or path.
    device : str
        'cpu' or 'cuda' / 0.
    allow_fallback : bool
        If True, falls back to best available local model when requested model is missing.

    Returns
    -------
    (model_instance, actual_model_name, info_dict)
    """
    model_filename = Path(model_name_or_path).name

    # Check cache
    cache_key = (model_filename, str(device))
    if cache_key in _LOADED_MODELS_CACHE:
        cached = _LOADED_MODELS_CACHE[cache_key]
        return cached["model"], cached["name"], cached["info"]

    local_path = find_model_path(model_filename)
    actual_model_name = model_filename
    fallback_used = False
    warning_msg = None

    if not local_path:
        if not allow_fallback:
            guide = get_manual_placement_guide(model_filename)
            info = {
                "success": False,
                "error": f"Model '{model_filename}' is not available locally.",
                "guide": guide,
                "fallback_used": False,
            }
            return None, "", info

        # Find best local alternative
        fallback_name, fallback_path = get_best_available_model(model_filename)
        if not fallback_path:
            guide = get_manual_placement_guide(model_filename)
            info = {
                "success": False,
                "error": f"No valid YOLO models found locally in '{MODELS_DIR}' or project root.",
                "guide": guide,
                "fallback_used": False,
            }
            return None, "", info

        fallback_used = True
        actual_model_name = fallback_name
        local_path = fallback_path
        warning_msg = (
            f"Requested model '{model_filename}' is not available locally. "
            f"Automatically using locally available '{fallback_name}' ({fallback_path.stat().st_size / (1024*1024):.1f} MB). "
            f"To use '{model_filename}', place it in '{MODELS_DIR}\\{model_filename}'."
        )
        logger.warning(warning_msg)

    # Now load using Ultralytics YOLO with absolute local path to prevent any GitHub download attempt
    try:
        from ultralytics import YOLO

        # Pass absolute local file path to YOLO so it never queries GitHub
        model = YOLO(str(local_path))

        info = {
            "success": True,
            "requested_model": model_filename,
            "loaded_model": actual_model_name,
            "model_path": str(local_path),
            "fallback_used": fallback_used,
            "warning": warning_msg,
            "device": device,
        }

        # Cache instance
        _LOADED_MODELS_CACHE[cache_key] = {
            "model": model,
            "name": actual_model_name,
            "info": info,
        }

        return model, actual_model_name, info

    except Exception as exc:
        logger.error(f"Failed to load YOLO model from {local_path}: {exc}")
        return None, "", {
            "success": False,
            "error": f"Failed to load model from {local_path}: {exc}",
            "guide": get_manual_placement_guide(model_filename),
            "fallback_used": fallback_used,
        }

