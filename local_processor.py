"""ATCC Local Processing Agent.

Runs entirely on the user's computer:
- Initializes YOLO models locally
- Reads video from local camera, video file, or assigned stream
- Executes vehicle detection, tracking, ROI density, and accident detection on local CPU/GPU
- Renders bounding boxes, tracking labels, and danger banners locally
- Transmits only structured metadata (counts, density, accidents) to the central server
- Closes and releases all resources cleanly on termination
"""
import argparse
import logging
import os
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import cv2
import torch

from .model_manager import (
    load_model_safely,
    get_available_local_models,
    MODELS_DIR,
)
from .video_processor import LocalVideoProcessor
from .accident_processor import format_accident_alert_payload
from .communication import ServerClient, get_hardware_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [LocalAgent] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("atcc.local_processor")

# Global reference for local MJPEG preview streaming
_LATEST_PROCESSED_JPEG = None
_JPEG_LOCK = threading.Lock()


class LocalMJPEGHandler(BaseHTTPRequestHandler):
    """Serves locally processed frames to localhost only (zero server bandwidth)."""

    def do_GET(self):
        global _LATEST_PROCESSED_JPEG
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            while True:
                with _JPEG_LOCK:
                    jpeg = _LATEST_PROCESSED_JPEG
                if jpeg is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except Exception:
                        break
                time.sleep(0.04)
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = (
                "<html><head><title>ATCC Local Engine Preview</title></head>"
                "<body style='background:#111; color:#eee; text-align:center; font-family:sans-serif;'>"
                "<h2>⚡ ATCC Local Processing Agent — Live Stream</h2>"
                "<p>Processing locally on your computer (0 KB server bandwidth used for video inference)</p>"
                "<img src='/stream' style='max-width:90%; border:2px solid #444; border-radius:8px;'/>"
                "</body></html>"
            )
            self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logging in terminal


def start_local_preview_server(port: int = 8585):
    """Start local MJPEG HTTP server on localhost."""
    try:
        server = HTTPServer(("127.0.0.1", port), LocalMJPEGHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        logger.info(f"Local browser preview running at http://127.0.0.1:{port}/")
        return server
    except Exception as exc:
        logger.warning(f"Could not start local preview server on port {port}: {exc}")
        return None


def open_video_source(source_str: str) -> tuple[cv2.VideoCapture | None, bool]:
    """Open a video capture from camera index, file path, or stream URL."""
    # Check if integer camera index
    if source_str.isdigit():
        cap = cv2.VideoCapture(int(source_str))
        return cap, False

    # Check if local file
    if os.path.exists(source_str):
        cap = cv2.VideoCapture(source_str)
        return cap, True

    # Check YouTube URL
    if "youtube.com" in source_str or "youtu.be" in source_str:
        logger.info(f"Resolving YouTube stream via yt-dlp: {source_str}")
        try:
            import yt_dlp
            ydl_opts = {
                "format": "best[ext=mp4]/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_str, download=False)
                direct_url = info.get("url")
                if direct_url:
                    cap = cv2.VideoCapture(direct_url)
                    return cap, False
        except Exception as exc:
            logger.error(f"Failed to resolve YouTube stream: {exc}")

    # Fallback to direct OpenCV capture
    cap = cv2.VideoCapture(source_str)
    return cap, False


def run_local_agent(
    server_url: str = "http://localhost:8502",
    username: str = "highway_operator",
    password: str = "user123",
    video_source: str = "0",
    feed_id: int = 1,
    camera_id: str = "CAM-01",
    feed_title: str = "Local Camera Feed",
    model_name: str = "yolov8n.pt",
    device: str = "auto",
    view_mode: str = "window",
    img_size: int = 640,
    conf_thresh: float = 0.30,
    frame_stride: int = 1,
    stop_event: threading.Event = None,
):
    """Main execution function of the Local ATCC Processing Agent."""
    global _LATEST_PROCESSED_JPEG
    logger.info("=" * 65)
    logger.info("ATCC CLIENT-SIDE LOCAL PROCESSING ENGINE")
    logger.info("=" * 65)

    # 1. Hardware Detection
    hw = get_hardware_info()
    logger.info(f"Hardware: {hw['device_name']} | Platform: {hw['platform']}")
    logger.info(f"CPU Utilization: {hw['cpu_percent']}% | RAM: {hw['memory_percent']}%")

    if device == "auto":
        target_device = 0 if hw["cuda_available"] else "cpu"
    else:
        target_device = device if (hw["cuda_available"] or device == "cpu") else "cpu"
    logger.info(f"Using Inference Device: {target_device}")

    # 2. Local Model Loading (Safely from models/ directory without GitHub crashes)
    logger.info(f"Loading vehicle detection model '{model_name}' locally...")
    vehicle_model, actual_model_name, model_info = load_model_safely(
        model_name,
        device=target_device,
        allow_fallback=True,
    )
    if not vehicle_model:
        logger.error(f"Failed to load model: {model_info.get('error')}")
        if model_info.get("guide"):
            print("\n" + model_info["guide"] + "\n")
        return False

    if model_info.get("fallback_used"):
        logger.warning(f"⚠️ {model_info['warning']}")
    else:
        logger.info(f"Loaded '{actual_model_name}' from {model_info['model_path']}")

    # Load auto-rickshaw model if available locally
    auto_model, _, _ = load_model_safely("best.pt", device=target_device, allow_fallback=False)
    if auto_model:
        logger.info("Custom 3-wheeler auto-rickshaw model loaded successfully.")

    # 3. Authenticate with Central Server
    client = ServerClient(server_url=server_url)
    auth_ok = client.authenticate(username, password)
    if not auth_ok:
        logger.warning(
            f"Could not authenticate with central server at {server_url}. "
            "Running in Standalone Local Mode (local processing active, telemetry buffered)."
        )

    # 4. Open Video Source
    logger.info(f"Opening video source: '{video_source}'...")
    cap, is_file = open_video_source(video_source)
    if not cap or not cap.isOpened():
        logger.error(f"Failed to open video source: '{video_source}'. Check camera connection or file path.")
        return False

    fps_nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    logger.info(f"Video source opened successfully. Nominal FPS: {fps_nominal:.1f}")

    # 5. Start Local Preview Server if requested
    if view_mode == "web":
        start_local_preview_server(port=8585)

    # 6. Initialize Local Processing Engine
    processor = LocalVideoProcessor(
        vehicle_model=vehicle_model,
        auto_model=auto_model,
        device=target_device,
        img_size=img_size,
        conf_thresh=conf_thresh,
        frame_stride=frame_stride,
    )

    frame_count = 0
    last_heartbeat_time = 0.0
    last_telemetry_time = 0.0
    transmitted_accidents = set()

    logger.info("Local processing loop started. Press 'q' or Ctrl+C to terminate.")

    try:
        while cap.isOpened():
            if stop_event and stop_event.is_set():
                logger.info("Stop event signaled.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logger.warning("Video stream ended or disconnected.")
                    break

            frame_count += 1
            run_inference = (frame_count == 1) or (frame_count % frame_stride == 0)

            # --- HEAVY PROCESSING RUNS ENTIRELY ON THIS USER COMPUTER ---
            processed_frame, telemetry = processor.process_frame(frame, run_inference=run_inference)

            # Update JPEG for local web preview if active
            if view_mode == "web":
                _, jpeg_buf = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with _JPEG_LOCK:
                    _LATEST_PROCESSED_JPEG = jpeg_buf.tobytes()

            # Render locally on OpenCV window if requested
            if view_mode == "window":
                cv2.imshow(f"ATCC Local Engine — {feed_title} (Press Q to exit)", processed_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    logger.info("Exit requested via local window.")
                    break

            # --- SYNCHRONIZE STRUCTURED RESULTS ONLY WITH CENTRAL SERVER ---
            now = time.time()

            # Send heartbeat every 3 seconds
            if now - last_heartbeat_time >= 3.0:
                client.send_heartbeat(
                    feed_id=feed_id,
                    feed_title=feed_title,
                    model_name=actual_model_name,
                    fps=telemetry["fps"],
                    total_frames=frame_count,
                    status="active",
                )
                last_heartbeat_time = now

            # Send traffic metrics telemetry every 1.5 seconds
            if now - last_telemetry_time >= 1.5:
                client.send_telemetry(feed_id=feed_id, camera_id=camera_id, telemetry=telemetry)
                last_telemetry_time = now

            # If accidents confirmed, immediately transmit structured accident event
            if telemetry.get("accidents"):
                for acc in telemetry["accidents"]:
                    inc_key = acc.get("incident_key")
                    if inc_key not in transmitted_accidents:
                        payload = format_accident_alert_payload(
                            event=acc,
                            feed_id=feed_id,
                            feed_title=feed_title,
                            camera_id=camera_id,
                            user_id=client.user.get("id", 0) if client.user else 0,
                        )
                        sent = client.send_accident_alert(payload)
                        if sent:
                            transmitted_accidents.add(inc_key)

    except KeyboardInterrupt:
        logger.info("Local agent interrupted by user.")
    finally:
        # Clean Resource Release
        logger.info("Cleaning up and releasing local camera, video, and GPU resources...")
        cap.release()
        cv2.destroyAllWindows()
        client.notify_stop(feed_id=feed_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All local resources released safely. Local engine stopped.")

    return True


def main():
    """Command-line interface for the ATCC Local Processing Agent."""
    parser = argparse.ArgumentParser(description="ATCC Local Processing Engine")
    parser.add_argument("--server", default="http://localhost:8502", help="Central ATCC Server URL")
    parser.add_argument("--user", default="highway_operator", help="ATCC Username")
    parser.add_argument("--password", default="user123", help="ATCC Password")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, file path, or stream URL)")
    parser.add_argument("--feed-id", type=int, default=1, help="Feed ID in ATCC system")
    parser.add_argument("--camera-id", default="CAM-01", help="Camera Identifier")
    parser.add_argument("--title", default="Local Feed", help="Feed title")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model (e.g. yolov8n.pt, yolov8x.pt)")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda:0")
    parser.add_argument("--view", default="window", choices=["window", "web", "none"], help="Local viewing mode")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold")
    args = parser.parse_args()

    run_local_agent(
        server_url=args.server,
        username=args.user,
        password=args.password,
        video_source=args.source,
        feed_id=args.feed_id,
        camera_id=args.camera_id,
        feed_title=args.title,
        model_name=args.model,
        device=args.device,
        view_mode=args.view,
        img_size=args.imgsz,
        conf_thresh=args.conf,
        frame_stride=args.stride,
    )


if __name__ == "__main__":
    main()

