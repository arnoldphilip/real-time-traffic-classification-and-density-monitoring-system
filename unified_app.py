"""Automatic Traffic Classification & Counting (ATCC) Unified Application.

Includes:
- Admin & User Authentication with bcrypt
- Role-based Dashboards (Admin full control, User own feeds only)
- Live MP4 Feeds with automatic playback & looping
- Three-dot menu for Pause/Resume and Delete operations
- YOLOv8 vehicle detection, centroid tracking, ROI density estimation & SQLite logging
"""
import os
import math
import time
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

import importlib
import auth
import database

# Force reload database and auth in case Streamlit kept cached modules in memory
importlib.reload(database)
importlib.reload(auth)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ATCC_MODEL_PATH = "yolov8n.pt"
AUTO_MODEL_PATH = "best.pt"
DB_PATH = database.DB_PATH


def setup_database():
    """Create ATCC database tables and initialize schema."""
    database.init_database(DB_PATH)


setup_database()


def get_gpu_info():
    """Return GPU information if PyTorch CUDA or MPS is available."""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return {
                "available": True,
                "type": "CUDA",
                "name": gpu_name,
                "count": device_count,
                "vram": f"{vram_gb:.1f} GB",
            }
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "available": True,
                "type": "MPS",
                "name": "Apple Silicon GPU",
                "count": 1,
                "vram": "Shared",
            }
    except Exception:
        pass
    return {"available": False, "type": "CPU", "name": "N/A", "count": 0, "vram": "N/A"}


# COCO classes used by ATCC
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
HEAVY_VEHICLES = {"bus", "truck"}

AVAILABLE_MODELS = {
    "YOLOv8 Nano (yolov8n.pt) - Lightweight": "yolov8n.pt",
    "YOLOv8 Small (yolov8s.pt) - Better Accuracy": "yolov8s.pt",
    "YOLOv8 Medium (yolov8m.pt) - High Accuracy (Recommended)": "yolov8m.pt",
    "YOLOv8 Large (yolov8l.pt) - Very High Precision": "yolov8l.pt",
    "YOLOv8 Extra Large (yolov8x.pt) - Maximum Precision": "yolov8x.pt",
}


@st.cache_resource
def load_vehicle_model(model_path):
    """Load selected YOLO model for vehicle detection."""
    try:
        return YOLO(model_path)
    except Exception as exc:
        st.error(f"Failed to load vehicle model ({model_path}): {exc}")
        return None


@st.cache_resource
def load_auto_model(auto_path=AUTO_MODEL_PATH):
    """Load auto-rickshaw detection model."""
    if not os.path.exists(auto_path):
        return None
    try:
        return YOLO(auto_path)
    except Exception as exc:
        st.error(f"Failed to load auto-rickshaw model ({auto_path}): {exc}")
        return None


class VehicleTracker:
    """Centroid tracker with class label smoothing and a single vehicle counting line."""

    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.class_history = defaultdict(list)
        self.max_disappeared = max_disappeared
        self.counting_line = {"y": 0.5, "counted": set()}
        self.counts = defaultdict(int)

    def update(self, detections, frame_height, nms_iou=0.35):
        detections = apply_nms(detections, iou_thresh=nms_iou)
        input_centroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            input_centroids.append((cx, cy, detection))

        if not input_centroids:
            for object_id in list(self.disappeared):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        elif not self.objects:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            distances = []
            for i, obj in enumerate(object_centroids):
                for j, inp in enumerate(input_centroids):
                    dist = math.hypot(obj[0] - inp[0], obj[1] - inp[1])
                    ex_box = obj[2]["bbox"]
                    cand_box = inp[2]["bbox"]
                    iou = box_iou(ex_box, cand_box)
                    effective_dist = dist * (1.0 - 0.5 * iou)
                    distances.append((effective_dist, i, j))

            distances.sort(key=lambda item: item[0])
            used_objects = set()
            used_inputs = set()

            for distance, object_idx, input_idx in distances:
                if object_idx in used_objects or input_idx in used_inputs:
                    continue
                if distance < 75:
                    object_id = object_ids[object_idx]
                    inp = input_centroids[input_idx]
                    det = dict(inp[2])

                    self.class_history[object_id].append(det.get("class", "vehicle"))
                    classes = self.class_history[object_id]
                    if "auto" in classes:
                        smoothed_class = "auto"
                    else:
                        smoothed_class = max(set(classes), key=classes.count)
                    det["class"] = smoothed_class

                    self.objects[object_id] = (inp[0], inp[1], det)
                    self.disappeared[object_id] = 0
                    used_objects.add(object_idx)
                    used_inputs.add(input_idx)

            for i, inp in enumerate(input_centroids):
                if i not in used_inputs:
                    self.register(inp)

            for i, object_id in enumerate(object_ids):
                if i not in used_objects:
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

        self.deduplicate_tracks(iou_thresh=nms_iou)
        self.check_lane_crossings(frame_height)
        return self.objects

    def register(self, centroid):
        det = dict(centroid[2])
        bbox = det["bbox"]
        for existing_id, existing_centroid in self.objects.items():
            ex_box = existing_centroid[2]["bbox"]
            if box_iou(bbox, ex_box) >= 0.35:
                return

        self.class_history[self.next_object_id].append(det.get("class", "vehicle"))
        self.objects[self.next_object_id] = (centroid[0], centroid[1], det)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deduplicate_tracks(self, iou_thresh=0.35):
        object_ids = list(self.objects.keys())
        to_remove = set()
        for i in range(len(object_ids)):
            id1 = object_ids[i]
            if id1 in to_remove or id1 not in self.objects:
                continue
            box1 = self.objects[id1][2]["bbox"]
            for j in range(i + 1, len(object_ids)):
                id2 = object_ids[j]
                if id2 in to_remove or id2 not in self.objects:
                    continue
                box2 = self.objects[id2][2]["bbox"]
                if box_iou(box1, box2) >= iou_thresh:
                    cls1 = self.objects[id1][2].get("class", "vehicle")
                    cls2 = self.objects[id2][2].get("class", "vehicle")
                    if cls1 == "auto" and cls2 != "auto":
                        to_remove.add(id2)
                    elif cls2 == "auto" and cls1 != "auto":
                        to_remove.add(id1)
                        break
                    else:
                        conf1 = self.objects[id1][2].get("confidence", 0.0)
                        conf2 = self.objects[id2][2].get("confidence", 0.0)
                        if conf1 >= conf2:
                            to_remove.add(id2)
                        else:
                            to_remove.add(id1)
                            break
        for obj_id in to_remove:
            self.deregister(obj_id)

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)
        self.class_history.pop(object_id, None)

    def check_lane_crossings(self, frame_height):
        line_y = int(self.counting_line["y"] * frame_height)
        counted = self.counting_line["counted"]
        for object_id, centroid in self.objects.items():
            if object_id in counted or centroid[1] <= line_y:
                continue
            counted.add(object_id)
            detection = centroid[2]
            vehicle_type = detection.get("class", "vehicle")
            self.counts[vehicle_type] += 1

    def count_in_roi(self, roi_bounds, frame_width, frame_height):
        ymin_pct, xmin_pct, ymax_pct, xmax_pct = roi_bounds
        roi_y1 = ymin_pct * frame_height
        roi_x1 = xmin_pct * frame_width
        roi_y2 = ymax_pct * frame_height
        roi_x2 = xmax_pct * frame_width

        roi_type_counts = defaultdict(int)
        roi_objects = {}

        for obj_id, centroid in self.objects.items():
            cx, cy, det = centroid
            if roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2:
                vehicle_type = det.get("class", "vehicle")
                roi_type_counts[vehicle_type] += 1
                roi_objects[obj_id] = centroid

        roi_total = sum(roi_type_counts.values())
        return roi_total, roi_type_counts, roi_objects


class TrafficSignalManager:
    """ATCC traffic-control logic based on detected vehicle counts."""

    def __init__(self):
        self.signal_state = "normal"
        self.signal_change_time = None
        self.signal_duration = 45

    def check_traffic_conditions(self, vehicle_counts):
        current_time = time.time()
        if self.signal_change_time and current_time - self.signal_change_time < self.signal_duration:
            remaining = int(self.signal_duration - (current_time - self.signal_change_time))
            return f"Signal Priority Active ({remaining}s remaining)"

        if self.signal_change_time:
            self.signal_state = "normal"
            self.signal_change_time = None

        heavy = sum(vehicle_counts.get(v, 0) for v in HEAVY_VEHICLES)
        total = sum(vehicle_counts.values())

        if heavy > 5 or total > 15:
            self.signal_state = "priority"
            self.signal_change_time = current_time
            database.save_traffic_event(
                "signal_change",
                f"Heavy vehicles: {heavy}, Total: {total}",
                lane="main",
                trigger_count=total,
            )
            return f"Signal Priority Active ({self.signal_duration}s) - Heavy: {heavy}, Total: {total}"

        return "Traffic flow normal"


def box_iou(box_a, box_b):
    """Return IoU for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def apply_nms(detections, iou_thresh=0.35):
    """Apply Non-Maximum Suppression to eliminate duplicate bounding boxes."""
    if not detections:
        return []

    sorted_dets = sorted(
        detections,
        key=lambda d: (d.get("class") == "auto", d.get("confidence", 0.0)),
        reverse=True,
    )
    keep = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        remaining = []
        for det in sorted_dets:
            iou = box_iou(best["bbox"], det["bbox"])
            if iou < iou_thresh:
                remaining.append(det)
        sorted_dets = remaining

    return keep


def process_frame(
    frame,
    vehicle_model,
    auto_model,
    tracker,
    signal_manager,
    conf_thresh=0.30,
    auto_conf_thresh=0.25,
    iou_thresh=0.35,
    img_size=640,
    device=0,
    run_inference=True,
    roi_bounds=(0.20, 0.00, 0.95, 1.00),
    low_thresh=4,
    med_thresh=10,
):
    """Detect, classify, track and count vehicles for ATCC with Traffic Density Estimation."""
    processed = frame.copy()
    h, w = frame.shape[:2]
    vehicle_detections = []

    if run_inference:
        with torch.inference_mode():
            if vehicle_model is not None:
                results = vehicle_model.predict(
                    frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    imgsz=img_size,
                    device=device,
                    verbose=False,
                )
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        if class_id not in VEHICLE_CLASSES:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].item())
                        vehicle_type = VEHICLE_CLASSES[class_id]
                        vehicle_detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "class": vehicle_type,
                            "confidence": confidence,
                        })

            if auto_model is not None:
                auto_results = auto_model.predict(
                    frame,
                    conf=auto_conf_thresh,
                    iou=iou_thresh,
                    imgsz=img_size,
                    device=device,
                    verbose=False,
                    classes=[2],
                )
                for result in auto_results:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        if class_id != 2:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].item())
                        vehicle_detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "class": "auto",
                            "confidence": confidence,
                        })

        auto_boxes = [d["bbox"] for d in vehicle_detections if d["class"] == "auto"]
        if auto_boxes:
            vehicle_detections = [
                d for d in vehicle_detections
                if d["class"] == "auto"
                or not any(box_iou(d["bbox"], abox) >= 0.15 for abox in auto_boxes)
            ]

        vehicle_detections = apply_nms(vehicle_detections, iou_thresh=iou_thresh)
        tracker.update(vehicle_detections, h, nms_iou=iou_thresh)
    else:
        tracker.check_lane_crossings(h)

    roi_total, roi_type_counts, roi_objects = tracker.count_in_roi(roi_bounds, w, h)

    if roi_total <= low_thresh:
        density_level = "Low"
        density_bgr = (0, 255, 0)
        badge_color = "#28a745"
    elif roi_total <= med_thresh:
        density_level = "Medium"
        density_bgr = (0, 255, 255)
        badge_color = "#ffc107"
    else:
        density_level = "High"
        density_bgr = (0, 0, 255)
        badge_color = "#dc3545"

    ymin_pct, xmin_pct, ymax_pct, xmax_pct = roi_bounds
    roi_y1 = int(ymin_pct * h)
    roi_x1 = int(xmin_pct * w)
    roi_y2 = int(ymax_pct * h)
    roi_x2 = int(xmax_pct * w)

    overlay = processed.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), density_bgr, -1)
    cv2.addWeighted(overlay, 0.10, processed, 0.90, 0, processed)
    cv2.rectangle(processed, (roi_x1, roi_y1), (roi_x2, roi_y2), density_bgr, 2)

    roi_header = f"ROI DENSITY: {density_level.upper()} ({roi_total} Vehicles)"
    cv2.putText(
        processed, roi_header, (roi_x1 + 10, max(25, roi_y1 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, density_bgr, 2
    )

    for object_id, centroid in tracker.objects.items():
        cx, cy, detection = centroid
        x1, y1, x2, y2 = detection["bbox"]
        vehicle_type = detection.get("class", "vehicle")
        confidence = detection.get("confidence", 0.0)

        in_roi = object_id in roi_objects
        color = (255, 0, 255) if vehicle_type == "auto" else ((0, 0, 255) if vehicle_type in HEAVY_VEHICLES else (0, 255, 0))
        thickness = 3 if in_roi else 2
        cv2.rectangle(processed, (x1, y1), (x2, y2), color, thickness)

        label = f"#{object_id} {vehicle_type}: {confidence:.2f}"
        if in_roi:
            label += " [ROI]"

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_y = max(y1, text_size[1] + 10)
        cv2.rectangle(
            processed,
            (x1, label_y - text_size[1] - 10),
            (x1 + text_size[0], label_y),
            color,
            -1,
        )
        cv2.putText(
            processed, label, (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
        )

    line_y = int(tracker.counting_line["y"] * h)
    cv2.line(processed, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(
        processed, "COUNTING LINE", (10, max(20, line_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )

    signal_status = signal_manager.check_traffic_conditions(tracker.counts)

    total_vehicles = sum(tracker.counts.values())
    cv2.putText(
        processed, f"Line Crossings: {total_vehicles} | ROI Density: {density_level}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
    )
    y_offset = 55
    for vehicle_type, count in tracker.counts.items():
        cv2.putText(
            processed, f"  {vehicle_type}: {count}", (15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        y_offset += 22

    cv2.putText(
        processed, signal_status, (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )

    density_info = {
        "density_level": density_level,
        "roi_total": roi_total,
        "roi_counts": {
            "Car": roi_type_counts.get("car", 0),
            "Bus": roi_type_counts.get("bus", 0),
            "Truck": roi_type_counts.get("truck", 0),
            "Bike": roi_type_counts.get("motorcycle", 0) + roi_type_counts.get("bike", 0),
            "Auto": roi_type_counts.get("auto", 0),
        },
        "raw_type_counts": dict(roi_type_counts),
        "badge_color": badge_color,
        "roi_bounds": roi_bounds,
    }

    return processed, tracker.counts, signal_status, density_info


def save_data_to_database(vehicle_counts):
    """Save line crossing counts to database."""
    try:
        for vehicle_type, count in vehicle_counts.items():
            database.save_vehicle_count(vehicle_type, count, lane="total")
    except Exception as exc:
        print(f"Database error saving vehicle counts: {exc}")


def save_density_to_database(roi_total, density_level, roi_type_counts, roi_bounds=None):
    """Save traffic density records to database."""
    try:
        database.save_density_to_database(roi_total, density_level, roi_type_counts, roi_bounds or (0.2, 0.0, 0.95, 1.0))
    except Exception as exc:
        print(f"Database error saving traffic density: {exc}")


def get_stream_capture(video_url: str):
    """
    Open an OpenCV VideoCapture instance.
    Supports local files, direct MP4 URLs, RTSP streams, and YouTube video URLs
    (resolves stream manifest via yt-dlp).
    """
    if not video_url:
        return None, "Empty video URL provided."

    resolved_url = video_url.strip()
    is_youtube = ("youtube.com" in resolved_url.lower()) or ("youtu.be" in resolved_url.lower())

    if is_youtube:
        try:
            import yt_dlp
            ydl_opts = {
                "format": "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]/best",
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(resolved_url, download=False)
                resolved_url = info.get("url") or resolved_url
        except Exception as exc:
            return None, f"Failed to extract YouTube stream: {exc}"

    cap = cv2.VideoCapture(resolved_url)
    if not cap.isOpened():
        return None, f"Unable to open video stream from '{video_url}'. Please verify URL or file accessibility."
    return cap, None


def process_video(
    cap,
    vehicle_model,
    auto_model,
    conf_thresh=0.25,
    auto_conf_thresh=0.18,
    iou_thresh=0.45,
    img_size=480,
    device=0,
    frame_stride=1,
    roi_bounds=(0.20, 0.00, 0.95, 1.00),
    low_thresh=4,
    med_thresh=10,
    stop_key="stop_stream",
):
    """Process video or live feed with real-time YOLO tracking and density metrics."""
    st.subheader("⚡ ATCC Live Video Stream Processing")

    stop_col1, stop_col2 = st.columns([1, 4])
    with stop_col1:
        stop_button = st.button("⏹️ Stop Stream Processing", key=stop_key)

    # Prominent traffic alert banner (persists when High until traffic becomes Low)
    alert_placeholder = st.empty()
    density_metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    data_placeholder = st.empty()

    high_alert_active = False

    tracker = VehicleTracker()
    signal_manager = TrafficSignalManager()

    frame_count = 0
    last_save = datetime.now()
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if is_file else 600
    fps_history = []

    while cap.isOpened():
        if stop_button:
            st.info("Stream processing stopped by user.")
            break

        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            # If looping is enabled for video file stream, rewind
            if is_file and cap.get(cv2.CAP_PROP_POS_FRAMES) >= max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame_count += 1

        h, w = frame.shape[:2]
        if w > 854:
            frame = cv2.resize(frame, (854, int(854 * h / w)))

        run_inference = (frame_count == 1) or (frame_count % frame_stride == 0)

        processed, counts, signal_status, density_info = process_frame(
            frame,
            vehicle_model,
            auto_model,
            tracker,
            signal_manager,
            conf_thresh=conf_thresh,
            auto_conf_thresh=auto_conf_thresh,
            iou_thresh=iou_thresh,
            img_size=img_size,
            device=device,
            run_inference=run_inference,
            roi_bounds=roi_bounds,
            low_thresh=low_thresh,
            med_thresh=med_thresh,
        )

        elapsed = time.time() - t0
        current_fps = 1.0 / max(elapsed, 0.001)
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # --- Traffic Density Alert: show banner when HIGH, clear when LOW ---
        d_level_current = density_info["density_level"].lower()
        if d_level_current == "high" and not high_alert_active:
            alert_placeholder.error(
                "🚨 **HIGH TRAFFIC ALERT** — Traffic density is HIGH. Please monitor the scene carefully."
            )
            high_alert_active = True
        elif d_level_current in ("low", "medium") and high_alert_active:
            alert_placeholder.empty()
            high_alert_active = False

        cv2.putText(
            processed,
            f"FPS: {avg_fps:.1f} ({'GPU' if device != 'cpu' else 'CPU'})",
            (processed.shape[1] - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if frame_count == 1 or frame_count % frame_stride == 0:
            d_level = density_info["density_level"]
            r_total = density_info["roi_total"]
            type_counts = density_info["roi_counts"]
            b_color = density_info["badge_color"]

            with density_metrics_placeholder.container():
                m1, m2, m3, m4, m5, m6 = st.columns([2, 2, 1.5, 1.5, 1.5, 1.5])
                with m1:
                    st.markdown(
                        f"""
                        <div style="background-color:{b_color}; padding:8px 12px; border-radius:8px; text-align:center; color:white; font-family:sans-serif;">
                            <div style="font-size: 11px; font-weight: bold; letter-spacing: 1px;">TRAFFIC DENSITY</div>
                            <div style="font-size: 22px; font-weight: 800; margin-top:-2px;">{d_level.upper()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.metric("ROI Vehicles", f"{r_total}")
                with m3:
                    st.metric("Cars", type_counts.get("Car", 0))
                with m4:
                    st.metric("Buses", type_counts.get("Bus", 0))
                with m5:
                    st.metric("Trucks", type_counts.get("Truck", 0))
                with m6:
                    st.metric("Bikes", type_counts.get("Bike", 0))

        _, jpeg_buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_placeholder.image(
            jpeg_buf.tobytes(),
            caption=f"Frame {frame_count}/{int(max_frames) if is_file else 'live'} | Processing Speed: {avg_fps:.1f} FPS",
        )

        now = datetime.now()
        if (now - last_save).seconds >= 10:
            save_data_to_database(counts)
            save_density_to_database(
                roi_total=density_info["roi_total"],
                density_level=density_info["density_level"],
                roi_type_counts=density_info["raw_type_counts"],
                roi_bounds=density_info["roi_bounds"],
            )
            last_save = now

        if frame_count % 10 == 0 or frame_count == 1:
            total_v = sum(counts.values())
            status_placeholder.markdown(
                f"**Status:** Active | **Speed:** {avg_fps:.1f} FPS | "
                f"**ROI Traffic Density:** `{density_info['density_level']}` ({density_info['roi_total']} in ROI) | "
                f"**Total Line Crossings:** {total_v} | **Signal:** {signal_status}"
            )
            data_placeholder.json({
                "Traffic Density Level": density_info["density_level"],
                "Vehicles in ROI": density_info["roi_total"],
                "ROI Vehicle Types": density_info["roi_counts"],
                "Line Crossing Counts": dict(counts),
                "Signal Status": signal_status,
                "FPS": round(avg_fps, 1),
            })

        if not is_file and frame_count >= 600:
            break

    cap.release()
    st.success("ATCC feed stream processing ended.")


# -------------------------------------------------------------
# Views: Authentication, Admin & User Dashboards
# -------------------------------------------------------------

def render_login_view():
    """Render Login & Registration screen."""
    st.markdown(
        """
        <style>
        .login-hero {
            background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 60%, #0d3b2e 100%);
            border-radius: 16px;
            padding: 36px 24px 28px 24px;
            text-align: center;
            margin-bottom: 28px;
        }
        .login-hero h1 { color: #4a90d9; font-size: 2.4rem; margin:0; letter-spacing:1px; }
        .login-hero p  { color: #8eacc9; font-size: 14px; margin-top: 8px; }
        .badge-pill {
            display: inline-block;
            background: rgba(74,144,217,0.15);
            border: 1px solid rgba(74,144,217,0.4);
            color: #4a90d9;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin: 4px 3px;
        }
        </style>
        <div class="login-hero">
            <div style="font-size:52px; margin-bottom:10px;">🚦</div>
            <h1>ATCC Platform</h1>
            <p>Automatic Traffic Classification &amp; Counting</p>
            <div style="margin-top:12px;">
                <span class="badge-pill">🤖 AI-Powered</span>
                <span class="badge-pill">📹 Live Feeds</span>
                <span class="badge-pill">📊 Real-time Analytics</span>
                <span class="badge-pill">🔒 Role-based Access</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_col1, login_col2, login_col3 = st.columns([1, 2, 1])
    with login_col2:
        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Register User"])

        with tab_login:
            st.markdown("#### Account Login")
            with st.form("login_form"):
                username_input = st.text_input("Username", placeholder="e.g. admin or highway_operator")
                password_input = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_login = st.form_submit_button("Sign In to ATCC", use_container_width=True)

            if submit_login:
                user, msg = auth.authenticate(username_input, password_input)
                if user:
                    auth.login_user(user)
                    st.success(f"Welcome back, {user.get('full_name') or user['username']}!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("---")
            st.markdown("**⚡ Quick Login Credentials (Demo):**")
            c_adm, c_u1, c_u2 = st.columns(3)
            with c_adm:
                if st.button("🛡️ Admin\n(admin123)", use_container_width=True):
                    u, _ = auth.authenticate("admin", "admin123")
                    if u:
                        auth.login_user(u)
                        st.rerun()
            with c_u1:
                if st.button("🛣️ User 101\n(user123)", use_container_width=True):
                    u, _ = auth.authenticate("highway_operator", "user123")
                    if u:
                        auth.login_user(u)
                        st.rerun()
            with c_u2:
                if st.button("🏙️ User 102\n(user123)", use_container_width=True):
                    u, _ = auth.authenticate("city_operator", "user123")
                    if u:
                        auth.login_user(u)
                        st.rerun()

        with tab_register:
            st.markdown("#### Create New User Account")
            with st.form("register_form"):
                reg_username = st.text_input("Username*", placeholder="Unique username")
                reg_name = st.text_input("Full Name", placeholder="Your full name")
                reg_email = st.text_input("Email", placeholder="operator@traffic.local")
                reg_pass1 = st.text_input("Password*", type="password")
                reg_pass2 = st.text_input("Confirm Password*", type="password")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True)

            if submit_reg:
                if not reg_username or not reg_pass1:
                    st.error("Username and password are required.")
                elif reg_pass1 != reg_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = database.create_user(
                        reg_username, reg_pass1, role="user", full_name=reg_name, email=reg_email
                    )
                    if ok:
                        st.success(msg + " You can now sign in.")
                    else:
                        st.error(msg)


def render_live_feeds_view(current_user: dict, is_admin_all_view: bool = False, model_args: dict = None):
    """
    Render Video Feeds with:
    - Autoplay and looping
    - Three-dot hover/click menu: Pause/Resume and Delete
    - Strict user isolation (Users see only their own feeds; Admins see all)
    - Live ATCC YOLO detection trigger
    """
    is_admin_user = current_user.get("role") == "admin"
    section_title = "🎥 All Live Video Feeds" if (is_admin_user and is_admin_all_view) else "🎥 My Live Video Feeds"
    st.subheader(section_title)

    # --- Add New Video Feed Section ---
    with st.expander("➕ Add Live Video Feed (MP4 / RTSP / Stream URL)", expanded=False):
        with st.form(f"add_feed_form_{'admin' if is_admin_all_view else 'user'}"):
            f_title = st.text_input("Camera / Feed Title*", placeholder="e.g. Highway Northbound Camera")
            f_url = st.text_input(
                "Video Stream / MP4 / RTSP URL*",
                placeholder="https://example.com/feed.mp4  or  rtsp://camera-ip/stream",
                help="Supports direct MP4 links, RTSP streams, and local video file paths.",
            )
            c_loop, c_status = st.columns(2)
            with c_loop:
                f_loop = st.checkbox("Loop Video Continuously", value=True)
            with c_status:
                f_status = st.selectbox("Initial Feed Status", ["active", "paused"])

            target_owner_id = None
            if is_admin_user:
                users_list = database.get_all_users()
                user_options = {f"{u['username']} (ID: {u['id']}, {u['role']})": u["id"] for u in users_list}
                selected_owner_label = st.selectbox(
                    "Assign Stream Link To User:*",
                    list(user_options.keys()),
                    help="The selected user will be the only standard user who can see and manage this stream link."
                )
                target_owner_id = user_options[selected_owner_label]

            feed_submitted = st.form_submit_button("Add & Assign Live Feed", use_container_width=True)

        if feed_submitted:
            ok, msg, new_id = database.add_video_feed(
                title=f_title,
                video_url=f_url,
                current_user=current_user,
                loop=f_loop,
                status=f_status,
                target_user_id=target_owner_id,
            )
            if ok:
                st.success(msg)
                time.sleep(0.3)
                st.rerun()
            else:
                st.error(msg)

    # --- Search and Filter Controls ---
    if is_admin_user:
        f_col1, f_col2, f_col3 = st.columns([3, 1, 2])
        with f_col1:
            search_query = st.text_input("🔍 Search Feeds by Title or URL", placeholder="Type keyword...", label_visibility="collapsed")
        with f_col2:
            status_filter = st.selectbox("Status Filter", ["all", "active", "paused"], label_visibility="collapsed")
        with f_col3:
            all_users_list = database.get_all_users()
            user_filter_options = {"All Users (All Streams)": "all"}
            user_filter_options.update({f"Assigned: {u['username']} (ID: {u['id']})": u["id"] for u in all_users_list})
            selected_user_filter = st.selectbox("Filter by Assigned User", list(user_filter_options.keys()), label_visibility="collapsed")
            assigned_user_id = user_filter_options[selected_user_filter]
    else:
        f_col1, f_col2 = st.columns([3, 1])
        with f_col1:
            search_query = st.text_input("🔍 Search Feeds by Title or URL", placeholder="Type keyword...", label_visibility="collapsed")
        with f_col2:
            status_filter = st.selectbox("Status Filter", ["all", "active", "paused"], label_visibility="collapsed")
        assigned_user_id = None

    # Fetch feeds according to authorization rules
    feeds = database.get_video_feeds(current_user, search_query=search_query, status_filter=status_filter, assigned_user_id=assigned_user_id)

    if not feeds:
        st.info("No video feeds found. Click '➕ Add Live Video Feed' above to add your first camera feed.")
        return

    st.markdown(f"**Showing {len(feeds)} camera feed(s)**")

    # Render video cards in 2 columns
    for idx, feed in enumerate(feeds):
        fid = feed["id"]
        status = feed["status"]
        is_loop = bool(feed["loop"])
        owner_str = feed.get("owner_username") or f"User #{feed['user_id']}"

        with st.container():
            # Compact card header: just title + status indicator + actions menu
            left_col, right_col = st.columns([5, 1])
            with left_col:
                dot_color = "#28a745" if status == "active" else "#9e9e9e"
                st.markdown(
                    f'<span style="color:{dot_color}; font-size:10px;">&#9679;</span> '
                    f'<strong style="font-size:15px;">{feed["title"]}</strong>',
                    unsafe_allow_html=True,
                )
            with right_col:
                # Three-dot popover menu for Pause, Resume, and Delete operations
                with st.popover("⋮ Actions", use_container_width=True):
                    st.markdown(f"**{feed['title']}**")
                    if status == "active":
                        if st.button("⏸️ Pause Feed", key=f"pause_{fid}", use_container_width=True):
                            ok, msg = database.pause_video_feed(fid, current_user)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    else:
                        if st.button("▶️ Resume Feed", key=f"resume_{fid}", use_container_width=True):
                            ok, msg = database.resume_video_feed(fid, current_user)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)

                    new_loop = st.checkbox("🔁 Loop Video", value=is_loop, key=f"loop_chk_{fid}")
                    if new_loop != is_loop:
                        database.toggle_loop_video_feed(fid, new_loop, current_user)
                        st.rerun()

                    st.markdown("---")
                    if st.button("🗑️ Delete Feed", key=f"del_{fid}", use_container_width=True):
                        ok, msg = database.delete_video_feed(fid, current_user)
                        if ok:
                            st.warning(msg)
                            st.rerun()
                        else:
                            st.error(msg)

                    if is_admin_user:
                        st.markdown("---")
                        st.markdown("👤 **Assign Stream to User:**")
                        all_users_for_assign = database.get_all_users()
                        assign_opts = {f"{u['username']} (ID: {u['id']})": u["id"] for u in all_users_for_assign}
                        cur_idx = 0
                        for u_idx, (u_lbl, uid) in enumerate(assign_opts.items()):
                            if uid == feed["user_id"]:
                                cur_idx = u_idx
                                break
                        sel_user_choice = st.selectbox(
                            "Choose User:",
                            list(assign_opts.keys()),
                            index=cur_idx,
                            key=f"reassign_sel_{fid}"
                        )
                        if st.button("🔗 Reassign Stream", key=f"reassign_btn_{fid}", use_container_width=True):
                            new_uid = assign_opts[sel_user_choice]
                            ok, msg = database.assign_video_feed(fid, new_uid, current_user)
                            if ok:
                                st.success(msg)
                                time.sleep(0.3)
                                st.rerun()
                            else:
                                st.error(msg)

            # Small video preview (wrapped in a narrow column to reduce display size)
            if status == "active":
                preview_col, spacer_col = st.columns([2, 3])
                with preview_col:
                    try:
                        st.video(
                            feed["video_url"],
                            autoplay=True,
                            loop=is_loop,
                            muted=True,
                        )
                    except Exception as exc:
                        st.caption(f"Preview unavailable: {exc}")

                # ATCC YOLO Inference Execution Button
                with st.expander("⚡ Run ATCC Live Detection on this Stream"):
                    if st.button(f"Start YOLO Detection ({feed['title']})", key=f"run_yolo_{fid}"):
                        with st.spinner("Connecting to live video stream..."):
                            cap, err_msg = get_stream_capture(feed["video_url"])
                        if not cap or not cap.isOpened():
                            st.error(
                                err_msg or (
                                    f"Unable to open video stream from '{feed['video_url']}'. "
                                    "Please verify the stream URL is accessible."
                                )
                            )
                        else:
                            if model_args:
                                process_video(
                                    cap,
                                    model_args["vehicle_model"],
                                    model_args["auto_model"],
                                    conf_thresh=model_args["conf_thresh"],
                                    auto_conf_thresh=model_args["auto_conf_thresh"],
                                    iou_thresh=model_args["iou_thresh"],
                                    img_size=model_args["img_size"],
                                    device=model_args["device"],
                                    frame_stride=model_args["frame_stride"],
                                    roi_bounds=model_args["roi_bounds"],
                                    low_thresh=model_args["low_thresh"],
                                    med_thresh=model_args["med_thresh"],
                                    stop_key=f"stop_{fid}",
                                )
            else:
                st.caption("⏸️ Feed paused — open ⋮ Actions to resume.")

            # Collapsed details section (hidden by default, shown on click)
            with st.expander("ℹ️ Details", expanded=False):
                status_label = "🟢 Active" if status == "active" else "⏸️ Paused"
                owner_label = f"👤 Assigned to: {owner_str}" if is_admin_user else "👤 Assigned to You"
                loop_label = "🔁 Looping" if is_loop else "No Loop"
                st.caption(f"{status_label}  |  {owner_label}  |  {loop_label}")
                st.caption(f"Source: `{feed['video_url']}`  •  Feed ID: `{fid}`")

            st.divider()


def render_all_videos_view(current_user: dict, model_args: dict = None):
    """Admin view: Full table and management view of all user video records."""
    if not auth.is_admin(current_user):
        st.error("Access denied: Administrator privileges required.")
        return

    st.subheader("📹 All Videos Management (Admin)")
    st.markdown("Full administrative view of all registered video camera feeds across all user accounts.")

    feeds = database.get_video_feeds(current_user)

    if not feeds:
        st.info("No video records exist.")
        return

    # Stream Link Assignment Quick Tool
    with st.expander("⚡ Assign / Transfer Stream Link to User Account", expanded=True):
        st.caption("Admin can assign any video stream link directly to any user. That user will only see the streams assigned to them.")
        col_st1, col_st2, col_st3 = st.columns([3, 3, 2])
        with col_st1:
            feed_choices = {f"#{f['id']}: {f['title']} (Assigned to: {f.get('owner_username') or f['user_id']})": f["id"] for f in feeds}
            selected_f_lbl = st.selectbox("Select Stream Link:", list(feed_choices.keys()), key="mgr_feed_sel")
            selected_f_id = feed_choices[selected_f_lbl]
        with col_st2:
            all_u = database.get_all_users()
            user_choices = {f"{u['username']} (ID: {u['id']}, Role: {u['role']})": u["id"] for u in all_u}
            selected_u_lbl = st.selectbox("Assign To User Account:", list(user_choices.keys()), key="mgr_user_sel")
            selected_u_id = user_choices[selected_u_lbl]
        with col_st3:
            st.write("")
            st.write("")
            if st.button("🔗 Confirm Assignment", use_container_width=True, key="mgr_btn_assign"):
                ok, msg = database.assign_video_feed(selected_f_id, selected_u_id, current_user)
                if ok:
                    st.success(msg)
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(msg)

    df_feeds = pd.DataFrame(feeds)
    display_cols = ["id", "title", "user_id", "owner_username", "status", "loop", "video_url", "created_at"]
    available_cols = [c for c in display_cols if c in df_feeds.columns]

    st.dataframe(df_feeds[available_cols], use_container_width=True)

    st.markdown("---")
    render_live_feeds_view(current_user, is_admin_all_view=True, model_args=model_args)


def render_admin_dashboard(current_user: dict):
    """Admin Overview Dashboard with system KPI metrics and audit trail."""
    stats = database.get_database_stats()
    all_users = database.get_all_users()
    all_feeds = database.get_video_feeds(current_user)
    active_feeds_cnt = sum(1 for f in all_feeds if f["status"] == "active")
    paused_feeds_cnt = len(all_feeds) - active_feeds_cnt

    # --- KPI metric cards ---
    k1, k2, k3, k4, k5 = st.columns(5)
    def _kpi(col, icon, label, value, bg="#1a2744"):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:{bg};
                    border-radius:12px;
                    padding:16px 14px 12px 14px;
                    text-align:center;
                    border: 1px solid rgba(255,255,255,0.08);
                ">
                    <div style="font-size:28px;">{icon}</div>
                    <div style="font-size:26px; font-weight:900; color:#e6edf3; margin:4px 0 2px 0;">{value}</div>
                    <div style="font-size:11px; color:#8eacc9; text-transform:uppercase; letter-spacing:0.8px;">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    _kpi(k1, "👥", "Total Users",       len(all_users),                   "#1a2744")
    _kpi(k2, "🟢", "Active Feeds",      active_feeds_cnt,                  "#0d3b2e")
    _kpi(k3, "⏸️", "Paused Feeds",      paused_feeds_cnt,                  "#3b2a0d")
    _kpi(k4, "🚗", "Vehicle Crossings", stats.get("VehicleCounts", 0),     "#2a1a44")
    _kpi(k5, "💾", "DB Size",           f"{stats.get('db_file_size_kb',0)} KB", "#1a3344")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            "<div style='font-weight:700; font-size:14px; color:#4a90d9; margin-bottom:10px;'>📹 Live Feed Status</div>",
            unsafe_allow_html=True,
        )
        if all_feeds:
            for f in all_feeds[:6]:
                dot = "🟢" if f["status"] == "active" else "⏸️"
                owner = f.get("owner_username") or f"User #{f['user_id']}"
                st.markdown(
                    f"""
                    <div style="
                        display:flex; align-items:center; justify-content:space-between;
                        background:#f8fafc; border-radius:8px; padding:8px 12px; margin-bottom:6px;
                        border-left: 3px solid {'#28a745' if f['status']=='active' else '#9e9e9e'};
                    ">
                        <div>
                            <span style="font-weight:600; font-size:13px;">{f['title']}</span>
                            <span style="color:#666; font-size:11px; margin-left:8px;">→ {owner}</span>
                        </div>
                        <span style="font-size:14px;">{dot}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No feeds configured.")

    with col_right:
        st.markdown(
            "<div style='font-weight:700; font-size:14px; color:#4a90d9; margin-bottom:10px;'>🛡️ Recent Activity Audit</div>",
            unsafe_allow_html=True,
        )
        logs = database.get_activity_logs(limit=8)
        if logs:
            df_logs = pd.DataFrame(logs)[["timestamp", "username", "action", "details"]]
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("No activity logged yet.")


def render_manage_users_view(current_user: dict):
    """Admin-only User Management view."""
    if not auth.is_admin(current_user):
        st.error("Access denied: Administrator privileges required.")
        return

    st.subheader("👥 User Management (Admin)")

    with st.expander("➕ Add New User", expanded=False):
        with st.form("create_user_admin_form"):
            c1, c2 = st.columns(2)
            with c1:
                u_username = st.text_input("Username*")
                u_name = st.text_input("Full Name")
                u_email = st.text_input("Email")
            with c2:
                u_pass = st.text_input("Password*", type="password")
                u_role = st.selectbox("Role", ["user", "admin"])

            create_sub = st.form_submit_button("Create Account", use_container_width=True)

        if create_sub:
            ok, msg = database.create_user(
                u_username, u_pass, role=u_role, full_name=u_name, email=u_email, current_user=current_user
            )
            if ok:
                st.success(msg)
                time.sleep(0.3)
                st.rerun()
            else:
                st.error(msg)

    users = database.get_all_users()
    st.markdown(
        f"<span style='color:#4a90d9; font-weight:700;'>{len(users)}</span> registered accounts",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    for u in users:
        is_self = (u["id"] == current_user["id"])
        u_avatar = (u["full_name"] or u["username"])[0].upper()
        u_role_color = "#e63946" if u["role"] == "admin" else "#2a9d8f"
        u_status_color = "#28a745" if u["status"] == "active" else "#dc3545"
        u_status_icon = "🟢" if u["status"] == "active" else "🔴"

        with st.container():
            left, right = st.columns([5, 2])
            with left:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:14px; padding:10px 0;">
                        <div style="
                            background:{u_role_color}; color:white; width:42px; height:42px;
                            border-radius:50%; display:flex; align-items:center; justify-content:center;
                            font-size:18px; font-weight:bold; flex-shrink:0;
                        ">{u_avatar}</div>
                        <div>
                            <div style="font-weight:700; font-size:14px;">
                                {u['full_name'] or u['username']}
                                <code style="font-size:11px; background:#f0f4f8; padding:1px 6px; border-radius:4px; margin-left:6px;">{u['username']}</code>
                                {'<span style="font-size:11px; color:#aaa; margin-left:6px;">(You)</span>' if is_self else ''}
                            </div>
                            <div style="font-size:12px; color:#666; margin-top:3px;">
                                📧 {u['email'] or 'No email'} &nbsp;•&nbsp;
                                <span style="color:{u_role_color}; font-weight:600;">{'🛡️ Admin' if u['role']=='admin' else '👤 User'}</span> &nbsp;•&nbsp;
                                <span style="color:{u_status_color};">{u_status_icon} {u['status'].capitalize()}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with right:
                if not is_self:
                    btn_a, btn_b = st.columns(2)
                    with btn_a:
                        toggle_label = "Deactivate" if u["status"] == "active" else "Activate"
                        target_status = "inactive" if u["status"] == "active" else "active"
                        if st.button(toggle_label, key=f"stat_{u['id']}", use_container_width=True):
                            database.update_user_status(u["id"], target_status, current_user)
                            st.rerun()
                    with btn_b:
                        if st.button("🗑️ Delete", key=f"del_u_{u['id']}", use_container_width=True):
                            ok, msg = database.delete_user(u["id"], current_user)
                            if ok:
                                st.warning(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                else:
                    st.caption("Current session")
            st.divider()


def render_analytics_view():
    """Traffic Density, Crossing Counts, and Events Analytics View."""
    st.subheader("📈 Traffic Density & Flow Analytics")

    conn = database.get_db_connection()
    density_df = pd.read_sql_query("SELECT * FROM TrafficDensity ORDER BY id DESC LIMIT 100", conn)
    counts_df = pd.read_sql_query("SELECT * FROM VehicleCounts ORDER BY id DESC LIMIT 100", conn)
    events_df = pd.read_sql_query("SELECT * FROM TrafficEvents ORDER BY id DESC LIMIT 100", conn)
    logs_df = pd.read_sql_query("SELECT * FROM ActivityLogs ORDER BY id DESC LIMIT 100", conn)
    conn.close()

    tab_density, tab_counts, tab_events, tab_logs = st.tabs([
        "🏙️ Traffic Density", "🚗 Vehicle Crossings", "🚨 Traffic Events", "🛡️ Activity Audit Logs"
    ])

    with tab_density:
        st.markdown("#### Traffic Density Distribution & Timeline")
        if not density_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Density Level Breakdown**")
                st.bar_chart(density_df["density_level"].value_counts())
            with c2:
                st.markdown("**ROI Total Vehicles Over Time**")
                if "timestamp" in density_df.columns:
                    st.line_chart(density_df.set_index("timestamp")["total_vehicles"])
            st.dataframe(density_df, use_container_width=True)
        else:
            st.info("No traffic density records logged yet.")

    with tab_counts:
        st.markdown("#### Vehicle Crossing Counts")
        if not counts_df.empty:
            st.bar_chart(counts_df["vehicle_type"].value_counts())
            st.dataframe(counts_df, use_container_width=True)
        else:
            st.info("No vehicle crossing records logged yet.")

    with tab_events:
        st.markdown("#### Automated Signal Control Events")
        if not events_df.empty:
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("No traffic signal events recorded yet.")

    with tab_logs:
        st.markdown("#### System Activity Logs")
        if not logs_df.empty:
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No activity logs.")


def render_settings_view(model_args: dict, target_device: str):
    """System configuration, detector tuning, and standalone file testing."""
    st.subheader("⚙️ System Configuration & Diagnostics")

    cfg = auth.load_config()
    st.markdown("#### Active ATCC JSON Configuration (`atcc_config.json`)")
    st.json(cfg)

    st.markdown("---")
    st.markdown("#### 🧪 Standalone Media Diagnostic Tools")
    source = st.radio(
        "Choose Diagnostic Source:",
        ("Upload Video", "Upload Photo", "Webcam"),
        horizontal=True,
    )

    if source == "Upload Video":
        uploaded = st.file_uploader("Upload MP4 / AVI / MOV video", type=["mp4", "avi", "mov"])
        if uploaded:
            st.video(uploaded)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1] or ".mp4") as tmp:
                tmp.write(uploaded.read())
                path = tmp.name
            if st.button("Start ATCC Processing on File"):
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    st.error("Could not open video file.")
                else:
                    process_video(
                        cap,
                        model_args["vehicle_model"],
                        model_args["auto_model"],
                        conf_thresh=model_args["conf_thresh"],
                        auto_conf_thresh=model_args["auto_conf_thresh"],
                        iou_thresh=model_args["iou_thresh"],
                        img_size=model_args["img_size"],
                        device=target_device,
                        frame_stride=model_args["frame_stride"],
                        roi_bounds=model_args["roi_bounds"],
                        low_thresh=model_args["low_thresh"],
                        med_thresh=model_args["med_thresh"],
                    )
            try:
                os.unlink(path)
            except OSError:
                pass

    elif source == "Upload Photo":
        uploaded = st.file_uploader("Upload Photo", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.image(image, caption="Original Image", use_container_width=True)
            if st.button("Analyze Image"):
                tracker = VehicleTracker()
                signal_manager = TrafficSignalManager()
                processed, counts, signal_status, density_info = process_frame(
                    frame,
                    model_args["vehicle_model"],
                    model_args["auto_model"],
                    tracker,
                    signal_manager,
                    conf_thresh=model_args["conf_thresh"],
                    auto_conf_thresh=model_args["auto_conf_thresh"],
                    iou_thresh=model_args["iou_thresh"],
                    img_size=model_args["img_size"],
                    device=target_device,
                    roi_bounds=model_args["roi_bounds"],
                    low_thresh=model_args["low_thresh"],
                    med_thresh=model_args["med_thresh"],
                )
                st.image(
                    cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                    caption="Detection & Density Estimation Result",
                    use_container_width=True,
                )
                st.json({
                    "Traffic Density Level": density_info["density_level"],
                    "Vehicles in ROI": density_info["roi_total"],
                    "Vehicle Counts in ROI": density_info["roi_counts"],
                    "Line Crossing Counts": dict(counts),
                    "Signal Status": signal_status,
                })
                save_density_to_database(
                    roi_total=density_info["roi_total"],
                    density_level=density_info["density_level"],
                    roi_type_counts=density_info["raw_type_counts"],
                    roi_bounds=density_info["roi_bounds"],
                )

    elif source == "Webcam":
        st.warning("Webcam mode runs for approximately 600 frames.")
        if st.button("Start Webcam"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open camera. Verify camera device connection.")
            else:
                process_video(
                    cap,
                    model_args["vehicle_model"],
                    model_args["auto_model"],
                    conf_thresh=model_args["conf_thresh"],
                    auto_conf_thresh=model_args["auto_conf_thresh"],
                    iou_thresh=model_args["iou_thresh"],
                    img_size=model_args["img_size"],
                    device=target_device,
                    frame_stride=model_args["frame_stride"],
                    roi_bounds=model_args["roi_bounds"],
                    low_thresh=model_args["low_thresh"],
                    med_thresh=model_args["med_thresh"],
                )


# -------------------------------------------------------------
# Main Application Entry Point
# -------------------------------------------------------------

def _page_header(title: str, subtitle: str, icon: str, color: str = "#1e3a5f"):
    """Render a styled page section header banner."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {color} 0%, #0d1b2a 100%);
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <span style="font-size:36px; line-height:1;">{icon}</span>
            <div>
                <div style="color:white; font-size:22px; font-weight:800; letter-spacing:0.3px;">{title}</div>
                <div style="color:#a0b8d0; font-size:13px; margin-top:2px;">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="ATCC — Traffic Intelligence Platform",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Global CSS overrides for a cleaner, more modern look
    st.markdown(
        """
        <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1b2a 0%, #16213e 100%);
        }
        [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
        /* Pill-style metric cards */
        div[data-testid="metric-container"] {
            background: #f0f4f8;
            border-radius: 10px;
            padding: 10px 14px;
            border: 1px solid #dde3ea;
        }
        /* Divider color */
        hr { border-color: #2a3a4a !important; }
        /* Sidebar nav buttons: style real Streamlit buttons to look like nav items */
        [data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            border: none !important;
            border-left: 3px solid transparent !important;
            color: #c9d1d9 !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            text-align: left !important;
            padding: 9px 14px !important;
            border-radius: 6px !important;
            width: 100% !important;
            margin: 1px 0 !important;
            transition: background 0.15s ease !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(74,144,217,0.12) !important;
            border-left: 3px solid #4a90d9 !important;
            color: #4a90d9 !important;
        }
        /* Expander header styling */
        .streamlit-expanderHeader { font-weight: 600 !important; }
        /* Feed card divider */
        [data-testid="stDivider"] { margin: 6px 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    auth.init_session_state()
    config = auth.load_config()
    current_user = auth.get_current_user()

    # If user is not authenticated, render login page
    if not current_user:
        render_login_view()
        return

    # User is authenticated
    is_admin_user = (current_user.get("role") == "admin")

    # ----------------------------------------------------------------
    # SIDEBAR — Branding
    # ----------------------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 18px 0 10px 0;">
            <div style="font-size:38px;">🚦</div>
            <div style="font-size:19px; font-weight:900; color:#4a90d9; letter-spacing:2px;">ATCC</div>
            <div style="font-size:10px; color:#6b7280; letter-spacing:1px; text-transform:uppercase;">
                Traffic Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<hr style='margin:6px 0 14px 0;'>", unsafe_allow_html=True)

    # ---- User identity card ----
    display_name = current_user.get("full_name") or current_user["username"]
    avatar_letter = display_name[0].upper()
    role_color = "#e63946" if is_admin_user else "#2a9d8f"
    role_label = "Administrator" if is_admin_user else "Operator"
    role_icon = "🛡️" if is_admin_user else "👤"
    st.sidebar.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 12px 14px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <div style="
                background:{role_color};
                color:white;
                width:38px; height:38px;
                border-radius:50%;
                display:flex; align-items:center; justify-content:center;
                font-size:18px; font-weight:bold; flex-shrink:0;
            ">{avatar_letter}</div>
            <div style="overflow:hidden;">
                <div style="font-weight:bold; font-size:13px; color:#e6edf3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{display_name}</div>
                <div style="font-size:11px; color:{role_color}; margin-top:2px;">{role_icon} {role_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Navigation Menu ----
    # Admin nav items: (label, icon, key)
    ADMIN_NAV = [
        ("Dashboard",     "📊", "nav_dashboard"),
        ("Live Feeds",    "📹", "nav_livefeeds"),
        ("All Videos",    "🎬", "nav_allvideos"),
        ("Manage Users",  "👥", "nav_users"),
        ("Analytics",     "📈", "nav_analytics"),
        ("Settings",      "⚙️",  "nav_settings"),
    ]
    USER_NAV = [
        ("My Live Feeds", "📹", "nav_myfeeds"),
        ("Settings",      "⚙️",  "nav_usettings"),
    ]

    nav_items = ADMIN_NAV if is_admin_user else USER_NAV

    # Use session state to track active page
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = nav_items[0][0]

    st.sidebar.markdown(
        f"<div style='font-size:10px; color:#4a90d9; letter-spacing:1.5px; text-transform:uppercase; "
        f"font-weight:bold; margin-bottom:6px; padding-left:4px;'>"
        f"{'ADMIN CONSOLE' if is_admin_user else 'OPERATOR PANEL'}</div>",
        unsafe_allow_html=True,
    )

    selected_nav = st.session_state.active_nav
    for label, icon, key in nav_items:
        is_active = (selected_nav == label)
        # Show the active page highlighted via markdown, then a real clickable button
        if is_active:
            st.sidebar.markdown(
                f"""
                <div style="
                    background:#1e3a5f;
                    border-left:3px solid #4a90d9;
                    border-radius:6px;
                    padding:9px 14px;
                    margin:2px 0;
                    display:flex; align-items:center; gap:10px;
                    pointer-events:none;
                ">
                    <span style="font-size:16px;">{icon}</span>
                    <span style="color:#4a90d9; font-size:13px; font-weight:bold;">{label}</span>
                    <span style="margin-left:auto; color:#4a90d9; font-size:10px;">●</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.sidebar.button(f"{icon}  {label}", key=key, use_container_width=True):
                st.session_state.active_nav = label
                st.rerun()
    selected_nav = st.session_state.active_nav

    # ---- Logout ----
    st.sidebar.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    if st.sidebar.button("🚪  Sign Out", use_container_width=True, key="logout_btn"):
        auth.logout_user()
        st.session_state.pop("active_nav", None)
        st.rerun()

    # ---- Compute & Model Settings ----
    st.sidebar.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size:10px; color:#4a90d9; letter-spacing:1.5px; text-transform:uppercase; font-weight:bold; margin-bottom:6px; padding-left:4px;'>DETECTION ENGINE</div>",
        unsafe_allow_html=True,
    )

    gpu_info = get_gpu_info()
    device_map = {}
    if gpu_info["available"]:
        device_map["Auto (GPU Standard)"] = 0
        device_map[f"GPU: {gpu_info['name']} ({gpu_info['vram']})"] = 0
        device_map["CPU Mode"] = "cpu"
        st.sidebar.success(f"🚀 GPU: {gpu_info['name']}")
    else:
        device_map["CPU Mode (Standard)"] = "cpu"
        st.sidebar.info("🐢 CPU Mode")

    selected_device_key = st.sidebar.selectbox("Inference Device:", options=list(device_map.keys()), index=0)
    target_device = device_map[selected_device_key]

    selected_model_label = st.sidebar.selectbox("YOLO Model:", options=list(AVAILABLE_MODELS.keys()), index=0)
    model_path = AVAILABLE_MODELS[selected_model_label]

    with st.sidebar.expander("🎯 Advanced Tuning", expanded=False):
        frame_stride = st.select_slider("Frame Stride:", options=[1, 2, 3, 4], value=1)
        conf_thresh = st.slider("Confidence:", 0.05, 0.90, 0.30, 0.05)
        auto_conf_thresh = st.slider("Auto-Rickshaw Conf:", 0.05, 0.90, 0.18, 0.05)
        iou_thresh = st.slider("NMS IoU:", 0.10, 0.80, 0.35, 0.05)
        img_size = st.select_slider("Resolution:", options=[480, 640, 800, 960], value=480)
        roi_y_range = st.slider("ROI Y-Range:", 0.0, 1.0, (0.20, 0.95), 0.05)
        roi_x_range = st.slider("ROI X-Range:", 0.0, 1.0, (0.00, 1.00), 0.05)
        c_low, c_med = st.columns(2)
        with c_low:
            low_thresh = st.number_input("Low Max", 1, 50, 4)
        with c_med:
            med_thresh = st.number_input("Med Max", 2, 100, 10)
    # Defaults when expander not interacted with
    if "frame_stride" not in dir():
        frame_stride = 1
    roi_bounds = (roi_y_range[0], roi_x_range[0], roi_y_range[1], roi_x_range[1])

    with st.spinner("Loading YOLO models..."):
        vehicle_model = load_vehicle_model(model_path)
        auto_model = load_auto_model()

    model_args = {
        "vehicle_model": vehicle_model,
        "auto_model": auto_model,
        "conf_thresh": conf_thresh,
        "auto_conf_thresh": auto_conf_thresh,
        "iou_thresh": iou_thresh,
        "img_size": img_size,
        "device": target_device,
        "frame_stride": frame_stride,
        "roi_bounds": roi_bounds,
        "low_thresh": low_thresh,
        "med_thresh": med_thresh,
    }

    # ----------------------------------------------------------------
    # MAIN AREA — Top header bar
    # ----------------------------------------------------------------
    if is_admin_user:
        nav_icon_map = {"Dashboard": "📊", "Live Feeds": "📹", "All Videos": "🎬",
                        "Manage Users": "👥", "Analytics": "📈", "Settings": "⚙️"}
        nav_sub_map = {
            "Dashboard": "System overview, KPIs & audit logs",
            "Live Feeds": "Monitor and manage all live camera streams",
            "All Videos": "Full admin view & stream assignment",
            "Manage Users": "Create, activate, and manage user accounts",
            "Analytics": "Traffic density & vehicle crossing analytics",
            "Settings": "System configuration, diagnostics & testing",
        }
        header_color = "#1a2744"
    else:
        nav_icon_map = {"My Live Feeds": "📹", "Settings": "⚙️"}
        nav_sub_map = {
            "My Live Feeds": "Your assigned camera streams",
            "Settings": "System configuration & diagnostics",
        }
        header_color = "#0d3b2e"

    page_icon = nav_icon_map.get(selected_nav, "🚦")
    page_sub  = nav_sub_map.get(selected_nav, "ATCC Traffic Platform")
    _page_header(selected_nav, page_sub, page_icon, header_color)

    # ----------------------------------------------------------------
    # RENDER SELECTED VIEW
    # ----------------------------------------------------------------
    if selected_nav == "Dashboard":
        render_admin_dashboard(current_user)
    elif selected_nav in ["Live Feeds", "My Live Feeds"]:
        render_live_feeds_view(current_user, is_admin_all_view=False, model_args=model_args)
    elif selected_nav == "All Videos":
        render_all_videos_view(current_user, model_args=model_args)
    elif selected_nav == "Manage Users":
        render_manage_users_view(current_user)
    elif selected_nav == "Analytics":
        render_analytics_view()
    elif selected_nav == "Settings":
        render_settings_view(model_args, target_device)


if __name__ == "__main__":
    main()
