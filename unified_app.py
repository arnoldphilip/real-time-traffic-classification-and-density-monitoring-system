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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ATCC_MODEL_PATH = "yolov8n.pt"
AUTO_MODEL_PATH = "best.pt"
DB_PATH = "unified_traffic_database.db"


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


# COCO classes used by ATCC.
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


def setup_database():
    """Create only the database tables required by the traffic monitoring system."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS VehicleCounts (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            vehicle_type TEXT,
            count INTEGER,
            lane TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TrafficEvents (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            event_type TEXT,
            description TEXT,
            lane TEXT,
            trigger_count INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TrafficDensity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total_vehicles INTEGER,
            density_level TEXT,
            car_count INTEGER,
            bus_count INTEGER,
            truck_count INTEGER,
            bike_count INTEGER,
            auto_count INTEGER,
            roi_bounds TEXT
        )
    """)
    conn.commit()
    conn.close()


setup_database()


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
        # Apply NMS to incoming detections to remove immediate spatial duplicates
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
        """Remove duplicate active tracker boxes on the same physical vehicle."""
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
        """Count active tracked vehicles inside the specified ROI bounds (ymin, xmin, ymax, xmax)."""
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
    """Traffic-control analysis based on detected vehicle counts."""

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
            conn = sqlite3.connect(DB_PATH)
            conn.execute(
                """INSERT INTO TrafficEvents
                   (timestamp, event_type, description, lane, trigger_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    "signal_change",
                    f"Heavy vehicles: {heavy}, Total: {total}",
                    "main",
                    total,
                ),
            )
            conn.commit()
            conn.close()
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
    """Apply Non-Maximum Suppression to eliminate duplicate overlapping bounding boxes."""
    if not detections:
        return []

    # Sort detections by confidence descending (giving preference to specialized auto class if tie)
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
            # Calculate overlap IoU
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
    """Detect, classify and track vehicles and estimate traffic density."""
    processed = frame.copy()
    h, w = frame.shape[:2]
    vehicle_detections = []

    if run_inference:
        with torch.inference_mode():
            # Existing detector: car / motorcycle / bus / truck.
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

            # Dedicated South-Asian detector: use only its 3-wheeler class as AUTO.
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

    # --- ROI & Traffic Density Calculation ---
    roi_total, roi_type_counts, roi_objects = tracker.count_in_roi(roi_bounds, w, h)

    if roi_total <= low_thresh:
        density_level = "Low"
        density_bgr = (0, 255, 0)      # Green
        badge_color = "#28a745"
    elif roi_total <= med_thresh:
        density_level = "Medium"
        density_bgr = (0, 255, 255)    # Yellow/Cyan
        badge_color = "#ffc107"
    else:
        density_level = "High"
        density_bgr = (0, 0, 255)      # Red
        badge_color = "#dc3545"

    # Draw ROI Box & Region Overlay
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

    # Draw boxes with tracker object ID, smoothed vehicle class, and confidence score
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

    # Single sleek vehicle count line
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
    try:
        conn = sqlite3.connect(DB_PATH)
        timestamp = datetime.now().isoformat()
        for vehicle_type, count in vehicle_counts.items():
            conn.execute(
                "INSERT INTO VehicleCounts (timestamp, vehicle_type, count, lane) VALUES (?, ?, ?, ?)",
                (timestamp, vehicle_type, count, "total"),
            )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"Database error saving vehicle counts: {exc}")


def save_density_to_database(roi_total, density_level, roi_type_counts, roi_bounds=None):
    """Store timestamp, total vehicle count, vehicle-type counts, and density level in SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        timestamp = datetime.now().isoformat()
        car_c = roi_type_counts.get("car", 0)
        bus_c = roi_type_counts.get("bus", 0)
        truck_c = roi_type_counts.get("truck", 0)
        bike_c = roi_type_counts.get("motorcycle", 0) + roi_type_counts.get("bike", 0)
        auto_c = roi_type_counts.get("auto", 0)

        conn.execute(
            """INSERT INTO TrafficDensity 
               (timestamp, total_vehicles, density_level, car_count, bus_count, truck_count, bike_count, auto_count, roi_bounds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                timestamp,
                roi_total,
                density_level,
                car_c,
                bus_c,
                truck_c,
                bike_c,
                auto_c,
                str(roi_bounds) if roi_bounds else "default",
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        print(f"Database error saving traffic density: {exc}")


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
):
    st.subheader("Real-Time Traffic Video & Camera Monitoring")
    
    # Real-time Traffic Density Dashboard metric placeholders
    density_metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    data_placeholder = st.empty()

    tracker = VehicleTracker()
    signal_manager = TrafficSignalManager()

    frame_count = 0
    last_save = datetime.now()
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if is_file else 600
    fps_history = []

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
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

        cv2.putText(
            processed,
            f"FPS: {avg_fps:.1f} ({'GPU' if device != 'cpu' else 'CPU'})",
            (processed.shape[1] - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Update Live Traffic Density Dashboard Cards continuously
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
                    st.metric("ROI Vehicle Count", f"{r_total}")
                with m3:
                    st.metric("Cars", type_counts.get("Car", 0))
                with m4:
                    st.metric("Buses", type_counts.get("Bus", 0))
                with m5:
                    st.metric("Trucks", type_counts.get("Truck", 0))
                with m6:
                    st.metric("Bikes", type_counts.get("Bike", 0))

        # Fast JPEG encoding reduces Streamlit WebSocket payload from 2.5MB to 50KB (10x faster frame rendering)
        _, jpeg_buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_placeholder.image(
            jpeg_buf.tobytes(),
            caption=f"Frame {frame_count}/{int(max_frames) if is_file else 'live'} | Speed: {avg_fps:.1f} FPS",
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

        # Update text/JSON widgets every 10 frames to eliminate Streamlit DOM re-render lag
        if frame_count % 10 == 0 or frame_count == 1:
            total_v = sum(counts.values())
            status_placeholder.markdown(
                f"**Frame:** {frame_count}/{int(max_frames) if is_file else 'live'} | **Processing Speed:** {avg_fps:.1f} FPS  \n"
                f"**ROI Traffic Density:** `{density_info['density_level']}` ({density_info['roi_total']} vehicles in ROI)  \n"
                f"**Total Vehicles (Line Crossed):** {total_v} | **Signal Status:** {signal_status}"
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
    frame_placeholder.empty()
    st.success("Traffic monitoring completed.")


def main():
    st.set_page_config(page_title="REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM", layout="wide")
    st.title("REAL TIME TRAFFIC CLASSIFICATION AND DENSITY MONITORING SYSTEM")
    st.markdown("Real-time vehicle detection, classification, tracking, traffic-density estimation and monitoring.")

    st.sidebar.markdown("### ⚡ Compute & GPU Speedup")
    gpu_info = get_gpu_info()

    device_map = {}
    if gpu_info["available"]:
        device_map["Auto (GPU Standard)"] = 0
        device_map[f"GPU: {gpu_info['name']} ({gpu_info['vram']})"] = 0
        device_map["CPU Mode"] = "cpu"
        st.sidebar.success(f"🚀 GPU Active: {gpu_info['name']}")
    else:
        device_map["CPU Mode (Standard)"] = "cpu"
        st.sidebar.info("🐢 Compute Mode: CPU")
        with st.sidebar.expander("💡 Enable GPU Speedup Instructions"):
            st.markdown("""
            To utilize NVIDIA GPU hardware for 5-10x faster inference speed:
            ```bash
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
            ```
            """)

    selected_device_key = st.sidebar.selectbox(
        "Inference Compute Device:",
        options=list(device_map.keys()),
        index=0,
        help="GPU acceleration speeds up frame processing and video analysis up to 10x faster."
    )
    target_device = device_map[selected_device_key]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Accuracy & Speed Tuning")
    selected_model_label = st.sidebar.selectbox(
        "Vehicle Detector Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,  # Default to Nano model for fastest real-time performance
        help="YOLOv8 Nano (Fastest/Lightweight) is recommended for high FPS video. Medium/Large models provide extra accuracy for complex scenes."
    )
    model_path = AVAILABLE_MODELS[selected_model_label]

    frame_stride = st.sidebar.select_slider(
        "Frame Skip / Stride (Speed Booster):",
        options=[1, 2, 3, 4],
        value=1,
        help="1 = Process every frame (>30-60 FPS on GPU). 2 = Process every 2nd frame (60+ FPS). 3 = Process every 3rd frame."
    )

    conf_thresh = st.sidebar.slider(
        "Vehicle Confidence Threshold:",
        min_value=0.05, max_value=0.90, value=0.30, step=0.05,
        help="Lower confidence detects smaller/distant vehicles; higher confidence reduces false positives."
    )

    auto_conf_thresh = st.sidebar.slider(
        "Auto-Rickshaw Confidence Threshold:",
        min_value=0.05, max_value=0.90, value=0.18, step=0.05,
        help="Confidence threshold for dedicated 3-wheeler auto-rickshaw detector."
    )

    iou_thresh = st.sidebar.slider(
        "Box Deduplication NMS IoU Threshold:",
        min_value=0.10, max_value=0.80, value=0.35, step=0.05,
        help="Controls strictness of duplicate bounding box removal. Lower values suppress overlapping boxes around the same vehicle more aggressively."
    )

    img_size = st.sidebar.select_slider(
        "Inference Image Resolution (imgsz):",
        options=[480, 640, 800, 960],
        value=480,
        help="480p gives ultra-high FPS (>40-60 FPS). 640 is standard real-time resolution."
    )

    with st.spinner(f"Loading {selected_model_label}..."):
        vehicle_model = load_vehicle_model(model_path)
        auto_model = load_auto_model()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏙️ Region of Interest (ROI) & Density")
    roi_y_range = st.sidebar.slider(
        "ROI Y-Axis Range (% of Frame):",
        min_value=0.0, max_value=1.0, value=(0.20, 0.95), step=0.05,
        help="Vertical boundary for ROI vehicle density counting."
    )
    roi_x_range = st.sidebar.slider(
        "ROI X-Axis Range (% of Frame):",
        min_value=0.0, max_value=1.0, value=(0.00, 1.00), step=0.05,
        help="Horizontal boundary for ROI vehicle density counting."
    )
    roi_bounds = (roi_y_range[0], roi_x_range[0], roi_y_range[1], roi_x_range[1])

    c_low, c_med = st.sidebar.columns(2)
    with c_low:
        low_thresh = st.number_input("Low Max", min_value=1, max_value=50, value=4, help="Max ROI vehicles for Low density")
    with c_med:
        med_thresh = st.number_input("Medium Max", min_value=2, max_value=100, value=10, help="Max ROI vehicles for Medium density")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.markdown(f"**Vehicle Detection Model:** {'Active (' + model_path + ')' if vehicle_model else 'Unavailable'}")
    st.sidebar.markdown(f"**Auto Model:** {'Active' if auto_model else 'Unavailable'}")
    if gpu_info["available"] and target_device != "cpu":
        st.sidebar.markdown(f"**Hardware Acceleration:** 🚀 GPU ({gpu_info['name']})")
    else:
        st.sidebar.markdown("**Hardware Acceleration:** 🐢 CPU")
    st.sidebar.markdown("**Vehicle classes:** Auto, Car, Motorcycle, Bus, Truck")

    with st.expander("💡 Tips for Ultra-High FPS (>30-60+ FPS)", expanded=False):
        st.markdown("""
        - **GPU Acceleration**: Select `Auto (GPU Standard)` for NVIDIA GTX 1650 hardware acceleration.
        - **Model Selection**: Use `YOLOv8 Nano` for maximum speed (>40-60 FPS).
        - **Inference Resolution**: Set `imgsz` to `480` for maximum GPU throughput.
        - **Frame Stride**: Set to `1` for full frame processing or `2` for double FPS.
        """)

    source = st.sidebar.radio(
        "Input Source:",
        ("Upload Video", "Upload Photo", "Use Webcam (Experimental)"),
    )

    if source == "Upload Video":
        st.header("Video File Upload")
        uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded:
            st.video(uploaded)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1] or ".mp4") as tmp:
                tmp.write(uploaded.read())
                path = tmp.name
            if st.button("Start Traffic Monitoring"):
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    st.error("Could not open the uploaded video.")
                else:
                    process_video(
                        cap,
                        vehicle_model,
                        auto_model,
                        conf_thresh=conf_thresh,
                        auto_conf_thresh=auto_conf_thresh,
                        iou_thresh=iou_thresh,
                        img_size=img_size,
                        device=target_device,
                        frame_stride=frame_stride,
                        roi_bounds=roi_bounds,
                        low_thresh=low_thresh,
                        med_thresh=med_thresh,
                    )
            try:
                os.unlink(path)
            except OSError:
                pass

    elif source == "Upload Photo":
        st.header("Image Analysis")
        uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            st.image(image, caption="Original Image", use_container_width=True)
            if st.button("Analyze Image"):
                tracker = VehicleTracker()
                signal_manager = TrafficSignalManager()
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
                    device=target_device,
                    roi_bounds=roi_bounds,
                    low_thresh=low_thresh,
                    med_thresh=med_thresh,
                )
                st.image(
                    cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                    caption="Traffic Classification & Density Monitoring Result",
                    use_container_width=True,
                )
                
                # Display Traffic Density Dashboard Metrics for Photo
                d_level = density_info["density_level"]
                r_total = density_info["roi_total"]
                type_counts = density_info["roi_counts"]
                b_color = density_info["badge_color"]

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
                    st.metric("ROI Vehicle Count", f"{r_total}")
                with m3:
                    st.metric("Cars", type_counts.get("Car", 0))
                with m4:
                    st.metric("Buses", type_counts.get("Bus", 0))
                with m5:
                    st.metric("Trucks", type_counts.get("Truck", 0))
                with m6:
                    st.metric("Bikes", type_counts.get("Bike", 0))

                st.json({
                    "Traffic Density Level": density_info["density_level"],
                    "Vehicles in ROI": density_info["roi_total"],
                    "Vehicle Counts in ROI": density_info["roi_counts"],
                    "Line Crossing Counts": dict(counts),
                    "Signal Status": signal_status,
                })

                # Save photo density result to SQLite database
                save_density_to_database(
                    roi_total=density_info["roi_total"],
                    density_level=density_info["density_level"],
                    roi_type_counts=density_info["raw_type_counts"],
                    roi_bounds=density_info["roi_bounds"],
                )

    else:
        st.header("Webcam Input")
        st.warning("Webcam mode runs for approximately 600 frames.")
        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open camera. Check camera permissions.")
            else:
                process_video(
                    cap,
                    vehicle_model,
                    auto_model,
                    conf_thresh=conf_thresh,
                    auto_conf_thresh=auto_conf_thresh,
                    iou_thresh=iou_thresh,
                    img_size=img_size,
                    device=target_device,
                    frame_stride=frame_stride,
                    roi_bounds=roi_bounds,
                    low_thresh=low_thresh,
                    med_thresh=med_thresh,
                )

    st.markdown("---")
    st.header("Traffic Monitoring Database & Density Analytics")
    if st.button("View Traffic Records"):
        conn = sqlite3.connect(DB_PATH)
        vehicle_df = pd.read_sql_query(
            "SELECT * FROM VehicleCounts ORDER BY timestamp DESC LIMIT 50", conn
        )
        events_df = pd.read_sql_query(
            "SELECT * FROM TrafficEvents ORDER BY timestamp DESC LIMIT 50", conn
        )
        density_df = pd.read_sql_query(
            "SELECT * FROM TrafficDensity ORDER BY timestamp DESC LIMIT 50", conn
        )
        conn.close()

        st.subheader("Traffic Density Estimation Records")
        st.dataframe(density_df)

        if not density_df.empty:
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Traffic Density Level Distribution")
                density_counts = density_df["density_level"].value_counts()
                st.bar_chart(density_counts)
            with col_chart2:
                st.markdown("#### Vehicles in ROI Over Time")
                if "total_vehicles" in density_df.columns:
                    st.line_chart(density_df.set_index("timestamp")["total_vehicles"])

        st.subheader("Vehicle Crossing Counts")
        st.dataframe(vehicle_df)
        st.subheader("Traffic Events")
        st.dataframe(events_df)


if __name__ == "__main__":
    main()

