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
import streamlit.components.v1 as components
from scene_cut import SceneCutDetector
import torch
from PIL import Image
from ultralytics import YOLO

import importlib
import auth
import database
import local_engine.model_manager as model_mgr
import server_api

# Force reload database and auth in case Streamlit kept cached modules in memory
importlib.reload(database)
importlib.reload(auth)

# Start Central Server REST API in background on port 8502 (for local processing agents)
try:
    server_api.start_server_api(host="0.0.0.0", port=8502)
except Exception:
    pass

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ATCC_MODEL_PATH = "yolov8n.pt"
AUTO_MODEL_PATH = "best.pt"
# All frames are scaled to this width before detection/tracking so that every pixel-based threshold
# (speeds, overlaps, ROI) behaves the same for 480p, 720p, 1080p or 4K sources. 0 = keep native size.
PROCESS_MAX_WIDTH = 960
ROAD_TYPE_MODES = {
    "Auto-detect": "UNKNOWN",
    "Highway": "HIGHWAY",
    "City street": "NORMAL_STREET",
    "Intersection": "INTERSECTION",
    "Residential road": "RESIDENTIAL_ROAD",
}
DB_PATH = database.DB_PATH


def setup_database():
    """Create ATCC database tables and initialize schema."""
    database.init_database(DB_PATH)


setup_database()


@st.cache_resource
def _close_stale_accidents_once():
    """Run once per server start: old unresolved accidents from past sessions
    must not show DANGER when no video is running."""
    return database.expire_stale_accident_events()


_close_stale_accidents_once()


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
    """Safely load selected YOLO model for vehicle detection without GitHub connection crashes."""
    model, loaded_name, info = model_mgr.load_model_safely(
        model_path,
        device="cpu",
        allow_fallback=True,
    )
    if not model:
        st.error(f"Failed to load vehicle model ({model_path}): {info.get('error')}")
        if info.get("guide"):
            st.info(info["guide"])
        return None
    return model


@st.cache_resource
def load_auto_model(auto_path=AUTO_MODEL_PATH):
    """Load auto-rickshaw detection model from local models directory or root."""
    model, _, _ = model_mgr.load_model_safely(auto_path, device="cpu", allow_fallback=False)
    return model


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
                pred_vx = obj[2].get("vx", 0.0)
                pred_vy = obj[2].get("vy", 0.0)
                pred_cx = obj[0] + pred_vx
                pred_cy = obj[1] + pred_vy
                ex_box = obj[2]["bbox"]
                ex_cls = obj[2].get("class", "vehicle")

                for j, inp in enumerate(input_centroids):
                    dist = math.hypot(pred_cx - inp[0], pred_cy - inp[1])
                    cand_box = inp[2]["bbox"]
                    cand_cls = inp[2].get("class", "vehicle")
                    iou = box_iou(ex_box, cand_box)
                    effective_dist = dist * (1.0 - 0.5 * iou)
                    if ex_cls != cand_cls:
                        effective_dist *= 1.25
                    distances.append((effective_dist, i, j))

            distances.sort(key=lambda item: item[0])
            used_objects = set()
            used_inputs = set()

            for distance, object_idx, input_idx in distances:
                if object_idx in used_objects or input_idx in used_inputs:
                    continue
                obj = object_centroids[object_idx]
                prev_cx, prev_cy = obj[0], obj[1]
                prev_vx = obj[2].get("vx", 0.0)
                prev_vy = obj[2].get("vy", 0.0)
                max_gate = max(80.0, math.hypot(prev_vx, prev_vy) * 2.8)

                if distance < max_gate:
                    object_id = object_ids[object_idx]
                    inp = input_centroids[input_idx]
                    det = dict(inp[2])

                    inst_vx = inp[0] - prev_cx
                    inst_vy = inp[1] - prev_cy
                    vx = 0.65 * inst_vx + 0.35 * prev_vx
                    vy = 0.65 * inst_vy + 0.35 * prev_vy
                    det["vx"] = vx
                    det["vy"] = vy
                    det["track_age"] = obj[2].get("track_age", 0) + 1

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

        det["vx"] = 0.0
        det["vy"] = 0.0
        det["track_age"] = 1
        self.class_history[self.next_object_id].append(det.get("class", "vehicle"))
        self.objects[self.next_object_id] = (centroid[0], centroid[1], det)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deduplicate_tracks(self, iou_thresh=0.35):
        """
        Remove duplicate tracker entries for the same physical vehicle.

        CRITICAL: Do NOT merge two genuinely different vehicles that happen to overlap
        in the camera frame due to a collision or perspective foreshortening.
        Only merge tracks when BOTH boxes heavily overlap AND centroids are very close
        (< 25px), indicating the same physical YOLO detection tracked twice.
        """
        object_ids = list(self.objects.keys())
        to_remove = set()
        for i in range(len(object_ids)):
            id1 = object_ids[i]
            if id1 in to_remove or id1 not in self.objects:
                continue
            box1 = self.objects[id1][2]["bbox"]
            p1 = (self.objects[id1][0], self.objects[id1][1])
            for j in range(i + 1, len(object_ids)):
                id2 = object_ids[j]
                if id2 in to_remove or id2 not in self.objects:
                    continue
                box2 = self.objects[id2][2]["bbox"]
                p2 = (self.objects[id2][0], self.objects[id2][1])
                centroid_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                iou = box_iou(box1, box2)
                # Only deduplicate when BOTH: high IoU AND centroids are very close
                # This prevents merging two vehicles that are genuinely in a collision
                if iou >= iou_thresh and centroid_dist < 25.0:
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

    def get_active_objects(self):
        """Return only active objects detected in the current frame (disappeared == 0)."""
        return {oid: self.objects[oid] for oid in self.objects if self.disappeared.get(oid, 0) == 0}


# Road-Aware Accident Detection & Incident Processing Engine
from local_engine.accident_processor import (
    ground_footprint_iou,
    is_camera_boundary,
    LocalAccidentDetector as AccidentDetector,
    format_accident_alert_payload,
)


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
    accident_detector=None,
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
    """Detect, classify, track, count vehicles and detect accidents for ATCC."""
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
                        bw = x2 - x1
                        bh = y2 - y1
                        if bw * bh < 400 or bw < 14 or bh < 14:
                            continue

                        # Aspect ratio filter: eliminate tree trunks, vertical poles, horizontal wires
                        aspect = float(bw) / float(bh)
                        if aspect < 0.28 or aspect > 4.2:
                            continue

                        # Sky & Horizon exclusion: vehicles cannot drive in sky
                        if float(y2) <= 0.05 * h:
                            continue

                        # Road ROI & Exclusion Zone Filtering: strictly reject houses, trees, sidewalks, buildings
                        if accident_detector is not None and hasattr(accident_detector, "road_roi") and accident_detector.road_roi is not None:
                            if not accident_detector.road_roi.bbox_ground_on_road((x1, y1, x2, y2), w, h):
                                continue

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
                        bw = x2 - x1
                        bh = y2 - y1
                        if bw * bh < 400 or bw < 14 or bh < 14:
                            continue

                        aspect = float(bw) / float(bh)
                        if aspect < 0.28 or aspect > 4.2:
                            continue

                        if float(y2) <= 0.05 * h:
                            continue

                        if accident_detector is not None and hasattr(accident_detector, "road_roi") and accident_detector.road_roi is not None:
                            if not accident_detector.road_roi.bbox_ground_on_road((x1, y1, x2, y2), w, h):
                                continue

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

    # --- Accident & Suspicious Traffic Detection (Evaluated before rendering) ---
    accidents = []
    suspicious_events = []
    if accident_detector is not None:
        accidents, suspicious_events = accident_detector.update(
            tracker.objects, w, h, frame=frame, disappeared=getattr(tracker, "disappeared", None)
        )

    # Collect accident-involved vehicle IDs and damage info
    accident_vehicle_ids = set()
    accident_vehicle_meta = {}
    for acc in accidents:
        for vid in acc.get("vehicle_ids", []):
            accident_vehicle_ids.add(vid)
            dmg_info = acc.get("damage_assessment", {}).get(vid, {})
            accident_vehicle_meta[vid] = {
                "conf": acc.get("confidence", "90%"),
                "severity": acc.get("severity", "HIGH"),
                "damage": dmg_info,
            }

    # Collect verifying / suspicious vehicle IDs
    suspicious_vehicle_ids = set()
    if accident_detector is not None:
        for cand in getattr(accident_detector, "verifying_candidates", []):
            for vid in cand.get("vehicles", []):
                suspicious_vehicle_ids.add(vid)

    # Render tracked vehicles with status color-coding:
    # 🔴 RED: Confirmed accident | 🟡 YELLOW: Checking/Suspicious | 🟢 GREEN: Normal traffic | 🔵 BLUE: Damage
    for object_id, centroid in tracker.objects.items():
        cx, cy, detection = centroid
        x1, y1, x2, y2 = detection["bbox"]
        vehicle_type = detection.get("class", "vehicle")
        confidence = detection.get("confidence", 0.0)
        in_roi = object_id in roi_objects

        if object_id in accident_vehicle_ids:
            # 🔴 Confirmed Accident Vehicle
            meta = accident_vehicle_meta.get(object_id, {})
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(processed, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (255, 255, 255), 1)

            # 🔵 Blue indicator if physical vehicle damage verified
            dmg = meta.get("damage", {})
            if dmg.get("damage_detected", False):
                cv2.rectangle(processed, (x1 - 2, y1 - 2), (x1 + 18, y1 + 18), (255, 120, 0), 2)
                cv2.rectangle(processed, (x2 - 18, y2 - 18), (x2 + 2, y2 + 2), (255, 120, 0), 2)

            label = f"ACCIDENT #{object_id} {vehicle_type.upper()} [{meta.get('conf')}]"
            tag_color = (0, 0, 200)

        elif object_id in suspicious_vehicle_ids:
            # 🟡 Suspicious / Investigating Interaction
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 215, 255), 2)
            label = f"CHECKING #{object_id} {vehicle_type}"
            tag_color = (0, 160, 210)

        else:
            # 🟢 Normal Vehicles
            color = (255, 0, 255) if vehicle_type == "auto" else ((0, 0, 255) if vehicle_type in HEAVY_VEHICLES else (0, 255, 0))
            thickness = 3 if in_roi else 2
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, thickness)
            label = f"#{object_id} {vehicle_type}: {confidence:.2f}"
            if in_roi:
                label += " [ROI]"
            tag_color = color

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        label_y = max(y1, text_size[1] + 10)
        cv2.rectangle(
            processed,
            (x1, label_y - text_size[1] - 8),
            (x1 + text_size[0] + 6, label_y),
            tag_color,
            -1,
        )
        cv2.putText(
            processed, label, (x1 + 3, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA,
        )

    line_y = int(tracker.counting_line["y"] * h)
    cv2.line(processed, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(
        processed, "COUNTING LINE", (10, max(20, line_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )

    signal_status = signal_manager.check_traffic_conditions(tracker.counts)

    # Clean, non-overlapping HUD overlay
    total_vehicles = sum(tracker.counts.values())
    hud_bg = (20, 25, 35)
    cv2.rectangle(processed, (8, 8), (380, 52), hud_bg, -1)
    cv2.rectangle(processed, (8, 8), (380, 52), (100, 116, 139), 1)
    cv2.putText(
        processed, f"Line Crossings: {total_vehicles} | ROI: {density_level.upper()}", (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.putText(
        processed, f"Signal: {signal_status}", (16, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 240), 1, cv2.LINE_AA
    )

    # Secondary Incident Scene Outline & Alert Banner
    if accidents:
        for acc in accidents:
            ax1, ay1, ax2, ay2 = acc["bbox"]
            ax1, ay1 = max(0, int(ax1)), max(0, int(ay1))
            ax2, ay2 = min(w - 1, int(ax2)), min(h - 1, int(ay2))
            cv2.rectangle(processed, (ax1, ay1), (ax2, ay2), (0, 0, 180), 1)

        sev = accidents[0].get("severity", "HIGH")
        banner_w = 380
        bx1 = max(10, w // 2 - banner_w // 2)
        bx2 = min(w - 10, bx1 + banner_w)
        cv2.rectangle(processed, (bx1, 8), (bx2, 42), (0, 0, 200), -1)
        cv2.rectangle(processed, (bx1, 8), (bx2, 42), (255, 255, 255), 2)
        cv2.putText(processed, f"!!! HARD ACCIDENT [{sev}] DETECTED !!!", (bx1 + 12, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

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

    return processed, tracker.counts, signal_status, density_info, accidents, suspicious_events


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


def get_accident_audio_and_notification_script(
    location: str,
    time_str: str,
    confidence: str,
    incident_id: str = "ACC-0001",
    vehicle_ids: str = "",
) -> str:
    """HTML+JS for a loud, unmissable accident alert (siren, voice, flashing banner,
    browser notification). Must be rendered with streamlit.components.v1.html so the
    script actually runs (st.markdown does NOT execute <script>). Fires once per incident."""
    import json as _json
    cfg = _json.dumps({
        "loc": str(location), "time": str(time_str), "conf": str(confidence),
        "inc": str(incident_id), "vids": str(vehicle_ids),
    })
    return """
<script>
(function() {
  var C = __CFG__;
  var top = window.parent || window;
  try { top.__atcc_seen = top.__atcc_seen || {}; } catch(e) { top = window; top.__atcc_seen = top.__atcc_seen || {}; }
  if (top.__atcc_seen[C.inc]) return;          // once per incident
  top.__atcc_seen[C.inc] = true;

  // 1) Loud siren (3 sweeps, ~3.5 s)
  try {
    var AC = window.AudioContext || window.webkitAudioContext;
    var ctx = new AC(); if (ctx.state === 'suspended') ctx.resume();
    var t0 = ctx.currentTime, osc = ctx.createOscillator(), g = ctx.createGain();
    osc.type = 'square';
    for (var i = 0; i < 8; i++) osc.frequency.setValueAtTime(i % 2 ? 640 : 980, t0 + i * 0.45);
    g.gain.setValueAtTime(0.5, t0);
    g.gain.setValueAtTime(0.5, t0 + 3.4);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 3.6);
    osc.connect(g); g.connect(ctx.destination); osc.start(t0); osc.stop(t0 + 3.6);
  } catch(e) { console.log('siren blocked', e); }

  // 2) Voice warning
  try {
    var u = new SpeechSynthesisUtterance('Warning. Accident detected at ' + C.loc + '.');
    u.rate = 0.95; u.volume = 1; u.lang = 'en-US';
    (top.speechSynthesis || window.speechSynthesis).speak(u);
  } catch(e) {}

  // 3) Full-width flashing banner on the main page
  try {
    var doc = top.document;
    var old = doc.getElementById('atcc-alert'); if (old) old.remove();
    if (!doc.getElementById('atcc-alert-css')) {
      var st = doc.createElement('style'); st.id = 'atcc-alert-css';
      st.textContent = '@keyframes atccflash{0%,100%{background:#b00000}50%{background:#ff1a1a}}' +
        '#atcc-alert{position:fixed;top:0;left:0;right:0;z-index:2147483647;color:#fff;' +
        'font:700 20px/1.4 sans-serif;padding:18px 24px;text-align:center;' +
        'animation:atccflash .5s infinite;box-shadow:0 6px 30px rgba(0,0,0,.6)}' +
        '#atcc-alert button{margin-left:18px;padding:6px 14px;font-weight:700;cursor:pointer;border:0;border-radius:6px}';
      doc.head.appendChild(st);
    }
    var bar = doc.createElement('div'); bar.id = 'atcc-alert';
    bar.innerHTML = '🚨 ACCIDENT DETECTED — ' + C.loc + ' | ' + C.time + ' | Vehicles #' + C.vids +
                    ' | ' + C.conf + ' <button id="atcc-dismiss">DISMISS</button>';
    doc.body.appendChild(bar);
    doc.getElementById('atcc-dismiss').onclick = function(){ bar.remove(); };
    setTimeout(function(){ if (bar.parentNode) bar.remove(); }, 15000);
  } catch(e) { console.log('banner failed', e); }

  // 4) Tab title flash + browser notification
  try {
    var d = top.document, orig = d.title, n = 0;
    var iv = setInterval(function(){ d.title = (n++ % 2) ? orig : '🚨 ACCIDENT DETECTED'; if (n > 20) { clearInterval(iv); d.title = orig; } }, 600);
  } catch(e) {}
  try {
    if ('Notification' in top) {
      var show = function(){ new top.Notification('🚨 TRAFFIC ACCIDENT DETECTED!',
        { body: C.loc + ' | ' + C.time + ' | ' + C.inc + ' | Vehicles #' + C.vids, requireInteraction: true }); };
      if (top.Notification.permission === 'granted') show();
      else if (top.Notification.permission !== 'denied') top.Notification.requestPermission().then(function(p){ if (p === 'granted') show(); });
    }
  } catch(e) {}
})();
</script>
""".replace("__CFG__", cfg)


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
    feed_title="Live Stream",
):
    """Process video or live feed with real-time YOLO tracking, density metrics and accident detection."""
    st.subheader("⚡ ATCC Live Video Stream Processing")

    stop_col1, stop_col2 = st.columns([1, 4])
    with stop_col1:
        stop_button = st.button("⏹️ Stop Stream Processing", key=stop_key)

    # Live Status & Alert Banners
    live_status_placeholder = st.empty()
    accident_alert_placeholder = st.empty()
    audio_notification_placeholder = st.empty()
    alert_placeholder = st.empty()
    density_metrics_placeholder = st.empty()
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    diagnostics_placeholder = st.empty()
    data_placeholder = st.empty()

    high_alert_active = False
    accident_alert_active = False
    # Incident tracking across frames: incident_id -> dict
    active_incidents = {}
    notified_incidents = set(st.session_state.setdefault("dispatched_notifications", set()))

    tracker = VehicleTracker()
    signal_manager = TrafficSignalManager()
    def _new_accident_detector():
        return AccidentDetector(
            sensitivity=st.session_state.get("accident_sensitivity", "Balanced"),
            road_type=ROAD_TYPE_MODES.get(st.session_state.get("road_type_mode", "Auto-detect"), "UNKNOWN"),
        )

    accident_detector = _new_accident_detector()
    scene_idx = 0
    cut_detector = None
    scene_notice_until = 0

    frame_count = 0
    last_save = datetime.now()
    is_file = cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
    max_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if is_file else 600
    fps_history = []

    # Safe device fallback if CUDA requested on CPU-only system
    actual_device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"

    while cap.isOpened():
        if stop_button:
            st.info("Stream processing stopped by user.")
            break

        t0 = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            if is_file and cap.get(cv2.CAP_PROP_POS_FRAMES) >= max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame_count += 1

        # Resolution-independent processing (see PROCESS_MAX_WIDTH)
        if PROCESS_MAX_WIDTH and frame.shape[1] > PROCESS_MAX_WIDTH:
            scale = PROCESS_MAX_WIDTH / float(frame.shape[1])
            frame = cv2.resize(frame, (PROCESS_MAX_WIDTH, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)

        # Camera switch / stream reconnect: old tracks, trajectories and incidents must not carry over
        if cut_detector is None:
            _fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cut_detector = SceneCutDetector(fps=_fps if _fps > 1 else 25.0, stride=max(1, int(frame_stride)))
        if cut_detector.update(frame):
            scene_idx += 1
            tracker = VehicleTracker()
            signal_manager = TrafficSignalManager()
            accident_detector = _new_accident_detector()
            accident_alert_placeholder.empty()
            accident_alert_active = False
            scene_notice_until = frame_count + 60

        run_inference = (frame_count == 1) or (frame_count % frame_stride == 0)

        processed, counts, signal_status, density_info, accidents, suspicious_events = process_frame(
            frame,
            vehicle_model,
            auto_model,
            tracker,
            signal_manager,
            accident_detector=accident_detector,
            conf_thresh=conf_thresh,
            auto_conf_thresh=auto_conf_thresh,
            iou_thresh=iou_thresh,
            img_size=img_size,
            device=actual_device,
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

        now_dt = datetime.now()
        date_time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        time_display = now_dt.strftime("%I:%M %p")

        # --- REAL-TIME ACCIDENT DETECTION & DEDUPLICATED NOTIFICATIONS ---
        if accidents:
            acc = accidents[0]
            vids = ", ".join(str(v) for v in acc.get("vehicle_ids", []))
            acc_type = acc.get("type", "Vehicle Collision")
            conf_str = acc.get("confidence", "92%")
            conf_pct = acc.get("confidence_pct", 92)
            severity = acc.get("severity", "HIGH")
            damage_dict = acc.get("damage_assessment", {})
            cand_key = acc.get("incident_key", f"coll_{vids}")
            evidence = acc.get("evidence", {})
            evidence_str = " | ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in evidence.items()) if evidence else f"Confidence: {conf_str}"

            # Format vehicle damage details
            dmg_html_list = []
            for v_id, d_data in damage_dict.items():
                if d_data.get("damage_detected"):
                    dmg_html_list.append(
                        f"🔹 <b>Vehicle #{v_id} ({d_data.get('vehicle_class', 'vehicle')}):</b> {d_data.get('damage_category')} "
                        f"| {d_data.get('severity_level')} severity (Confidence: {d_data.get('confidence_pct')}%)"
                    )
                else:
                    dmg_html_list.append(
                        f"🔹 <b>Vehicle #{v_id} ({d_data.get('vehicle_class', 'vehicle')}):</b> {d_data.get('damage_category')}"
                    )
            damage_display_html = "<br>".join(dmg_html_list) if dmg_html_list else "Visual inspection completed: no severe structural perimeter distortion detected."
            damage_summary_str = " | ".join(f"#{v}: {d.get('damage_category')}" for v, d in damage_dict.items())

            # Deduplicated Incident Record Management via SQLite
            db_id, canonical_id, is_new_incident = database.save_or_update_accident_event(
                incident_id=acc.get("incident_id") or f"scene{scene_idx}:{cand_key}",
                feed_title=feed_title,
                location=feed_title,
                confidence=conf_str,
                confidence_pct=conf_pct,
                vehicle_ids=vids,
                frame_number=frame_count,
                accident_type=acc_type,
                evidence_details=evidence_str,
                severity=severity,
                damage_assessment=damage_summary_str,
            )
            acc["incident_id"] = canonical_id

            formatted_alert_msg = (
                f"🚨 TRAFFIC ACCIDENT DETECTED [{severity}]! Camera: {feed_title} | Location: {feed_title} | "
                f"Time: {time_display} | Incident ID: {canonical_id} | "
                f"Vehicles Involved: Vehicle #{vids} | Confidence: {conf_str} | Status: UNRESOLVED. "
                f"Immediate attention required."
            )

            # 1. Trigger Audio & Browser Notification (EXACTLY ONCE per verified incident)
            if is_new_incident and (canonical_id not in notified_incidents):
                notified_incidents.add(canonical_id)
                st.session_state["dispatched_notifications"].add(canonical_id)
                audio_script = get_accident_audio_and_notification_script(
                    location=feed_title,
                    time_str=time_display,
                    confidence=conf_str,
                    incident_id=canonical_id,
                    vehicle_ids=vids,
                )
                with audio_notification_placeholder:
                    components.html(audio_script, height=0)

            sev_badge_color = "#dc3545" if severity in ("CRITICAL", "HIGH") else ("#d39e00" if severity == "MEDIUM" else "#5c5c00")

            # 2. Danger Symbol Display & Prominent Responsive Alert Banner
            accident_alert_placeholder.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #490000 0%, #a00000 50%, #490000 100%);
                    border: 3px solid #ff3333;
                    border-radius: 12px;
                    padding: 16px 20px;
                    margin-bottom: 14px;
                    color: #ffffff;
                    box-shadow: 0 0 20px rgba(220, 53, 69, 0.6);
                    box-sizing: border-box;
                    width: 100%;
                ">
                    <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
                        <span style="font-size:46px; line-height:1; animation: dangerBlink 1s infinite;">⚠️</span>
                        <div style="flex:1; min-width:260px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                                <span style="font-size:22px; font-weight:900; letter-spacing:0.5px; color:#ffffff;">
                                    HARD ACCIDENT DETECTED
                                </span>
                                <div style="display:flex; gap:6px;">
                                    <span style="background:{sev_badge_color}; color:white; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:bold;">
                                        {severity} SEVERITY
                                    </span>
                                    <span style="background:#dc3545; color:white; padding:3px 12px; border-radius:12px; font-size:12px; font-weight:bold;">
                                        🔴 DANGER
                                    </span>
                                </div>
                            </div>
                            <div style="font-size:13px; color:#ffcccc; margin-top:4px; display:flex; gap:12px; flex-wrap:wrap;">
                                <span>📅 <b>Time:</b> {date_time_str}</span>
                                <span>📍 <b>Camera:</b> {feed_title}</span>
                                <span>🆔 <b>Incident:</b> <span style="background:rgba(255,255,255,0.2); padding:1px 6px; border-radius:6px;">{canonical_id}</span></span>
                                <span>🎯 <b>Confidence:</b> <span style="background:#ff3333; padding:1px 8px; border-radius:8px; font-weight:bold;">{conf_str}</span></span>
                            </div>
                            <div style="margin-top:8px; background:rgba(0,0,0,0.35); padding:8px 12px; border-radius:6px; font-size:13px; line-height:1.4; border-left:4px solid #ff4d4d; word-break:break-word;">
                                {formatted_alert_msg}
                            </div>
                            <div style="margin-top:8px; background:rgba(10,35,70,0.55); padding:8px 12px; border-radius:6px; font-size:12px; line-height:1.4; border-left:4px solid #38bdf8;">
                                <div style="color:#7dd3fc; font-weight:bold; margin-bottom:2px;">🔵 VEHICLE DAMAGE ANALYSIS:</div>
                                <div style="color:#e0f2fe;">{damage_display_html}</div>
                            </div>
                            <div style="margin-top:6px; font-size:12px; color:#ffe6e6; display:flex; gap:12px; flex-wrap:wrap;">
                                <span>🚗 <b>Vehicles:</b> #{vids}</span>
                                <span>📊 <b>Evidence:</b> {evidence_str}</span>
                                <span>⚙️ <b>Status:</b> UNRESOLVED (Immediate Attention)</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 3. Live System Condition -> 🔴 DANGER
            live_status_placeholder.markdown(
                f"""
                <div style="background:#490000; border-left:5px solid #dc3545; padding:8px 14px; border-radius:8px; margin-bottom:10px; color:#ffc9c9; font-weight:bold; font-size:14px; display:flex; align-items:center; gap:10px;">
                    <span style="font-size:18px;">🔴</span>
                    <span>SYSTEM CONDITION: <b>DANGER – HARD ACCIDENT VERIFIED ({canonical_id})</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            accident_alert_active = True

        elif suspicious_events:
            susp = suspicious_events[0]
            # 3. Live System Condition -> 🟡 VERIFYING
            live_status_placeholder.markdown(
                f"""
                <div style="background:#3b2a00; border-left:5px solid #ffc107; padding:8px 14px; border-radius:8px; margin-bottom:10px; color:#fff0aa; font-weight:bold; font-size:14px; display:flex; align-items:center; gap:10px;">
                    <span style="font-size:18px;">🟡</span>
                    <span>SYSTEM CONDITION: <b>VERIFYING – INVESTIGATING SUSPICIOUS ACTIVITY</b> ({susp.get('type', 'Under Review')})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            accident_alert_placeholder.empty()
            accident_alert_active = False

        else:
            # 3. Live System Condition -> 🟢 NORMAL
            live_status_placeholder.markdown(
                """
                <div style="background:#002b11; border-left:5px solid #28a745; padding:8px 14px; border-radius:8px; margin-bottom:10px; color:#c7f9cc; font-weight:bold; font-size:14px; display:flex; align-items:center; gap:10px;">
                    <span style="font-size:18px;">🟢</span>
                    <span>SYSTEM CONDITION: <b>NORMAL – ALL FEEDS OPERATING NORMALLY</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            accident_alert_placeholder.empty()
            accident_alert_active = False

        # --- Traffic Density Alert ---
        d_level_current = density_info["density_level"].lower()
        if d_level_current == "high" and not high_alert_active:
            alert_placeholder.warning(
                "⚠️ **HIGH TRAFFIC DENSITY ALERT** — Congestion threshold exceeded in ROI. Monitor traffic flow."
            )
            high_alert_active = True
        elif d_level_current in ("low", "medium") and high_alert_active:
            alert_placeholder.empty()
            high_alert_active = False

        if frame_count == 1 or frame_count % frame_stride == 0:
            d_level = density_info["density_level"]
            r_total = density_info["roi_total"]
            type_counts = density_info["roi_counts"]
            b_color = density_info["badge_color"]

            with density_metrics_placeholder.container():
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                with m1:
                    st.markdown(
                        f"""
                        <div style="background-color:{b_color}; padding:6px 10px; border-radius:8px; text-align:center; color:white;">
                            <div style="font-size: 10px; font-weight: bold; letter-spacing: 0.5px;">DENSITY</div>
                            <div style="font-size: 18px; font-weight: 800; margin-top:-2px;">{d_level.upper()}</div>
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

        _, jpeg_buf = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_placeholder.image(
            jpeg_buf.tobytes(),
            caption=f"Camera: {feed_title} | Frame #{frame_count} | {avg_fps:.1f} FPS"
                    + (" | ⚠️ HARD ACCIDENT VERIFIED" if accidents else ""),
            use_container_width=True,
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
                + (f" | 🚨 **ACCIDENT ACTIVE**" if accidents else "")
            )
            data_placeholder.json({
                "Traffic Density Level": density_info["density_level"],
                "Vehicles in ROI": density_info["roi_total"],
                "ROI Vehicle Types": density_info["roi_counts"],
                "Line Crossing Counts": dict(counts),
                "Signal Status": signal_status,
                "Road Condition": accident_detector.road_condition() if accident_detector is not None else None,
                "Camera Scene #": scene_idx + 1,
                "Frame Size": f"{processed.shape[1]}x{processed.shape[0]}",
                "FPS": round(avg_fps, 1),
            })
            if accident_detector is not None and accident_detector.road_condition()["mismatch"]:
                _rc = accident_detector.road_condition()
                status_placeholder.warning(
                    f"⚠️ Road type is set to {_rc['road_type']} but traffic here behaves like "
                    f"{_rc['observed_type']} ({_rc['observed_confidence_pct']}% confidence). "
                    f"Switch 'Road Type' in the sidebar to Auto-detect for correct accident rules."
                )

            if accident_detector is not None:
                diag_records = accident_detector.get_diagnostics()
                if diag_records or accidents or suspicious_events:
                    with diagnostics_placeholder.container():
                        with st.expander("🔍 Live Accident Detection & Collision Diagnostics", expanded=bool(accidents)):
                            d1, d2, d3, d4 = st.columns(4)
                            with d1:
                                st.metric("System Condition", accident_detector.system_condition)
                            with d2:
                                st.metric("Active Tracks", len(tracker.get_active_objects()))
                            with d3:
                                st.metric("Candidates", len(accident_detector.candidates))
                            with d4:
                                st.metric("Active Confirmed", len(accident_detector.active_confirmed_map))

                            if diag_records:
                                st.caption("Ground Footprint & Kinematic Telemetry Log:")
                                for rec in reversed(diag_records[-5:]):
                                    pair_str = f"Vehicles #{rec.get('pair', [0, 0])[0]} & #{rec.get('pair', [0, 0])[1]}"
                                    st.markdown(
                                        f"• **F{rec.get('frame', 0)} [{rec.get('status', 'NORMAL')}]** {pair_str} — "
                                        f"2D Box IoU: `{rec.get('box_iou', 0.0):.2f}` | "
                                        f"Ground Footprint IoU: `{rec.get('ground_iou', 0.0):.2f}` | "
                                        f"Reason: *{rec.get('reason', 'Evaluating')}*"
                                    )

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

                # ATCC YOLO Inference Execution Options (Client-Side Agent vs In-App)
                with st.expander("⚡ Run ATCC Live Detection (Client-Side or In-App)", expanded=False):
                    proc_tab1, proc_tab2 = st.tabs(["🖥️ User Computer (Local Agent)", "⚡ In-App Desktop Mode"])

                    with proc_tab1:
                        st.markdown("""
                        **Recommended Architecture:** Video decoding, YOLO detection, vehicle tracking, and accident verification run on **your own computer**.
                        The central server receives only structured telemetry (counts, density, accidents).
                        """)
                        cli_cmd = (
                            f"python -m local_engine.local_processor "
                            f"--server http://localhost:8502 "
                            f"--user {current_user['username']} "
                            f"--feed-id {fid} "
                            f'--source "{feed["video_url"]}" '
                            f'--title "{feed["title"]}"'
                        )
                        st.code(cli_cmd, language="bash")
                        st.caption("Copy and run this command in your computer's terminal to process this feed locally.")

                        # Check if any active local agent is currently handling this feed
                        agents = database.get_active_agents(timeout_seconds=20)
                        matching_agent = next((a for a in agents if a.get("feed_id") == fid and a.get("status") == "active"), None)
                        if matching_agent:
                            st.success(
                                f"🟢 Agent Connected! Running on {matching_agent.get('username')}'s PC "
                                f"({matching_agent.get('device_info')}) at {matching_agent.get('fps', 0):.1f} FPS."
                            )

                    with proc_tab2:
                        st.caption("Runs inference inside this app instance (suitable for single-machine / desktop deployments):")
                        if st.button(f"Start In-App YOLO Detection ({feed['title']})", key=f"run_yolo_{fid}"):
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
                                        feed_title=feed["title"],
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


def render_system_status_banner(current_user: dict):
    """
    Prominent system status indicator displayed at top of dashboard.
    Shows one of three mandatory conditions:
    - 🟢 NORMAL – No accident detected.
    - 🟡 WARNING – Suspicious traffic activity detected / Incident being acknowledged.
    - 🔴 DANGER – Accident detected.
    """
    summary = database.get_system_accident_summary()
    cond = summary["condition"]
    latest = summary["latest_event"]

    if cond == "DANGER":
        latest_loc = latest["location"] if latest else "Traffic Camera"
        latest_time = latest["timestamp"] if latest else "Just now"
        latest_conf = latest.get("confidence", "92%") if latest else "92%"
        inc_id = (latest.get("incident_id") or f"ACC-{latest.get('id', 1):04d}") if latest else "ACC-0001"
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #5c0000 0%, #b30000 50%, #5c0000 100%);
                border: 3px solid #ff4d4d;
                border-radius: 12px;
                padding: 16px 20px;
                margin-bottom: 20px;
                color: white;
                box-shadow: 0 0 22px rgba(220, 53, 69, 0.7);
                animation: dangerBorderPulse 1.8s infinite;
                width: 100%;
                box-sizing: border-box;
            ">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                    <div style="display: flex; align-items: center; gap: 16px; flex-wrap: wrap;">
                        <span style="font-size: 38px; line-height: 1; animation: dangerBlink 1s infinite;">⚠️</span>
                        <div>
                            <div style="font-size: 20px; font-weight: 900; letter-spacing: 0.5px;">
                                🔴 SYSTEM STATUS: DANGER — HARD ACCIDENT DETECTED
                            </div>
                            <div style="font-size: 13px; color: #ffcccc; margin-top: 3px; display:flex; gap:12px; flex-wrap:wrap;">
                                <span>Camera / Location: <b>{latest_loc}</b></span>
                                <span>Time: <b>{latest_time}</b></span>
                                <span>Incident ID: <b style="background:rgba(255,255,255,0.2); padding:1px 6px; border-radius:4px;">{inc_id}</b></span>
                                <span>Confidence: <b>{latest_conf}</b></span>
                                <span style="background:#dc3545; padding:1px 8px; border-radius:10px; font-weight:bold;">
                                    {summary['unresolved_count']} Unresolved Incident(s)
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif cond == "WARNING":
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #3d2b00 0%, #7a5700 50%, #3d2b00 100%);
                border: 2px solid #ffc107;
                border-radius: 10px;
                padding: 12px 20px;
                margin-bottom: 18px;
                color: #fff3cd;
            ">
                <div style="display:flex; align-items:center; gap:14px;">
                    <span style="font-size: 28px;">🟡</span>
                    <div>
                        <div style="font-size: 16px; font-weight: 800;">
                            SYSTEM STATUS: WARNING — SUSPICIOUS TRAFFIC ACTIVITY / INCIDENT UNDER REVIEW
                        </div>
                        <div style="font-size: 12px; color: #ffe69c; margin-top: 2px;">
                            {summary['acknowledged_count']} incident(s) currently acknowledged & under active investigation.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #002e12 0%, #005a24 50%, #002e12 100%);
                border: 2px solid #28a745;
                border-radius: 10px;
                padding: 10px 18px;
                margin-bottom: 18px;
                color: #d4edda;
            ">
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="font-size: 22px;">🟢</span>
                    <div style="font-size: 14px; font-weight: 700;">
                        SYSTEM STATUS: NORMAL TRAFFIC — NO ACCIDENTS DETECTED
                    </div>
                    <span style="margin-left:auto; font-size:12px; color:#8ce99a;">All CCTV feeds operating normally</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_accident_alert_history_panel(current_user: dict, title: str = "🚨 Accident Alert History", expanded: bool = True):
    """
    Dedicated Accident Alert History panel on the dashboard.
    - Displays all detected accident alerts with latest on top.
    - Maintains timestamps for every alert.
    - Highlights new / unresolved alerts in red.
    - Stores previous alerts for historical review.
    - Provides interactive options to acknowledge or clear/resolve alerts.
    """
    summary = database.get_system_accident_summary()
    unresolved_cnt = summary["unresolved_count"]
    acknowledged_cnt = summary["acknowledged_count"]
    resolved_cnt = summary["resolved_count"]
    total_cnt = unresolved_cnt + acknowledged_cnt + resolved_cnt

    st.markdown(f"### {title}")

    # Top KPI Metrics row for Accident Alerts
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("🔴 Unresolved Alerts", unresolved_cnt)
    with m2:
        st.metric("🟡 Acknowledged", acknowledged_cnt)
    with m3:
        st.metric("🟢 Resolved Incidents", resolved_cnt)
    with m4:
        st.metric("📋 Total Alerts Logged", total_cnt)

    # Audio Alarm Testing & Global Alert Clearance Controls
    col_tools1, col_tools2 = st.columns([2, 1])
    with col_tools1:
        if st.button("🔊 Test Warning Siren Audio Alert", key=f"test_audio_{title[:5]}"):
            components.html(
                get_accident_audio_and_notification_script(
                    "Test Location (CCTV)", datetime.now().strftime("%I:%M %p"), "95%",
                    incident_id=f"TEST-{int(time.time())}", vehicle_ids="1,2"),
                height=0,
            )
            st.success("Test alarm triggered! Emergency warning siren sound played.")
    with col_tools2:
        if st.button("🧹 Clear All Active Alerts", key=f"clear_all_btn_{title[:5]}", type="secondary", use_container_width=True):
            username = current_user.get("username", "Operator")
            cleared_cnt = database.clear_all_active_accidents(username=username)
            if "dispatched_notifications" in st.session_state:
                st.session_state["dispatched_notifications"].clear()
            st.success(f"Successfully cleared {cleared_cnt} active alerts from dashboard. Historical records and evidence preserved.")
            time.sleep(0.3)
            st.rerun()

    tab_unresolved, tab_ack, tab_resolved, tab_all = st.tabs([
        f"🔴 Unresolved ({unresolved_cnt})",
        f"🟡 Acknowledged ({acknowledged_cnt})",
        f"🟢 Resolved ({resolved_cnt})",
        f"📋 All Alerts ({total_cnt})",
    ])

    panel_prefix = "".join(c for c in title if c.isalnum()) or "panel"

    def _render_alert_card(row: dict, tab_prefix: str = "tab"):
        a_id = row["id"]
        status = row.get("status", "unresolved")
        resolved = int(row.get("resolved", 0))
        acknowledged = int(row.get("acknowledged", 0))

        if resolved:
            card_border = "#28a745"
            card_bg = "#f0fff4"
            status_badge = "🟢 RESOLVED"
            badge_bg = "#28a745"
        elif acknowledged:
            card_border = "#ffc107"
            card_bg = "#fffdf0"
            status_badge = "🟡 ACKNOWLEDGED (Investigating)"
            badge_bg = "#d39e00"
        else:
            card_border = "#dc3545"
            card_bg = "#fff5f5"
            status_badge = "🔴 UNRESOLVED (Immediate Attention)"
            badge_bg = "#dc3545"

        acc_type = row.get("accident_type") or "Vehicle Collision"
        inc_id = row.get("incident_id") or f"ACC-{a_id:04d}"
        loc = row.get("location") or row.get("feed_title") or "CCTV Camera"
        ts = row.get("timestamp", "N/A")
        last_up = row.get("last_updated") or ts
        conf = row.get("confidence", "90%")
        vids = row.get("vehicle_ids", "N/A")
        sev = row.get("severity") or "HIGH"
        dmg_note = row.get("damage_assessment") or ""
        evidence_info = row.get("evidence_details") or "Multi-frame kinetic & temporal verification"
        msg = row.get("alert_message") or (
            f"🚨 TRAFFIC ACCIDENT DETECTED [{sev}]! Camera: {loc} Location: {loc} Time: {ts} Incident ID: {inc_id} Vehicles Involved: Vehicle #{vids} Confidence: {conf} Status: UNRESOLVED. Immediate attention required."
        )

        st.markdown(
            f"""
            <div style="
                background: {card_bg};
                border-left: 5px solid {card_border};
                border-top: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
                border-bottom: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 14px 18px;
                margin-bottom: 12px;
                box-sizing: border-box;
                width: 100%;
                {'box-shadow: 0 0 12px rgba(220,53,69,0.35);' if not resolved and not acknowledged else ''}
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
                        <span style="font-size:24px;">⚠️</span>
                        <div>
                            <span style="font-size:16px; font-weight:800; color:#1e293b;">{acc_type}</span>
                            <span style="margin-left:8px; background:rgba(0,0,0,0.06); padding:2px 8px; border-radius:6px; font-weight:bold; font-size:12px; color:#334155;">{inc_id}</span>
                            <span style="margin-left:8px; background:{'#dc3545' if sev in ('CRITICAL','HIGH') else '#d39e00'}; color:white; padding:1px 7px; border-radius:4px; font-weight:bold; font-size:11px;">{sev}</span>
                            <span style="margin-left:8px; font-size:12px; color:#64748b;">📅 {ts}</span>
                        </div>
                    </div>
                    <span style="background:{badge_bg}; color:white; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:bold;">
                        {status_badge}
                    </span>
                </div>
                <div style="margin-top:10px; background:rgba(0,0,0,0.04); padding:10px 14px; border-radius:8px; font-size:13px; color:#0f172a; border-left:3px solid {card_border}; word-break:break-word;">
                    {msg}
                </div>
                <div style="margin-top:6px; font-size:12px; color:#0369a1; background:#f0f9ff; padding:6px 10px; border-radius:6px; border-left:3px solid #0284c7;">
                    🔵 <b>Damage Status:</b> {dmg_note if dmg_note else 'UNKNOWN (Unable to Verify)'}
                </div>
                <div style="margin-top:8px; font-size:12px; color:#475569; display:flex; gap:14px; flex-wrap:wrap;">
                    <span>📍 <b>Camera:</b> {loc}</span>
                    <span>🗺️ <b>Location:</b> {row.get('location') or loc}</span>
                    <span>🎯 <b>Collision Confidence:</b> <b>{conf}</b></span>
                    <span>🚗 <b>Vehicles:</b> #{vids}</span>
                    <span>🎞️ <b>Frame:</b> #{row.get('frame_number', 0)}</span>
                    <span>🕒 <b>Last Observed:</b> {last_up}</span>
                </div>
                <div style="margin-top:6px; font-size:11px; color:#64748b; background:rgba(0,0,0,0.02); padding:5px 8px; border-radius:4px;">
                    📊 <b>Collision Evidence:</b> {evidence_info}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Buttons (Acknowledge, Resolve, Individual Dismissal, Clear)
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1.5, 1.5, 1.4, 1.2])
        username = current_user.get("username", "Operator")

        if not resolved and not acknowledged:
            with btn_col1:
                if st.button("👁️ Acknowledge", key=f"ack_{panel_prefix}_{tab_prefix}_{a_id}", use_container_width=True):
                    database.acknowledge_accident_event(a_id, username=username)
                    st.success(f"Alert #{a_id} acknowledged by {username}.")
                    time.sleep(0.3)
                    st.rerun()
            with btn_col2:
                if st.button("✅ Confirm & Resolve", key=f"res_{panel_prefix}_{tab_prefix}_{a_id}", use_container_width=True):
                    database.resolve_accident_event(a_id, username=username)
                    st.success(f"Alert #{a_id} marked resolved by {username}.")
                    time.sleep(0.3)
                    st.rerun()
            with btn_col3:
                if st.button("❌ Dismiss Alert", key=f"dism_{panel_prefix}_{tab_prefix}_{a_id}", use_container_width=True):
                    database.dismiss_accident_event(a_id, username=username)
                    st.info(f"Alert #{a_id} dismissed from active view (evidence preserved in history).")
                    time.sleep(0.3)
                    st.rerun()

        elif not resolved and acknowledged:
            with btn_col1:
                st.caption(f"Acknowledged by: `{row.get('acknowledged_by') or 'Operator'}`")
            with btn_col2:
                if st.button("✅ Confirm & Resolve", key=f"res2_{panel_prefix}_{tab_prefix}_{a_id}", use_container_width=True):
                    database.resolve_accident_event(a_id, username=username)
                    st.success(f"Alert #{a_id} marked resolved by {username}.")
                    time.sleep(0.3)
                    st.rerun()
            with btn_col3:
                if st.button("❌ Dismiss Alert", key=f"dism2_{panel_prefix}_{tab_prefix}_{a_id}", use_container_width=True):
                    database.dismiss_accident_event(a_id, username=username)
                    st.info(f"Alert #{a_id} dismissed from active view (evidence preserved in history).")
                    time.sleep(0.3)
                    st.rerun()

        else:
            with btn_col1:
                st.caption(f"Status: `{status.upper()}` by `{row.get('resolved_by') or row.get('acknowledged_by') or 'Operator'}`")

        if auth.is_admin(current_user):
            with btn_col4:
                if st.button("🗑️ Clear Record", key=f"clr_{panel_prefix}_{tab_prefix}_{a_id}", use_container_width=True):
                    database.clear_accident_event(a_id, username=username)
                    st.warning(f"Alert #{a_id} cleared from active history.")
                    time.sleep(0.3)
                    st.rerun()

        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    with tab_unresolved:
        unresolved_events = database.get_recent_accident_events(limit=50, status_filter="unresolved")
        if unresolved_events:
            for ev in unresolved_events:
                _render_alert_card(ev, tab_prefix="unres")
        else:
            st.info("🟢 No unresolved accident alerts. All traffic cameras report normal conditions.")

    with tab_ack:
        ack_events = database.get_recent_accident_events(limit=50, status_filter="acknowledged")
        if ack_events:
            for ev in ack_events:
                _render_alert_card(ev, tab_prefix="ack")
        else:
            st.info("No acknowledged accident alerts pending resolution.")

    with tab_resolved:
        resolved_events = database.get_recent_accident_events(limit=50, status_filter="resolved")
        if resolved_events:
            for ev in resolved_events:
                _render_alert_card(ev, tab_prefix="res")
        else:
            st.info("No resolved accident history records.")

    with tab_all:
        all_events = database.get_recent_accident_events(limit=50, status_filter="all")
        if all_events:
            for ev in all_events:
                _render_alert_card(ev, tab_prefix="all")
        else:
            st.info("No accident alerts logged yet in the database.")


def render_accident_alerts_view(current_user: dict):
    """Dedicated full-page Accident Alerts Manager and Real-Time Event Dashboard."""
    render_accident_alert_history_panel(current_user, title="🚨 Accident Alert History & Incident Management")


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

    st.markdown("---")
    # Connected Local Processing Engines Monitoring Panel
    st.markdown(
        "<div style='font-weight:700; font-size:16px; color:#4a90d9; margin-bottom:6px;'>🖥️ Connected Local Processing Engines (User Computers)</div>",
        unsafe_allow_html=True,
    )
    st.caption("Heavy video frames, YOLO inference, and accident analysis run on each user's computer. The central server receives only structured telemetry.")

    active_agents = database.get_active_agents(timeout_seconds=20)
    if active_agents:
        agent_rows = []
        for a in active_agents:
            st_badge = "🟢 Active" if a["status"] == "active" else ("⏸️ Stopped" if a["status"] == "stopped" else "⚠️ Stale")
            agent_rows.append({
                "Status": st_badge,
                "Agent ID": a["agent_id"],
                "User": a.get("username", "N/A"),
                "Feed": a.get("feed_title", f"Feed #{a.get('feed_id')}"),
                "Hardware / Device": a.get("device_info", "CPU"),
                "Model": a.get("model_name", "N/A"),
                "FPS": f"{a.get('fps', 0.0):.1f}",
                "CPU %": f"{a.get('cpu_percent', 0.0):.1f}%",
                "RAM %": f"{a.get('memory_percent', 0.0):.1f}%",
                "Frames": a.get("total_frames", 0),
                "Last Heartbeat": a.get("last_heartbeat", ""),
            })
        st.dataframe(pd.DataFrame(agent_rows), use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ No client processing agents currently reporting. Local engines can connect to: `http://localhost:8502`")

    with st.expander("🚀 Run Local Processing Agent on a User's Computer", expanded=False):
        st.markdown("""
        **To process video feeds on a client computer instead of this server:**
        1. Open a terminal on the client computer inside the ATCC folder.
        2. Run the local processor command:
        ```bash
        python -m local_engine.local_processor --server http://<server-ip>:8502 --user highway_operator --password user123 --source 0
        ```
        3. All YOLO inference, vehicle tracking, ROI density, and accident detection run locally on that computer.
        4. Structured results, counts, density, and accident alerts are synchronized to this dashboard in real-time.
        """)

    st.markdown("---")
    render_accident_alert_history_panel(current_user, title="🚨 Live Accident Alert History & Incident Management")


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
    """Traffic Density, Crossing Counts, Accident Events, and Activity Analytics View."""

    conn = database.get_db_connection()
    density_df = pd.read_sql_query("SELECT * FROM TrafficDensity ORDER BY id DESC LIMIT 100", conn)
    counts_df  = pd.read_sql_query("SELECT * FROM VehicleCounts ORDER BY id DESC LIMIT 100", conn)
    events_df  = pd.read_sql_query("SELECT * FROM TrafficEvents ORDER BY id DESC LIMIT 100", conn)
    logs_df    = pd.read_sql_query("SELECT * FROM ActivityLogs ORDER BY id DESC LIMIT 100", conn)
    # Accident events (table may not exist yet on old databases)
    try:
        accident_df = pd.read_sql_query("SELECT * FROM AccidentEvents ORDER BY id DESC LIMIT 100", conn)
    except Exception:
        accident_df = pd.DataFrame()
    conn.close()

    # Prominent accident banner at top of analytics if unresolved accidents exist
    if not accident_df.empty:
        unresolved = accident_df[accident_df.get("resolved", pd.Series([0]*len(accident_df))) == 0]
        if not unresolved.empty:
            st.error(
                f"🚨 **{len(unresolved)} Unresolved Accident Event(s) Detected!** "
                f"Review the **Accident Events** tab below and dispatch emergency response as needed."
            )

    tab_accidents, tab_density, tab_counts, tab_events, tab_logs = st.tabs([
        "🚨 Accident Events", "🏙️ Traffic Density", "🚗 Vehicle Crossings",
        "⚡ Signal Events", "🛡️ Activity Logs"
    ])

    with tab_accidents:
        st.markdown("#### 🚨 Detected Accident / Collision Events")
        if not accident_df.empty:
            for _, row in accident_df.iterrows():
                resolved = int(row.get("resolved", 0))
                bg = "#fff8f8" if not resolved else "#f0fff4"
                border = "#dc3545" if not resolved else "#28a745"
                status_label = "⚠️ UNRESOLVED" if not resolved else "✅ Resolved"
                st.markdown(
                    f"""
                    <div style="background:{bg}; border-left:4px solid {border}; border-radius:8px;
                                padding:12px 16px; margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <span style="font-size:18px;">🚨</span>
                                <strong style="font-size:14px; margin-left:8px;">
                                    Accident at {row.get('location','Unknown')}
                                </strong>
                                <span style="margin-left:12px; font-size:12px; color:#666;">
                                    {row.get('timestamp','')}
                                </span>
                            </div>
                            <span style="font-size:12px; font-weight:600; color:{border};">{status_label}</span>
                        </div>
                        <div style="margin-top:6px; font-size:13px; color:#333;">
                            <b>Feed:</b> {row.get('feed_title','N/A')} &nbsp;|&nbsp;
                            <b>Confidence:</b> {row.get('confidence','N/A')} &nbsp;|&nbsp;
                            <b>Vehicles Involved:</b> #{row.get('vehicle_ids','N/A')} &nbsp;|&nbsp;
                            <b>Frame:</b> {row.get('frame_number','N/A')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not resolved:
                    if st.button(f"✅ Mark Resolved (ID {row['id']})", key=f"resolve_acc_{row['id']}"):
                        database.resolve_accident_event(int(row["id"]))
                        st.success("Accident event marked as resolved.")
                        st.rerun()
        else:
            st.info("No accident events recorded yet. Run ATCC live detection on a video stream to begin monitoring.")

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
                        feed_title="Uploaded Diagnostic Video",
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
                processed, counts, signal_status, density_info, _accidents, _suspicious = process_frame(
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
                    feed_title="Live Webcam Feed",
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

        /* Accident Detection Danger & Warning CSS animations */
        @keyframes dangerBlink {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.25; transform: scale(1.08); }
        }
        @keyframes dangerBorderPulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.8); }
            70% { box-shadow: 0 0 0 16px rgba(220, 53, 69, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
        }
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

    # Fetch live system accident summary for badges
    accident_summary = database.get_system_accident_summary()
    unresolved_count = accident_summary["unresolved_count"]
    acc_badge = f" ({unresolved_count})" if unresolved_count > 0 else ""

    # ---- Navigation Menu ----
    # Admin nav items: (label, icon, key)
    ADMIN_NAV = [
        ("Dashboard",       "📊", "nav_dashboard"),
        ("Live Feeds",      "📹", "nav_livefeeds"),
        (f"Accident Alerts{acc_badge}", "🚨", "nav_accidents"),
        ("All Videos",      "🎬", "nav_allvideos"),
        ("Manage Users",    "👥", "nav_users"),
        ("Analytics",       "📈", "nav_analytics"),
        ("Settings",        "⚙️",  "nav_settings"),
    ]
    USER_NAV = [
        ("My Live Feeds",   "📹", "nav_myfeeds"),
        (f"Accident Alerts{acc_badge}", "🚨", "nav_uaccidents"),
        ("Settings",        "⚙️",  "nav_usettings"),
    ]

    nav_items = ADMIN_NAV if is_admin_user else USER_NAV

    # Use session state to track active page
    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "Dashboard" if is_admin_user else "My Live Feeds"

    st.sidebar.markdown(
        f"<div style='font-size:10px; color:#4a90d9; letter-spacing:1.5px; text-transform:uppercase; "
        f"font-weight:bold; margin-bottom:6px; padding-left:4px;'>"
        f"{'ADMIN CONSOLE' if is_admin_user else 'OPERATOR PANEL'}</div>",
        unsafe_allow_html=True,
    )

    current_active_nav = st.session_state.active_nav
    for label, icon, key in nav_items:
        clean_item_label = label.split(" (")[0]
        clean_current_active = current_active_nav.split(" (")[0]
        is_active = (clean_current_active == clean_item_label)
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
                st.session_state.active_nav = clean_item_label
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
    st.sidebar.selectbox(
        "🚨 Accident Sensitivity:",
        options=["Strict", "Balanced", "Sensitive"],
        index=1,
        key="accident_sensitivity",
        help="Strict = fewest false alarms (may miss crashes). Sensitive = catches more crashes, more false alarms.",
    )
    st.sidebar.selectbox(
        "🛣️ Road Type:",
        options=list(ROAD_TYPE_MODES.keys()),
        index=0,
        key="road_type_mode",
        help="Auto-detect learns the road type from how vehicles move (highway = fast straight flow, "
             "intersection = frequent turns). Picking a type here forces that rule set.",
    )
    target_device = device_map[selected_device_key]

    # Model Selection with Local Availability Detection & Placement Guide
    local_status = model_mgr.get_available_local_models()
    model_options = []
    model_lookup = {}
    for filename, meta in local_status.items():
        if filename == "best.pt":
            continue
        is_avail = meta.get("available", False)
        tag = " [Ready - Local]" if is_avail else " [Not Downloaded]"
        opt_label = f"{meta['label']}{tag}"
        model_options.append(opt_label)
        model_lookup[opt_label] = filename

    selected_model_label = st.sidebar.selectbox("YOLO Model:", options=model_options, index=0)
    model_path = model_lookup[selected_model_label]

    # If requested model is not present locally, guide the user and provide on-demand download
    if not model_mgr.find_model_path(model_path):
        st.sidebar.warning(f"⚠️ '{model_path}' is not installed locally in `models/`.")
        col_dl, col_info = st.sidebar.columns([1, 1])
        with col_dl:
            if st.sidebar.button(f"📥 Download {model_path}", key=f"btn_dl_{model_path}", use_container_width=True):
                with st.spinner(f"Downloading {model_path} with retry & validation..."):
                    ok_dl, msg_dl = model_mgr.download_model_with_retry(model_path)
                    if ok_dl:
                        st.sidebar.success(f"Downloaded {model_path}!")
                        st.rerun()
                    else:
                        st.sidebar.error(msg_dl)
        st.sidebar.caption(f"📁 Manual placement: Copy `{model_path}` into: `ATCC/models/{model_path}`")
        st.sidebar.info(f"💡 Fallback active: Using locally available model until `{model_path}` is placed.")

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

    clean_nav = st.session_state.active_nav.split(" (")[0]

    # ----------------------------------------------------------------
    # MAIN AREA — Top header bar & System Condition Banner
    # ----------------------------------------------------------------
    if is_admin_user:
        nav_icon_map = {
            "Dashboard": "📊",
            "Live Feeds": "📹",
            "Accident Alerts": "🚨",
            "All Videos": "🎬",
            "Manage Users": "👥",
            "Analytics": "📈",
            "Settings": "⚙️",
        }
        nav_sub_map = {
            "Dashboard": "System overview, KPIs, audit logs & accident alerts",
            "Live Feeds": "Monitor and manage live CCTV camera feeds",
            "Accident Alerts": "Real-time road accident detection, danger alerts & incident history",
            "All Videos": "Full admin view & stream assignment",
            "Manage Users": "Create, activate, and manage user accounts",
            "Analytics": "Traffic density & vehicle crossing analytics",
            "Settings": "System configuration, diagnostics & testing",
        }
        header_color = "#1a2744"
    else:
        nav_icon_map = {
            "My Live Feeds": "📹",
            "Accident Alerts": "🚨",
            "Settings": "⚙️",
        }
        nav_sub_map = {
            "My Live Feeds": "Your assigned camera streams",
            "Accident Alerts": "Real-time road accident detection & incident alert history",
            "Settings": "System configuration & diagnostics",
        }
        header_color = "#0d3b2e"

    page_icon = nav_icon_map.get(clean_nav, "🚦")
    page_sub  = nav_sub_map.get(clean_nav, "ATCC Traffic Platform")
    _page_header(clean_nav, page_sub, page_icon, header_color)

    # Global Mandatory System Status Display (🟢 NORMAL / 🟡 WARNING / 🔴 DANGER)
    render_system_status_banner(current_user)

    # ----------------------------------------------------------------
    # RENDER SELECTED VIEW
    # ----------------------------------------------------------------
    if clean_nav == "Dashboard":
        render_admin_dashboard(current_user)
    elif clean_nav in ["Live Feeds", "My Live Feeds"]:
        render_live_feeds_view(current_user, is_admin_all_view=False, model_args=model_args)
    elif clean_nav == "Accident Alerts":
        render_accident_alerts_view(current_user)
    elif clean_nav == "All Videos":
        render_all_videos_view(current_user, model_args=model_args)
    elif clean_nav == "Manage Users":
        render_manage_users_view(current_user)
    elif clean_nav == "Analytics":
        render_analytics_view()
    elif clean_nav == "Settings":
        render_settings_view(model_args, target_device)


if __name__ == "__main__":
    main()