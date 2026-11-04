"""Local Video Processing Engine for ATCC.

Executes on the client machine:
- YOLO inference (CPU or CUDA)
- Vehicle classification & tracking
- ROI density classification
- Traffic signal management
- Real-time accident detection
- Full video rendering with visual overlays (boxes, banners, warnings)
"""
import math
import time
from collections import defaultdict
import cv2
import torch
from .accident_processor import LocalAccidentDetector, box_iou

# Classes recognized by ATCC
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
HEAVY_VEHICLES = {"bus", "truck"}


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


class VehicleTracker:
    """Centroid tracker with class label smoothing, counting line, and collision-safe deduplication."""

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
            p1 = (self.objects[id1][0], self.objects[id1][1])
            for j in range(i + 1, len(object_ids)):
                id2 = object_ids[j]
                if id2 in to_remove or id2 not in self.objects:
                    continue
                box2 = self.objects[id2][2]["bbox"]
                p2 = (self.objects[id2][0], self.objects[id2][1])
                centroid_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                iou = box_iou(box1, box2)
                # Deduplicate only if high IoU and centroid distance < 25px
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

        heavy_count = vehicle_counts.get("bus", 0) + vehicle_counts.get("truck", 0)
        total_count = sum(vehicle_counts.values())

        if heavy_count >= 3 or total_count >= 15:
            self.signal_state = "extended_green"
            self.signal_change_time = current_time
            return "Heavy Traffic: Green Extended (+45s)"

        return "Normal Cycle (Active)"


class LocalVideoProcessor:
    """Unified video processing engine that runs completely on the user's computer."""

    def __init__(
        self,
        vehicle_model=None,
        auto_model=None,
        device="cpu",
        img_size=640,
        conf_thresh=0.30,
        auto_conf_thresh=0.25,
        iou_thresh=0.35,
        frame_stride=1,
        roi_bounds=(0.20, 0.00, 0.95, 1.00),
        low_thresh=4,
        med_thresh=10,
    ):
        self.vehicle_model = vehicle_model
        self.auto_model = auto_model
        # Safe device selection
        if device != "cpu" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.auto_conf_thresh = auto_conf_thresh
        self.iou_thresh = iou_thresh
        self.frame_stride = frame_stride
        self.roi_bounds = roi_bounds
        self.low_thresh = low_thresh
        self.med_thresh = med_thresh

        self.tracker = VehicleTracker()
        self.signal_manager = TrafficSignalManager()
        self.accident_detector = LocalAccidentDetector()

        self.frame_counter = 0
        self.fps_history = []

    def process_frame(self, frame, run_inference=True) -> tuple[object, dict]:
        """Process a single frame and return rendered frame + structured telemetry."""
        t_start = time.time()
        processed = frame.copy()
        h, w = frame.shape[:2]
        self.frame_counter += 1

        vehicle_detections = []

        if run_inference:
            with torch.inference_mode():
                if self.vehicle_model is not None:
                    try:
                        results = self.vehicle_model.predict(
                            frame,
                            conf=self.conf_thresh,
                            iou=self.iou_thresh,
                            imgsz=self.img_size,
                            device=self.device,
                            verbose=False,
                        )
                        for result in results:
                            for box in result.boxes:
                                class_id = int(box.cls[0].item())
                                if class_id not in VEHICLE_CLASSES:
                                    continue
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                confidence = float(box.conf[0].item())
                                vehicle_detections.append({
                                    "bbox": (x1, y1, x2, y2),
                                    "class": VEHICLE_CLASSES[class_id],
                                    "confidence": confidence,
                                })
                    except Exception as exc:
                        pass

                if self.auto_model is not None:
                    try:
                        auto_results = self.auto_model.predict(
                            frame,
                            conf=self.auto_conf_thresh,
                            iou=self.iou_thresh,
                            imgsz=self.img_size,
                            device=self.device,
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
                    except Exception:
                        pass

            # NMS and deduplication between models
            auto_boxes = [d["bbox"] for d in vehicle_detections if d["class"] == "auto"]
            if auto_boxes:
                vehicle_detections = [
                    d for d in vehicle_detections
                    if d["class"] == "auto" or not any(box_iou(d["bbox"], abox) >= 0.15 for abox in auto_boxes)
                ]

            vehicle_detections = apply_nms(vehicle_detections, iou_thresh=self.iou_thresh)
            self.tracker.update(vehicle_detections, h, nms_iou=self.iou_thresh)
        else:
            self.tracker.check_lane_crossings(h)

        # ROI Counting & Density Classification
        roi_total, roi_type_counts, roi_objects = self.tracker.count_in_roi(self.roi_bounds, w, h)

        if roi_total <= self.low_thresh:
            density_level = "Low"
            density_bgr = (0, 255, 0)
        elif roi_total <= self.med_thresh:
            density_level = "Medium"
            density_bgr = (0, 255, 255)
        else:
            density_level = "High"
            density_bgr = (0, 0, 255)

        # Render ROI overlay
        ymin_pct, xmin_pct, ymax_pct, xmax_pct = self.roi_bounds
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

        # Render tracked vehicle bounding boxes
        for object_id, centroid in self.tracker.objects.items():
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

        # Counting line
        line_y = int(self.tracker.counting_line["y"] * h)
        cv2.line(processed, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(
            processed, "COUNTING LINE", (10, max(20, line_y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        signal_status = self.signal_manager.check_traffic_conditions(self.tracker.counts)

        # HUD Overlay
        total_line_crossings = sum(self.tracker.counts.values())
        cv2.rectangle(processed, (8, 8), (380, 52), (20, 25, 35), -1)
        cv2.rectangle(processed, (8, 8), (380, 52), (100, 116, 139), 1)
        cv2.putText(
            processed, f"Crossings: {total_line_crossings} | ROI: {density_level.upper()}", (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            processed, f"Signal: {signal_status}", (16, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 240), 1, cv2.LINE_AA
        )

        # Accident Detection (Road-aware with active tracks filter)
        accidents, suspicious_events = self.accident_detector.update(
            self.tracker.objects, w, h, frame=frame, disappeared=getattr(self.tracker, "disappeared", None)
        )

        # Render Accident Danger Overlays
        for acc in accidents:
            ax1, ay1, ax2, ay2 = acc["bbox"]
            ax1, ay1 = max(0, int(ax1)), max(0, int(ay1))
            ax2, ay2 = min(w - 1, int(ax2)), min(h - 1, int(ay2))
            cv2.rectangle(processed, (ax1, ay1), (ax2, ay2), (0, 0, 255), 3)
            cv2.rectangle(processed, (ax1 + 2, ay1 + 2), (ax2 - 2, ay2 - 2), (255, 255, 255), 1)

            vids_tag = ','.join(map(str, acc.get('vehicle_ids', [])))
            acc_label = f"! HARD ACCIDENT #{vids_tag} [{acc.get('confidence', '90%')}]"
            text_size = cv2.getTextSize(acc_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            label_y = (ay1 - 8) if ay1 > 28 else (ay1 + text_size[1] + 10)
            label_x = max(4, min(ax1, w - text_size[0] - 12))
            cv2.rectangle(processed, (label_x - 2, label_y - text_size[1] - 4), (label_x + text_size[0] + 6, label_y + 4), (0, 0, 200), -1)
            cv2.rectangle(processed, (label_x - 2, label_y - text_size[1] - 4), (label_x + text_size[0] + 6, label_y + 4), (255, 255, 255), 1)
            cv2.putText(processed, acc_label, (label_x + 2, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        if accidents:
            banner_w = 360
            bx1 = max(10, w // 2 - banner_w // 2)
            bx2 = min(w - 10, bx1 + banner_w)
            cv2.rectangle(processed, (bx1, 8), (bx2, 44), (0, 0, 200), -1)
            cv2.rectangle(processed, (bx1, 8), (bx2, 44), (255, 255, 255), 2)
            cv2.putText(processed, "!!! HARD ACCIDENT DETECTED !!!", (bx1 + 18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS calculation
        elapsed = max(0.001, time.time() - t_start)
        instant_fps = 1.0 / elapsed
        self.fps_history.append(instant_fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        current_fps = sum(self.fps_history) / len(self.fps_history)

        # Structured telemetry payload for server sync
        telemetry = {
            "frame_number": self.frame_counter,
            "fps": round(current_fps, 1),
            "density_level": density_level,
            "total_roi_vehicles": roi_total,
            "line_crossings": total_line_crossings,
            "counts": dict(self.tracker.counts),
            "roi_counts": {
                "car": roi_type_counts.get("car", 0),
                "motorcycle": roi_type_counts.get("motorcycle", 0),
                "bus": roi_type_counts.get("bus", 0),
                "truck": roi_type_counts.get("truck", 0),
                "auto": roi_type_counts.get("auto", 0),
            },
            "signal_status": signal_status,
            "system_condition": self.accident_detector.system_condition,
            "accidents": accidents,
            "suspicious_events": suspicious_events,
        }

        return processed, telemetry

