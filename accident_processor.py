"""Road-Aware Intelligent Accident Detection & Incident Processing Engine for ATCC.

Distinguishes between different road environments (HIGHWAY, NORMAL_STREET, INTERSECTION, RESIDENTIAL_ROAD, UNKNOWN):
- 5-State Decision Machine: NORMAL -> SUSPICIOUS_BEHAVIOUR -> POSSIBLE_COLLISION -> CONFIRMED_ACCIDENT -> ACCIDENT_RESOLVED
- 4-Tier Alert Levels: GREEN (Normal), YELLOW (Suspicious), ORANGE (Possible), RED (Confirmed Danger)
- Road-specific behavioral intelligence:
    * NORMAL_STREET / INTERSECTION: Red traffic signals, queue stops, intersection turns, pedestrian stops,
      motorcycle lane filtering, and roadside parking are recognized as NORMAL traffic.
    * HIGHWAY: High-speed cruising, smooth lane changes, overtaking, traffic jam slow-downs are recognized as NORMAL;
      shoulder breakdowns are distinguished from lane crashes; sudden braking + kinetic arrest + ground impact confirms crashes.
- Ground-contact tire footprint IoU (bottom 30% patch) eliminates 2D camera perspective overlap.
- Multi-frame temporal verification window with kinetic arrest, impulse deceleration, and trajectory deflection.
- Integrated VehicleDamageAnalyzer and RoadROIManager with strict exclusion zones (houses, trees, sidewalks).
- Full frame-by-frame debug telemetry.
"""
import math
from collections import defaultdict
from datetime import datetime
import cv2

try:
    from .road_roi import RoadROIManager, RoadType, RoadClassifier
    from .damage_analyzer import VehicleDamageAnalyzer
except ImportError:
    from road_roi import RoadROIManager, RoadType, RoadClassifier
    from damage_analyzer import VehicleDamageAnalyzer


class AccidentState:
    NORMAL = "NORMAL"                         # State 1: Green - Normal traffic flow (No collision)
    POSSIBLE_IMPACT = "POSSIBLE IMPACT"       # State 2: Orange - Collision evidence is being verified (DO NOT generate alert)
    POSSIBLE_COLLISION = "POSSIBLE IMPACT"    # Backward-compatible alias
    SUSPICIOUS = "POSSIBLE IMPACT"            # Backward-compatible alias
    CONFIRMED_ACCIDENT = "CONFIRMED ACCIDENT" # State 3: Red - Actual collision evidence verified (Generate red alert)
    RESOLVED = "ACCIDENT_RESOLVED"            # State 4: Handled / archived incident


class AlertLevel:
    GREEN = "GREEN"     # Normal Traffic
    YELLOW = "YELLOW"   # Suspicious / Checking
    ORANGE = "ORANGE"   # Possible Impact (verifying)
    RED = "RED"         # Confirmed Accident


def box_iou(box1: tuple, box2: tuple) -> float:
    """Calculate 2D Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def ground_footprint_iou(box1: tuple, box2: tuple, bottom_pct: float = 0.30) -> float:
    """Calculate IoU of the road contact footprint (bottom 30% of each bounding box).

    In camera perspective, vehicles in adjacent lanes have overlapping 2D boxes,
    but their ground tire contact patches on the asphalt NEVER intersect (IoU = 0).
    """
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    gy1_a = y2_a - (y2_a - y1_a) * bottom_pct
    gy2_a = y2_a
    gy1_b = y2_b - (y2_b - y1_b) * bottom_pct
    gy2_b = y2_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(gy1_a, gy1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(gy2_a, gy2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(1.0, (x2_a - x1_a) * (gy2_a - gy1_a))
    area_b = max(1.0, (x2_b - x1_b) * (gy2_b - gy1_b))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def is_camera_boundary(bbox: tuple, frame_w: int, frame_h: int) -> bool:
    """Check if vehicle is entering or exiting at the camera FOV boundaries."""
    x1, y1, x2, y2 = bbox
    if y2 >= 0.88 * frame_h:
        return True
    if y1 <= 0.05 * frame_h:
        return True
    if x1 <= 0.02 * frame_w or x2 >= 0.98 * frame_w:
        return True
    return False


class LocalAccidentDetector:
    """Intelligent Road-Aware Accident Detection & Incident Management Engine."""

    CONF_THRESHOLD_PCT = 85
    VERIFY_WINDOW_FRAMES = 10
    COOLDOWN_FRAMES = 200

    # Sensitivity presets. "Strict" = original behaviour (very few false alarms, may miss
    # real crashes). "Sensitive" catches more crashes but may raise more false alarms.
    SENSITIVITY_PRESETS = {
        "Strict":    dict(conf=85, verify=10, iou2d=0.15, pre_speed=2.2, decel=0.35,
                          arrest=0.50, g_iou=0.06, single_pre=7.5, single_arrest=0.60,
                          wreck_still=0, wreck_prior=99.0, wreck_overlap=1.0, wreck_moved=99.0),
        "Balanced":  dict(conf=65, verify=8,  iou2d=0.12, pre_speed=1.8, decel=0.25,
                          arrest=0.35, g_iou=0.04, single_pre=6.0, single_arrest=0.50,
                          wreck_still=15, wreck_prior=4.5, wreck_overlap=0.40, wreck_moved=2.5),
        "Sensitive": dict(conf=50, verify=6,  iou2d=0.08, pre_speed=1.4, decel=0.15,
                          arrest=0.20, g_iou=0.02, single_pre=5.0, single_arrest=0.40,
                          wreck_still=10, wreck_prior=2.5, wreck_overlap=0.30, wreck_moved=1.5),
    }

    def __init__(
        self,
        confidence_threshold: int = 85,
        verify_window: int = 10,
        road_roi: "RoadROIManager" = None,
        road_type: str = RoadType.UNKNOWN,
        sensitivity: str = None,
    ):
        self.conf_threshold = confidence_threshold
        self.verify_window = verify_window
        self.sensitivity = sensitivity if sensitivity in self.SENSITIVITY_PRESETS else "Strict"
        self.P = dict(self.SENSITIVITY_PRESETS[self.sensitivity])
        if sensitivity in self.SENSITIVITY_PRESETS:
            self.conf_threshold = self.P["conf"]
            self.verify_window = self.P["verify"]
        self.cooldown_frames = self.COOLDOWN_FRAMES

        # Road ROI & Environment Intelligence
        self.road_roi = road_roi or RoadROIManager(road_type=road_type)
        self.road_type = self.road_roi.road_type
        # UNKNOWN = automatic: the classifier infers the road type from vehicle motion.
        # Any explicit type is a manual override and is never replaced by the classifier.
        self.road_type_locked = self.road_type != RoadType.UNKNOWN
        self.road_classifier = RoadClassifier()

        self.damage_analyzer = VehicleDamageAnalyzer()
        self.crop_buffer = defaultdict(list)

        self.trajectories = defaultdict(list)
        self.velocities = {}
        self.max_travel_speeds = defaultdict(float)

        self.candidates = {}
        self.active_confirmed_map = {}
        self.active_accidents = []
        self.suspicious_events = []
        self.verifying_candidates = []
        self.cooldown_map = {}
        self.frame_idx = 0

        # 5-State Machine & Alert Level
        self.accident_state = AccidentState.NORMAL
        self.alert_level = AlertLevel.GREEN
        self.system_condition = "NORMAL"

        # Telemetry & Diagnostics
        self.debug_telemetry = []
        self.diagnostic_records = []
        self.frame_telemetry_table = []
        self.last_eval_log = {}
        self.prev_frame_gray = None

        # Estimated highway flow vector (dx, dy)
        self.road_flow_vector = (0.0, 1.0)

    def _log(self, msg: str):
        self.debug_telemetry.append(msg)
        if len(self.debug_telemetry) > 250:
            self.debug_telemetry.pop(0)

    def _raw_speed(self, hist, lookback=4):
        """Compute raw average speed over last `lookback` frames."""
        if len(hist) < 2:
            return 0.0
        speeds = []
        for i in range(max(0, len(hist) - lookback), len(hist)):
            if i > 0:
                dx = hist[i]["cx"] - hist[i - 1]["cx"]
                dy = hist[i]["cy"] - hist[i - 1]["cy"]
                speeds.append(math.hypot(dx, dy))
        return sum(speeds) / max(1, len(speeds))

    def get_diagnostics(self) -> list[dict]:
        """Return structured diagnostic records of recent evaluations."""
        return self.diagnostic_records[-30:]

    def get_debug_telemetry_table(self) -> list[dict]:
        """Return full frame-by-frame debug telemetry records."""
        return self.frame_telemetry_table[-50:]

    def road_condition(self) -> dict:
        """Verified road condition for the dashboard: type in use, how it was chosen, and whether
        the observed traffic behaviour agrees with it."""
        inferred = self.road_classifier.inferred_road_type
        analysed = self.road_classifier.frames_analyzed
        ready = inferred != RoadType.UNKNOWN
        mismatch = bool(self.road_type_locked and ready and inferred != self.road_type
                        and self.road_classifier.confidence_pct >= 75)
        return {
            "road_type": self.road_type,
            "source": "manual" if self.road_type_locked else "auto",
            "observed_type": inferred if ready else "UNKNOWN (collecting traffic data)",
            "observed_confidence_pct": self.road_classifier.confidence_pct if ready else 0,
            "frames_analyzed": analysed,
            "mismatch": mismatch,
        }

    def set_road_type(self, road_type: str):
        """Update road classification (HIGHWAY, NORMAL_STREET, INTERSECTION, RESIDENTIAL_ROAD, UNKNOWN)."""
        if road_type in RoadType.ALL_TYPES:
            self.road_type = road_type
            self.road_roi.road_type = road_type
            self.road_type_locked = road_type != RoadType.UNKNOWN
            self._log(f"[CONFIG] Road type set to: {road_type}")

    def update(
        self,
        tracker_objects: dict,
        frame_w: int,
        frame_h: int,
        frame=None,
        disappeared: dict = None,
    ) -> tuple[list, list]:
        """Update detector state with tracked objects, filtering coasted/ghost tracks."""
        self.frame_idx += 1
        self.active_accidents = []
        self.suspicious_events = []
        self.verifying_candidates = []

        # 1. Active Objects Filter: Ignore disappeared / coasted tracks
        if disappeared is not None:
            active_objects = {oid: obj for oid, obj in tracker_objects.items() if disappeared.get(oid, 0) == 0}
        else:
            active_objects = tracker_objects

        active_oids = set(active_objects.keys())

        # 2. Maintain active confirmed accidents while scene is active
        for cand_key, alert_data in list(self.active_confirmed_map.items()):
            if self.frame_idx - alert_data.get("confirmed_frame", 0) < 180:
                self.active_accidents.append(alert_data)
            else:
                self.active_confirmed_map.pop(cand_key, None)

        # 3. Update active vehicle trajectories & velocity vectors
        flow_dx_list = []
        flow_dy_list = []
        current_frame_telemetry = []

        for oid in active_oids:
            cx, cy, det = active_objects[oid]
            bbox = det["bbox"]
            vclass = det.get("class", "vehicle")
            x1, y1, x2, y2 = bbox
            w = max(1.0, float(x2 - x1))
            h_box = max(1.0, float(y2 - y1))
            aspect = w / h_box
            area = w * h_box

            hist = self.trajectories[oid]
            if hist:
                prev = hist[-1]
                frame_gap = max(1, self.frame_idx - prev["frame"])
                raw_dx = (cx - prev["cx"]) / frame_gap
                raw_dy = (cy - prev["cy"]) / frame_gap
                raw_disp = math.hypot(raw_dx, raw_dy)
                raw_speed = min(raw_disp, 50.0)

                # Cap abrupt tracker jumps
                if raw_disp > 50.0:
                    raw_dx = prev.get("vx", 0.0)
                    raw_dy = prev.get("vy", 0.0)
                    raw_speed = math.hypot(raw_dx, raw_dy)

                prev_vx, prev_vy = self.velocities.get(oid, (raw_dx, raw_dy))
                vx = 0.65 * raw_dx + 0.35 * prev_vx
                vy = 0.65 * raw_dy + 0.35 * prev_vy
                ema_speed = math.hypot(vx, vy)
            else:
                raw_dx, raw_dy = 0.0, 0.0
                vx, vy = 0.0, 0.0
                ema_speed = 0.0
                raw_speed = 0.0

            self.velocities[oid] = (vx, vy)
            if raw_speed > self.max_travel_speeds[oid]:
                self.max_travel_speeds[oid] = raw_speed

            # Stable heading: only update when vehicle is actually moving (> 1.8 px/f)
            if raw_speed > 1.8:
                heading = math.atan2(raw_dy, raw_dx)
                flow_dx_list.append(raw_dx)
                flow_dy_list.append(raw_dy)
            else:
                heading = hist[-1]["heading"] if hist else 0.0

            hist.append({
                "frame": self.frame_idx,
                "cx": cx,
                "cy": cy,
                "bbox": bbox,
                "w": w,
                "h": h_box,
                "aspect": aspect,
                "area": area,
                "vx": vx,
                "vy": vy,
                "speed": ema_speed,
                "raw_speed": raw_speed,
                "heading": heading,
                "class": vclass,
            })
            if len(hist) > 40:
                hist.pop(0)

            # Buffer image crops for pre-impact visual damage inspection
            if frame is not None and getattr(frame, "shape", None) is not None:
                cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(frame_w, int(x2)), min(frame_h, int(y2))
                if cx2 - cx1 >= 15 and cy2 - cy1 >= 15:
                    crop = frame[cy1:cy2, cx1:cx2].copy()
                    self.crop_buffer[oid].append((self.frame_idx, crop))
                    if len(self.crop_buffer[oid]) > 30:
                        self.crop_buffer[oid].pop(0)

            # Check road ROI status
            is_on_road = self.road_roi.bbox_ground_on_road(bbox, frame_w, frame_h)
            is_shoulder = self.road_roi.is_in_shoulder(cx, cy, frame_w, frame_h)
            is_signal_stop = self.road_roi.is_in_signal_zone(cx, cy, frame_w, frame_h)
            is_intersection = self.road_roi.is_in_intersection_zone(cx, cy, frame_w, frame_h)

            roi_status_tag = "ROAD" if is_on_road else "OFF_ROAD"
            if is_shoulder:
                roi_status_tag = "SHOULDER"
            elif is_signal_stop:
                roi_status_tag = "SIGNAL_ZONE"
            elif is_intersection:
                roi_status_tag = "INTERSECTION_ZONE"

            # Log telemetry row
            current_frame_telemetry.append({
                "frame": self.frame_idx,
                "vehicle_id": oid,
                "class": vclass,
                "bbox": bbox,
                "center": (round(cx, 1), round(cy, 1)),
                "speed_px_f": round(ema_speed, 2),
                "heading_deg": round(math.degrees(heading), 1),
                "road_roi_status": roi_status_tag,
                "evidence": "Tracking",
                "state": self.accident_state,
                "reason": "Normal flow",
            })

            # Stage 2: Sudden hard braking detection (SUSPICIOUS ACTIVITY - YELLOW)
            # Only if NOT stopped at traffic light or stop line
            if len(hist) >= 6:
                pre_raw = [h.get("raw_speed", h["speed"]) for h in hist[-6:-1]]
                avg_prior_raw = sum(pre_raw) / max(1, len(pre_raw))
                curr_raw = hist[-1].get("raw_speed", hist[-1]["speed"])

                # Normal street braking for pedestrian or signal queue is NORMAL
                is_normal_street_stop = (
                    self.road_type in (RoadType.NORMAL_STREET, RoadType.INTERSECTION)
                    and (is_signal_stop or curr_raw < 1.0)
                )

                if (
                    avg_prior_raw >= 6.0
                    and curr_raw < avg_prior_raw * 0.35
                    and not is_camera_boundary(bbox, frame_w, frame_h)
                    and not is_normal_street_stop
                    and not is_shoulder
                ):
                    self.suspicious_events.append({
                        "type": "Sudden Hard Braking",
                        "vehicle_id": oid,
                        "bbox": bbox,
                        "severity": "WARNING",
                        "details": f"Vehicle #{oid} hard braking: {avg_prior_raw:.1f} -> {curr_raw:.1f} px/f",
                    })

        self.frame_telemetry_table.extend(current_frame_telemetry)

        # Update road flow vector
        if flow_dx_list and flow_dy_list:
            avg_fx = sum(flow_dx_list) / len(flow_dx_list)
            avg_fy = sum(flow_dy_list) / len(flow_dy_list)
            mag = math.hypot(avg_fx, avg_fy)
            if mag > 0.5:
                self.road_flow_vector = (avg_fx / mag, avg_fy / mag)

        # Road classification runs ALWAYS (as a shadow check when the type was set manually) so the
        # dashboard can verify the road condition. It only changes the rules when type is automatic.
        inferred = self.road_classifier.update_with_frame_trajectories(self.trajectories)
        if not self.road_type_locked and inferred != RoadType.UNKNOWN and inferred != self.road_type:
            self.road_type = inferred
            self.road_roi.road_type = inferred
            self._log(f"[AUTO-CLASSIFIER] Road classified as: {inferred}")

        # 4. Stage 3: Collision Candidate Identification (Road-aware)
        self._identify_candidates(list(active_oids), active_objects, frame_w, frame_h)
        self._identify_wreck_candidates(list(active_oids), active_objects, frame_w, frame_h)

        # 5. Stage 4: Multi-Stage Consecutive Frame Verification
        self._verify_candidates(active_objects, frame_w, frame_h, active_oids, frame=frame)

        # 6. Cleanup expired trajectories
        for oid in list(self.trajectories.keys()):
            if oid not in active_oids and (self.frame_idx - self.trajectories[oid][-1]["frame"] > 30):
                del self.trajectories[oid]
                self.velocities.pop(oid, None)
                self.max_travel_speeds.pop(oid, None)
                self.crop_buffer.pop(oid, None)

        # 7. Populate suspicious events with active verifying candidates
        for cand in self.verifying_candidates:
            self.suspicious_events.append({
                "type": f"Investigating {cand['type']}",
                "vehicle_ids": cand["vehicles"],
                "bbox": cand["bbox"],
                "severity": "WARNING",
                "details": f"Investigating collision interaction ({cand['progress_pct']}% verified)",
            })

        # 8. State Machine & Alert Level Assignment (NORMAL -> POSSIBLE IMPACT -> CONFIRMED ACCIDENT)
        if self.active_accidents:
            self.accident_state = AccidentState.CONFIRMED_ACCIDENT
            self.alert_level = AlertLevel.RED
            self.system_condition = "DANGER"
        elif self.verifying_candidates:
            self.accident_state = AccidentState.POSSIBLE_IMPACT
            self.alert_level = AlertLevel.ORANGE
            self.system_condition = "POSSIBLE IMPACT"
        else:
            self.accident_state = AccidentState.NORMAL
            self.alert_level = AlertLevel.GREEN
            self.system_condition = "NORMAL"

        if frame is not None and getattr(frame, "size", 0) > 0:
            try:
                self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass

        return self.active_accidents, self.suspicious_events

    def _identify_candidates(self, object_ids: list, tracker_objects: dict, frame_w: int, frame_h: int):
        """Identify potential collision candidates using ground footprint overlap and road-specific kinematics."""
        # A. Multi-Vehicle Collision Candidates
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                id1, id2 = object_ids[i], object_ids[j]
                pair = (min(id1, id2), max(id1, id2))
                cand_key = f"pair_{pair[0]}_{pair[1]}"

                if self.frame_idx - self.cooldown_map.get(cand_key, -9999) < self.cooldown_frames:
                    continue
                if cand_key in self.candidates:
                    continue

                hist1 = self.trajectories[id1]
                hist2 = self.trajectories[id2]
                if len(hist1) < 5 or len(hist2) < 5:
                    continue

                box1 = tracker_objects[id1][2]["bbox"]
                box2 = tracker_objects[id2][2]["bbox"]
                cls1 = tracker_objects[id1][2].get("class", "vehicle")
                cls2 = tracker_objects[id2][2].get("class", "vehicle")

                # Boundary Check: Discard vehicles exiting or entering camera edges
                if is_camera_boundary(box1, frame_w, frame_h) or is_camera_boundary(box2, frame_w, frame_h):
                    continue

                # Road Surface Containment: Both vehicles must have tire ground contact inside road ROI
                if not (self.road_roi.bbox_ground_on_road(box1, frame_w, frame_h) and
                        self.road_roi.bbox_ground_on_road(box2, frame_w, frame_h)):
                    continue

                # 2D Bounding Box Overlap
                iou_2d = box_iou(box1, box2)
                if iou_2d < self.P["iou2d"]:
                    continue

                # GROUND FOOTPRINT ANALYSIS
                g_iou = ground_footprint_iou(box1, box2, bottom_pct=0.30)

                cx1 = (box1[0] + box1[2]) / 2.0
                cx2 = (box2[0] + box2[2]) / 2.0
                w1 = max(1.0, float(box1[2] - box1[0]))
                w2 = max(1.0, float(box2[2] - box2[0]))
                lateral_ground_gap = abs(cx1 - cx2)

                # Separate Lane Check: In adjacent highway lanes, 2D boxes overlap but ground footprint IoU is 0
                if g_iou < 0.04 and lateral_ground_gap >= 0.40 * min(w1, w2):
                    self.diagnostic_records.append({
                        "frame": self.frame_idx,
                        "pair": [id1, id2],
                        "status": "NORMAL",
                        "reason": f"Adjacent lanes / passing: ground IoU={g_iou:.2f}, lateral gap={lateral_ground_gap:.0f}px",
                        "box_iou": round(iou_2d, 2),
                        "ground_iou": round(g_iou, 2),
                    })
                    continue

                # Kinematic Analysis: Pre-impact speeds
                def raw_avg_speed(hist, start, end):
                    speeds = []
                    for k in range(max(1, start), min(len(hist), end)):
                        dx = hist[k]["cx"] - hist[k-1]["cx"]
                        dy = hist[k]["cy"] - hist[k-1]["cy"]
                        speeds.append(min(math.hypot(dx, dy), 50.0))
                    return sum(speeds) / max(1, len(speeds))

                lb1 = len(hist1)
                lb2 = len(hist2)
                v1_pre = raw_avg_speed(hist1, lb1 - 5, lb1 - 1)
                v2_pre = raw_avg_speed(hist2, lb2 - 5, lb2 - 1)
                max_pre = max(v1_pre, v2_pre)

                if max_pre < self.P["pre_speed"]:
                    continue

                # Closing Rate
                look = min(4, lb1 - 1, lb2 - 1)
                if look < 1:
                    continue
                p1_prev = (hist1[-(look+1)]["cx"], hist1[-(look+1)]["cy"])
                p2_prev = (hist2[-(look+1)]["cx"], hist2[-(look+1)]["cy"])
                p1_curr = (hist1[-1]["cx"], hist1[-1]["cy"])
                p2_curr = (hist2[-1]["cx"], hist2[-1]["cy"])
                dist_prev = math.hypot(p1_prev[0] - p2_prev[0], p1_prev[1] - p2_prev[1])
                dist_curr = math.hypot(p1_curr[0] - p2_curr[0], p1_curr[1] - p2_curr[1])
                closing_rate = (dist_prev - dist_curr) / max(1, look)

                # Deceleration Impulse
                curr_raw1 = min(hist1[-1].get("raw_speed", hist1[-1]["speed"]), 50.0)
                curr_raw2 = min(hist2[-1].get("raw_speed", hist2[-1]["speed"]), 50.0)
                decel_ratio = max(
                    (v1_pre - curr_raw1) / max(0.1, v1_pre),
                    (v2_pre - curr_raw2) / max(0.1, v2_pre),
                )
                if decel_ratio < self.P["decel"]:
                    continue

                # Approach Heading Difference
                h1_pre = hist1[-(look+1)]["heading"]
                h2_pre = hist2[-(look+1)]["heading"]
                h_diff = abs(h1_pre - h2_pre)
                if h_diff > math.pi:
                    h_diff = 2 * math.pi - h_diff
                h_diff_deg = math.degrees(h_diff)

                # --- ROAD TYPE BEHAVIORAL DISCRIMINATORS ---

                # 1. Normal Overtaking Check (Highway & Street):
                # Moving in same direction (h_diff < 35°), both maintain positive forward travel (speed >= 3.0)
                if h_diff_deg < 35.0:
                    if curr_raw1 >= 2.8 and curr_raw2 >= 2.8 and g_iou < 0.10:
                        self.diagnostic_records.append({
                            "frame": self.frame_idx,
                            "pair": [id1, id2],
                            "status": "NORMAL",
                            "reason": f"Normal overtaking: forward travel maintained (v1={curr_raw1:.1f}, v2={curr_raw2:.1f})",
                            "box_iou": round(iou_2d, 2),
                            "ground_iou": round(g_iou, 2),
                        })
                        continue

                # 2. Motorcycle Lane Filtering Check (Normal Street & Highway):
                # Two-wheelers filtering between cars without collision impact
                if "motorcycle" in (cls1, cls2) or "auto" in (cls1, cls2):
                    if g_iou < 0.08 and decel_ratio < 0.60:
                        self.diagnostic_records.append({
                            "frame": self.frame_idx,
                            "pair": [id1, id2],
                            "status": "NORMAL",
                            "reason": f"Motorcycle lane filtering / gap travel (cls1={cls1}, cls2={cls2}, g_iou={g_iou:.2f})",
                        })
                        continue

                # 3. Traffic Signal Stop / Queue Check (Normal Street & Intersection):
                # When cars queue up at red light or stop line smoothly
                if self.road_type in (RoadType.NORMAL_STREET, RoadType.INTERSECTION):
                    mid_y = (box1[3] + box2[3]) / 2.0
                    in_signal = self.road_roi.is_in_signal_zone(cx1, mid_y, frame_w, frame_h)
                    if in_signal or (curr_raw1 < 1.0 and curr_raw2 < 1.0 and g_iou < 0.12 and h_diff_deg < 30.0):
                        self.diagnostic_records.append({
                            "frame": self.frame_idx,
                            "pair": [id1, id2],
                            "status": "NORMAL",
                            "reason": f"Traffic light queue stop: cars #{id1} and #{id2} queued at signal line",
                        })
                        continue

                # 4. Parking at Roadside Check (Street & Residential):
                if self.road_type in (RoadType.NORMAL_STREET, RoadType.RESIDENTIAL_ROAD):
                    if (cx1 <= 0.15 * frame_w or cx1 >= 0.85 * frame_w) and (curr_raw1 < 0.8 or curr_raw2 < 0.8) and g_iou < 0.08:
                        continue

                union_box = (
                    min(box1[0], box2[0]),
                    min(box1[1], box2[1]),
                    max(box1[2], box2[2]),
                    max(box1[3], box2[3]),
                )

                self._log(f"[F{self.frame_idx}] CANDIDATE REGISTERED: {cand_key} | "
                          f"2D_IoU={iou_2d:.2f}, Ground_IoU={g_iou:.2f}, v_pre={max_pre:.1f}px/f, decel={decel_ratio:.0%}")

                self.candidates[cand_key] = {
                    "key": cand_key,
                    "type": "Vehicle Collision",
                    "vehicles": [pair[0], pair[1]],
                    "start_frame": self.frame_idx,
                    "frames_observed": 0,
                    "v_pre": {pair[0]: v1_pre, pair[1]: v2_pre},
                    "headings_pre": {pair[0]: h1_pre, pair[1]: h2_pre},
                    "h_diff_deg_pre": h_diff_deg,
                    "max_ground_iou": g_iou,
                    "max_box_iou": iou_2d,
                    "bbox": union_box,
                    "arrest_frames": 0,
                    "deflection_deg": 0.0,
                    "aspect_change_pct": 0.0,
                    "status": AccidentState.POSSIBLE_IMPACT,
                }

        # B. Single-Vehicle Roadside Crash / Rollover Candidates
        for oid in object_ids:
            cand_key = f"single_{oid}"
            if self.frame_idx - self.cooldown_map.get(cand_key, -9999) < self.cooldown_frames:
                continue
            if cand_key in self.candidates:
                continue

            hist = self.trajectories[oid]
            if len(hist) < 8:
                continue

            box = tracker_objects[oid][2]["bbox"]
            if is_camera_boundary(box, frame_w, frame_h):
                continue

            # Road Surface Containment: Vehicle tire ground contact must be inside road ROI
            if not self.road_roi.bbox_ground_on_road(box, frame_w, frame_h):
                continue

            # Must have high pre-impact speed
            pre_raw = [min(h.get("raw_speed", 0.0), 50.0) for h in hist[-8:-2]]
            avg_pre_raw = sum(pre_raw) / max(1, len(pre_raw))
            if avg_pre_raw < self.P["single_pre"]:
                continue

            curr_raw = min(hist[-1].get("raw_speed", 0.0), 50.0)
            if curr_raw > 1.5:  # Must have suffered complete road stop
                continue

            ref_idx = max(0, len(hist) - 8)
            aspect_init = hist[ref_idx]["aspect"]
            aspect_curr = hist[-1]["aspect"]
            aspect_ratio_diff = abs(aspect_curr - aspect_init) / max(0.1, aspect_init)

            head_ref = hist[ref_idx]["heading"]
            head_curr = hist[-1]["heading"]
            angle_diff = abs(head_curr - head_ref)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            deg_diff = math.degrees(angle_diff)

            # --- ROAD TYPE DISCRIMINATOR FOR SINGLE-VEHICLE CRASH ---
            # On NORMAL_STREET and INTERSECTION: A 60°-120° heading change is a NORMAL TURN!
            # It must NEVER be classified as a rollover or violent spin crash.
            if self.road_type in (RoadType.NORMAL_STREET, RoadType.INTERSECTION, RoadType.RESIDENTIAL_ROAD):
                # Only severe structural aspect inversion (rollover >= 50%) can trigger crash
                if aspect_ratio_diff < 0.50:
                    continue
            else:
                # Highway rollover / violent spin-out: requires aspect change >= 45% or spin >= 75° with high speed
                if not (aspect_ratio_diff >= 0.45 or (deg_diff >= 75.0 and avg_pre_raw >= 10.0)):
                    continue

            self.candidates[cand_key] = {
                "key": cand_key,
                "type": "Roadside Crash / Rollover",
                "vehicles": [oid],
                "start_frame": self.frame_idx,
                "frames_observed": 0,
                "v_pre": {oid: avg_pre_raw},
                "headings_pre": {oid: head_ref},
                "h_diff_deg_pre": 0.0,
                "max_ground_iou": 0.0,  # Honest metric: single vehicle has 0 multi-vehicle ground overlap
                "max_box_iou": 0.0,
                "bbox": box,
                "arrest_frames": 0,
                "deflection_deg": deg_diff,
                "aspect_change_pct": aspect_ratio_diff * 100.0,
                "status": AccidentState.POSSIBLE_IMPACT,
            }

    def _identify_wreck_candidates(self, object_ids: list, tracker_objects: dict, frame_w: int, frame_h: int):
        """Stationary vehicles by themselves are NOT proof of a crash (per strict actual-impact requirement).
        Only active collisions with kinematic impact can trigger alerts."""
        return

        info = {}
        for oid in object_ids:
            hist = self.trajectories.get(oid, [])
            if len(hist) < K + 2:
                continue
            tail = hist[-K:]
            if any(min(h.get("raw_speed", 0.0), 50.0) > 1.2 for h in tail):
                continue                                  # not stationary for K frames
            box = tracker_objects[oid][2]["bbox"]
            cx = (box[0] + box[2]) / 2.0
            if self.road_roi.is_in_exclusion(cx, float(box[3]), frame_w, frame_h):
                continue                                  # house / tree / sidewalk false detections
            run_start = len(hist) - K
            while run_start > 0 and min(hist[run_start - 1].get("raw_speed", 0.0), 50.0) <= 1.2:
                run_start -= 1
            prior = [min(h.get("raw_speed", 0.0), 50.0) for h in hist[max(0, run_start - 6):run_start]]
            last_moving = min(hist[run_start - 1].get("raw_speed", 0.0), 50.0) if run_start > 0 else 0.0
            prior_max = max(prior) if prior else 0.0
            # Abrupt = still moving fast on the frame right before stopping (gradual braking at a
            # red light has already slowed down by then)
            abrupt = prior_max if last_moving >= 0.75 * self.P["wreck_prior"] else 0.0
            info[oid] = {
                "box": box,
                "prior": abrupt,
                "moved": self.max_travel_speeds.get(oid, 0.0) >= self.P["wreck_moved"],
                "cy": (box[1] + box[3]) / 2.0, "cx": cx,
            }

        def in_signal(i):
            return self.road_type in (RoadType.NORMAL_STREET, RoadType.INTERSECTION) and \
                self.road_roi.is_in_signal_zone(i["cx"], i["box"][3], frame_w, frame_h)

        # Rule A: abrupt stop (fast -> stopped within ~6 frames) away from the stop line
        for oid, i in info.items():
            key = f"wreck_{oid}"
            if key in self.candidates or self.frame_idx - self.cooldown_map.get(key, -9999) < self.cooldown_frames:
                continue
            if i["prior"] >= self.P["wreck_prior"] and not in_signal(i):
                self._register_wreck([oid], key, i["prior"], "abrupt stop", 0.0, i["box"])

        # Rule B: two stopped vehicles whose boxes overlap heavily (one of them has driven before,
        # which rules out parked cars that merely overlap in perspective)
        ids = list(info.keys())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                o1, o2 = ids[a], ids[b]
                key = f"wreck_{min(o1, o2)}_{max(o1, o2)}"
                if key in self.candidates or self.frame_idx - self.cooldown_map.get(key, -9999) < self.cooldown_frames:
                    continue
                i1, i2 = info[o1], info[o2]
                if not (i1["moved"] or i2["moved"]):
                    continue
                b1, b2 = i1["box"], i2["box"]
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                small = max(1.0, min((b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])))
                contain = inter / small
                if contain >= self.P["wreck_overlap"] and not (in_signal(i1) and in_signal(i2)):
                    union = (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
                    self._register_wreck([o1, o2], key, max(i1["prior"], i2["prior"]), "stopped & overlapping", contain, union)

    def _register_wreck(self, vids, key, prior_speed, reason, contain, bbox):
        self._log(f"[F{self.frame_idx}] WRECK CANDIDATE {key}: {reason}, prior_speed={prior_speed:.1f}, overlap={contain:.2f}")
        self.candidates[key] = {
            "key": key,
            "type": "Stationary Wreck / Sudden Stop",
            "vehicles": list(vids),
            "start_frame": self.frame_idx,
            "frames_observed": 0,
            "v_pre": {v: prior_speed for v in vids},
            "headings_pre": {v: 0.0 for v in vids},
            "h_diff_deg_pre": 0.0,
            "max_ground_iou": min(1.0, contain),
            "max_box_iou": min(1.0, contain),
            "bbox": bbox,
            "arrest_frames": 0,
            "deflection_deg": 0.0,
            "aspect_change_pct": 0.0,
            "wreck_reason": reason,
            "wreck_abrupt": prior_speed >= self.P["wreck_prior"],
            "wreck_contain": contain,
            "status": "POSSIBLE_COLLISION",
        }

    def _verify_candidates(self, tracker_objects: dict, frame_w: int, frame_h: int, active_oids: set, frame=None):
        """Multi-stage consecutive-frame confirmation with road-type aware dismissal."""
        for cand_key in list(self.candidates.keys()):
            cand = self.candidates[cand_key]
            cand["frames_observed"] += 1
            cand_type = cand["type"]
            vids = cand["vehicles"]

            all_active = all(v in active_oids for v in vids)

            if cand_type == "Vehicle Collision":
                id1, id2 = vids[0], vids[1]
                if all_active:
                    box1 = tracker_objects[id1][2]["bbox"]
                    box2 = tracker_objects[id2][2]["bbox"]

                    # Dismissal 1: Camera boundary exit
                    if is_camera_boundary(box1, frame_w, frame_h) or is_camera_boundary(box2, frame_w, frame_h):
                        self._log(f"[F{self.frame_idx}] {cand_key} DISMISSED: Vehicle exited camera FOV boundary.")
                        del self.candidates[cand_key]
                        continue

                    iou_2d = box_iou(box1, box2)
                    g_iou = ground_footprint_iou(box1, box2, bottom_pct=0.30)
                    cand["max_ground_iou"] = max(cand["max_ground_iou"], g_iou)
                    cand["max_box_iou"] = max(cand["max_box_iou"], iou_2d)

                    hist1 = self.trajectories[id1]
                    hist2 = self.trajectories[id2]
                    sp1_raw = hist1[-1].get("raw_speed", hist1[-1]["speed"])
                    sp2_raw = hist2[-1].get("raw_speed", hist2[-1]["speed"])

                    # Dismissal 2: Vehicles continued normal travel / completed overtake
                    if g_iou < 0.04 and (sp1_raw >= 2.8 or sp2_raw >= 2.8):
                        self._log(f"[F{self.frame_idx}] {cand_key} DISMISSED: Completed overtake / continued normal flow.")
                        del self.candidates[cand_key]
                        continue

                    # Kinetic arrest check: vehicles halted at scene
                    if sp1_raw <= 1.8 and sp2_raw <= 1.8:
                        cand["arrest_frames"] += 1

                    # Measure deflection
                    th1_init = cand["headings_pre"].get(id1, 0.0)
                    th2_init = cand["headings_pre"].get(id2, 0.0)
                    th1_curr = hist1[-1]["heading"]
                    th2_curr = hist2[-1]["heading"]

                    def angle_diff(a, b):
                        d = abs(a - b)
                        return d if d <= math.pi else 2 * math.pi - d

                    defl1 = math.degrees(angle_diff(th1_curr, th1_init))
                    defl2 = math.degrees(angle_diff(th2_curr, th2_init))
                    cand["deflection_deg"] = max(cand["deflection_deg"], defl1, defl2)

                    cand["bbox"] = (
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3]),
                    )
                else:
                    del self.candidates[cand_key]
                    continue

            elif cand_type == "Roadside Crash / Rollover":
                oid = vids[0]
                if oid in active_oids:
                    box = tracker_objects[oid][2]["bbox"]
                    if is_camera_boundary(box, frame_w, frame_h):
                        del self.candidates[cand_key]
                        continue
                    hist = self.trajectories[oid]
                    sp_raw = hist[-1].get("raw_speed", hist[-1]["speed"])
                    if sp_raw <= 1.5:
                        cand["arrest_frames"] += 1
                    elif sp_raw >= 3.5:
                        del self.candidates[cand_key]
                        continue
                else:
                    del self.candidates[cand_key]
                    continue

            elif cand_type == "Stationary Wreck / Sudden Stop":
                if all_active:
                    boxes = [tracker_objects[v][2]["bbox"] for v in vids]
                    cand["bbox"] = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                                    max(b[2] for b in boxes), max(b[3] for b in boxes))
                    speeds = [min(self.trajectories[v][-1].get("raw_speed", 0.0), 50.0) for v in vids]
                    if max(speeds) >= 3.5:               # someone drove off -> not a wreck
                        del self.candidates[cand_key]
                        continue
                    if max(speeds) <= 1.5:
                        cand["arrest_frames"] += 1
                else:
                    del self.candidates[cand_key]
                    continue

            # Stage 4: Comprehensive Evaluation at end of Verification Window
            if cand["frames_observed"] >= self.verify_window:
                v_pre_max = max(cand["v_pre"].values())
                arrest_ratio = min(1.0, cand["arrest_frames"] / float(max(1, self.verify_window - 2)))
                max_g_iou = cand.get("max_ground_iou", 0.0)
                defl_deg = cand.get("deflection_deg", 0.0)
                asp_chg = cand.get("aspect_change_pct", 0.0)
                h_diff_pre = cand.get("h_diff_deg_pre", 0.0)

                # Evidence-based scores (calibrated to real physics, realistic cap at 92%)
                if len(vids) == 1:
                    # Single-vehicle rollover / barrier crash: evaluate kinetic impulse, arrest, and violent rollover/spin
                    impulse_score = min(30.0, (v_pre_max / 8.0) * 30.0)
                    arrest_score = min(30.0, arrest_ratio * 30.0)
                    rollover_score = min(35.0, (asp_chg / 50.0) * 35.0) if asp_chg >= 40.0 else 0.0
                    spin_score = min(25.0, (defl_deg / 90.0) * 25.0) if (defl_deg >= 70.0 and self.road_type == RoadType.HIGHWAY) else 0.0
                    composite_conf = int(impulse_score + arrest_score + max(rollover_score, spin_score))
                    is_confirmed = (
                        composite_conf >= self.conf_threshold and
                        arrest_ratio >= self.P["single_arrest"] and
                        (defl_deg >= 70.0 or asp_chg >= 45.0) and
                        v_pre_max >= 7.0
                    )
                else:
                    # Multi-vehicle collision: require ground contact IoU, deceleration impulse, and post-impact arrest
                    g_score = min(30.0, (max_g_iou / 0.18) * 30.0)
                    impulse_score = min(25.0, (v_pre_max / 5.0) * 25.0)
                    arrest_score = min(25.0, arrest_ratio * 25.0)
                    impact_evidence = min(15.0, max(defl_deg / 35.0, (h_diff_pre / 90.0) * 0.75) * 15.0)
                    composite_conf = int(g_score + impulse_score + arrest_score + impact_evidence)
                    is_confirmed = (
                        composite_conf >= self.conf_threshold and
                        arrest_ratio >= self.P["arrest"] and
                        max_g_iou >= self.P["g_iou"] and
                        v_pre_max >= self.P["pre_speed"]
                    )

                composite_conf = min(92, max(0, composite_conf))

                cand["confidence_pct"] = composite_conf
                self._log(f"[F{self.frame_idx}] {cand_key} EVALUATED: conf={composite_conf}%, arrest={arrest_ratio:.0%}, "
                          f"g_iou={max_g_iou:.2f}, defl={defl_deg:.0f}deg, confirmed={is_confirmed}")

                if is_confirmed:
                    # Multi-vehicle visual damage analysis
                    damage_results = {}
                    for vid in vids:
                        pre_crop = None
                        if vid in self.crop_buffer and len(self.crop_buffer[vid]) > 0:
                            pre_crop = self.crop_buffer[vid][0][1]
                        post_crop = None
                        if vid in tracker_objects and frame is not None and getattr(frame, "shape", None) is not None:
                            det = tracker_objects[vid][2]
                            bx1, by1, bx2, by2 = det["bbox"]
                            bx1, by1 = max(0, int(bx1)), max(0, int(by1))
                            bx2, by2 = min(frame_w, int(bx2)), min(frame_h, int(by2))
                            if bx2 - bx1 >= 15 and by2 - by1 >= 15:
                                post_crop = frame[by1:by2, bx1:bx2].copy()

                        vclass = tracker_objects[vid][2].get("class", "vehicle") if vid in tracker_objects else "vehicle"
                        dmg = self.damage_analyzer.analyze_vehicle_damage(
                            pre_crop, post_crop, vid, vehicle_class=vclass, impact_angle_deg=cand.get("deflection_deg", 0.0)
                        )
                        damage_results[vid] = dmg

                    # Severity classification: CRITICAL / HIGH / MEDIUM / LOW
                    has_severe_dmg = any(d.get("severity_level") == "SEVERE" for d in damage_results.values())
                    if (max_g_iou >= 0.25 and arrest_ratio >= 0.70 and v_pre_max >= 7.0) or has_severe_dmg:
                        severity = "CRITICAL"
                    elif max_g_iou >= 0.12 and v_pre_max >= 4.5:
                        severity = "HIGH"
                    elif arrest_ratio >= 0.50:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    individual_boxes = {}
                    for v in vids:
                        if v in tracker_objects:
                            individual_boxes[v] = tracker_objects[v][2]["bbox"]
                        elif v in active_oids:
                            individual_boxes[v] = active_oids[v][2]["bbox"]

                    conf_str = f"{composite_conf}%"
                    confirmed_event = {
                        "type": cand_type,
                        "road_type": self.road_type,
                        "vehicle_ids": vids,
                        "bbox": cand["bbox"],
                        "individual_bboxes": individual_boxes,
                        "severity": severity,
                        "confidence": conf_str,
                        "confidence_pct": composite_conf,
                        "details": (
                            f"HARD ROAD ACCIDENT DETECTED [{self.road_type}]! "
                            f"Severity: {severity} | "
                            f"Pre-impact speed: {v_pre_max:.1f} px/f, "
                            f"Arrest: {int(arrest_ratio*100)}%, "
                            f"Ground Overlap: {int(max_g_iou*100)}%"
                        ),
                        "damage_assessment": damage_results,
                        "evidence": {
                            "road_type": self.road_type,
                            "severity": severity,
                            "ground_footprint_overlap": "N/A (Single Vehicle)" if len(vids) == 1 else f"{int(max_g_iou*100)}%",
                            "scene_kinetic_arrest": f"{int(arrest_ratio*100)}%",
                            "pre_impact_impulse": f"{int(min(100, (v_pre_max / 5.0) * 100))}%",
                            "trajectory_deflection": f"{int(defl_deg)}°",
                        },
                        "incident_key": cand["key"],
                        "confirmed_frame": self.frame_idx,
                    }
                    self.active_accidents.append(confirmed_event)
                    self.cooldown_map[cand["key"]] = self.frame_idx
                    self.active_confirmed_map[cand["key"]] = confirmed_event
                    self._log(f"[F{self.frame_idx}] *** ACCIDENT CONFIRMED: {cand['key']} at {conf_str} [{self.road_type}] ***")
                else:
                    if composite_conf >= 50 or arrest_ratio >= 0.40:
                        self.suspicious_events.append({
                            "type": "Possible Collision Interaction",
                            "vehicle_ids": vids,
                            "bbox": cand["bbox"],
                            "severity": "WARNING",
                            "details": f"Investigating traffic incident ({composite_conf}% evidence - non-critical)",
                        })
                del self.candidates[cand_key]
            else:
                self.verifying_candidates.append({
                    "type": cand["type"],
                    "vehicles": vids,
                    "bbox": cand["bbox"],
                    "progress_pct": int((cand["frames_observed"] / float(self.verify_window)) * 100),
                    "max_ground_iou": cand.get("max_ground_iou", 0.0),
                })


def format_accident_alert_payload(
    event: dict,
    feed_id: int,
    feed_title: str,
    camera_id: str,
    user_id: int,
) -> dict:
    """Format structured accident payload for central server transmission."""
    now = datetime.now()
    vids_str = ", ".join(f"#{v}" for v in event.get("vehicle_ids", []))
    incident_id = f"ACC-{feed_id}-{int(now.timestamp())}"
    road_type = event.get("road_type", "HIGHWAY")
    severity = event.get("severity", "HIGH")

    return {
        "incident_id": incident_id,
        "feed_id": feed_id,
        "feed_title": feed_title,
        "camera_id": camera_id,
        "location": f"Camera {camera_id} ({feed_title})",
        "road_type": road_type,
        "user_id": user_id,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "accident_type": event.get("type", "Vehicle Collision"),
        "confidence_pct": event.get("confidence_pct", 90),
        "severity": severity,
        "vehicle_ids": vids_str,
        "individual_bboxes": event.get("individual_bboxes", {}),
        "damage_assessment": event.get("damage_assessment", {}),
        "alert_message": (
            f"🚨 TRAFFIC ACCIDENT ALERT [{severity} - {road_type}]! Road accident detected on Camera {camera_id} "
            f"({feed_title}) at {now.strftime('%I:%M %p')}. Vehicles involved: {vids_str}. "
            f"Immediate emergency dispatch recommended!"
        ),
        "evidence_details": str(event.get("evidence", {})),
        "status": "unresolved",
        "bbox": list(event.get("bbox", [0, 0, 0, 0])),
    }