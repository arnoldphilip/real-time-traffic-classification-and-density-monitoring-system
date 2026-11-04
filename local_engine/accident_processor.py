"""Road-Aware Accident Detection & Incident Processing Engine for ATCC.

Specialized for highway and urban traffic monitoring:
- Ground-contact footprint analysis (eliminates 2D perspective overlap between adjacent lanes)
- Road flow vector alignment & normal overtaking distinction
- Multi-frame kinetic arrest and evidence-based confidence calculation (no fake 100%)
- Camera boundary exit filtering (prevents exiting vehicles from triggering false crashes)
- Consecutive-frame multi-stage verification (NORMAL -> SUSPICIOUS -> POSSIBLE -> CONFIRMED)
- Full diagnostic telemetry reporting
"""
import math
from collections import defaultdict
from datetime import datetime
import cv2


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
    # Vehicles exiting off bottom or entering from top
    if y2 >= 0.86 * frame_h:
        return True
    if y1 <= 0.06 * frame_h:
        return True
    if x1 <= 0.03 * frame_w or x2 >= 0.97 * frame_w:
        return True
    return False


class LocalAccidentDetector:
    """Road-Aware Road Accident Verification & Incident Management Engine."""

    CONF_THRESHOLD_PCT = 85
    VERIFY_WINDOW_FRAMES = 10
    COOLDOWN_FRAMES = 200

    def __init__(self, confidence_threshold: int = 85, verify_window: int = 10):
        self.conf_threshold = confidence_threshold
        self.verify_window = verify_window
        self.cooldown_frames = self.COOLDOWN_FRAMES

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
        self.system_condition = "NORMAL"

        # Telemetry & Diagnostics
        self.debug_telemetry = []
        self.diagnostic_records = []
        self.last_eval_log = {}
        self.prev_frame_gray = None

        # Estimated highway flow vector (dx, dy)
        self.road_flow_vector = (0.0, 1.0)

    def _log(self, msg: str):
        self.debug_telemetry.append(msg)
        if len(self.debug_telemetry) > 200:
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
        return self.diagnostic_records[-25:]

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

        # 1. Active Objects Filter: Ignore disappeared / ghost tracks
        if disappeared is not None:
            active_objects = {oid: obj for oid, obj in tracker_objects.items() if disappeared.get(oid, 0) == 0}
        else:
            active_objects = tracker_objects

        active_oids = set(active_objects.keys())

        # 2. Maintain active confirmed accidents while vehicles remain on scene
        for cand_key, alert_data in list(self.active_confirmed_map.items()):
            if self.frame_idx - alert_data.get("confirmed_frame", 0) < 180:
                self.active_accidents.append(alert_data)
            else:
                self.active_confirmed_map.pop(cand_key, None)

        # 3. Update active vehicle trajectories
        flow_dx_list = []
        flow_dy_list = []

        for oid in active_oids:
            cx, cy, det = active_objects[oid]
            bbox = det["bbox"]
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

            # Stable heading: only update when vehicle is actually in motion (> 2.0 px/f)
            # This prevents 1-pixel centroid jitter on stationary vehicles from generating false 180° spin
            if raw_speed > 2.0:
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
            })
            if len(hist) > 40:
                hist.pop(0)

            # Stage 2: Sudden hard braking detection (SUSPICIOUS ACTIVITY - YELLOW)
            if len(hist) >= 6:
                pre_raw = [h.get("raw_speed", h["speed"]) for h in hist[-6:-1]]
                avg_prior_raw = sum(pre_raw) / max(1, len(pre_raw))
                curr_raw = hist[-1].get("raw_speed", hist[-1]["speed"])
                # Deceleration > 65% with significant prior speed, not at exit boundary
                if avg_prior_raw >= 6.0 and curr_raw < avg_prior_raw * 0.35 and not is_camera_boundary(bbox, frame_w, frame_h):
                    self.suspicious_events.append({
                        "type": "Sudden Hard Braking",
                        "vehicle_id": oid,
                        "bbox": bbox,
                        "severity": "WARNING",
                        "details": f"Vehicle #{oid} hard braking: {avg_prior_raw:.1f} -> {curr_raw:.1f} px/f",
                    })

        # Update road flow vector
        if flow_dx_list and flow_dy_list:
            avg_fx = sum(flow_dx_list) / len(flow_dx_list)
            avg_fy = sum(flow_dy_list) / len(flow_dy_list)
            mag = math.hypot(avg_fx, avg_fy)
            if mag > 0.5:
                self.road_flow_vector = (avg_fx / mag, avg_fy / mag)

        # 4. Stage 3: Collision Candidate Identification (Road-aware)
        self._identify_candidates(list(active_oids), active_objects, frame_w, frame_h)

        # 5. Stage 4: Multi-Stage Consecutive Frame Verification
        self._verify_candidates(active_objects, frame_w, frame_h, active_oids, frame=frame)

        # 6. Cleanup expired trajectories
        for oid in list(self.trajectories.keys()):
            if oid not in active_oids and (self.frame_idx - self.trajectories[oid][-1]["frame"] > 30):
                del self.trajectories[oid]
                self.velocities.pop(oid, None)
                self.max_travel_speeds.pop(oid, None)

        # 7. Populate suspicious events with active verifying candidates
        for cand in self.verifying_candidates:
            self.suspicious_events.append({
                "type": f"Investigating {cand['type']}",
                "vehicle_ids": cand["vehicles"],
                "bbox": cand["bbox"],
                "severity": "WARNING",
                "details": f"Investigating collision interaction ({cand['progress_pct']}% verified)",
            })

        # 8. Determine Live System Condition
        if self.active_accidents:
            self.system_condition = "DANGER"
        elif self.verifying_candidates or self.suspicious_events:
            self.system_condition = "VERIFYING"
        else:
            self.system_condition = "NORMAL"

        if frame is not None and getattr(frame, "size", 0) > 0:
            try:
                self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass

        return self.active_accidents, self.suspicious_events

    def _identify_candidates(self, object_ids: list, tracker_objects: dict, frame_w: int, frame_h: int):
        """Identify potential collision candidates using ground footprint overlap and kinematic approach."""
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

                # Boundary Check: Discard vehicles exiting or entering camera edges
                if is_camera_boundary(box1, frame_w, frame_h) or is_camera_boundary(box2, frame_w, frame_h):
                    continue

                # 2D Bounding Box Overlap
                iou_2d = box_iou(box1, box2)
                if iou_2d < 0.15:
                    continue

                # GROUND FOOTPRINT ANALYSIS (Requirement 2 & 3)
                # Tires on road asphalt: bottom 30% of each bounding box
                g_iou = ground_footprint_iou(box1, box2, bottom_pct=0.30)

                cx1 = (box1[0] + box1[2]) / 2.0
                cx2 = (box2[0] + box2[2]) / 2.0
                w1 = max(1.0, float(box1[2] - box1[0]))
                w2 = max(1.0, float(box2[2] - box2[0]))
                lateral_ground_gap = abs(cx1 - cx2)

                # Separate Lane Check: In adjacent highway lanes, 2D boxes overlap but ground footprint IoU is 0
                # and lateral gap is >= 40% of vehicle width
                if g_iou < 0.04 and lateral_ground_gap >= 0.40 * min(w1, w2):
                    self.diagnostic_records.append({
                        "frame": self.frame_idx,
                        "pair": [id1, id2],
                        "status": "NORMAL",
                        "reason": f"Adjacent lanes / overtaking: ground IoU={g_iou:.2f}, lateral gap={lateral_ground_gap:.0f}px",
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

                if max_pre < 2.5:
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
                if decel_ratio < 0.35:
                    continue

                # Approach Heading Difference
                h1_pre = hist1[-(look+1)]["heading"]
                h2_pre = hist2[-(look+1)]["heading"]
                h_diff = abs(h1_pre - h2_pre)
                if h_diff > math.pi:
                    h_diff = 2 * math.pi - h_diff
                h_diff_deg = math.degrees(h_diff)

                # Normal Overtaking Check (Requirement 3 & 4):
                # When moving in same direction on highway (h_diff < 35°):
                # If BOTH vehicles maintain positive forward motion (speed >= 3.0 px/f), they are OVERTAKING!
                if h_diff_deg < 35.0:
                    if curr_raw1 >= 3.0 and curr_raw2 >= 3.0 and g_iou < 0.10:
                        self.diagnostic_records.append({
                            "frame": self.frame_idx,
                            "pair": [id1, id2],
                            "status": "NORMAL",
                            "reason": f"Normal overtaking: both vehicles maintain forward highway travel (v1={curr_raw1:.1f}, v2={curr_raw2:.1f})",
                            "box_iou": round(iou_2d, 2),
                            "ground_iou": round(g_iou, 2),
                        })
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
                    "status": "POSSIBLE_ACCIDENT",
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

            # Must have high pre-impact speed
            pre_raw = [min(h.get("raw_speed", 0.0), 50.0) for h in hist[-8:-2]]
            avg_pre_raw = sum(pre_raw) / max(1, len(pre_raw))
            if avg_pre_raw < 7.5:
                continue

            curr_raw = min(hist[-1].get("raw_speed", 0.0), 50.0)
            if curr_raw > 1.5:  # Must have suffered complete road stop
                continue

            # True posture deformation: Rollover (aspect ratio inversion >= 40%) or violent spin >= 60°
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

            if aspect_ratio_diff >= 0.40 or deg_diff >= 60.0:
                self.candidates[cand_key] = {
                    "key": cand_key,
                    "type": "Roadside Crash / Rollover",
                    "vehicles": [oid],
                    "start_frame": self.frame_idx,
                    "frames_observed": 0,
                    "v_pre": {oid: avg_pre_raw},
                    "headings_pre": {oid: head_ref},
                    "h_diff_deg_pre": 0.0,
                    "max_ground_iou": 1.0,
                    "max_box_iou": 1.0,
                    "bbox": box,
                    "arrest_frames": 0,
                    "deflection_deg": deg_diff,
                    "aspect_change_pct": aspect_ratio_diff * 100.0,
                    "status": "POSSIBLE_ACCIDENT",
                }

    def _verify_candidates(self, tracker_objects: dict, frame_w: int, frame_h: int, active_oids: set, frame=None):
        """Multi-stage consecutive-frame confirmation with continuation and boundary dismissal."""
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
                    if g_iou < 0.04 and (sp1_raw >= 3.0 or sp2_raw >= 3.0):
                        self._log(f"[F{self.frame_idx}] {cand_key} DISMISSED: Completed overtake / continued normal flow.")
                        del self.candidates[cand_key]
                        continue

                    # Kinetic arrest check: both vehicles halted at scene
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
                    # Vehicle disappeared during verification: if near border, dismiss
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

            # Stage 4: Comprehensive Evaluation at end of Verification Window
            if cand["frames_observed"] >= self.verify_window:
                v_pre_max = max(cand["v_pre"].values())
                arrest_ratio = min(1.0, cand["arrest_frames"] / float(max(1, self.verify_window - 2)))
                max_g_iou = cand.get("max_ground_iou", 0.0)
                defl_deg = cand.get("deflection_deg", 0.0)
                asp_chg = cand.get("aspect_change_pct", 0.0)
                h_diff_pre = cand.get("h_diff_deg_pre", 0.0)

                # Evidence-based scores (calibrated to real physics, realistic cap at 95%)
                g_score = min(25.0, (max_g_iou / 0.20) * 25.0)
                impulse_score = min(25.0, (v_pre_max / 5.0) * 25.0)
                arrest_score = min(30.0, arrest_ratio * 30.0)
                impact_evidence = min(15.0, max(defl_deg / 35.0, asp_chg / 25.0, (h_diff_pre / 90.0) * 0.75) * 15.0)

                composite_conf = int(g_score + impulse_score + arrest_score + impact_evidence)
                composite_conf = min(95, max(0, composite_conf))

                # Hard collision confirmation requires:
                # 1. High composite evidence score
                # 2. Multi-frame kinetic arrest at the scene (>= 50%)
                # 3. Verified ground contact overlap (>= 0.06)
                # 4. Significant pre-impact kinetic energy
                if cand_type == "Vehicle Collision":
                    is_confirmed = (
                        composite_conf >= self.conf_threshold and
                        arrest_ratio >= 0.50 and
                        max_g_iou >= 0.06 and
                        v_pre_max >= 2.5
                    )
                else:
                    is_confirmed = (
                        composite_conf >= self.conf_threshold and
                        arrest_ratio >= 0.60 and
                        (defl_deg >= 45.0 or asp_chg >= 35.0) and
                        v_pre_max >= 5.0
                    )

                cand["confidence_pct"] = composite_conf
                self._log(f"[F{self.frame_idx}] {cand_key} EVALUATED: conf={composite_conf}%, arrest={arrest_ratio:.0%}, "
                          f"g_iou={max_g_iou:.2f}, defl={defl_deg:.0f}deg, confirmed={is_confirmed}")

                if is_confirmed:
                    conf_str = f"{composite_conf}%"
                    confirmed_event = {
                        "type": cand_type,
                        "vehicle_ids": vids,
                        "bbox": cand["bbox"],
                        "confidence": conf_str,
                        "confidence_pct": composite_conf,
                        "details": (
                            f"HARD ROAD ACCIDENT DETECTED! "
                            f"Pre-impact speed: {v_pre_max:.1f} px/f, "
                            f"Arrest: {int(arrest_ratio*100)}%, "
                            f"Ground Overlap: {int(max_g_iou*100)}%"
                        ),
                        "evidence": {
                            "ground_footprint_overlap": f"{int(max_g_iou*100)}%",
                            "scene_kinetic_arrest": f"{int(arrest_ratio*100)}%",
                            "pre_impact_impulse": f"{int(impulse_score*4)}%",
                            "trajectory_deflection": f"{int(defl_deg)}°",
                        },
                        "incident_key": cand["key"],
                        "confirmed_frame": self.frame_idx,
                    }
                    self.active_accidents.append(confirmed_event)
                    self.cooldown_map[cand["key"]] = self.frame_idx
                    self.active_confirmed_map[cand["key"]] = confirmed_event
                    self._log(f"[F{self.frame_idx}] *** ACCIDENT CONFIRMED: {cand['key']} at {conf_str} ***")
                else:
                    # If not confirmed, retain as Suspicious if evidence is moderate, else dismiss
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

    return {
        "incident_id": incident_id,
        "feed_id": feed_id,
        "feed_title": feed_title,
        "camera_id": camera_id,
        "location": f"Camera {camera_id} ({feed_title})",
        "user_id": user_id,
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "accident_type": event.get("type", "Vehicle Collision"),
        "confidence_pct": event.get("confidence_pct", 90),
        "vehicle_ids": vids_str,
        "alert_message": (
            f"🚨 TRAFFIC ACCIDENT ALERT! Road accident detected on Camera {camera_id} "
            f"({feed_title}) at {now.strftime('%I:%M %p')}. Vehicles involved: {vids_str}. "
            f"Immediate emergency dispatch recommended!"
        ),
        "evidence_details": str(event.get("evidence", {})),
        "status": "unresolved",
        "bbox": list(event.get("bbox", [0, 0, 0, 0])),
    }
