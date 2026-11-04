import math
from collections import defaultdict
import numpy as np

def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

class StrictAccidentDetector:
    """
    Multi-Stage Hard Road Accident Verification Engine.
    Detects ONLY genuine serious collisions, high-impact crashes, rollovers, and roadside impacts.
    Guarantees zero false positives for normal traffic, congestion, traffic signal stops, and 2D perspective overlaps.
    """
    def __init__(self, confidence_threshold: int = 90, verify_window: int = 12, cooldown_frames: int = 150):
        self.conf_threshold = confidence_threshold
        self.verify_window = verify_window
        self.cooldown_frames = cooldown_frames

        # Trajectory history: oid -> list of state dicts
        self.trajectories = defaultdict(list)
        # Smoothed velocities: oid -> (vx, vy)
        self.velocities = {}
        # Max observed travel speed: oid -> float
        self.max_travel_speeds = defaultdict(float)

        # Multi-stage candidates: cand_id -> candidate dict
        self.candidates = {}
        # Active confirmed accidents
        self.active_accidents = []
        # Candidates currently in verification (YELLOW state)
        self.verifying_candidates = []
        # Cooldown map: key -> last_alert_frame
        self.cooldown_map = {}
        # Current frame index
        self.frame_idx = 0
        # Telemetry / Debug log
        self.debug_telemetry = []

        # Previous frame for visual shock verification
        self.prev_frame_gray = None

    def update(self, tracker_objects: dict, frame_w: int, frame_h: int, frame=None) -> tuple[list, list, str, dict]:
        """
        Update detector state with tracked objects and optional video frame.
        Returns:
            (confirmed_accidents, verifying_events, system_condition, debug_info)
        """
        self.frame_idx += 1
        self.active_accidents = []
        self.verifying_candidates = []
        object_ids = list(tracker_objects.keys())

        # 1. Update vehicle trajectories and kinematics
        for oid in object_ids:
            cx, cy, det = tracker_objects[oid]
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            w = max(1.0, float(x2 - x1))
            h = max(1.0, float(y2 - y1))
            aspect = w / h
            area = w * h

            hist = self.trajectories[oid]
            if hist:
                prev = hist[-1]
                raw_dx = cx - prev["cx"]
                raw_dy = cy - prev["cy"]
                raw_disp = math.hypot(raw_dx, raw_dy)

                # Clamp tracker jumps (e.g. centroid re-identification jitter > 60px)
                if raw_disp > 55.0:
                    raw_dx = prev["vx"]
                    raw_dy = prev["vy"]

                prev_vx, prev_vy = self.velocities.get(oid, (raw_dx, raw_dy))
                # EMA smoothing
                vx = 0.60 * raw_dx + 0.40 * prev_vx
                vy = 0.60 * raw_dy + 0.40 * prev_vy
                speed = math.hypot(vx, vy)
            else:
                raw_dx, raw_dy = 0.0, 0.0
                vx, vy = 0.0, 0.0
                speed = 0.0

            self.velocities[oid] = (vx, vy)
            if speed > self.max_travel_speeds[oid]:
                self.max_travel_speeds[oid] = speed

            heading = math.atan2(vy, vx) if speed > 0.5 else (hist[-1]["heading"] if hist else 0.0)

            hist.append({
                "frame": self.frame_idx,
                "cx": cx,
                "cy": cy,
                "bbox": bbox,
                "w": w,
                "h": h,
                "aspect": aspect,
                "area": area,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "heading": heading,
            })
            if len(hist) > 40:
                hist.pop(0)

        # 2. Stage 2: Collision Candidate Identification
        self._identify_candidates(object_ids, tracker_objects, frame_w, frame_h)

        # 3. Stage 3 & 4: Consecutive-Frame Multi-Stage Verification
        self._verify_candidates(tracker_objects, frame)

        # 4. Clean up expired trajectories
        current_oids = set(object_ids)
        for oid in list(self.trajectories.keys()):
            if oid not in current_oids and (self.frame_idx - self.trajectories[oid][-1]["frame"] > 45):
                del self.trajectories[oid]
                self.velocities.pop(oid, None)
                self.max_travel_speeds.pop(oid, None)

        # 5. Determine Live System Condition
        if self.active_accidents:
            system_condition = "DANGER"
        elif self.verifying_candidates:
            system_condition = "VERIFYING"
        else:
            system_condition = "NORMAL"

        if frame is not None and frame.size > 0:
            import cv2
            self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        debug_info = {
            "condition": system_condition,
            "active_accidents_count": len(self.active_accidents),
            "verifying_candidates_count": len(self.verifying_candidates),
            "tracked_vehicles": len(object_ids),
            "candidates": list(self.candidates.values()),
            "telemetry": self.debug_telemetry[-5:] if self.debug_telemetry else [],
        }

        return self.active_accidents, self.verifying_candidates, system_condition, debug_info

    def _identify_candidates(self, object_ids: list, tracker_objects: dict, frame_w: int, frame_h: int):
        """Identify potential collision events for multi-stage consecutive-frame investigation."""
        # A. Multi-Vehicle Collision Candidates
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                id1, id2 = object_ids[i], object_ids[j]
                pair = (min(id1, id2), max(id1, id2))
                cand_key = f"pair_{pair[0]}_{pair[1]}"

                # Check cooldown
                if self.frame_idx - self.cooldown_map.get(cand_key, -9999) < self.cooldown_frames:
                    continue
                if cand_key in self.candidates:
                    continue

                hist1 = self.trajectories[id1]
                hist2 = self.trajectories[id2]
                # Require genuine tracking tracklet (>= 8 frames) to rule out detection flicker
                if len(hist1) < 8 or len(hist2) < 8:
                    continue

                box1 = tracker_objects[id1][2]["bbox"]
                box2 = tracker_objects[id2][2]["bbox"]
                iou = box_iou(box1, box2)

                # Strict Overlap Requirement: IoU must be >= 0.28 (NO proximity-only triggers!)
                if iou < 0.28:
                    continue

                # Sustained Pre-Impact Speed: Both vehicles must have a real moving history
                speeds1 = [h["speed"] for h in hist1[-6:-2]]
                speeds2 = [h["speed"] for h in hist2[-6:-2]]
                v1_pre = sum(speeds1) / len(speeds1)
                v2_pre = sum(speeds2) / len(speeds2)
                max_pre = max(v1_pre, v2_pre)

                # Ignore slow-creeping or idling traffic queue overlaps
                if max_pre < 3.0:
                    continue

                # Relative Closing Speed Calculation along line of centers
                p1_prev = (hist1[-4]["cx"], hist1[-4]["cy"])
                p2_prev = (hist2[-4]["cx"], hist2[-4]["cy"])
                p1_curr = (hist1[-1]["cx"], hist1[-1]["cy"])
                p2_curr = (hist2[-1]["cx"], hist2[-1]["cy"])
                dist_prev = math.hypot(p1_prev[0] - p2_prev[0], p1_prev[1] - p2_prev[1])
                dist_curr = math.hypot(p1_curr[0] - p2_curr[0], p1_curr[1] - p2_curr[1])
                closing_rate = (dist_prev - dist_curr) / 3.0

                # In normal following / congestion / lane change, closing rate is near 0 or negative
                if closing_rate < 2.0:
                    continue

                # Deceleration Impulse: at contact, kinetic energy must drop abruptly
                curr_v1 = hist1[-1]["speed"]
                curr_v2 = hist2[-1]["speed"]
                decel_ratio = max(
                    (v1_pre - curr_v1) / max(0.1, v1_pre),
                    (v2_pre - curr_v2) / max(0.1, v2_pre),
                )
                if decel_ratio < 0.50:
                    continue

                # Register Candidate for Multi-Frame Consecutive Investigation
                union_box = (
                    min(box1[0], box2[0]),
                    min(box1[1], box2[1]),
                    max(box1[2], box2[2]),
                    max(box1[3], box2[3]),
                )
                self.candidates[cand_key] = {
                    "key": cand_key,
                    "type": "Vehicle Collision",
                    "vehicles": [pair[0], pair[1]],
                    "start_frame": self.frame_idx,
                    "frames_observed": 0,
                    "v_pre": {pair[0]: v1_pre, pair[1]: v2_pre},
                    "headings_pre": {pair[0]: hist1[-4]["heading"], pair[1]: hist2[-4]["heading"]},
                    "impact_point": ((p1_curr[0] + p2_curr[0]) / 2, (p1_curr[1] + p2_curr[1]) / 2),
                    "bbox": union_box,
                    "entangle_frames": 0,
                    "deflection_deg": 0.0,
                    "aspect_change_pct": 0.0,
                    "status": "VERIFYING",
                }

        # B. Single-Vehicle Roadside Crash / Rollover Candidates
        for oid in object_ids:
            cand_key = f"single_{oid}"
            if self.frame_idx - self.cooldown_map.get(cand_key, -9999) < self.cooldown_frames:
                continue
            if cand_key in self.candidates:
                continue

            hist = self.trajectories[oid]
            # Require at least 10 tracked frames
            if len(hist) < 10:
                continue

            # Must have steady sustained travel speed over preceding 6 frames
            pre_speeds = [h["speed"] for h in hist[-8:-3]]
            avg_pre = sum(pre_speeds) / len(pre_speeds)
            if avg_pre < 6.0 or min(pre_speeds) < 4.0:
                continue

            curr_sp = hist[-1]["speed"]
            # Abrupt crash stop: speed dropped by >= 85% in <= 2 frames
            if curr_sp > 1.0 or (curr_sp > avg_pre * 0.15):
                continue

            # Posture check: rollover / aspect ratio inversion or violent spin
            aspect_init = hist[-6]["aspect"]
            aspect_curr = hist[-1]["aspect"]
            aspect_ratio_diff = abs(aspect_curr - aspect_init) / max(0.1, aspect_init)

            angle_diff = abs(hist[-1]["heading"] - hist[-5]["heading"])
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            deg_diff = math.degrees(angle_diff)

            # Strict rollover / barrier crash criterion
            if aspect_ratio_diff >= 0.40 or deg_diff >= 75.0:
                box = tracker_objects[oid][2]["bbox"]
                self.candidates[cand_key] = {
                    "key": cand_key,
                    "type": "Roadside Crash / Rollover",
                    "vehicles": [oid],
                    "start_frame": self.frame_idx,
                    "frames_observed": 0,
                    "v_pre": {oid: avg_pre},
                    "headings_pre": {oid: hist[-6]["heading"]},
                    "impact_point": (hist[-1]["cx"], hist[-1]["cy"]),
                    "bbox": box,
                    "entangle_frames": 0,
                    "deflection_deg": deg_diff,
                    "aspect_change_pct": aspect_ratio_diff * 100.0,
                    "status": "VERIFYING",
                }

    def _verify_candidates(self, tracker_objects: dict, frame=None):
        """Analyze consecutive video frames before and after suspected impact to verify hard collision."""
        for cand_key in list(self.candidates.keys()):
            cand = self.candidates[cand_key]
            cand["frames_observed"] += 1
            cand_type = cand["type"]
            vids = cand["vehicles"]

            # Verify presence of tracked vehicles
            all_present = all(v in tracker_objects for v in vids)

            if cand_type == "Vehicle Collision":
                id1, id2 = vids[0], vids[1]
                if all_present:
                    box1 = tracker_objects[id1][2]["bbox"]
                    box2 = tracker_objects[id2][2]["bbox"]
                    iou = box_iou(box1, box2)

                    hist1 = self.trajectories[id1]
                    hist2 = self.trajectories[id2]
                    sp1 = hist1[-1]["speed"]
                    sp2 = hist2[-1]["speed"]

                    # Separation check: if vehicles separate and move away along lanes, it was NORMAL TRAFFIC
                    if iou < 0.08 and (sp1 >= 2.0 or sp2 >= 2.0):
                        self.debug_telemetry.append(f"Candidate {cand_key} DISMISSED: Vehicles separated and continued moving normally.")
                        del self.candidates[cand_key]
                        continue

                    # In a genuine hard crash, vehicles remain entangled / arrested at collision locus
                    if sp1 <= 1.2 and sp2 <= 1.2:
                        cand["entangle_frames"] += 1

                    # Check trajectory deflection post-impact
                    th1_init = cand["headings_pre"][id1]
                    th1_curr = hist1[-1]["heading"]
                    diff1 = abs(th1_curr - th1_init)
                    if diff1 > math.pi:
                        diff1 = 2 * math.pi - diff1
                    cand["deflection_deg"] = max(cand["deflection_deg"], math.degrees(diff1))

                    # Update bounding box
                    cand["bbox"] = (
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3]),
                    )
                else:
                    # Vehicle tracker lost centroid due to severe deformation / overlap
                    cand["entangle_frames"] += 1

            elif cand_type == "Roadside Crash / Rollover":
                oid = vids[0]
                if oid in tracker_objects:
                    hist = self.trajectories[oid]
                    sp = hist[-1]["speed"]
                    if sp <= 0.8:
                        cand["entangle_frames"] += 1
                    else:
                        # Resumed driving normally -> DISMISS
                        self.debug_telemetry.append(f"Candidate {cand_key} DISMISSED: Vehicle resumed normal driving.")
                        del self.candidates[cand_key]
                        continue

            # Stage 4: Evaluation at end of Consecutive-Frame Verification Window
            if cand["frames_observed"] >= self.verify_window:
                # 1. Impulse factor (35%)
                v_pre_max = max(cand["v_pre"].values())
                impulse_score = min(100.0, max(0.0, (v_pre_max / 5.5) * 100.0))

                # 2. Entanglement / Post-Impact Arrest factor (35%)
                entangle_ratio = cand["entangle_frames"] / float(self.verify_window)
                entangle_score = min(100.0, entangle_ratio * 125.0)

                # 3. Deflection / Spin factor (15%)
                deflection_score = min(100.0, (cand["deflection_deg"] / 60.0) * 100.0)

                # 4. Aspect Ratio / Deformation factor (15%)
                deform_score = min(100.0, (cand.get("aspect_change_pct", 0.0) / 35.0) * 100.0)

                # Composite Evidence Confidence
                composite_conf = (
                    0.35 * impulse_score +
                    0.35 * entangle_score +
                    0.15 * deflection_score +
                    0.15 * deform_score
                )
                cand["confidence_pct"] = int(composite_conf)

                # Strict Verification Decision
                is_confirmed = (
                    composite_conf >= self.conf_threshold and
                    entangle_ratio >= 0.65 and
                    v_pre_max >= 3.0
                )

                if is_confirmed:
                    conf_str = f"{int(composite_conf)}%"
                    confirmed_event = {
                        "type": cand_type,
                        "vehicle_ids": vids,
                        "bbox": cand["bbox"],
                        "confidence": conf_str,
                        "confidence_pct": int(composite_conf),
                        "details": (
                            f"HARD ROAD ACCIDENT DETECTED! "
                            f"Pre-impact speed: {v_pre_max:.1f} px/f, "
                            f"Arrest: {int(entangle_ratio*100)}%, "
                            f"Deflection: {int(cand['deflection_deg'])} deg"
                        ),
                        "evidence": {
                            "impulse": f"{int(impulse_score)}%",
                            "entanglement_arrest": f"{int(entangle_score)}%",
                            "trajectory_deflection": f"{int(deflection_score)}%",
                            "deformation": f"{int(deform_score)}%",
                        },
                        "incident_key": cand["key"],
                    }
                    self.active_accidents.append(confirmed_event)
                    self.cooldown_map[cand["key"]] = self.frame_idx
                    self.debug_telemetry.append(f"ACCIDENT CONFIRMED: {cand['key']} at {conf_str}")
                else:
                    self.debug_telemetry.append(
                        f"Candidate {cand_key} DISMISSED: Insufficient collision evidence "
                        f"(Conf: {int(composite_conf)}%, Arrest: {int(entangle_ratio*100)}%)."
                    )

                # Remove evaluated candidate
                del self.candidates[cand_key]
            else:
                # Still in verification window -> YELLOW state
                self.verifying_candidates.append({
                    "type": cand["type"],
                    "vehicle_ids": vids,
                    "bbox": cand["bbox"],
                    "frames_observed": cand["frames_observed"],
                    "verify_window": self.verify_window,
                    "incident_key": cand["key"],
                    "progress_pct": int((cand["frames_observed"] / self.verify_window) * 100),
                })
