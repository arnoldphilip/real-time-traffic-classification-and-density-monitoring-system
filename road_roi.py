"""Road ROI, Road Classification & Perspective Analysis Module for ATCC.

Ensures vehicle accident detection and monitoring occurs ONLY on valid road surfaces
and dynamically adapts detection rules to different road environments:
- Road types: HIGHWAY, NORMAL_STREET, INTERSECTION, RESIDENTIAL_ROAD, UNKNOWN
- Multi-point polygon road boundaries and trapezoidal perspective zones
- Exclusion zones for houses, trees, sidewalks, and buildings (strictly rejects false candidates)
- Specialized zones: Road shoulder, signal stop line zones, intersection turning zones
- Multi-frame intelligent RoadClassifier (infers road type from traffic kinematics)
- Perspective depth calibration
"""
import math
from typing import List, Tuple, Optional, Union, Dict, Any
import cv2
import numpy as np


class RoadType:
    HIGHWAY = "HIGHWAY"
    NORMAL_STREET = "NORMAL_STREET"
    INTERSECTION = "INTERSECTION"
    RESIDENTIAL_ROAD = "RESIDENTIAL_ROAD"
    UNKNOWN = "UNKNOWN"

    ALL_TYPES = [HIGHWAY, NORMAL_STREET, INTERSECTION, RESIDENTIAL_ROAD, UNKNOWN]


class RoadROIManager:
    """Manages road region-of-interest (ROI), exclusion zones, and camera perspective scaling."""

    def __init__(
        self,
        polygon_pct: Optional[List[Tuple[float, float]]] = None,
        rect_pct: Optional[Tuple[float, float, float, float]] = None,
        vanishing_y_pct: float = 0.15,
        road_type: str = RoadType.HIGHWAY,
        exclusion_polygons_pct: Optional[List[List[Tuple[float, float]]]] = None,
        shoulder_polygons_pct: Optional[List[List[Tuple[float, float]]]] = None,
        signal_zones_pct: Optional[List[List[Tuple[float, float]]]] = None,
        intersection_zones_pct: Optional[List[List[Tuple[float, float]]]] = None,
    ):
        """
        Initialize Road ROI Manager.

        Args:
            polygon_pct: Points defining drivable road polygon in normalized [0.0, 1.0] coordinates.
            rect_pct: Fallback (ymin_pct, xmin_pct, ymax_pct, xmax_pct) if polygon not provided.
            vanishing_y_pct: Estimated vanishing point horizon (0.0 to 1.0) for perspective depth scaling.
            road_type: HIGHWAY, NORMAL_STREET, INTERSECTION, RESIDENTIAL_ROAD, or UNKNOWN.
            exclusion_polygons_pct: List of polygons covering houses, trees, sidewalks (strictly excluded).
            shoulder_polygons_pct: Polygons covering highway shoulders / breakdown lanes.
            signal_zones_pct: Polygons covering traffic light stop lines.
            intersection_zones_pct: Polygons covering turning junction areas.
        """
        self.road_type = road_type if road_type in RoadType.ALL_TYPES else RoadType.HIGHWAY
        self.vanishing_y_pct = max(0.0, min(0.60, vanishing_y_pct))

        if polygon_pct and len(polygon_pct) >= 3:
            self.polygon_pct = [(float(x), float(y)) for x, y in polygon_pct]
        elif rect_pct:
            ymin, xmin, ymax, xmax = rect_pct
            self.polygon_pct = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
            ]
        else:
            # Default generous camera view (excludes extreme 2% frame margins)
            self.polygon_pct = [
                (0.00, 0.02),
                (1.00, 0.02),
                (1.00, 0.99),
                (0.00, 0.99),
            ]

        self.exclusion_polygons_pct = exclusion_polygons_pct or []
        self.shoulder_polygons_pct = shoulder_polygons_pct or []
        self.signal_zones_pct = signal_zones_pct or []
        self.intersection_zones_pct = intersection_zones_pct or []

    @classmethod
    def from_rectangle_pct(cls, ymin: float, xmin: float, ymax: float, xmax: float, road_type: str = RoadType.HIGHWAY) -> "RoadROIManager":
        """Construct from rectangle percentages."""
        return cls(rect_pct=(ymin, xmin, ymax, xmax), road_type=road_type)

    def get_pixel_polygon(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Convert normalized points to integer pixel polygon coordinates for cv2."""
        pts = [[int(x * frame_w), int(y * frame_h)] for x, y in self.polygon_pct]
        return np.array(pts, dtype=np.int32)

    def is_point_on_road(self, cx_px: float, cy_px: float, frame_w: int, frame_h: int) -> bool:
        """Check if an image point lies strictly inside the road boundary."""
        poly = self.get_pixel_polygon(frame_w, frame_h)
        dist = cv2.pointPolygonTest(poly, (float(cx_px), float(cy_px)), False)
        return dist >= 0

    def is_in_exclusion(self, cx_px: float, cy_px: float, frame_w: int, frame_h: int) -> bool:
        """Check if an image point lies inside an exclusion zone (houses, trees, buildings, sidewalk)."""
        if not self.exclusion_polygons_pct:
            return False
        for ex_pts in self.exclusion_polygons_pct:
            if len(ex_pts) < 3:
                continue
            poly = np.array([[int(x * frame_w), int(y * frame_h)] for x, y in ex_pts], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (float(cx_px), float(cy_px)), False) >= 0:
                return True
        return False

    def is_in_shoulder(self, cx_px: float, cy_px: float, frame_w: int, frame_h: int) -> bool:
        """Check if vehicle is on the highway emergency shoulder / roadside breakdown lane."""
        if not self.shoulder_polygons_pct:
            # Fallback: lateral 12% on either side of frame in highway mode
            if self.road_type == RoadType.HIGHWAY:
                return (cx_px <= 0.12 * frame_w) or (cx_px >= 0.88 * frame_w)
            return False
        for sh_pts in self.shoulder_polygons_pct:
            if len(sh_pts) < 3:
                continue
            poly = np.array([[int(x * frame_w), int(y * frame_h)] for x, y in sh_pts], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (float(cx_px), float(cy_px)), False) >= 0:
                return True
        return False

    def is_in_signal_zone(self, cx_px: float, cy_px: float, frame_w: int, frame_h: int) -> bool:
        """Check if vehicle is stopped at or approaching a traffic light stop line."""
        if not self.signal_zones_pct:
            # In NORMAL_STREET / INTERSECTION, center stop line region is typically middle third vertically
            if self.road_type in (RoadType.NORMAL_STREET, RoadType.INTERSECTION):
                return (0.35 * frame_h <= cy_px <= 0.70 * frame_h)
            return False
        for sz_pts in self.signal_zones_pct:
            if len(sz_pts) < 3:
                continue
            poly = np.array([[int(x * frame_w), int(y * frame_h)] for x, y in sz_pts], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (float(cx_px), float(cy_px)), False) >= 0:
                return True
        return False

    def is_in_intersection_zone(self, cx_px: float, cy_px: float, frame_w: int, frame_h: int) -> bool:
        """Check if vehicle is inside a turning junction / intersection cross zone."""
        if self.road_type == RoadType.INTERSECTION:
            return True
        if not self.intersection_zones_pct:
            return False
        for iz_pts in self.intersection_zones_pct:
            if len(iz_pts) < 3:
                continue
            poly = np.array([[int(x * frame_w), int(y * frame_h)] for x, y in iz_pts], dtype=np.int32)
            if cv2.pointPolygonTest(poly, (float(cx_px), float(cy_px)), False) >= 0:
                return True
        return False

    def bbox_ground_on_road(self, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
        """
        Check if vehicle's road contact patch (bottom center) is inside road polygon
        AND NOT inside any exclusion zone (houses, trees, roadside buildings).
        """
        x1, y1, x2, y2 = bbox
        ground_x = (x1 + x2) / 2.0
        ground_y = float(y2)

        # 1. Must be on road
        if not self.is_point_on_road(ground_x, ground_y, frame_w, frame_h):
            return False

        # 2. Must NOT be in an exclusion zone (house/tree/sidewalk)
        if self.is_in_exclusion(ground_x, ground_y, frame_w, frame_h):
            return False

        return True

    def bbox_intersects_road(
        self, bbox: Tuple[int, int, int, int], frame_w: int, frame_h: int, min_overlap_pct: float = 0.20
    ) -> bool:
        """Check whether a bounding box has substantial overlap with the road area."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        ground_y = float(y2)

        if self.is_point_on_road(cx, ground_y, frame_w, frame_h):
            return not self.is_in_exclusion(cx, ground_y, frame_w, frame_h)
        if self.is_point_on_road(cx, cy, frame_w, frame_h):
            return not self.is_in_exclusion(cx, cy, frame_w, frame_h)

        corners = [
            (x1, y1), (x2, y1),
            (x1, y2), (x2, y2),
            (cx, y1), (cx, y2),
        ]
        hits = sum(
            1 for px, py in corners
            if self.is_point_on_road(px, py, frame_w, frame_h) and not self.is_in_exclusion(px, py, frame_w, frame_h)
        )
        return (hits / len(corners)) >= min_overlap_pct

    def get_perspective_scale(self, cy_px: float, frame_h: int) -> float:
        """Calculate perspective scaling factor based on vertical position."""
        y_norm = max(0.0, min(1.0, cy_px / max(1.0, float(frame_h))))
        if y_norm <= self.vanishing_y_pct:
            return 0.35
        depth_progress = (y_norm - self.vanishing_y_pct) / max(0.01, 1.0 - self.vanishing_y_pct)
        scale = 0.35 + 0.65 * (depth_progress ** 1.2)
        return max(0.30, min(1.20, scale))

    def scale_threshold(self, base_val: float, cy_px: float, frame_h: int) -> float:
        """Scale a pixel distance or speed threshold by camera perspective."""
        return base_val * self.get_perspective_scale(cy_px, frame_h)

    def draw_roi(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 220, 255),
        alpha: float = 0.10,
        draw_border: bool = True,
    ) -> np.ndarray:
        """Draw transparent road boundary overlay on frame."""
        h, w = frame.shape[:2]
        poly = self.get_pixel_polygon(w, h)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poly], color)

        # Draw exclusion zones in red tint
        for ex_pts in self.exclusion_polygons_pct:
            if len(ex_pts) >= 3:
                ex_poly = np.array([[int(x * w), int(y * h)] for x, y in ex_pts], dtype=np.int32)
                cv2.fillPoly(overlay, [ex_poly], (0, 0, 160))

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        if draw_border:
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            top_pt = poly[0]
            label_x = max(10, min(w - 220, int(top_pt[0])))
            label_y = max(20, int(top_pt[1]) - 6)
            cv2.putText(
                frame, f"ROAD ZONE [{self.road_type}]", (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )
        return frame

    def to_dict(self) -> dict:
        """Serialize configuration to dict."""
        return {
            "road_type": self.road_type,
            "polygon_pct": self.polygon_pct,
            "vanishing_y_pct": self.vanishing_y_pct,
            "exclusion_polygons_pct": self.exclusion_polygons_pct,
            "shoulder_polygons_pct": self.shoulder_polygons_pct,
            "signal_zones_pct": self.signal_zones_pct,
            "intersection_zones_pct": self.intersection_zones_pct,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RoadROIManager":
        """Deserialize from dict."""
        if not data:
            return cls()
        return cls(
            polygon_pct=data.get("polygon_pct"),
            vanishing_y_pct=data.get("vanishing_y_pct", 0.15),
            road_type=data.get("road_type", RoadType.HIGHWAY),
            exclusion_polygons_pct=data.get("exclusion_polygons_pct", []),
            shoulder_polygons_pct=data.get("shoulder_polygons_pct", []),
            signal_zones_pct=data.get("signal_zones_pct", []),
            intersection_zones_pct=data.get("intersection_zones_pct", []),
        )


class RoadClassifier:
    """
    Infers road environment (HIGHWAY vs NORMAL_STREET vs INTERSECTION vs RESIDENTIAL)
    from multi-frame trajectory kinematics over time. Never guesses from a single frame.
    """

    def __init__(self, sample_window: int = 50):
        self.sample_window = sample_window
        self.speeds = []
        self.turn_angles_deg = []
        self.flow_vectors = []
        self.inferred_road_type = RoadType.UNKNOWN
        self.confidence_pct = 50
        self.frames_analyzed = 0

    def update_with_frame_trajectories(self, trajectories: dict) -> str:
        """Update road classification with active vehicle motion vectors."""
        self.frames_analyzed += 1

        for oid, hist in trajectories.items():
            if len(hist) >= 5:
                # Speed sample
                sp = hist[-1].get("raw_speed", hist[-1].get("speed", 0.0))
                if sp > 0.5:
                    self.speeds.append(sp)

                # Turning angle sample over 6-frame span
                if len(hist) >= 8:
                    th_init = hist[-7]["heading"]
                    th_curr = hist[-1]["heading"]
                    dth = abs(th_curr - th_init)
                    if dth > math.pi:
                        dth = 2 * math.pi - dth
                    deg = math.degrees(dth)
                    self.turn_angles_deg.append(deg)

        if len(self.speeds) > 200:
            self.speeds = self.speeds[-200:]
        if len(self.turn_angles_deg) > 100:
            self.turn_angles_deg = self.turn_angles_deg[-100:]

        # Multi-frame decision
        if len(self.speeds) >= 20 and self.frames_analyzed >= 25:
            avg_speed = sum(self.speeds) / len(self.speeds)
            turning_fraction = sum(1 for deg in self.turn_angles_deg if deg >= 45.0) / max(1, len(self.turn_angles_deg))

            if turning_fraction >= 0.25:
                # Vehicles routinely turn 45° - 90° -> INTERSECTION
                self.inferred_road_type = RoadType.INTERSECTION
                self.confidence_pct = min(92, int(60 + turning_fraction * 50))
            elif avg_speed >= 6.5 and turning_fraction < 0.10:
                # Fast linear flow with minimal turning -> HIGHWAY
                self.inferred_road_type = RoadType.HIGHWAY
                self.confidence_pct = min(95, int(65 + (avg_speed / 10.0) * 30))
            elif avg_speed < 3.0:
                # Low speed urban / residential
                self.inferred_road_type = RoadType.RESIDENTIAL_ROAD
                self.confidence_pct = 80
            else:
                # Moderate speed urban travel
                self.inferred_road_type = RoadType.NORMAL_STREET
                self.confidence_pct = 85

        return self.inferred_road_type
