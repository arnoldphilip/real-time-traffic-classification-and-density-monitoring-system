"""Vehicle Damage Analysis Engine for ATCC.

Analyzes structural integrity of vehicles involved in collisions:
- Compares pre-impact and post-impact vehicle crops
- Uses structural contour deviation, pixel difference analysis, and aspect distortion
- Categorizes damage location (Front, Rear, Side, Structural/Rollover)
- Provides calibrated confidence with honest reporting ("Unable to verify" if crops too small/occluded)
"""
import math
from typing import Dict, Optional, Tuple, Any
import cv2
import numpy as np


class VehicleDamageAnalyzer:
    """Estimates physical vehicle deformation from pre- and post-collision image crops."""

    DAMAGE_CATEGORIES = [
        "Front-End Impact (Bumper/Hood Deformation)",
        "Rear-End Impact (Trunk/Bumper Crumple)",
        "Lateral/Side-Impact Collision Deformation",
        "Severe Structural Deformation / Rollover",
        "Minor Surface Abrasion / Scratching",
        "Unable to Verify (Insufficient Resolution or Occlusion)",
    ]

    def __init__(self, min_crop_dim: int = 30):
        self.min_crop_dim = min_crop_dim

    def analyze_vehicle_damage(
        self,
        pre_crop: Optional[np.ndarray],
        post_crop: Optional[np.ndarray],
        vehicle_id: int,
        vehicle_class: str = "vehicle",
        impact_angle_deg: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Analyze structural difference between pre-impact and post-impact vehicle images.

        Returns structured result dict:
        {
            "vehicle_id": int,
            "vehicle_class": str,
            "damage_detected": bool,
            "damage_category": str,
            "severity_level": "NONE" | "MINOR" | "MODERATE" | "SEVERE",
            "confidence_pct": int,
            "deformation_index": float,
            "affected_quadrant": str,
            "reliable": bool,
            "assessment_note": str,
        }
        """
        # Baseline fallback when crops are unavailable
        if pre_crop is None or post_crop is None:
            return {
                "vehicle_id": vehicle_id,
                "vehicle_class": vehicle_class,
                "damage_detected": False,
                "damage_category": "UNKNOWN (Unable to Verify - Crops Unavailable)",
                "severity_level": "NONE",
                "confidence_pct": 30,
                "deformation_index": 0.0,
                "affected_quadrant": "Unknown",
                "reliable": False,
                "assessment_note": "Visual crops not available for frame-by-frame differential inspection.",
            }

        # Check resolution adequacy
        h1, w1 = pre_crop.shape[:2]
        h2, w2 = post_crop.shape[:2]

        if min(h1, w1, h2, w2) < self.min_crop_dim:
            return {
                "vehicle_id": vehicle_id,
                "vehicle_class": vehicle_class,
                "damage_detected": False,
                "damage_category": "UNKNOWN (Unable to Verify - Low Resolution / Occluded)",
                "severity_level": "NONE",
                "confidence_pct": 25,
                "deformation_index": 0.0,
                "affected_quadrant": "Unknown",
                "reliable": False,
                "assessment_note": f"Vehicle bounding crop too small ({w1}x{h1}px) for reliable damage verification.",
            }

        # Normalize resolution to common reference size (128 x 128)
        norm_size = (128, 128)
        try:
            pre_norm = cv2.resize(pre_crop, norm_size, interpolation=cv2.INTER_LINEAR)
            post_norm = cv2.resize(post_crop, norm_size, interpolation=cv2.INTER_LINEAR)

            pre_gray = cv2.cvtColor(pre_norm, cv2.COLOR_BGR2GRAY)
            post_gray = cv2.cvtColor(post_norm, cv2.COLOR_BGR2GRAY)

            # Contrast equalization to mitigate illumination changes
            pre_eq = cv2.equalizeHist(pre_gray)
            post_eq = cv2.equalizeHist(post_gray)

            # 1. Pixel-level absolute difference
            diff = cv2.absdiff(pre_eq, post_eq)
            _, thresh_diff = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
            diff_ratio = float(np.count_nonzero(thresh_diff)) / float(thresh_diff.size)

            # 2. Structural Edge Map Comparison
            pre_edges = cv2.Canny(pre_eq, 50, 150)
            post_edges = cv2.Canny(post_eq, 50, 150)
            edge_diff = cv2.absdiff(pre_edges, post_edges)
            edge_diff_ratio = float(np.count_nonzero(edge_diff)) / float(edge_diff.size)

            # 3. Quadrant analysis (Top, Bottom, Left, Right) to locate impact region
            h, w = norm_size
            mid_y, mid_x = h // 2, w // 2
            q_top = np.count_nonzero(thresh_diff[0:mid_y, :])
            q_bottom = np.count_nonzero(thresh_diff[mid_y:h, :])
            q_left = np.count_nonzero(thresh_diff[:, 0:mid_x])
            q_right = np.count_nonzero(thresh_diff[:, mid_x:w])

            quadrant_counts = {
                "Front / Top": q_top,
                "Rear / Bottom": q_bottom,
                "Left Lateral": q_left,
                "Right Lateral": q_right,
            }
            primary_quadrant = max(quadrant_counts, key=quadrant_counts.get)

            # 4. Aspect Ratio Distortion
            aspect1 = float(w1) / max(1.0, float(h1))
            aspect2 = float(w2) / max(1.0, float(h2))
            aspect_diff = abs(aspect2 - aspect1) / max(0.1, aspect1)

            # 5. Combined Deformation Index (0.0 to 1.0)
            deformation_index = float(
                0.45 * diff_ratio + 0.35 * edge_diff_ratio + 0.20 * min(1.0, aspect_diff)
            )

            # Calibrated damage thresholding
            if deformation_index < 0.14:
                return {
                    "vehicle_id": vehicle_id,
                    "vehicle_class": vehicle_class,
                    "damage_detected": False,
                    "damage_category": "Minor Surface Abrasion / Scratching",
                    "severity_level": "MINOR",
                    "confidence_pct": min(70, int(50 + (1.0 - deformation_index) * 20)),
                    "deformation_index": round(deformation_index, 3),
                    "affected_quadrant": primary_quadrant,
                    "reliable": True,
                    "assessment_note": "No significant structural perimeter crumple detected; minor contact only.",
                }

            # Map primary deformation to category
            if aspect_diff >= 0.35:
                category = "Severe Structural Deformation / Rollover"
                sev = "SEVERE"
                conf = min(75, int(60 + aspect_diff * 30))
            elif "Front" in primary_quadrant:
                category = "Front-End Impact (Bumper/Hood Deformation)"
                sev = "MODERATE" if deformation_index < 0.30 else "SEVERE"
                conf = min(75, int(55 + deformation_index * 45))
            elif "Rear" in primary_quadrant:
                category = "Rear-End Impact (Trunk/Bumper Crumple)"
                sev = "MODERATE" if deformation_index < 0.30 else "SEVERE"
                conf = min(75, int(55 + deformation_index * 45))
            else:
                category = "Lateral/Side-Impact Collision Deformation"
                sev = "MODERATE" if deformation_index < 0.28 else "SEVERE"
                conf = min(75, int(55 + deformation_index * 45))

            return {
                "vehicle_id": vehicle_id,
                "vehicle_class": vehicle_class,
                "damage_detected": True,
                "damage_category": category,
                "severity_level": sev,
                "confidence_pct": conf,
                "deformation_index": round(deformation_index, 3),
                "affected_quadrant": primary_quadrant,
                "reliable": True,
                "assessment_note": (
                    f"Structural distortion verified in {primary_quadrant} region "
                    f"(Deformation Index: {deformation_index:.2f}, Edge variance: {edge_diff_ratio:.0%})."
                ),
            }

        except Exception as e:
            return {
                "vehicle_id": vehicle_id,
                "vehicle_class": vehicle_class,
                "damage_detected": False,
                "damage_category": "UNKNOWN (Unable to Verify)",
                "severity_level": "NONE",
                "confidence_pct": 20,
                "deformation_index": 0.0,
                "affected_quadrant": "Unknown",
                "reliable": False,
                "assessment_note": f"Analysis encountered processing exception: {str(e)}",
            }

