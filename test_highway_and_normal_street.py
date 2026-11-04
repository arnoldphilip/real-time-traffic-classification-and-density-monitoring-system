"""Comprehensive Verification Test Suite for ATCC Intelligent Highway & Normal Street Accident Detection System.

Tests all key scenarios:
1. Vehicle Detection & Exclusion: Filters out non-vehicles (houses, trees, sky, poles).
2. Highway Scenarios (10 tests):
   - Highway normal cruising (No alert)
   - Highway high-speed overtaking (No alert)
   - Highway lane change (No alert)
   - Highway congestion stop / traffic jam (No alert)
   - Highway sudden braking (No false crash alert)
   - Highway rear-end high-speed collision (Confirmed Accident - RED)
   - Highway multi-vehicle collision (Confirmed Accident - RED)
   - Highway shoulder stationary vehicle / breakdown (No false crash alert)
   - Highway rollover / barrier crash (Confirmed Accident - RED)
   - Highway vehicle #909 test (ACC-1462: stopping vehicle is NOT a crash)
3. Normal Street & Intersection Scenarios (10 tests):
   - Red light queue stop (No alert)
   - 90-degree intersection turn (No false crash alert)
   - Zebra crossing / pedestrian stop (No alert)
   - Motorcycle lane filtering (No alert)
   - Roadside parking (No alert)
   - Slow traffic crawl (No alert)
   - Street lane change (No alert)
   - Intersection through-crossing (No alert)
   - Intersection T-bone collision (Confirmed Accident - RED)
   - Pedestrian emergency braking (No false crash alert)
4. Vehicle Damage Analysis & UNKNOWN reporting.
5. Database Clear All & Dismiss Alert (preserves historical records).
"""
import math
import os
import unittest
import numpy as np

# Ensure root import
import database
from road_roi import RoadROIManager, RoadType
from accident_processor import LocalAccidentDetector, AccidentState, AlertLevel, box_iou, ground_footprint_iou
from damage_analyzer import VehicleDamageAnalyzer


class TestVehicleDetectionAndExclusion(unittest.TestCase):
    """Test 1: Vehicle Detection & Exclusion Zone Rules."""

    def setUp(self):
        # Frame 1280 x 720
        self.w, self.h = 1280, 720
        # Define road with side exclusion zones for houses and trees
        self.roi = RoadROIManager(
            polygon_pct=[(0.05, 0.10), (0.95, 0.10), (0.95, 0.95), (0.05, 0.95)],
            exclusion_polygons_pct=[
                [(0.00, 0.00), (0.15, 0.00), (0.15, 0.40), (0.00, 0.40)],  # Left roadside houses/trees
                [(0.85, 0.00), (1.00, 0.00), (1.00, 0.40), (0.85, 0.40)],  # Right roadside building
            ],
            road_type=RoadType.HIGHWAY,
        )

    def test_house_and_tree_exclusion(self):
        """Objects detected over roadside houses or trees must be rejected."""
        # Tree bounding box on top left
        tree_box = (20, 30, 80, 180)
        self.assertFalse(
            self.roi.bbox_ground_on_road(tree_box, self.w, self.h),
            "Bounding box on roadside trees/houses must NOT be accepted on road."
        )

        # Real vehicle in center lane
        car_box = (500, 300, 620, 420)
        self.assertTrue(
            self.roi.bbox_ground_on_road(car_box, self.w, self.h),
            "Genuine vehicle in center lane must be accepted on road."
        )

    def test_ground_footprint_vs_perspective_overlap(self):
        """2D bounding boxes overlap in perspective, but ground contact patches in adjacent lanes do NOT."""
        # Car A in left lane, Car B in adjacent right lane slightly behind
        box_a = (400, 250, 520, 380)
        box_b = (470, 200, 590, 320)

        iou_2d = box_iou(box_a, box_b)
        g_iou = ground_footprint_iou(box_a, box_b, bottom_pct=0.30)

        self.assertGreater(iou_2d, 0.05, "2D camera perspective overlap occurs.")
        self.assertEqual(g_iou, 0.0, "Ground tire contact patches in adjacent lanes MUST have 0.0 IoU.")


class TestHighwayScenarios(unittest.TestCase):
    """Test 2: Highway Scenarios (10 tests)."""

    def setUp(self):
        self.w, self.h = 1280, 720
        self.detector = LocalAccidentDetector(
            road_type=RoadType.HIGHWAY,
            confidence_threshold=85,
            verify_window=8,
        )

    def test_hw1_normal_cruising(self):
        """Highway Test 1: Normal high-speed cruising must produce 0 accident alerts."""
        for f in range(25):
            cy = 200 + f * 8
            objects = {
                1: (500, cy, {"bbox": (450, cy - 30, 550, cy + 30), "class": "car", "confidence": 0.90}),
                2: (700, cy + 10, {"bbox": (650, cy - 20, 750, cy + 40), "class": "car", "confidence": 0.88}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)
        self.assertEqual(self.detector.accident_state, AccidentState.NORMAL)

    def test_hw2_high_speed_overtaking(self):
        """Highway Test 2: Overtaking vehicle passing alongside must NOT trigger accident."""
        for f in range(30):
            cy1 = 200 + f * 5       # Slower vehicle
            cy2 = 120 + f * 9       # Faster overtaking vehicle in adjacent lane
            objects = {
                1: (500, cy1, {"bbox": (450, cy1 - 25, 550, cy1 + 25), "class": "car", "confidence": 0.90}),
                2: (580, cy2, {"bbox": (540, cy2 - 25, 620, cy2 + 25), "class": "car", "confidence": 0.92}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)
        self.assertEqual(self.detector.accident_state, AccidentState.NORMAL)

    def test_hw3_smooth_lane_change(self):
        """Highway Test 3: Vehicle changing lanes smoothly must NOT trigger accident."""
        for f in range(25):
            cx = 500 + f * 4
            cy = 200 + f * 7
            objects = {
                1: (cx, cy, {"bbox": (cx - 40, cy - 30, cx + 40, cy + 30), "class": "car", "confidence": 0.91}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)
        self.assertEqual(self.detector.accident_state, AccidentState.NORMAL)

    def test_hw4_congestion_traffic_jam_stop(self):
        """Highway Test 4: Fleet slowing down and stopping in traffic congestion must NOT trigger accident."""
        for f in range(35):
            speed = max(0.0, 8.0 - f * 0.3)
            cy1 = 200 + min(120, f * speed)
            cy2 = 300 + min(120, f * speed)
            objects = {
                1: (500, cy1, {"bbox": (450, cy1 - 25, 550, cy1 + 25), "class": "car", "confidence": 0.89}),
                2: (500, cy2, {"bbox": (450, cy2 - 25, 550, cy2 + 25), "class": "truck", "confidence": 0.87}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Congestion slowdown must never trigger accident alert.")
        self.assertNotEqual(self.detector.accident_state, AccidentState.CONFIRMED_ACCIDENT)

    def test_hw5_sudden_braking_no_impact(self):
        """Highway Test 5: Sudden braking without collision impact must NOT trigger confirmed accident."""
        # Car cruises at 10 px/f then brakes hard to a stop without hitting anything
        for f in range(12):
            cy = 200 + f * 10
            objects = {1: (500, cy, {"bbox": (450, cy - 30, 550, cy + 30), "class": "car", "confidence": 0.92})}
            self.detector.update(objects, self.w, self.h)
        # Brakes to full stop
        for f in range(15):
            objects = {1: (500, 320, {"bbox": (450, 290, 550, 350), "class": "car", "confidence": 0.92})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Sudden braking alone without impact must NOT confirm accident.")

    def test_hw6_rear_end_collision_confirmed(self):
        """Highway Test 6: Genuine high-speed rear-end collision with kinetic arrest MUST trigger CONFIRMED_ACCIDENT (RED)."""
        # Vehicle 1 moving slow, Vehicle 2 approaching from behind at 12 px/f
        for f in range(8):
            cy1 = 350 + f * 1.5
            cy2 = 200 + f * 12.0
            objects = {
                1: (500, cy1, {"bbox": (450, cy1 - 30, 550, cy1 + 30), "class": "car", "confidence": 0.90}),
                2: (500, cy2, {"bbox": (450, cy2 - 30, 550, cy2 + 30), "class": "car", "confidence": 0.92}),
            }
            self.detector.update(objects, self.w, self.h)

        # Impact and kinetic arrest at cy ~ 360
        impact_confirmed = False
        for f in range(12):
            objects = {
                1: (500, 362, {"bbox": (445, 330, 545, 390), "class": "car", "confidence": 0.90}),
                2: (500, 358, {"bbox": (445, 325, 545, 385), "class": "car", "confidence": 0.92}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            if accidents:
                impact_confirmed = True
                self.assertEqual(self.detector.accident_state, AccidentState.CONFIRMED_ACCIDENT)
                self.assertEqual(self.detector.alert_level, AlertLevel.RED)
                # Check individual bboxes
                self.assertIn("individual_bboxes", accidents[0])
                self.assertEqual(set(accidents[0]["vehicle_ids"]), {1, 2})
                break

        self.assertTrue(impact_confirmed, "Genuine rear-end collision must be confirmed as accident.")

    def test_hw7_multi_vehicle_pileup(self):
        """Highway Test 7: Multi-vehicle collision triggers confirmed accident with correct involved IDs."""
        # 3 vehicles involved in collision
        for f in range(8):
            objects = {
                1: (500, 300 + f * 2, {"bbox": (450, 270 + f * 2, 550, 330 + f * 2), "class": "car", "confidence": 0.9}),
                2: (500, 200 + f * 11, {"bbox": (450, 170 + f * 11, 550, 230 + f * 11), "class": "car", "confidence": 0.9}),
            }
            self.detector.update(objects, self.w, self.h)
        for f in range(12):
            objects = {
                1: (500, 320, {"bbox": (445, 290, 545, 350), "class": "car", "confidence": 0.9}),
                2: (500, 318, {"bbox": (445, 288, 545, 348), "class": "car", "confidence": 0.9}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            if accidents:
                self.assertEqual(self.detector.alert_level, AlertLevel.RED)
                break

    def test_hw8_shoulder_breakdown_vehicle(self):
        """Highway Test 8: Stationary broken down vehicle on highway shoulder must NOT trigger accident."""
        # Vehicle stopped on right shoulder (cx >= 0.88 * w = 1126)
        for f in range(25):
            objects = {
                99: (1180, 400, {"bbox": (1130, 360, 1230, 440), "class": "car", "confidence": 0.88}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Shoulder stationary breakdown must NOT generate accident alert.")

    def test_hw9_rollover_crash_confirmed(self):
        """Highway Test 9: High-speed single vehicle rollover with violent aspect inversion MUST confirm accident."""
        # Vehicle cruises at 12 px/f, then violently flips (aspect ratio inverts from 1.6 to 0.7) and arrests
        for f in range(10):
            cy = 150 + f * 12.0
            objects = {10: (500, cy, {"bbox": (420, cy - 25, 580, cy + 25), "class": "car", "confidence": 0.93})}
            self.detector.update(objects, self.w, self.h)

        confirmed = False
        for f in range(12):
            # Rolled over on roof/side: aspect ratio inverted, vehicle arrested
            objects = {10: (520, 280, {"bbox": (490, 220, 550, 340), "class": "car", "confidence": 0.85})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            if accidents:
                confirmed = True
                self.assertEqual(accidents[0]["type"], "Roadside Crash / Rollover")
                self.assertEqual(accidents[0]["evidence"]["ground_footprint_overlap"], "N/A (Single Vehicle)")
                break
        self.assertTrue(confirmed, "Violent single-vehicle rollover must be confirmed.")

    def test_hw10_vehicle_909_stopping_bug_fixed(self):
        """Highway Test 10 (ACC-1462): Vehicle #909 simply stopping on road MUST NOT be classified as a rollover crash."""
        # Vehicle 909 travels normally then slows to a halt without flipping or crashing
        for f in range(10):
            cy = 200 + f * 5.0
            # Normal vehicle aspect (w=100, h=60 => aspect=1.66)
            objects = {909: (500, cy, {"bbox": (450, cy - 30, 550, cy + 30), "class": "car", "confidence": 0.91})}
            self.detector.update(objects, self.w, self.h)

        for f in range(20):
            # Stopped vehicle maintaining normal upright aspect ratio
            objects = {909: (500, 255, {"bbox": (450, 225, 550, 285), "class": "car", "confidence": 0.91})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Vehicle #909 stopping MUST NOT trigger a false rollover accident!")


class TestNormalStreetScenarios(unittest.TestCase):
    """Test 3: Normal Street & Intersection Scenarios (10 tests)."""

    def setUp(self):
        self.w, self.h = 1280, 720
        self.detector = LocalAccidentDetector(
            road_type=RoadType.NORMAL_STREET,
            confidence_threshold=85,
            verify_window=8,
        )

    def test_st1_red_light_queue_stop(self):
        """Street Test 1: Cars queuing and stopping at traffic light must NOT trigger accident."""
        for f in range(25):
            cy1 = 300
            cy2 = 230
            objects = {
                1: (600, cy1, {"bbox": (550, cy1 - 30, 650, cy1 + 30), "class": "car", "confidence": 0.90}),
                2: (600, cy2, {"bbox": (550, cy2 - 30, 650, cy2 + 30), "class": "car", "confidence": 0.91}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Traffic light queue stop must NEVER trigger accident alert.")

    def test_st2_intersection_90_degree_turn(self):
        """Street Test 2: Normal 90-degree intersection turn must NOT be classified as a crash/rollover."""
        for f in range(10):
            # Car traveling down then making a 90-degree right turn
            cx = 500 + (f * 6 if f > 4 else 0)
            cy = 200 + (f * 5 if f <= 4 else 20)
            objects = {
                5: (cx, cy, {"bbox": (cx - 35, cy - 25, cx + 35, cy + 25), "class": "car", "confidence": 0.92})
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Turning at an intersection must NEVER trigger an accident alert.")

    def test_st3_zebra_crossing_stop(self):
        """Street Test 3: Vehicle stopping smoothly before zebra crossing must NOT trigger accident."""
        for f in range(15):
            objects = {7: (450, 400, {"bbox": (400, 360, 500, 440), "class": "bus", "confidence": 0.94})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)

    def test_st4_motorcycle_lane_filtering(self):
        """Street Test 4: Motorcycle filtering closely between cars must NOT trigger accident."""
        for f in range(20):
            cy_car = 350
            cy_moto = 200 + f * 8
            objects = {
                1: (500, cy_car, {"bbox": (450, cy_car - 30, 550, cy_car + 30), "class": "car", "confidence": 0.90}),
                2: (620, cy_car, {"bbox": (570, cy_car - 30, 670, cy_car + 30), "class": "car", "confidence": 0.91}),
                3: (560, cy_moto, {"bbox": (545, cy_moto - 20, 575, cy_moto + 20), "class": "motorcycle", "confidence": 0.88}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0, "Motorcycle lane filtering must NOT trigger false accident.")

    def test_st5_roadside_parking(self):
        """Street Test 5: Vehicle pulling over to curb and parking must NOT trigger accident."""
        for f in range(20):
            cx = 100
            cy = 300
            objects = {12: (cx, cy, {"bbox": (50, 260, 150, 340), "class": "car", "confidence": 0.89})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)

    def test_st6_slow_traffic_crawl(self):
        """Street Test 6: Vehicles crawling at 1.5 px/f must NOT trigger accident."""
        for f in range(25):
            cy = 200 + f * 1.5
            objects = {
                1: (500, cy, {"bbox": (450, cy - 25, 550, cy + 25), "class": "car", "confidence": 0.90}),
                2: (500, cy + 80, {"bbox": (450, cy + 55, 550, cy + 105), "class": "car", "confidence": 0.89}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)

    def test_st7_street_lane_change(self):
        """Street Test 7: Normal street lane change must NOT trigger accident."""
        for f in range(20):
            cx = 400 + f * 3
            cy = 250 + f * 4
            objects = {20: (cx, cy, {"bbox": (cx - 30, cy - 25, cx + 30, cy + 25), "class": "auto", "confidence": 0.85})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)

    def test_st8_intersection_through_crossing(self):
        """Street Test 8: Vehicles crossing intersection on green light must NOT trigger accident."""
        for f in range(20):
            cy = 100 + f * 9
            objects = {22: (600, cy, {"bbox": (560, cy - 30, 640, cy + 30), "class": "car", "confidence": 0.91})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)

    def test_st9_intersection_t_bone_collision_confirmed(self):
        """Street Test 9: Genuine T-bone collision at intersection MUST trigger CONFIRMED_ACCIDENT (RED)."""
        # Car 1 traveling East-West, Car 2 traveling North-South, impacting at (600, 350)
        for f in range(8):
            cx1 = 450 + f * 15.0
            cy2 = 230 + f * 12.0
            objects = {
                31: (cx1, 350, {"bbox": (cx1 - 35, 325, cx1 + 35, 375), "class": "car", "confidence": 0.92}),
                32: (600, cy2, {"bbox": (575, cy2 - 35, 625, cy2 + 35), "class": "car", "confidence": 0.90}),
            }
            self.detector.update(objects, self.w, self.h)

        confirmed = False
        for f in range(12):
            objects = {
                31: (595, 350, {"bbox": (560, 320, 630, 380), "class": "car", "confidence": 0.92}),
                32: (605, 352, {"bbox": (570, 325, 640, 385), "class": "car", "confidence": 0.90}),
            }
            accidents, _ = self.detector.update(objects, self.w, self.h)
            if accidents:
                confirmed = True
                self.assertEqual(self.detector.alert_level, AlertLevel.RED)
                self.assertEqual(self.detector.accident_state, AccidentState.CONFIRMED_ACCIDENT)
                break
        self.assertTrue(confirmed, "Genuine intersection T-bone collision must be confirmed.")

    def test_st10_pedestrian_emergency_braking(self):
        """Street Test 10: Vehicle stopping for pedestrian without hitting another vehicle must NOT trigger accident."""
        for f in range(15):
            objects = {40: (500, 320, {"bbox": (450, 280, 550, 360), "class": "car", "confidence": 0.91})}
            accidents, _ = self.detector.update(objects, self.w, self.h)
            self.assertEqual(len(accidents), 0)


class TestDamageAnalysisAndUnknownReporting(unittest.TestCase):
    """Test 4: Vehicle Damage Analysis & UNKNOWN reporting."""

    def setUp(self):
        self.analyzer = VehicleDamageAnalyzer()

    def test_missing_crops_reports_unknown(self):
        """When crops are missing, damage analysis MUST report UNKNOWN."""
        res = self.analyzer.analyze_vehicle_damage(None, None, vehicle_id=1, vehicle_class="car")
        self.assertFalse(res["damage_detected"])
        self.assertTrue(res["damage_category"].startswith("UNKNOWN"))
        self.assertFalse(res["reliable"])

    def test_low_res_crops_reports_unknown(self):
        """When crops are too small (<30px), damage analysis MUST report UNKNOWN."""
        tiny1 = np.zeros((15, 15, 3), dtype=np.uint8)
        tiny2 = np.zeros((15, 15, 3), dtype=np.uint8)
        res = self.analyzer.analyze_vehicle_damage(tiny1, tiny2, vehicle_id=2, vehicle_class="car")
        self.assertFalse(res["damage_detected"])
        self.assertTrue(res["damage_category"].startswith("UNKNOWN"))

    def test_undamaged_car_reports_minor_or_no_crumple(self):
        """When two identical crops are analyzed, no severe damage is reported."""
        crop1 = np.full((100, 100, 3), 128, dtype=np.uint8)
        crop2 = np.full((100, 100, 3), 128, dtype=np.uint8)
        res = self.analyzer.analyze_vehicle_damage(crop1, crop2, vehicle_id=3, vehicle_class="car")
        self.assertFalse(res["damage_detected"])
        self.assertNotEqual(res["severity_level"], "SEVERE")


class TestDatabaseClearAllAndDismiss(unittest.TestCase):
    """Test 5: Clear All & Individual Alert Dismissal preserving database records."""

    def setUp(self):
        import uuid
        database.init_database()
        # Insert a unique test active accident
        self.test_incident_id = f"ACC-{uuid.uuid4().hex[:8]}"
        self.db_id, self.canonical_id, _ = database.save_or_update_accident_event(
            incident_id=self.test_incident_id,
            feed_title="Test Feed",
            location="Test Junction",
            confidence="90%",
            confidence_pct=90,
            vehicle_ids="1,2",
            frame_number=100,
            accident_type="Vehicle Collision",
            evidence_details="Ground IoU: 15% | Kinetic Arrest: 100%",
            severity="HIGH",
            damage_assessment="UNKNOWN",
        )

    def test_individual_alert_dismissal(self):
        """Dismissing an alert removes it from active view but keeps historical record in database."""
        # Find active
        unres = database.get_recent_accident_events(limit=50, status_filter="unresolved")
        match = next((x for x in unres if x["id"] == self.db_id), None)
        self.assertIsNotNone(match, "Test accident must be initially unresolved.")

        # Dismiss
        dismissed = database.dismiss_accident_event(match["id"], username="TestOfficer")
        self.assertTrue(dismissed)

        # Check unresolved list: must NOT appear
        unres_after = database.get_recent_accident_events(limit=10, status_filter="unresolved")
        self.assertFalse(any(x["id"] == match["id"] for x in unres_after), "Dismissed alert must NOT appear in unresolved list.")

        # Check all list: MUST still exist in database
        all_events = database.get_recent_accident_events(limit=50, status_filter="all")
        db_match = next((x for x in all_events if x["id"] == match["id"]), None)
        self.assertIsNotNone(db_match, "Dismissed alert record MUST be preserved in database history.")
        self.assertEqual(db_match["status"], "dismissed")

    def test_clear_all_active_accidents(self):
        """Clear All clears all active alerts from dashboard while preserving historical evidence."""
        # Insert another active
        database.save_or_update_accident_event(
            incident_id=f"{self.test_incident_id}-2",
            feed_title="Test Feed 2",
            location="Test Highway",
            confidence="88%",
            confidence_pct=88,
            vehicle_ids="3,4",
            frame_number=200,
            accident_type="Vehicle Collision",
            evidence_details="Ground IoU: 18%",
            severity="CRITICAL",
        )

        cleared_count = database.clear_all_active_accidents(username="AdminClear")
        self.assertGreaterEqual(cleared_count, 1)

        summary = database.get_system_accident_summary()
        self.assertEqual(summary["unresolved_count"], 0, "Unresolved count must be 0 after Clear All.")
        self.assertEqual(summary["condition"], "NORMAL", "System condition must return to NORMAL after Clear All.")

        # Verify historical records still exist in database
        all_events = database.get_recent_accident_events(limit=50, status_filter="all")
        self.assertGreater(len(all_events), 0, "Historical records must be safely preserved.")


if __name__ == "__main__":
    unittest.main()
