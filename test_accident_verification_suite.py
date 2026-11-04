"""
Comprehensive 12-Scenario Test Suite for ATCC Accident Detection, Verification,
Deduplication, and Incident Management Lifecycle.
"""
import os
import sys
import math
import sqlite3
# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import database
import unified_app

def setup_clean_db():
    database.init_database()
    conn = database.get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM AccidentEvents")
    conn.commit()
    conn.close()

def run_all_tests():
    print("=" * 75)
    print("RUNNING ATCC ACCIDENT DETECTION & INCIDENT MANAGEMENT 12-TEST SUITE")
    print("=" * 75)
    setup_clean_db()

    # -------------------------------------------------------------
    # TEST 1: Normal Traffic -> No accident alert (Condition: NORMAL)
    # -------------------------------------------------------------
    print("\n[TEST 1] Normal Traffic Flow...")
    detector = unified_app.AccidentDetector()
    for f in range(25):
        # 3 vehicles moving steadily down the road at 4 px/f in separate lanes
        tracker_objects = {
            1: (100.0, 50.0 + f * 4.0, {"bbox": (80, int(35 + f * 4.0), 120, int(65 + f * 4.0)), "class_name": "car"}),
            2: (220.0, 40.0 + f * 4.0, {"bbox": (200, int(25 + f * 4.0), 240, int(55 + f * 4.0)), "class_name": "car"}),
            3: (340.0, 60.0 + f * 4.0, {"bbox": (320, int(45 + f * 4.0), 360, int(75 + f * 4.0)), "class_name": "bus"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
    assert len(accidents) == 0, f"Expected 0 accidents, got {len(accidents)}"
    assert detector.system_condition == "NORMAL", f"Expected NORMAL condition, got {detector.system_condition}"
    print("  ✓ PASS: Normal traffic produced 0 alerts and maintained condition NORMAL.")

    # -------------------------------------------------------------
    # TEST 2: Vehicle Passing Normally -> No accident alert
    # -------------------------------------------------------------
    print("\n[TEST 2] Vehicle Passing / Overtaking Normally...")
    detector = unified_app.AccidentDetector()
    for f in range(25):
        # Vehicle 1 driving at 3 px/f, Vehicle 2 overtaking at 6 px/f in adjacent lane
        tracker_objects = {
            10: (150.0, 50.0 + f * 3.0, {"bbox": (130, int(35 + f * 3.0), 170, int(65 + f * 3.0)), "class_name": "car"}),
            11: (200.0, 10.0 + f * 6.0, {"bbox": (180, int(0 + f * 6.0), 220, int(25 + f * 6.0)), "class_name": "car"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
    assert len(accidents) == 0, "Passing vehicles falsely triggered an accident!"
    assert detector.system_condition == "NORMAL"
    print("  ✓ PASS: Normal overtaking produced 0 alerts and maintained condition NORMAL.")

    # -------------------------------------------------------------
    # TEST 3: Vehicle Stopping at Traffic Signal -> No accident alert
    # -------------------------------------------------------------
    print("\n[TEST 3] Vehicle Stopping at Traffic Signal...")
    detector = unified_app.AccidentDetector()
    # Decelerate smoothly over 15 frames from 5 px/f to 0, then stay stopped for 20 frames
    y = 50.0
    for f in range(35):
        if f < 15:
            speed = 5.0 * (1.0 - f / 15.0)
            y += speed
        tracker_objects = {
            15: (200.0, y, {"bbox": (180, int(y - 15), 220, int(y + 15)), "class_name": "car"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
    assert len(accidents) == 0, "Traffic signal stop falsely triggered an accident!"
    assert detector.system_condition == "NORMAL"
    print("  ✓ PASS: Stopping at a traffic signal produced 0 alerts and maintained condition NORMAL.")

    # -------------------------------------------------------------
    # TEST 4: Stationary Roadside Objects -> No accident alert
    # -------------------------------------------------------------
    print("\n[TEST 4] Stationary Roadside Objects...")
    detector = unified_app.AccidentDetector()
    for f in range(30):
        tracker_objects = {
            20: (50.0, 300.0, {"bbox": (30, 280, 70, 320), "class_name": "car"}), # stationary roadside car
            21: (200.0, 50.0 + f * 5.0, {"bbox": (180, int(35 + f * 5.0), 220, int(65 + f * 5.0)), "class_name": "truck"}), # moving car
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
    assert len(accidents) == 0, "Stationary roadside object falsely triggered an accident!"
    assert detector.system_condition == "NORMAL"
    print("  ✓ PASS: Stationary roadside objects produced 0 alerts and maintained condition NORMAL.")

    # -------------------------------------------------------------
    # TEST 5: Two Vehicles Passing Close Without Collision -> No alert
    # -------------------------------------------------------------
    print("\n[TEST 5] Two Vehicles Passing Close Together (Transient 2D overlap)...")
    detector = unified_app.AccidentDetector()
    for f in range(30):
        # Vehicles cross paths in 2D perspective between frames 10-14, then separate and keep driving
        y1 = 50.0 + f * 4.0
        y2 = 250.0 - f * 4.0
        tracker_objects = {
            30: (200.0, y1, {"bbox": (180, int(y1 - 15), 220, int(y1 + 15)), "class_name": "car"}),
            31: (210.0, y2, {"bbox": (190, int(y2 - 15), 230, int(y2 + 15)), "class_name": "car"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
    assert len(accidents) == 0, "Vehicles passing close falsely triggered a confirmed accident!"
    assert detector.system_condition == "NORMAL"
    print("  ✓ PASS: Vehicles passing close separated normally with 0 false alerts.")

    # -------------------------------------------------------------
    # TEST 6: Suspicious Sudden Braking -> Suspicious Activity (NOT confirmed accident)
    # -------------------------------------------------------------
    print("\n[TEST 6] Suspicious Sudden Braking...")
    detector = unified_app.AccidentDetector()
    y = 50.0
    has_suspicious_flagged = False
    for f in range(15):
        speed = 7.0 if f < 8 else 1.0  # sudden hard braking without collision
        y += speed
        tracker_objects = {
            40: (200.0, y, {"bbox": (180, int(y - 15), 220, int(y + 15)), "class_name": "car"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
        if susp:
            has_suspicious_flagged = True
    assert len(accidents) == 0, "Sudden braking falsely triggered a confirmed accident!"
    assert has_suspicious_flagged, "Expected suspicious activity event to be flagged for hard braking!"
    print("  ✓ PASS: Sudden braking flagged as Suspicious Activity without confirmed accident.")

    # -------------------------------------------------------------
    # TEST 7: Actual Vehicle Collision -> Confirmed Accident & Notification
    # -------------------------------------------------------------
    print("\n[TEST 7] Actual Hard Vehicle Collision...")
    detector = unified_app.AccidentDetector(confidence_threshold=90, verify_window=10)
    confirmed = []
    # Approach phase (frames 0 to 7)
    for f in range(8):
        y1 = 50.0 + f * 5.0
        y2 = 130.0 - f * 5.0
        tracker_objects = {
            50: (200.0, y1, {"bbox": (180, int(y1 - 15), 220, int(y1 + 15)), "class_name": "car"}),
            51: (205.0, y2, {"bbox": (185, int(y2 - 15), 225, int(y2 + 15)), "class_name": "car"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
        if accidents:
            confirmed = accidents

    # Impact & Post-Impact Entanglement / Arrest Phase (frames 8 to 25)
    # Both vehicles collide violently at y=90, bounding boxes intersect (IoU >= 0.65), and remain arrested
    for f in range(8, 26):
        tracker_objects = {
            50: (200.0, 90.0, {"bbox": (180, 75, 220, 105), "class_name": "car"}),
            51: (205.0, 92.0, {"bbox": (185, 77, 225, 107), "class_name": "car"}),
        }
        accidents, susp = detector.update(tracker_objects, 640, 480)
        if accidents:
            confirmed = accidents

    assert len(confirmed) > 0, "Failed to confirm actual hard vehicle collision!"
    acc = confirmed[0]
    assert acc["confidence_pct"] >= 90, f"Confidence {acc['confidence_pct']}% below 90% threshold!"
    assert detector.system_condition == "DANGER"
    print(f"  ✓ PASS: Hard collision verified with {acc['confidence']} confidence (Condition: DANGER).")

    # -------------------------------------------------------------
    # TEST 8: Same Accident Visible for 100 Frames -> 1 DB Record, 1 Siren/Notif
    # -------------------------------------------------------------
    print("\n[TEST 8] Deduplication Across 100 Consecutive Frames...")
    incident_key = acc["incident_key"]
    vids_str = ", ".join(map(str, acc["vehicle_ids"]))
    first_db_id = None
    first_inc_id = None
    notification_triggers = 0

    for f in range(100):
        db_id, inc_id, is_new = database.save_or_update_accident_event(
            incident_id=incident_key,
            feed_title="Highway North CCTV",
            location="Highway North CCTV",
            confidence=acc["confidence"],
            confidence_pct=acc["confidence_pct"],
            vehicle_ids=vids_str,
            frame_number=f + 1,
            accident_type="Vehicle Collision",
        )
        if f == 0:
            first_db_id = db_id
            first_inc_id = inc_id
        if is_new:
            notification_triggers += 1

    # Verify database has exactly 1 row
    conn = database.get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*), id, incident_id, frame_number FROM AccidentEvents")
    row = c.fetchone()
    total_db_records = row[0]
    last_frame = row[3]
    conn.close()

    assert total_db_records == 1, f"Expected exactly 1 database record, found {total_db_records}!"
    assert notification_triggers == 1, f"Expected exactly 1 notification trigger, got {notification_triggers}!"
    assert last_frame == 100, f"Expected frame_number to be updated to 100, got {last_frame}"
    print(f"  ✓ PASS: 100 consecutive frames produced exactly 1 DB record ({first_inc_id}) and 1 notification trigger.")

    # -------------------------------------------------------------
    # TEST 9: Dashboard Refresh While Accident Active -> Visible, No Duplicate
    # -------------------------------------------------------------
    print("\n[TEST 9] Dashboard Refresh While Accident Active...")
    summary = database.get_system_accident_summary()
    assert summary["condition"] == "DANGER", f"Expected condition DANGER, got {summary['condition']}"
    assert summary["unresolved_count"] == 1
    recent = database.get_recent_accident_events(limit=5, status_filter="unresolved")
    assert len(recent) == 1
    assert recent[0]["incident_id"] == first_inc_id
    print("  ✓ PASS: Dashboard refresh correctly displays active unresolved incident without duplicate alerts.")

    # -------------------------------------------------------------
    # TEST 10: Acknowledging an Accident -> Status: Acknowledged (WARNING)
    # -------------------------------------------------------------
    print("\n[TEST 10] Acknowledging Accident Incident...")
    database.acknowledge_accident_event(first_inc_id, username="TrafficDispatcher")
    summary_after_ack = database.get_system_accident_summary()
    assert summary_after_ack["condition"] == "WARNING", f"Expected WARNING, got {summary_after_ack['condition']}"
    assert summary_after_ack["acknowledged_count"] == 1
    assert summary_after_ack["unresolved_count"] == 0

    recent_ack = database.get_recent_accident_events(status_filter="acknowledged")
    assert len(recent_ack) == 1
    assert recent_ack[0]["acknowledged_by"] == "TrafficDispatcher"
    print("  ✓ PASS: Accident successfully acknowledged by TrafficDispatcher (Condition: WARNING).")

    # -------------------------------------------------------------
    # TEST 11: Resolving an Accident -> Status: Resolved (NORMAL)
    # -------------------------------------------------------------
    print("\n[TEST 11] Confirming & Resolving Accident Incident...")
    database.resolve_accident_event(first_inc_id, username="SiteSupervisor")
    summary_after_res = database.get_system_accident_summary()
    assert summary_after_res["condition"] == "NORMAL", f"Expected NORMAL, got {summary_after_res['condition']}"
    assert summary_after_res["resolved_count"] == 1
    assert summary_after_res["unresolved_count"] == 0
    assert summary_after_res["acknowledged_count"] == 0

    recent_res = database.get_recent_accident_events(status_filter="resolved")
    assert len(recent_res) == 1
    assert recent_res[0]["resolved_by"] == "SiteSupervisor"
    print("  ✓ PASS: Incident resolved by SiteSupervisor. System condition reset to NORMAL.")

    # -------------------------------------------------------------
    # TEST 12: Genuinely New Accident -> New Incident ID & 1 New Notification
    # -------------------------------------------------------------
    print("\n[TEST 12] Genuinely New Accident Event...")
    new_db_id, new_inc_id, new_is_new = database.save_or_update_accident_event(
        incident_id="pair_88_89",
        feed_title="Intersection South CCTV",
        location="Intersection South CCTV",
        confidence="94%",
        confidence_pct=94,
        vehicle_ids="88, 89",
        frame_number=1,
        accident_type="Vehicle Collision",
    )
    assert new_is_new is True, "Expected new_is_new to be True for new incident!"
    assert new_inc_id != first_inc_id, f"Expected distinct incident ID, got {new_inc_id} matching old {first_inc_id}"
    assert new_db_id != first_db_id

    summary_new = database.get_system_accident_summary()
    assert summary_new["condition"] == "DANGER"
    assert summary_new["unresolved_count"] == 1
    assert summary_new["resolved_count"] == 1

    conn = database.get_db_connection()
    c = conn.cursor()
    c.execute("SELECT incident_id, status FROM AccidentEvents ORDER BY id ASC")
    all_rows = c.fetchall()
    conn.close()
    assert len(all_rows) == 2
    print(f"  ✓ PASS: Genuinely new accident created incident {new_inc_id} with initial notification.")

    print("\n" + "=" * 75)
    print("ALL 12 ACCIDENT DETECTION & INCIDENT MANAGEMENT TESTS PASSED PERFECTLY!")
    print("=" * 75)

if __name__ == "__main__":
    run_all_tests()
