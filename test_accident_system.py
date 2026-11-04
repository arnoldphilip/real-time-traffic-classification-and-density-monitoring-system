"""
Accident Detection and Alert Notification System verification tests.
"""
import time
from datetime import datetime
import database
from unified_app import AccidentDetector, get_accident_audio_and_notification_script


def test_accident_detector_collision():
    detector = AccidentDetector()
    frame_w, frame_h = 854, 480

    # Simulate two vehicles moving toward each other and colliding
    # Frame 1 to 5: moving toward each other
    for f in range(5):
        tracks = {
            1: (100 + f * 10, 200, {"bbox": (100 + f * 10 - 20, 180, 100 + f * 10 + 20, 220), "class": "car", "confidence": 0.9}),
            2: (200 - f * 10, 200, {"bbox": (200 - f * 10 - 20, 180, 200 - f * 10 + 20, 220), "class": "car", "confidence": 0.88}),
        }
        detector.update(tracks, frame_w, frame_h)

    # Frame 6 to 10: overlapping and stopped (collision!)
    accidents = []
    for f in range(5):
        tracks = {
            1: (145, 200, {"bbox": (125, 180, 165, 220), "class": "car", "confidence": 0.9}),
            2: (150, 200, {"bbox": (130, 180, 170, 220), "class": "car", "confidence": 0.88}),
        }
        accidents, _ = detector.update(tracks, frame_w, frame_h)

    assert len(accidents) > 0, "AccidentDetector failed to detect collision between overlapping vehicles"
    acc = accidents[0]
    assert acc["type"] == "Vehicle Collision"
    assert 1 in acc["vehicle_ids"] and 2 in acc["vehicle_ids"]
    assert "%" in acc["confidence"]
    assert acc["confidence_pct"] >= 75
    print(f"[PASS] test_accident_detector_collision (Confidence: {acc['confidence']})")


def test_accident_detector_sudden_crash():
    detector = AccidentDetector()
    frame_w, frame_h = 854, 480

    # Moving at high speed (12 px/frame)
    for f in range(8):
        tracks = {
            10: (50 + f * 12, 150, {"bbox": (50 + f * 12 - 20, 130, 50 + f * 12 + 20, 170), "class": "truck", "confidence": 0.92})
        }
        detector.update(tracks, frame_w, frame_h)

    # Abruptly stop (speed = 0) for 6 frames
    accidents = []
    for f in range(6):
        tracks = {
            10: (146, 150, {"bbox": (126, 130, 166, 170), "class": "truck", "confidence": 0.92})
        }
        accidents, _ = detector.update(tracks, frame_w, frame_h)

    assert len(accidents) > 0, "AccidentDetector failed to detect sudden crash / hard stop"
    assert accidents[0]["type"] == "Sudden Crash / Hard Stop"
    print(f"[PASS] test_accident_detector_sudden_crash (Confidence: {accidents[0]['confidence']})")


def test_database_accident_workflow():
    database.init_database()

    # 1. Save an accident event
    event_id = database.save_accident_event(
        feed_title="CCTV Camera 01 (Highway North)",
        location="CCTV Camera 01 - KM 42 Northbound",
        confidence="89%",
        vehicle_ids="10, 14",
        frame_number=120,
        accident_type="Vehicle Collision",
    )
    assert event_id > 0, "Failed to insert accident event"

    # 2. Check system condition transitions to DANGER
    summary_danger = database.get_system_accident_summary()
    assert summary_danger["condition"] == "DANGER", f"Expected DANGER, got {summary_danger['condition']}"
    assert summary_danger["unresolved_count"] >= 1
    print("[PASS] test_database_accident_workflow: Condition DANGER verified")

    # 3. Acknowledge alert -> condition transitions to WARNING
    database.acknowledge_accident_event(event_id, username="highway_operator")
    # Mark any other old unresolved test events resolved for clean test
    conn = database.get_db_connection()
    conn.execute("UPDATE AccidentEvents SET resolved = 1 WHERE id != ? AND resolved = 0 AND acknowledged = 0", (event_id,))
    conn.commit()
    conn.close()

    summary_warn = database.get_system_accident_summary()
    assert summary_warn["condition"] == "WARNING", f"Expected WARNING, got {summary_warn['condition']}"
    assert summary_warn["acknowledged_count"] >= 1
    print("[PASS] test_database_accident_workflow: Condition WARNING verified on Acknowledge")

    # 4. Resolve alert -> condition transitions to NORMAL
    database.resolve_accident_event(event_id, username="admin")
    # Verify event fields
    recent = database.get_recent_accident_events(limit=5)
    ev = next(r for r in recent if r["id"] == event_id)
    assert ev["resolved"] == 1
    assert ev["acknowledged"] == 1
    assert ev["acknowledged_by"] == "highway_operator"
    assert ev["resolved_by"] == "admin"
    assert "🚨 TRAFFIC ACCIDENT ALERT!" in ev["alert_message"]

    summary_normal = database.get_system_accident_summary()
    assert summary_normal["unresolved_count"] == 0
    assert summary_normal["acknowledged_count"] == 0
    assert summary_normal["condition"] == "NORMAL"
    print("[PASS] test_database_accident_workflow: Condition NORMAL verified on Resolve")

    # 5. Clear test event
    database.clear_accident_event(event_id)
    cleared = [r for r in database.get_recent_accident_events(limit=20) if r["id"] == event_id]
    assert len(cleared) == 0, "Failed to clear accident event"
    print("[PASS] test_database_accident_workflow: Clear event verified")


def test_audio_and_notification_generator():
    html_script = get_accident_audio_and_notification_script("CCTV 01", "10:45 AM", "91%")
    assert "AudioContext" in html_script
    assert "Notification" in html_script
    assert "CCTV 01" in html_script
    print("[PASS] test_audio_and_notification_generator")


def main():
    print("Testing Real-Time Accident Detection and Alert Notification System...")
    test_accident_detector_collision()
    test_accident_detector_sudden_crash()
    test_database_accident_workflow()
    test_audio_and_notification_generator()
    print("\nALL ACCIDENT DETECTION & ALERT SYSTEM TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

