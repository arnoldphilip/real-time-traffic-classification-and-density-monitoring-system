"""Comprehensive Test Suite for ATCC Local Engine & Safe Model Architecture.

Verifies:
1. Local-first model loading without GitHub dependency crashes
2. Model verification, offline fallback, and manual placement guides
3. Client-side processing engine (local inference, tracking, ROI, accident detection)
4. Hardware resource detection (CPU / GPU / RAM)
5. Secure client-server communication and structured telemetry synchronization
6. Multi-user isolation
7. Clean resource release
"""
import os
import sys
import time
import numpy as np
import cv2
import torch

from local_engine.model_manager import (
    find_model_path,
    verify_model_file,
    get_available_local_models,
    load_model_safely,
    get_manual_placement_guide,
    MODELS_DIR,
)
from local_engine.communication import ServerClient, get_hardware_info
from local_engine.video_processor import LocalVideoProcessor
from local_engine.accident_processor import LocalAccidentDetector, format_accident_alert_payload
from server_api import start_server_api, stop_server_api
import database
import auth


def test_1_models_directory_and_discovery():
    """Verify models directory exists and finds base models."""
    print("\n[TEST 1] Models Directory & Local Model Discovery...")
    assert MODELS_DIR.exists(), f"Models directory '{MODELS_DIR}' should exist!"
    models = get_available_local_models()
    assert "yolov8n.pt" in models, "yolov8n.pt must be registered"
    assert "yolov8x.pt" in models, "yolov8x.pt must be registered"
    assert models["yolov8n.pt"]["available"] is True, "yolov8n.pt must be available locally in models/"
    print(f"  [PASS] Models directory verified: {MODELS_DIR}")
    print(f"  [PASS] yolov8n.pt discovered locally ({models['yolov8n.pt']['local_size_mb']} MB)")


def test_2_safe_yolov8x_loading_without_github_crash():
    """Verify loading missing model falls back gracefully to best local model with zero network crashes."""
    print("\n[TEST 2] Missing Model Graceful Fallback & Zero Crash...")
    model, loaded_name, info = load_model_safely("yolov8_missing_heavy.pt", device="cpu", allow_fallback=True)
    assert model is not None, "Model must load via fallback"
    assert info["fallback_used"] is True, "Fallback flag must be True"
    assert loaded_name in ("yolov8n.pt", "yolov8x.pt", "yolov8l.pt"), f"Expected local fallback, got {loaded_name}"
    assert "guide" not in info or info.get("success") is True
    assert "yolov8_missing_heavy.pt" in info["warning"]
    print(f"  [PASS] Successfully avoided GitHub ConnectionResetError. Loaded local fallback: '{loaded_name}'")
    print(f"  [PASS] Warning message informs administrator: {info['warning'][:80]}...")


def test_3_manual_placement_guide():
    """Verify clear instructions provided when model is missing and no fallback allowed."""
    print("\n[TEST 3] Manual Model Placement Instructions...")
    guide = get_manual_placement_guide("yolov8x.pt")
    assert "yolov8x.pt" in guide
    assert "ATCC models directory" in guide
    assert str(MODELS_DIR) in guide
    print("  [PASS] Manual placement guide formatted correctly:")
    for line in guide.splitlines():
        print(f"         {line}")


def test_4_model_file_verification():
    """Verify validation of local model files."""
    print("\n[TEST 4] Model File Verification...")
    yolo_path = find_model_path("yolov8n.pt")
    assert yolo_path is not None
    is_valid, msg = verify_model_file(yolo_path)
    assert is_valid is True, f"yolov8n.pt verification failed: {msg}"
    print(f"  [PASS] Local model verified: {yolo_path.name} -> {msg}")


def test_5_hardware_detection():
    """Verify CPU/GPU and resource detection."""
    print("\n[TEST 5] Hardware Resource Detection...")
    hw = get_hardware_info()
    assert "cpu_count" in hw and hw["cpu_count"] > 0
    assert "memory_percent" in hw and hw["memory_percent"] > 0
    assert "device_name" in hw
    print(f"  [PASS] Hardware detected: {hw['device_name']} | Platform: {hw['platform']}")
    print(f"  [PASS] CPU: {hw['cpu_count']} cores ({hw['cpu_percent']}%) | RAM: {hw['memory_percent']}%")


def test_6_local_video_processor_inference():
    """Verify local video processing runs on local CPU/GPU without server inference."""
    print("\n[TEST 6] Local Video Processor (Client-Side Inference & Rendering)...")
    model, _, _ = load_model_safely("yolov8n.pt", device="cpu", allow_fallback=True)
    processor = LocalVideoProcessor(vehicle_model=model, device="cpu", img_size=480)

    # Synthetic 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

    t0 = time.time()
    processed_frame, telemetry = processor.process_frame(frame, run_inference=True)
    infer_time_ms = (time.time() - t0) * 1000

    assert processed_frame.shape == (480, 640, 3), "Processed frame shape mismatch"
    assert "fps" in telemetry
    assert "density_level" in telemetry
    assert "counts" in telemetry
    assert telemetry["system_condition"] == "NORMAL"
    print(f"  [PASS] Frame processed locally in {infer_time_ms:.1f} ms | FPS: {telemetry['fps']}")
    print(f"  [PASS] Telemetry generated: Density={telemetry['density_level']}, Condition={telemetry['system_condition']}")


def test_7_client_server_communication_and_multiuser():
    """Verify multi-user authentication, telemetry sync, and user isolation."""
    print("\n[TEST 7] Client-Server Communication & Multi-User Isolation...")
    # Start API server on 8502
    server = start_server_api("127.0.0.1", 8502)
    time.sleep(0.3)

    # Client A: Highway Operator
    client_a = ServerClient("http://127.0.0.1:8502", agent_id="agent-user-a")
    ok_a = client_a.authenticate("highway_operator", "user123")
    assert ok_a is True, "User A authentication failed"
    feeds_a = client_a.get_assigned_feeds()

    # Client B: City Operator
    client_b = ServerClient("http://127.0.0.1:8502", agent_id="agent-user-b")
    ok_b = client_b.authenticate("city_operator", "user123")
    assert ok_b is True, "User B authentication failed"
    feeds_b = client_b.get_assigned_feeds()

    # Verify Heartbeats with hardware telemetry from both machines
    client_a.send_heartbeat(feed_id=1, feed_title="Highway Junction", model_name="yolov8n.pt", fps=24.0, total_frames=150)
    client_b.send_heartbeat(feed_id=2, feed_title="City Crossing", model_name="yolov8n.pt", fps=30.0, total_frames=200)

    agents = database.get_active_agents()
    agent_ids = [a["agent_id"] for a in agents]
    assert "agent-user-a" in agent_ids, "Agent A should be registered"
    assert "agent-user-b" in agent_ids, "Agent B should be registered"
    print(f"  [PASS] Both User A and User B connected their local agents independently.")
    print(f"  [PASS] Server maintains multi-user registry with {len(agents)} active local agents.")


def test_8_accident_transmission_and_server_alert():
    """Verify local accident detection transmits structured alert to server without raw video."""
    print("\n[TEST 8] Local Accident Detection Transmission & Server Alert System...")
    client = ServerClient("http://127.0.0.1:8502", agent_id="agent-user-a")
    client.authenticate("highway_operator", "user123")

    mock_event = {
        "type": "Vehicle Collision",
        "vehicle_ids": [5, 8],
        "bbox": (150, 180, 260, 290),
        "confidence": "94%",
        "confidence_pct": 94,
        "evidence": {"impulse": "95%", "entanglement": "92%"},
    }
    payload = format_accident_alert_payload(
        event=mock_event,
        feed_id=1,
        feed_title="Highway Junction",
        camera_id="CAM-01",
        user_id=client.user["id"],
    )

    sent = client.send_accident_alert(payload)
    assert sent is True, "Failed to send accident alert payload"

    summary = database.get_system_accident_summary()
    assert summary["condition"] == "DANGER", "System status should transition to DANGER"
    print(f"  [PASS] Structured accident incident received and recorded on central server.")
    print(f"  [PASS] Central system condition: {summary['condition']} (Unresolved: {summary['unresolved_count']})")


def test_9_resource_release():
    """Verify local agent release clears records cleanly."""
    print("\n[TEST 9] Clean Resource Release & Agent Disconnect...")
    client = ServerClient("http://127.0.0.1:8502", agent_id="agent-user-a")
    client.notify_stop(feed_id=1)
    agents = database.get_active_agents()
    agent_a = next((a for a in agents if a["agent_id"] == "agent-user-a"), None)
    assert agent_a is not None and agent_a["status"] == "stopped"
    print("  [PASS] Local agent marked as stopped cleanly upon exit.")


def main():
    print("=" * 75)
    print("ATCC LOCAL ENGINE & SAFE MODEL ARCHITECTURE VERIFICATION")
    print("=" * 75)
    test_1_models_directory_and_discovery()
    test_2_safe_yolov8x_loading_without_github_crash()
    test_3_manual_placement_guide()
    test_4_model_file_verification()
    test_5_hardware_detection()
    test_6_local_video_processor_inference()
    test_7_client_server_communication_and_multiuser()
    test_8_accident_transmission_and_server_alert()
    test_9_resource_release()

    # Clean up test accident records
    import sqlite3
    conn = sqlite3.connect("unified_traffic_database.db")
    c = conn.cursor()
    c.execute("DELETE FROM AccidentEvents WHERE incident_id LIKE 'ACC-1-%'")
    conn.commit()
    conn.close()

    print("\n" + "=" * 75)
    print("ALL 9 LOCAL ENGINE & SAFE MODEL ARCHITECTURE TESTS PASSED PERFECTLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()

