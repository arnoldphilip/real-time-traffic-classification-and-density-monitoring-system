"""Basic ATCC system verification tests."""
import os
import sqlite3
from unified_app import setup_database


def test_files():
    required = ["unified_app.py", "launcher.py", "requirements.txt", "yolov8n.pt"]
    missing = [name for name in required if not os.path.exists(name)]
    assert not missing, f"Missing required files: {missing}"


def test_database():
    setup_database()
    conn = sqlite3.connect("unified_traffic_database.db")
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    conn.close()
    assert "VehicleCounts" in tables
    assert "TrafficEvents" in tables
    assert "TrafficDensity" in tables
    assert "VideoFeeds" in tables
    assert "Users" in tables
    assert "SystemSettings" in tables


def test_dependencies():
    import yt_dlp  # noqa: F401
    import cv2  # noqa: F401
    import streamlit  # noqa: F401
    import pandas  # noqa: F401
    import bcrypt  # noqa: F401


def main():
    test_files()
    test_database()
    test_dependencies()
    print("ATCC system tests passed.")


if __name__ == "__main__":
    main()
