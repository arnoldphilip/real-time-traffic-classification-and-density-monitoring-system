"""Comprehensive integration and security tests for ATCC Dashboard."""
import json
import os
import sqlite3
import auth
import database


def test_config_json():
    """Verify atcc_config.json structure and required keys."""
    assert os.path.exists("atcc_config.json"), "atcc_config.json must exist"
    with open("atcc_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["project"] == "ATCC"
    assert cfg["authentication"]["enabled"] is True
    assert cfg["authentication"]["password_storage"] == "bcrypt_hash"
    assert "admin" in cfg["authentication"]["login_types"]
    assert "user" in cfg["authentication"]["login_types"]

    # Check admin permissions
    admin_perms = set(cfg["admin"]["permissions"])
    required_admin_perms = {
        "view_all_users", "add_user", "edit_user", "delete_user",
        "view_all_videos", "add_live_feed", "pause_live_feed",
        "resume_live_feed", "delete_live_feed", "view_analytics", "manage_database"
    }
    assert required_admin_perms.issubset(admin_perms), f"Missing admin perms: {required_admin_perms - admin_perms}"

    # Check user permissions
    user_perms = set(cfg["user"]["permissions"])
    required_user_perms = {
        "add_live_feed", "view_own_videos", "pause_own_video",
        "resume_own_video", "delete_own_video"
    }
    assert required_user_perms.issubset(user_perms), f"Missing user perms: {required_user_perms - user_perms}"
    assert cfg["user"]["access"] == "own_videos_only"

    # Check video settings
    vs = cfg["video_settings"]
    assert vs["autoplay"] is True
    assert vs["loop"] is True
    assert vs["hover_menu"]["enabled"] is True
    assert set(vs["hover_menu"]["options"]) == {"pause", "resume", "delete"}

    # Check database table list
    db_tables = set(cfg["database"]["tables"])
    expected_tables = {"Users", "VideoFeeds", "VehicleCounts", "TrafficEvents", "ActivityLogs"}
    assert expected_tables.issubset(db_tables)

    # Check interface navigation
    assert "Dashboard" in cfg["interface"]["admin_dashboard"]
    assert "Live Feeds" in cfg["interface"]["admin_dashboard"]
    assert "My Live Feeds" in cfg["interface"]["user_dashboard"]
    print("[PASS] test_config_json")


def test_videos_json():
    """Verify videos.json structure and validity of entries."""
    assert os.path.exists("videos.json"), "videos.json must exist"
    with open("videos.json", "r", encoding="utf-8") as f:
        vdata = json.load(f)

    assert "videos" in vdata
    videos = vdata["videos"]
    for v in videos:
        assert "id" in v
        assert "user_id" in v
        assert "title" in v
        assert "video_url" in v
        assert "status" in v
        assert "loop" in v
    print("[PASS] test_videos_json")


def test_database_tables():
    """Verify all database tables exist in SQLite."""
    database.init_database()
    conn = database.get_db_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()

    expected = {"Users", "VideoFeeds", "VehicleCounts", "TrafficEvents", "TrafficDensity", "ActivityLogs"}
    for t in expected:
        assert t in tables, f"Table {t} missing from SQLite database"
    print("[PASS] test_database_tables")


def test_authentication_and_bcrypt():
    """Verify bcrypt hashing and credential checks."""
    pw = "superSecretPassword123"
    hashed = auth.hash_password(pw)
    assert hashed != pw
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
    assert auth.verify_password(pw, hashed) is True
    assert auth.verify_password("wrongSecret", hashed) is False

    # Admin authentication
    admin_u, msg = auth.authenticate("admin", "admin123")
    assert admin_u is not None, f"Admin authentication failed: {msg}"
    assert admin_u["role"] == "admin"
    assert "password_hash" not in admin_u  # Safe dict

    # User authentication
    user101, msg = auth.authenticate("highway_operator", "user123")
    assert user101 is not None, f"User 101 auth failed: {msg}"
    assert user101["id"] == 101
    assert user101["role"] == "user"

    # Bad password
    bad_u, msg = auth.authenticate("admin", "badpassword")
    assert bad_u is None
    print("[PASS] test_authentication_and_bcrypt")


def test_strict_video_ownership_isolation():
    """
    CRITICAL SECURITY TEST:
    Users must never access or delete another user's videos, even by changing a URL or record ID.
    Admins can access all feeds.
    """
    admin_u = database.get_user_by_id(1)
    user101 = database.get_user_by_id(101)
    user102 = database.get_user_by_id(102)

    assert admin_u is not None
    assert user101 is not None
    assert user102 is not None

    # 1. Feed listing isolation
    feeds_admin = database.get_video_feeds(admin_u)
    feeds_u101 = database.get_video_feeds(user101)
    feeds_u102 = database.get_video_feeds(user102)

    # Admin sees feeds from all users
    all_user_ids_in_admin = {f["user_id"] for f in feeds_admin}
    assert 101 in all_user_ids_in_admin
    assert 102 in all_user_ids_in_admin

    # User 101 sees ONLY user 101's feeds
    for f in feeds_u101:
        assert f["user_id"] == 101, f"Security Breach: User 101 saw feed belonging to user {f['user_id']}"

    # User 102 sees ONLY user 102's feeds
    for f in feeds_u102:
        assert f["user_id"] == 102, f"Security Breach: User 102 saw feed belonging to user {f['user_id']}"

    # 2. Feed lookup by ID security check
    # Feed owned by User 102
    target_f102 = feeds_u102[0]["id"]
    target_f101 = feeds_u101[0]["id"]

    assert database.get_video_feed_by_id(target_f102, user101) is None, "Security Breach: User 101 accessed Feed belonging to User 102"
    assert database.get_video_feed_by_id(target_f102, user102) is not None, "User 102 should access their own feed"
    assert database.get_video_feed_by_id(target_f102, admin_u) is not None, "Admin must be able to view any feed"

    # 3. Unauthorized Pause/Resume/Delete blocked
    # User 101 tries to pause Feed owned by User 102
    ok, msg = database.pause_video_feed(target_f102, user101)
    assert not ok, f"Security Breach: User 101 was allowed to pause feed! ({msg})"

    # User 101 tries to resume Feed owned by User 102
    ok, msg = database.resume_video_feed(target_f102, user101)
    assert not ok, f"Security Breach: User 101 was allowed to resume feed! ({msg})"

    # User 101 tries to delete Feed owned by User 102
    ok, msg = database.delete_video_feed(target_f102, user101)
    assert not ok, f"Security Breach: User 101 was allowed to delete feed! ({msg})"

    # 4. Authorized Pause/Resume/Delete allowed
    # User 101 manages their OWN feed
    ok_pause, _ = database.pause_video_feed(target_f101, user101)
    assert ok_pause is True
    f101_check = database.get_video_feed_by_id(target_f101, user101)
    assert f101_check["status"] == "paused"

    ok_resume, _ = database.resume_video_feed(target_f101, user101)
    assert ok_resume is True
    f101_check = database.get_video_feed_by_id(target_f101, user101)
    assert f101_check["status"] == "active"

    # 5. User adding a video cannot forge target_user_id
    ok_add, _, new_feed_id = database.add_video_feed(
        title="Malicious Add Attempt",
        video_url="https://example.com/test.mp4",
        current_user=user101,
        target_user_id=102  # Attempting to assign to 102
    )
    assert ok_add is True
    added_feed = database.get_video_feed_by_id(new_feed_id, admin_u)
    assert added_feed["user_id"] == 101, "Security Breach: User 101 forged owner ID to 102!"

    # Clean up test feed
    database.delete_video_feed(new_feed_id, user101)
    print("[PASS] test_strict_video_ownership_isolation")


def test_admin_assign_stream_to_user():
    """
    Verify Admin can assign streams to users,
    and that users only see the streams assigned to them by Admin.
    """
    admin_u = database.get_user_by_id(1)
    user101 = database.get_user_by_id(101)
    user102 = database.get_user_by_id(102)

    # Admin creates a new stream assigned to user 101
    ok_add, _, fid = database.add_video_feed(
        title="Admin Assigned Stream Test",
        video_url="https://example.com/test_assign.mp4",
        current_user=admin_u,
        target_user_id=101,
    )
    assert ok_add is True

    # User 101 can see it; User 102 CANNOT see it
    assert database.get_video_feed_by_id(fid, user101) is not None
    assert database.get_video_feed_by_id(fid, user102) is None

    # Admin reassigns stream from user 101 to user 102
    ok_reassign, _ = database.assign_video_feed(fid, 102, admin_u)
    assert ok_reassign is True

    # User 102 can now see it; User 101 CANNOT see it anymore
    assert database.get_video_feed_by_id(fid, user102) is not None
    assert database.get_video_feed_by_id(fid, user101) is None

    # Standard user CANNOT reassign feeds
    ok_unauth, _ = database.assign_video_feed(fid, 101, user102)
    assert ok_unauth is False

    # Admin can filter feeds by assigned user
    feeds_u102 = database.get_video_feeds(admin_u, assigned_user_id=102)
    assert any(f["id"] == fid for f in feeds_u102)

    # Clean up test feed
    database.delete_video_feed(fid, admin_u)
    print("[PASS] test_admin_assign_stream_to_user")


def test_activity_logs():
    """Verify activity audit logging records actions."""
    logs = database.get_activity_logs(limit=20)
    assert len(logs) > 0
    actions = {l["action"] for l in logs}
    assert "INITIALIZE" in actions or "LOGIN_SUCCESS" in actions or "PAUSE_FEED" in actions
    print("[PASS] test_activity_logs")


def test_permanent_video_deletion():
    """
    CRITICAL TEST:
    Verify that when an admin deletes a video feed:
    1. It is permanently removed from the VideoFeeds SQLite table.
    2. It is permanently removed from videos.json.
    3. Standard users can no longer see the feed.
    4. Subsequent calls to database.init_database() DO NOT resurrect or regenerate the deleted feed.
    """
    admin_u = database.get_user_by_id(1)
    user101 = database.get_user_by_id(101)

    # 1. Add a temporary feed assigned to user101
    ok_add, _, fid = database.add_video_feed(
        title="Permanent Deletion Test Feed",
        video_url="https://youtu.be/test_deletion",
        current_user=admin_u,
        target_user_id=101,
    )
    assert ok_add is True
    assert database.get_video_feed_by_id(fid, user101) is not None

    # Check that videos.json was synchronized
    with open("videos.json", "r", encoding="utf-8") as f:
        vdata = json.load(f)
    assert any(v["id"] == fid for v in vdata.get("videos", []))

    # 2. Admin deletes the feed permanently
    ok_del, _ = database.delete_video_feed(fid, admin_u)
    assert ok_del is True

    # 3. Verify it is gone from SQLite and from user101
    assert database.get_video_feed_by_id(fid, user101) is None
    assert database.get_video_feed_by_id(fid, admin_u) is None

    # 4. Verify it is gone from videos.json
    with open("videos.json", "r", encoding="utf-8") as f:
        vdata2 = json.load(f)
    assert not any(v["id"] == fid for v in vdata2.get("videos", []))

    # 5. Call init_database multiple times and verify feed DOES NOT reappear
    database.init_database()
    database.init_database()
    assert database.get_video_feed_by_id(fid, admin_u) is None
    assert database.get_video_feed_by_id(fid, user101) is None

    print("[PASS] test_permanent_video_deletion")


def run_all_tests():
    print("Running ATCC Dashboard Test Suite...")
    test_config_json()
    test_videos_json()
    test_database_tables()
    test_authentication_and_bcrypt()
    test_strict_video_ownership_isolation()
    test_admin_assign_stream_to_user()
    test_activity_logs()
    test_permanent_video_deletion()
    print("\nALL ATCC DASHBOARD TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
