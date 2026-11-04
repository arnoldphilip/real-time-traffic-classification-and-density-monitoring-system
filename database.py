"""Database operations and security module for ATCC."""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from auth import hash_password, load_config

DB_PATH = "unified_traffic_database.db"
VIDEOS_JSON_PATH = "videos.json"


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Return an SQLite connection with Row factory enabled."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_database(db_path: str = DB_PATH):
    """Initialize all ATCC tables according to configuration and seed sample data."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            full_name TEXT,
            email TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT
        )
    """)

    # 2. VideoFeeds table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS VideoFeeds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            video_url TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            loop INTEGER DEFAULT 1,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES Users(id) ON DELETE CASCADE
        )
    """)

    # 3. VehicleCounts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS VehicleCounts (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            vehicle_type TEXT,
            count INTEGER,
            lane TEXT
        )
    """)

    # 4. TrafficEvents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TrafficEvents (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            event_type TEXT,
            description TEXT,
            lane TEXT,
            trigger_count INTEGER
        )
    """)

    # 5. TrafficDensity table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TrafficDensity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            total_vehicles INTEGER,
            density_level TEXT,
            car_count INTEGER,
            bus_count INTEGER,
            truck_count INTEGER,
            bike_count INTEGER,
            auto_count INTEGER,
            roi_bounds TEXT
        )
    """)

    # 6. ActivityLogs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ActivityLogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TEXT
        )
    """)

    # 7. AccidentEvents table — stores detected accident/collision events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS AccidentEvents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            feed_title TEXT,
            location TEXT,
            accident_type TEXT DEFAULT 'Vehicle Collision',
            confidence TEXT,
            vehicle_ids TEXT,
            frame_number INTEGER,
            alert_message TEXT,
            status TEXT DEFAULT 'unresolved',
            resolved INTEGER DEFAULT 0,
            acknowledged INTEGER DEFAULT 0,
            acknowledged_by TEXT,
            resolved_by TEXT
        )
    """)

    # Ensure schema migrations for any existing databases
    cursor.execute("PRAGMA table_info(AccidentEvents)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    columns_to_add = [
        ("incident_id", "TEXT"),
        ("last_updated", "TEXT"),
        ("feed_id", "INTEGER DEFAULT 0"),
        ("accident_type", "TEXT DEFAULT 'Vehicle Collision'"),
        ("confidence_pct", "INTEGER DEFAULT 90"),
        ("road_type", "TEXT DEFAULT 'HIGHWAY'"),
        ("severity", "TEXT DEFAULT 'HIGH'"),
        ("damage_assessment", "TEXT DEFAULT ''"),
        ("evidence_details", "TEXT"),
        ("alert_message", "TEXT"),
        ("status", "TEXT DEFAULT 'unresolved'"),
        ("resolved", "INTEGER DEFAULT 0"),
        ("acknowledged", "INTEGER DEFAULT 0"),
        ("notification_sent", "INTEGER DEFAULT 1"),
        ("acknowledged_by", "TEXT"),
        ("acknowledged_at", "TEXT"),
        ("resolved_by", "TEXT"),
        ("resolved_at", "TEXT"),
    ]
    for col_name, col_def in columns_to_add:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE AccidentEvents ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass

    # Ensure schema migrations for VideoFeeds
    cursor.execute("PRAGMA table_info(VideoFeeds)")
    existing_feed_cols = {row[1] for row in cursor.fetchall()}
    feed_cols_to_add = [
        ("road_type", "TEXT DEFAULT 'HIGHWAY'"),
        ("roi_polygon", "TEXT DEFAULT ''"),
        ("exclusion_zones", "TEXT DEFAULT ''"),
    ]
    for col_name, col_def in feed_cols_to_add:
        if col_name not in existing_feed_cols:
            try:
                cursor.execute(f"ALTER TABLE VideoFeeds ADD COLUMN {col_name} {col_def}")
            except Exception:
                pass

    # 8. LocalAgents table — tracks connected client processing engines
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS LocalAgents (
            agent_id TEXT PRIMARY KEY,
            user_id INTEGER,
            username TEXT,
            feed_id INTEGER,
            feed_title TEXT,
            model_name TEXT,
            device_info TEXT,
            fps REAL DEFAULT 0.0,
            cpu_percent REAL DEFAULT 0.0,
            memory_percent REAL DEFAULT 0.0,
            total_frames INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',
            last_heartbeat TEXT
        )
    """)

    conn.commit()

    # Seed default users if empty
    cursor.execute("SELECT COUNT(*) FROM Users")
    user_count = cursor.fetchone()[0]
    if user_count == 0:
        seed_users = [
            (1, "admin", hash_password("admin123"), "admin", "System Administrator", "admin@traffic.local", "active"),
            (101, "highway_operator", hash_password("user123"), "user", "Highway Camera Operator", "highway@traffic.local", "active"),
            (102, "city_operator", hash_password("user123"), "user", "City Road Operator", "city@traffic.local", "active"),
            (103, "traffic_user", hash_password("user123"), "user", "Standard User", "user@traffic.local", "active"),
        ]
        cursor.executemany("""
            INSERT INTO Users (id, username, password_hash, role, full_name, email, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, seed_users)
        conn.commit()
        log_activity_internal(cursor, 1, "admin", "INITIALIZE", "Database seeded with initial administrative and user accounts")

    # Seed sample video feeds if empty
    cursor.execute("SELECT COUNT(*) FROM VideoFeeds")
    feed_count = cursor.fetchone()[0]
    if feed_count == 0:
        seed_videos = []
        # Try loading from videos.json first
        if os.path.exists(VIDEOS_JSON_PATH):
            try:
                with open(VIDEOS_JSON_PATH, "r", encoding="utf-8") as vf:
                    data = json.load(vf)
                    for item in data.get("videos", []):
                        seed_videos.append((
                            item.get("id"),
                            item.get("user_id", 101),
                            item.get("title", "Live Traffic Camera"),
                            item.get("video_url", ""),
                            item.get("status", "active"),
                            1 if item.get("loop", True) else 0,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ))
            except Exception as e:
                print(f"Notice: Failed to load videos.json: {e}")

        # Fallback default feeds if videos.json empty or failed
        if not seed_videos:
            seed_videos = [
                (1, 101, "Highway Camera 1", "https://example.com/highway1.mp4", "active", 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                (2, 102, "City Road Camera", "https://example.com/cityroad.mp4", "active", 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ]

        cursor.executemany("""
            INSERT INTO VideoFeeds (id, user_id, title, video_url, status, loop, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, seed_videos)
        conn.commit()

    conn.close()


def log_activity_internal(cursor: sqlite3.Cursor, user_id: int, username: str, action: str, details: str):
    """Internal helper to insert into ActivityLogs given an active cursor."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO ActivityLogs (user_id, username, action, details, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, username, action, details, now_str))


def log_activity(user_id: int, username: str, action: str, details: str = ""):
    """Record an action in the ActivityLogs table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    log_activity_internal(cursor, user_id, username, action, details)
    conn.commit()
    conn.close()


# -------------------------------------------------------------
# User Operations
# -------------------------------------------------------------

def get_user_by_username(username: str) -> dict | None:
    """Retrieve user record by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    """Retrieve user record by user ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users() -> list:
    """Retrieve all users (admin operation)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role, full_name, email, status, created_at FROM Users ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_user(username: str, password: str, role: str = "user", full_name: str = "", email: str = "", current_user: dict = None) -> tuple[bool, str]:
    """Create a new user account."""
    if current_user and current_user.get("role") != "admin":
        return False, "Access denied: Only administrators can create new users."
    
    if not username or not password:
        return False, "Username and password are required."

    if get_user_by_username(username):
        return False, f"Username '{username}' already exists."

    conn = get_db_connection()
    cursor = conn.cursor()
    hashed = hash_password(password)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO Users (username, password_hash, role, full_name, email, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'active', ?)
    """, (username, hashed, role.lower(), full_name, email, now_str))
    new_user_id = cursor.lastrowid

    admin_id = current_user.get("id") if current_user else new_user_id
    admin_name = current_user.get("username") if current_user else username
    log_activity_internal(cursor, admin_id, admin_name, "CREATE_USER", f"Created user '{username}' with role '{role}'")

    conn.commit()
    conn.close()
    return True, f"User '{username}' created successfully."


def update_user_status(user_id: int, status: str, current_user: dict) -> tuple[bool, str]:
    """Change user status (active / inactive). Admin only."""
    if not current_user or current_user.get("role") != "admin":
        return False, "Access denied: Administrator privileges required."
    
    if user_id == current_user.get("id") and status != "active":
        return False, "Cannot deactivate your own administrator account."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE Users SET status = ? WHERE id = ?", (status, user_id))
    log_activity_internal(cursor, current_user["id"], current_user["username"], "UPDATE_USER_STATUS", f"Changed status of user ID {user_id} to '{status}'")
    conn.commit()
    conn.close()
    return True, f"User status updated to {status}."


def delete_user(user_id: int, current_user: dict) -> tuple[bool, str]:
    """Delete a user account and their associated video feeds. Admin only."""
    if not current_user or current_user.get("role") != "admin":
        return False, "Access denied: Administrator privileges required."
    
    if user_id == current_user.get("id"):
        return False, "Security check: You cannot delete your own administrative account."

    target_user = get_user_by_id(user_id)
    if not target_user:
        return False, "User not found."

    conn = get_db_connection()
    cursor = conn.cursor()
    # Delete feeds owned by this user
    cursor.execute("DELETE FROM VideoFeeds WHERE user_id = ?", (user_id,))
    # Delete user
    cursor.execute("DELETE FROM Users WHERE id = ?", (user_id,))
    log_activity_internal(cursor, current_user["id"], current_user["username"], "DELETE_USER", f"Deleted user '{target_user['username']}' (ID: {user_id}) and their video feeds")
    conn.commit()
    conn.close()
    return True, f"User '{target_user['username']}' deleted successfully."


# -------------------------------------------------------------
# Video Feed Operations with Strict Ownership Protection
# -------------------------------------------------------------

def get_video_feeds(current_user: dict, search_query: str = None, status_filter: str = None, assigned_user_id: int = None) -> list:
    """
    Retrieve video feeds based on user permissions.
    - Admin: can view all feeds, or filter by specific assigned user.
    - Standard user: can view ONLY feeds assigned to them by admin.
    """
    if not current_user:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    params = []
    where_clauses = []

    if current_user.get("role") != "admin":
        # Strict isolation: user can only see their own assigned feeds
        where_clauses.append("v.user_id = ?")
        params.append(current_user["id"])
    else:
        # Admin can optionally filter by a specific assigned user
        if assigned_user_id and assigned_user_id != "all":
            where_clauses.append("v.user_id = ?")
            params.append(int(assigned_user_id))

    if search_query:
        where_clauses.append("(v.title LIKE ? OR v.video_url LIKE ?)")
        params.extend([f"%{search_query}%", f"%{search_query}%"])

    if status_filter and status_filter != "all":
        where_clauses.append("v.status = ?")
        params.append(status_filter)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    query = f"""
        SELECT v.*, u.username as owner_username, u.full_name as owner_name
        FROM VideoFeeds v
        LEFT JOIN Users u ON v.user_id = u.id
        {where_sql}
        ORDER BY v.id DESC
    """
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_video_feed_by_id(video_id: int, current_user: dict) -> dict | None:
    """
    Retrieve a single video feed by ID with authorization verification.
    Users cannot access another user's video feed.
    """
    if not current_user:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.*, u.username as owner_username, u.full_name as owner_name
        FROM VideoFeeds v
        LEFT JOIN Users u ON v.user_id = u.id
        WHERE v.id = ?
    """, (video_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    feed = dict(row)
    # Check ownership security: Admins can view anything, users can only view their own
    if current_user.get("role") != "admin" and feed["user_id"] != current_user.get("id"):
        return None  # Access blocked

    return feed


def add_video_feed(
    title: str,
    video_url: str,
    current_user: dict,
    loop: bool = True,
    status: str = "active",
    target_user_id: int = None,
    road_type: str = "HIGHWAY",
    roi_polygon: str = "",
    exclusion_zones: str = "",
) -> tuple[bool, str, int | None]:
    """
    Add a new video feed with road classification and zone configurations.
    - Standard user always assigns feed to their own user_id.
    - Admin can assign feed to self or optionally specify target_user_id.
    """
    if not current_user:
        return False, "Authentication required.", None

    if not title or not video_url:
        return False, "Title and Video URL are required.", None

    # Determine assigned user_id
    if current_user.get("role") == "admin" and target_user_id:
        assigned_user_id = target_user_id
    else:
        assigned_user_id = current_user["id"]

    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO VideoFeeds (user_id, title, video_url, status, loop, created_at, road_type, roi_polygon, exclusion_zones)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (assigned_user_id, title, video_url, status, 1 if loop else 0, now_str, road_type, roi_polygon, exclusion_zones))
    new_id = cursor.lastrowid

    log_activity_internal(cursor, current_user["id"], current_user["username"], "ADD_FEED", f"Added video feed '{title}' [{road_type}] (ID: {new_id}, Owner: {assigned_user_id})")
    conn.commit()
    conn.close()
    return True, f"Video feed '{title}' added successfully.", new_id


def update_video_feed_road_config(video_id: int, road_type: str, roi_polygon: str = "", exclusion_zones: str = "") -> bool:
    """Update road type and polygon zones for a specific video feed."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE VideoFeeds
        SET road_type = ?,
            roi_polygon = coalesce(nullif(?, ''), roi_polygon),
            exclusion_zones = coalesce(nullif(?, ''), exclusion_zones)
        WHERE id = ?
    """, (road_type, roi_polygon, exclusion_zones, video_id))
    conn.commit()
    conn.close()
    return True


def pause_video_feed(video_id: int, current_user: dict) -> tuple[bool, str]:
    """
    Pause a live video feed.
    Strictly verifies ownership: Admins can pause any feed, users can pause only their own.
    """
    if not current_user:
        return False, "Authentication required."

    feed = get_video_feed_by_id(video_id, current_user)
    if not feed:
        return False, "Access denied or video feed not found."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE VideoFeeds SET status = 'paused' WHERE id = ?", (video_id,))
    log_activity_internal(cursor, current_user["id"], current_user["username"], "PAUSE_FEED", f"Paused video feed '{feed['title']}' (ID: {video_id})")
    conn.commit()
    conn.close()
    return True, f"Feed '{feed['title']}' paused."


def resume_video_feed(video_id: int, current_user: dict) -> tuple[bool, str]:
    """
    Resume a live video feed.
    Strictly verifies ownership: Admins can resume any feed, users can resume only their own.
    """
    if not current_user:
        return False, "Authentication required."

    feed = get_video_feed_by_id(video_id, current_user)
    if not feed:
        return False, "Access denied or video feed not found."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE VideoFeeds SET status = 'active' WHERE id = ?", (video_id,))
    log_activity_internal(cursor, current_user["id"], current_user["username"], "RESUME_FEED", f"Resumed video feed '{feed['title']}' (ID: {video_id})")
    conn.commit()
    conn.close()
    return True, f"Feed '{feed['title']}' resumed."


def delete_video_feed(video_id: int, current_user: dict) -> tuple[bool, str]:
    """
    Delete a video feed.
    Strictly verifies ownership: Admins can delete any feed, users can delete ONLY their own.
    """
    if not current_user:
        return False, "Authentication required."

    feed = get_video_feed_by_id(video_id, current_user)
    if not feed:
        return False, "Access denied or video feed not found."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM VideoFeeds WHERE id = ?", (video_id,))
    log_activity_internal(cursor, current_user["id"], current_user["username"], "DELETE_FEED", f"Deleted video feed '{feed['title']}' (ID: {video_id})")
    conn.commit()
    conn.close()
    return True, f"Feed '{feed['title']}' deleted successfully."


def toggle_loop_video_feed(video_id: int, loop: bool, current_user: dict) -> tuple[bool, str]:
    """Toggle loop playback on a video feed."""
    feed = get_video_feed_by_id(video_id, current_user)
    if not feed:
        return False, "Access denied or video feed not found."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE VideoFeeds SET loop = ? WHERE id = ?", (1 if loop else 0, video_id))
    conn.commit()
    conn.close()
    return True, f"Loop setting updated to {'Enabled' if loop else 'Disabled'}."


def assign_video_feed(video_id: int, target_user_id: int, current_user: dict) -> tuple[bool, str]:
    """
    Assign or reassign a video feed to a specific user.
    Admin only: Users can only view videos that are assigned to them by Admin.
    """
    if not current_user or current_user.get("role") != "admin":
        return False, "Access denied: Only administrators can assign feeds to users."

    target_user = get_user_by_id(target_user_id)
    if not target_user:
        return False, f"Target user ID {target_user_id} not found."

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, user_id FROM VideoFeeds WHERE id = ?", (video_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return False, f"Video feed ID {video_id} not found."

    old_owner_id = row["user_id"]
    cursor.execute("UPDATE VideoFeeds SET user_id = ? WHERE id = ?", (target_user_id, video_id))
    log_activity_internal(
        cursor,
        current_user["id"],
        current_user["username"],
        "ASSIGN_FEED",
        f"Assigned feed '{row['title']}' (ID: {video_id}) from User {old_owner_id} to '{target_user['username']}' (ID: {target_user_id})"
    )
    conn.commit()
    conn.close()
    return True, f"Feed '{row['title']}' assigned to '{target_user['username']}' successfully."


# -------------------------------------------------------------
# Activity Logs and System Stats
# -------------------------------------------------------------

def get_activity_logs(limit: int = 50, user_id: int = None) -> list:
    """Retrieve recent system activity logs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id:
        cursor.execute("SELECT * FROM ActivityLogs WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit))
    else:
        cursor.execute("SELECT * FROM ActivityLogs ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_database_stats() -> dict:
    """Return database record counts and file size statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = {}
    tables = ["Users", "VideoFeeds", "VehicleCounts", "TrafficEvents", "TrafficDensity", "ActivityLogs"]
    for t in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {t}")
            stats[t] = cursor.fetchone()[0]
        except Exception:
            stats[t] = 0
    conn.close()

    try:
        stats["db_file_size_kb"] = round(os.path.getsize(DB_PATH) / 1024, 2)
    except Exception:
        stats["db_file_size_kb"] = 0.0

    return stats


# -------------------------------------------------------------
# ATCC Traffic Detection Logging Functions
# -------------------------------------------------------------

def save_vehicle_count(vehicle_type: str, count: int, lane: str = "Lane 1"):
    """Record vehicle count crossing line."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO VehicleCounts (timestamp, vehicle_type, count, lane)
        VALUES (?, ?, ?, ?)
    """, (now_str, vehicle_type, count, lane))
    conn.commit()
    conn.close()


def save_traffic_event(event_type: str, description: str, lane: str = "Lane 1", trigger_count: int = 1):
    """Record a traffic signal or control event."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO TrafficEvents (timestamp, event_type, description, lane, trigger_count)
        VALUES (?, ?, ?, ?, ?)
    """, (now_str, event_type, description, lane, trigger_count))
    conn.commit()
    conn.close()


def save_density_to_database(roi_total: int, density_level: str, roi_type_counts: dict, roi_bounds: tuple):
    """Save traffic density estimation record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bounds_str = f"y:[{roi_bounds[0]:.2f}-{roi_bounds[2]:.2f}], x:[{roi_bounds[1]:.2f}-{roi_bounds[3]:.2f}]"
    cursor.execute("""
        INSERT INTO TrafficDensity (
            timestamp, total_vehicles, density_level,
            car_count, bus_count, truck_count, bike_count, auto_count, roi_bounds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        int(roi_total),
        str(density_level),
        int(roi_type_counts.get("car", 0)),
        int(roi_type_counts.get("bus", 0)),
        int(roi_type_counts.get("truck", 0)),
        int(roi_type_counts.get("motorcycle", 0)),
        int(roi_type_counts.get("auto", 0)),
        bounds_str,
    ))
    conn.commit()
    conn.close()



def save_or_update_accident_event(
    incident_id: str = None,
    feed_title: str = "Live Feed",
    location: str = "Traffic Camera",
    confidence: str = "90%",
    vehicle_ids: str = "",
    frame_number: int = 0,
    accident_type: str = "Vehicle Collision",
    alert_message: str = "",
    evidence_details: str = "",
    confidence_pct: int = 90,
    feed_id: int = 0,
    severity: str = "HIGH",
    damage_assessment: str = "",
    road_type: str = "HIGHWAY",
) -> tuple[int, str, bool]:
    """
    Save a new accident incident or update an existing incident across consecutive frames.
    Implements strict deduplication:
    - If incident_id exists or matches active incident on same feed with same vehicles, updates existing record.
    - If genuinely new, inserts exactly ONE record and returns is_new_incident=True.
    Returns:
        (db_id, incident_id, is_new_incident)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    time_display = now_dt.strftime("%I:%M %p")

    existing_row = None
    if incident_id:
        cursor.execute("SELECT id, incident_id, status, resolved, acknowledged FROM AccidentEvents WHERE incident_id = ?", (incident_id,))
        existing_row = cursor.fetchone()

    # Also check if incident_id key is stored in alert_message or evidence_details
    if not existing_row and incident_id:
        cursor.execute(
            "SELECT id, incident_id, status, resolved, acknowledged FROM AccidentEvents WHERE feed_title = ? AND resolved = 0 AND (alert_message LIKE ? OR evidence_details LIKE ?) ORDER BY id DESC LIMIT 1",
            (feed_title, f"%{incident_id}%", f"%{incident_id}%")
        )
        existing_row = cursor.fetchone()

    # If no match by incident_id, check for active un-resolved incident on same feed with overlapping vehicles within 180s
    # Fallback match by overlapping vehicle IDs: only when the caller gave NO incident key, and only
    # within 180 s. Tracker IDs restart on every camera/scene change, so matching on IDs alone would
    # silently merge a NEW accident into an old one and suppress its alert.
    if not existing_row and vehicle_ids and not incident_id:
        cursor.execute("""
            SELECT id, incident_id, status, resolved, acknowledged, vehicle_ids, timestamp
            FROM AccidentEvents
            WHERE feed_title = ? AND resolved = 0 AND COALESCE(last_updated, timestamp) >= ?
            ORDER BY id DESC LIMIT 5
        """, (feed_title, (now_dt - timedelta(seconds=180)).strftime("%Y-%m-%d %H:%M:%S")))
        active_candidates = cursor.fetchall()
        in_vids = {v.strip() for v in str(vehicle_ids).replace("#", "").split(",") if v.strip()}
        for cand in active_candidates:
            cand_vids = {v.strip() for v in str(cand["vehicle_ids"]).replace("#", "").split(",") if v.strip()}
            if in_vids and cand_vids and (in_vids.intersection(cand_vids)):
                existing_row = cand
                break

    # 1. Update Existing Incident Record (No new row, no duplicate notification)
    if existing_row:
        event_id = existing_row["id"]
        canonical_id = existing_row["incident_id"] or f"ACC-{event_id:04d}"
        cursor.execute("""
            UPDATE AccidentEvents
            SET last_updated = ?,
                frame_number = ?,
                confidence = ?,
                confidence_pct = ?,
                severity = coalesce(nullif(?, ''), severity),
                damage_assessment = coalesce(nullif(?, ''), damage_assessment),
                road_type = coalesce(nullif(?, ''), road_type),
                evidence_details = coalesce(nullif(?, ''), evidence_details),
                alert_message = coalesce(nullif(?, ''), alert_message)
            WHERE id = ?
        """, (now_str, frame_number, confidence, confidence_pct, severity, damage_assessment, road_type, evidence_details, alert_message, event_id))
        conn.commit()
        conn.close()
        return event_id, canonical_id, False

    # 2. Insert Genuinely New Accident Incident
    cursor.execute("SELECT MAX(id) FROM AccidentEvents")
    max_res = cursor.fetchone()[0]
    next_num = (max_res or 0) + 1
    # Only a real ACC-xxxx id is kept; detector keys (pair_/single_/wreck_/scene prefixes) are references.
    canonical_id = incident_id if (incident_id and str(incident_id).startswith("ACC-")) else f"ACC-{next_num:04d}"

    if not alert_message:
        alert_message = (
            f"🚨 TRAFFIC ACCIDENT DETECTED [{severity} - {road_type}]! Camera: {feed_title} | Location: {location} | "
            f"Time: {time_display} | Incident ID: {canonical_id} | Ref: {incident_id or ''} | "
            f"Vehicles Involved: #{vehicle_ids} | Confidence: {confidence} | Status: UNRESOLVED. "
            f"Immediate attention required."
        )

    cursor.execute("""
        INSERT INTO AccidentEvents (
            incident_id, timestamp, last_updated, feed_id, feed_title, location,
            accident_type, confidence, confidence_pct, severity, damage_assessment, road_type,
            vehicle_ids, frame_number, alert_message, evidence_details, status, resolved, acknowledged, notification_sent
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unresolved', 0, 0, 1)
    """, (
        canonical_id,
        now_str,
        now_str,
        feed_id,
        feed_title,
        location,
        accident_type,
        confidence,
        confidence_pct,
        severity,
        damage_assessment,
        road_type,
        vehicle_ids,
        frame_number,
        alert_message,
        evidence_details,
    ))
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id, canonical_id, True


def save_accident_event(
    feed_title: str,
    location: str,
    confidence: str,
    vehicle_ids: str,
    frame_number: int = 0,
    accident_type: str = "Vehicle Collision",
    alert_message: str = "",
    incident_id: str = None,
) -> int:
    """Record a detected accident/collision event to the AccidentEvents table (backward compatible)."""
    db_id, _, _ = save_or_update_accident_event(
        incident_id=incident_id,
        feed_title=feed_title,
        location=location,
        confidence=confidence,
        vehicle_ids=vehicle_ids,
        frame_number=frame_number,
        accident_type=accident_type,
        alert_message=alert_message,
    )
    return db_id


def get_recent_accident_events(limit: int = 50, status_filter: str = "all") -> list:
    """Retrieve accident events from database with status filtering (preserving historical records)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if status_filter == "unresolved":
        cursor.execute(
            "SELECT * FROM AccidentEvents WHERE resolved = 0 AND acknowledged = 0 AND status NOT IN ('cleared', 'dismissed', 'resolved') ORDER BY id DESC LIMIT ?",
            (limit,)
        )
    elif status_filter == "acknowledged":
        cursor.execute(
            "SELECT * FROM AccidentEvents WHERE acknowledged = 1 AND resolved = 0 AND status NOT IN ('cleared', 'dismissed', 'resolved') ORDER BY id DESC LIMIT ?",
            (limit,)
        )
    elif status_filter == "resolved":
        cursor.execute(
            "SELECT * FROM AccidentEvents WHERE (resolved = 1 OR status IN ('cleared', 'dismissed', 'resolved')) ORDER BY id DESC LIMIT ?",
            (limit,)
        )
    else:
        cursor.execute("SELECT * FROM AccidentEvents ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def acknowledge_accident_event(event_id, username: str = "Operator"):
    """Mark an accident event as acknowledged (being handled/investigated)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE AccidentEvents
        SET acknowledged = 1, status = 'acknowledged', acknowledged_by = ?, acknowledged_at = ?
        WHERE id = ? OR incident_id = ?
    """, (username, now_str, event_id, str(event_id)))
    conn.commit()
    conn.close()


def resolve_accident_event(event_id, username: str = "Operator"):
    """Mark an accident event as resolved with confirmation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE AccidentEvents
        SET resolved = 1, status = 'resolved', resolved_by = ?, resolved_at = ?
        WHERE id = ? OR incident_id = ?
    """, (username, now_str, event_id, str(event_id)))
    conn.commit()
    conn.close()


def dismiss_accident_event(event_id, username: str = "Operator") -> bool:
    """Dismiss an individual active alert from dashboard while safely preserving evidence in database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE AccidentEvents
        SET resolved = 1, status = 'dismissed', resolved_by = ?, resolved_at = ?
        WHERE id = ? OR incident_id = ?
    """, (username, now_str, event_id, str(event_id)))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def clear_all_active_accidents(username: str = "Operator") -> int:
    """Clear all active/unresolved accident alerts from the dashboard while preserving all historical records in database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        UPDATE AccidentEvents
        SET resolved = 1, status = 'cleared', resolved_by = ?, resolved_at = ?
        WHERE resolved = 0 OR status IN ('unresolved', 'acknowledged')
    """, (username, now_str))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected


def clear_accident_event(event_id, username: str = "Operator", hard_delete: bool = False):
    """Clear an accident event. Defaults to soft-clear (retaining evidence in history) unless hard_delete is requested."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if hard_delete:
        cursor.execute("DELETE FROM AccidentEvents WHERE id = ? OR incident_id = ?", (event_id, str(event_id)))
    else:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            UPDATE AccidentEvents
            SET resolved = 1, status = 'cleared', resolved_by = ?, resolved_at = ?
            WHERE id = ? OR incident_id = ?
        """, (username, now_str, event_id, str(event_id)))
    conn.commit()
    conn.close()


# An accident only counts as "live" if the detector updated it this recently.
# Detector refreshes `last_updated` every frame while a video/stream is running,
# so once processing stops, old rows stop triggering the DANGER/WARNING banner.
LIVE_ACCIDENT_WINDOW_SEC = 30


def expire_stale_accident_events(max_age_sec: int = LIVE_ACCIDENT_WINDOW_SEC) -> int:
    """Auto-close unresolved accident rows left over from earlier sessions.
    Call once at app start. Returns the number of rows closed."""
    cutoff = (datetime.now() - timedelta(seconds=max_age_sec)).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE AccidentEvents SET resolved = 1, status = 'expired' "
            "WHERE resolved = 0 AND COALESCE(last_updated, timestamp) < ?",
            (cutoff,),
        )
        n = cur.rowcount
        conn.commit()
        return n
    except Exception:
        return 0
    finally:
        conn.close()


def get_system_accident_summary(feed_title: str = None) -> dict:
    """
    Current LIVE accident status (optionally scoped to one feed_title):
    - DANGER : unresolved accident updated within LIVE_ACCIDENT_WINDOW_SEC
    - WARNING: acknowledged accident updated within the same window
    - NORMAL : nothing live (no video running / no detection)
    Resolved count is a lifetime total; history lives in the Alert History panel.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    feed_filter = " AND feed_title = ?" if feed_title else ""
    cutoff = (datetime.now() - timedelta(seconds=LIVE_ACCIDENT_WINDOW_SEC)).strftime("%Y-%m-%d %H:%M:%S")
    live = " AND COALESCE(last_updated, timestamp) >= ?"
    live_params = (cutoff, feed_title) if feed_title else (cutoff,)
    feed_params = (feed_title,) if feed_title else ()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM AccidentEvents WHERE resolved = 0 AND acknowledged = 0 AND status NOT IN ('cleared', 'dismissed', 'resolved'){live}{feed_filter}", live_params)
        unresolved_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM AccidentEvents WHERE acknowledged = 1 AND resolved = 0 AND status NOT IN ('cleared', 'dismissed', 'resolved'){live}{feed_filter}", live_params)
        acknowledged_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT COUNT(*) FROM AccidentEvents WHERE (resolved = 1 OR status IN ('cleared', 'dismissed', 'resolved')){feed_filter}", feed_params)
        resolved_count = cursor.fetchone()[0]

        if feed_title:
            cursor.execute("SELECT * FROM AccidentEvents WHERE feed_title = ? ORDER BY id DESC LIMIT 1", (feed_title,))
        else:
            cursor.execute("SELECT * FROM AccidentEvents ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        latest_event = dict(row) if row else None
    except Exception:
        unresolved_count = 0
        acknowledged_count = 0
        resolved_count = 0
        latest_event = None
    finally:
        conn.close()

    if unresolved_count > 0:
        condition = "DANGER"
        condition_label = "🔴 DANGER – Accident detected"
        color = "#dc3545"
    elif acknowledged_count > 0:
        condition = "WARNING"
        condition_label = "🟡 WARNING – Suspicious traffic activity / Accident being handled"
        color = "#ffc107"
    else:
        condition = "NORMAL"
        condition_label = "🟢 NORMAL – No accident detected"
        color = "#28a745"

    return {
        "condition": condition,
        "condition_label": condition_label,
        "color": color,
        "unresolved_count": unresolved_count,
        "acknowledged_count": acknowledged_count,
        "resolved_count": resolved_count,
        "latest_event": latest_event,
    }


# ---------------------------------------------------------------------------
# Local Processing Agents Operations
# ---------------------------------------------------------------------------

def register_or_update_agent(
    agent_id: str,
    user_id: int,
    username: str,
    feed_id: int,
    feed_title: str,
    model_name: str,
    device_info: str,
    fps: float,
    cpu_percent: float,
    memory_percent: float,
    total_frames: int,
    status: str = "active",
) -> bool:
    """Register or update heartbeat of a local processing agent running on a user machine."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO LocalAgents (
                agent_id, user_id, username, feed_id, feed_title, model_name,
                device_info, fps, cpu_percent, memory_percent, total_frames,
                status, last_heartbeat
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                user_id = excluded.user_id,
                username = excluded.username,
                feed_id = excluded.feed_id,
                feed_title = excluded.feed_title,
                model_name = excluded.model_name,
                device_info = excluded.device_info,
                fps = excluded.fps,
                cpu_percent = excluded.cpu_percent,
                memory_percent = excluded.memory_percent,
                total_frames = excluded.total_frames,
                status = excluded.status,
                last_heartbeat = excluded.last_heartbeat
        """, (
            agent_id, user_id, username, feed_id, feed_title, model_name,
            device_info, fps, cpu_percent, memory_percent, total_frames,
            status, now_str
        ))
        conn.commit()
        return True
    except Exception as exc:
        print(f"Error registering local agent {agent_id}: {exc}")
        return False
    finally:
        conn.close()


def get_active_agents(timeout_seconds: int = 15) -> list[dict]:
    """Return list of active local processing agents with recent heartbeats."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM LocalAgents ORDER BY last_heartbeat DESC")
        rows = cursor.fetchall()
        agents = []
        now = datetime.now()
        for r in rows:
            d = dict(r)
            try:
                hb = datetime.strptime(d["last_heartbeat"], "%Y-%m-%d %H:%M:%S")
                diff_sec = (now - hb).total_seconds()
                if diff_sec > timeout_seconds and d["status"] == "active":
                    d["status"] = "stale"
            except Exception:
                diff_sec = 999
            d["seconds_since_heartbeat"] = round(diff_sec, 1)
            agents.append(d)
        return agents
    except Exception as exc:
        print(f"Error getting active agents: {exc}")
        return []
    finally:
        conn.close()


def deactivate_agent(agent_id: str) -> bool:
    """Mark a local processing agent as stopped."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE LocalAgents SET status = 'stopped' WHERE agent_id = ?", (agent_id,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()