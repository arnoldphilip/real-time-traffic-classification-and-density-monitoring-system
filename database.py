"""Database operations and security module for ATCC."""
import os
import json
import sqlite3
from datetime import datetime
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


def add_video_feed(title: str, video_url: str, current_user: dict, loop: bool = True, status: str = "active", target_user_id: int = None) -> tuple[bool, str, int | None]:
    """
    Add a new video feed.
    - Standard user always assigns feed to their own user_id (cannot assign to others).
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
        INSERT INTO VideoFeeds (user_id, title, video_url, status, loop, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (assigned_user_id, title, video_url, status, 1 if loop else 0, now_str))
    new_id = cursor.lastrowid

    log_activity_internal(cursor, current_user["id"], current_user["username"], "ADD_FEED", f"Added video feed '{title}' (ID: {new_id}, Owner: {assigned_user_id})")
    conn.commit()
    conn.close()
    return True, f"Video feed '{title}' added successfully.", new_id


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

