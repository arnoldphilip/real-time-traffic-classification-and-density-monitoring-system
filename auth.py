"""Authentication and authorization module for ATCC."""
import json
import os
import bcrypt
import streamlit as st

CONFIG_PATH = "atcc_config.json"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Load configuration from atcc_config.json."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load config from {config_path}: {exc}")
    # Fallback to json/ directory if root not found
    fallback = os.path.join("json", config_path)
    if os.path.exists(fallback):
        try:
            with open(fallback, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "authentication": {"enabled": True, "login_types": ["admin", "user"], "password_storage": "bcrypt_hash"},
        "admin": {
            "permissions": [
                "view_all_users", "add_user", "edit_user", "delete_user",
                "view_all_videos", "add_live_feed", "pause_live_feed",
                "resume_live_feed", "delete_live_feed", "view_analytics", "manage_database"
            ]
        },
        "user": {
            "permissions": [
                "add_live_feed", "view_own_videos", "pause_own_video",
                "resume_own_video", "delete_own_video"
            ],
            "access": "own_videos_only"
        }
    }


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def get_role_permissions(role: str) -> list:
    """Retrieve allowed permissions list for given role from config."""
    config = load_config()
    role_section = config.get(role.lower(), {})
    return role_section.get("permissions", [])


def has_permission(user: dict, permission_name: str) -> bool:
    """Check if the given user dict has a specific permission."""
    if not user:
        return False
    role = user.get("role", "user").lower()
    if role == "admin":
        return True
    permissions = get_role_permissions(role)
    return permission_name in permissions


def authenticate(username: str, password: str) -> tuple[dict | None, str]:
    """Authenticate credentials against database."""
    from database import get_user_by_username, log_activity

    if not username or not password:
        return None, "Please enter both username and password."

    user = get_user_by_username(username.strip())
    if not user:
        return None, "Invalid username or password."

    if user.get("status") != "active":
        return None, "Account is disabled. Please contact an administrator."

    if verify_password(password, user["password_hash"]):
        log_activity(user["id"], user["username"], "LOGIN_SUCCESS", f"User logged in ({user['role']})")
        # Return user dict without password hash
        safe_user = {k: v for k, v in user.items() if k != "password_hash"}
        return safe_user, "Authentication successful."
    else:
        log_activity(user["id"], user["username"], "LOGIN_FAILED", "Failed login attempt (invalid password)")
        return None, "Invalid username or password."


# -------------------------------------------------------------
# Streamlit Session State Management
# -------------------------------------------------------------

def init_session_state():
    """Initialize authentication session state keys."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None


def login_user(user_dict: dict):
    """Set logged-in user in Streamlit session state."""
    st.session_state.authenticated = True
    st.session_state.current_user = {k: v for k, v in user_dict.items() if k != "password_hash"}


def logout_user():
    """Clear user session state."""
    from database import log_activity
    current = get_current_user()
    if current:
        log_activity(current.get("id", 0), current.get("username", "unknown"), "LOGOUT", "User logged out")
    st.session_state.authenticated = False
    st.session_state.current_user = None


def get_current_user() -> dict | None:
    """Get the current authenticated user dictionary from session state."""
    if st.session_state.get("authenticated", False):
        return st.session_state.get("current_user", None)
    return None


def is_admin(user: dict = None) -> bool:
    """Check if the user is an admin."""
    if user is None:
        user = get_current_user()
    if not user:
        return False
    return user.get("role", "").lower() == "admin"
