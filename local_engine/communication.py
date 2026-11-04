"""Client-Server Communication Module for ATCC Local Engine.

Handles authenticated HTTP communication between the local processing agent
running on the user's computer and the central ATCC server:
- Heartbeats with local CPU / GPU / RAM resource telemetry
- Traffic density & vehicle count synchronization
- Real-time accident incident reporting
"""
import json
import logging
import platform
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
import psutil
import torch

logger = logging.getLogger("atcc.communication")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Comms] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_hardware_info() -> dict:
    """Collect local computer hardware and resource usage statistics."""
    info = {
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "cuda_available": torch.cuda.is_available(),
        "device_name": "CPU Mode",
    }
    if torch.cuda.is_available():
        try:
            device_idx = 0
            dev_name = torch.cuda.get_device_name(device_idx)
            total_vram = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            info["device_name"] = f"GPU: {dev_name} ({total_vram:.1f} GB VRAM)"
        except Exception:
            info["device_name"] = "CUDA (Active)"
    else:
        info["device_name"] = f"CPU ({info['cpu_count']} Cores)"
    return info


class ServerClient:
    """Client for synchronizing local processing results with central ATCC server."""

    def __init__(self, server_url: str = "http://localhost:8502", agent_id: str = None):
        self.server_url = server_url.rstrip("/")
        self.agent_id = agent_id or f"agent-{int(time.time())}"
        self.token = None
        self.user = None
        self.is_authenticated = False

    def _post(self, endpoint: str, data: dict, timeout: int = 5) -> tuple[bool, dict | None]:
        url = f"{self.server_url}{endpoint}"
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"ATCC-LocalEngine/{self.agent_id}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                return True, resp_data
        except urllib.error.HTTPError as exc:
            try:
                err_data = json.loads(exc.read().decode("utf-8"))
                return False, err_data
            except Exception:
                return False, {"error": str(exc)}
        except Exception as exc:
            return False, {"error": str(exc)}

    def _get(self, endpoint: str, timeout: int = 5) -> tuple[bool, dict | None]:
        url = f"{self.server_url}{endpoint}"
        headers = {
            "User-Agent": f"ATCC-LocalEngine/{self.agent_id}",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                return True, resp_data
        except Exception as exc:
            return False, {"error": str(exc)}

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate with the central server using username and password."""
        logger.info(f"Authenticating with server {self.server_url} as '{username}'...")
        success, resp = self._post("/api/auth/login", {"username": username, "password": password})
        if success and resp and resp.get("success"):
            self.token = resp.get("token")
            self.user = resp.get("user")
            self.is_authenticated = True
            logger.info(f"Authentication successful for user '{username}' (Role: {self.user.get('role')}).")
            return True
        else:
            err = resp.get("error", "Unknown authentication error") if resp else "Server unreachable"
            logger.error(f"Authentication failed: {err}")
            return False

    def get_assigned_feeds(self) -> list[dict]:
        """Fetch video feeds assigned to this user from the central server."""
        success, resp = self._get("/api/feeds")
        if success and resp and resp.get("success"):
            return resp.get("feeds", [])
        return []

    def send_heartbeat(
        self,
        feed_id: int,
        feed_title: str,
        model_name: str,
        fps: float,
        total_frames: int,
        status: str = "active",
    ) -> bool:
        """Send heartbeat telemetry with local hardware metrics to central server."""
        hw = get_hardware_info()
        payload = {
            "agent_id": self.agent_id,
            "user_id": self.user.get("id", 0) if self.user else 0,
            "username": self.user.get("username", "anonymous") if self.user else "anonymous",
            "feed_id": feed_id,
            "feed_title": feed_title,
            "model_name": model_name,
            "device_info": hw["device_name"],
            "fps": fps,
            "cpu_percent": hw["cpu_percent"],
            "memory_percent": hw["memory_percent"],
            "total_frames": total_frames,
            "status": status,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        success, _ = self._post("/api/agent/heartbeat", payload, timeout=3)
        return success

    def send_telemetry(self, feed_id: int, camera_id: str, telemetry: dict) -> bool:
        """Send structured traffic metrics (counts, density) to central server."""
        payload = {
            "agent_id": self.agent_id,
            "user_id": self.user.get("id", 0) if self.user else 0,
            "feed_id": feed_id,
            "camera_id": camera_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fps": telemetry.get("fps", 0.0),
            "density_level": telemetry.get("density_level", "Low"),
            "total_roi_vehicles": telemetry.get("total_roi_vehicles", 0),
            "line_crossings": telemetry.get("line_crossings", 0),
            "counts": telemetry.get("counts", {}),
            "roi_counts": telemetry.get("roi_counts", {}),
            "system_condition": telemetry.get("system_condition", "NORMAL"),
        }
        success, _ = self._post("/api/agent/telemetry", payload, timeout=3)
        return success

    def send_accident_alert(self, payload: dict) -> bool:
        """Immediately transmit verified accident alert to central server."""
        logger.warning(f"[ACCIDENT ALERT] TRANSMITTING ALERT TO SERVER: {payload.get('incident_id')}")
        success, resp = self._post("/api/agent/accident", payload, timeout=5)
        if success:
            logger.info("Server acknowledged accident alert successfully.")
            return True
        else:
            logger.error(f"Failed to transmit accident alert to server: {resp}")
            return False

    def notify_stop(self, feed_id: int = 0):
        """Notify server that local processing has stopped."""
        self._post("/api/agent/stop", {"agent_id": self.agent_id, "feed_id": feed_id}, timeout=3)
        if self.is_authenticated:
            self.send_heartbeat(feed_id=feed_id, feed_title="Stopped", model_name="", fps=0.0, total_frames=0, status="stopped")
