"""ATCC Central Server REST API.

Provides endpoints for client-side processing engines:
- User authentication and session tokens
- Assigned video feed queries
- Heartbeat with local hardware metrics
- Traffic density & vehicle counts synchronization
- Real-time accident incident reporting
"""
import argparse
import json
import logging
import sys
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import auth
import database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ServerAPI] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("atcc.server_api")

# In-memory session token store: token -> {user_id, username, role, expires}
_ACTIVE_SESSIONS = {}
_SESSIONS_LOCK = threading.Lock()


def generate_token(user: dict) -> str:
    """Generate a session token for an authenticated user."""
    token = f"atcc_{uuid.uuid4().hex}"
    with _SESSIONS_LOCK:
        _ACTIVE_SESSIONS[token] = {
            "id": user["id"],
            "username": user["username"],
            "role": user.get("role", "user"),
            "full_name": user.get("full_name", ""),
            "created_at": time.time(),
        }
    return token


def verify_token(token: str) -> dict | None:
    """Validate bearer session token."""
    if not token:
        return None
    token = token.replace("Bearer ", "").strip()
    with _SESSIONS_LOCK:
        return _ACTIVE_SESSIONS.get(token)


class ATCCAPIHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for ATCC REST API."""

    def _send_json(self, status_code: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS pre-flight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)
        except Exception:
            return {}

    def _get_auth_user(self) -> dict | None:
        auth_header = self.headers.get("Authorization", "")
        return verify_token(auth_header)

    # -----------------------------------------------------------------------
    # GET Handlers
    # -----------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/api/health":
            agents = database.get_active_agents(timeout_seconds=15)
            active_cnt = sum(1 for a in agents if a["status"] == "active")
            self._send_json(200, {
                "status": "healthy",
                "service": "ATCC Central Server API",
                "timestamp": datetime.now().isoformat(),
                "active_agents": active_cnt,
            })
            return

        user = self._get_auth_user()
        if not user:
            self._send_json(401, {"success": False, "error": "Unauthorized. Please authenticate first."})
            return

        if path == "/api/feeds":
            feeds = database.get_video_feeds(current_user=user)
            self._send_json(200, {"success": True, "feeds": [dict(f) for f in feeds]})
            return

        if path == "/api/agent/status":
            agents = database.get_active_agents(timeout_seconds=20)
            self._send_json(200, {"success": True, "agents": agents})
            return

        if path == "/api/accidents/summary":
            summary = database.get_system_accident_summary()
            self._send_json(200, {"success": True, "summary": summary})
            return

        self._send_json(404, {"success": False, "error": f"Endpoint not found: {path}"})

    # -----------------------------------------------------------------------
    # POST Handlers
    # -----------------------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0]
        payload = self._read_json()

        # Public Login & Stop Endpoints
        if path == "/api/auth/login":
            username = payload.get("username", "")
            password = payload.get("password", "")
            safe_user, msg = auth.authenticate(username, password)
            if safe_user:
                token = generate_token(safe_user)
                logger.info(f"User '{username}' logged in successfully via API.")
                self._send_json(200, {
                    "success": True,
                    "token": token,
                    "user": safe_user,
                    "message": "Authentication successful.",
                })
            else:
                logger.warning(f"Failed API login attempt for username '{username}': {msg}")
                self._send_json(401, {"success": False, "error": msg})
            return

        if path == "/api/agent/stop":
            agent_id = payload.get("agent_id", "")
            if agent_id:
                database.deactivate_agent(agent_id)
            self._send_json(200, {"success": True})
            return

        # Authenticated Endpoints
        user = self._get_auth_user()
        if not user:
            self._send_json(401, {"success": False, "error": "Unauthorized. Invalid or expired token."})
            return

        # 1. Agent Heartbeat & Local Hardware Resource Reporting
        if path == "/api/agent/heartbeat":
            agent_id = payload.get("agent_id", "unknown")
            feed_id = payload.get("feed_id", 0)
            feed_title = payload.get("feed_title", "")
            model_name = payload.get("model_name", "")
            device_info = payload.get("device_info", "CPU")
            fps = float(payload.get("fps", 0.0))
            cpu_percent = float(payload.get("cpu_percent", 0.0))
            memory_percent = float(payload.get("memory_percent", 0.0))
            total_frames = int(payload.get("total_frames", 0))
            status = payload.get("status", "active")

            saved = database.register_or_update_agent(
                agent_id=agent_id,
                user_id=user["id"],
                username=user["username"],
                feed_id=feed_id,
                feed_title=feed_title,
                model_name=model_name,
                device_info=device_info,
                fps=fps,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                total_frames=total_frames,
                status=status,
            )
            self._send_json(200, {"success": saved})
            return

        # 2. Telemetry Ingestion (Counts, Density)
        if path == "/api/agent/telemetry":
            timestamp = payload.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            roi_counts = payload.get("roi_counts", {})
            density_level = payload.get("density_level", "Low")
            total_roi = payload.get("total_roi_vehicles", 0)

            # Persist to database TrafficDensity table
            conn = database.get_db_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO TrafficDensity (
                        timestamp, total_vehicles, density_level,
                        car_count, bus_count, truck_count, bike_count, auto_count, roi_bounds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, total_roi, density_level,
                    roi_counts.get("car", 0),
                    roi_counts.get("bus", 0),
                    roi_counts.get("truck", 0),
                    roi_counts.get("motorcycle", 0),
                    roi_counts.get("auto", 0),
                    str(payload.get("camera_id", "CAM-01")),
                ))
                conn.commit()
            except Exception as exc:
                logger.error(f"Error saving traffic density telemetry: {exc}")
            finally:
                conn.close()

            self._send_json(200, {"success": True})
            return

        # 3. Accident Alert Reporting
        if path == "/api/agent/accident":
            logger.warning(f"[ACCIDENT ALERT] RECEIVED EVENT FROM AGENT: {payload.get('incident_id')}")
            feed_id = int(payload.get("feed_id", 0))
            feed_title = payload.get("feed_title", "Local Stream")
            camera_id = payload.get("camera_id", "CAM-01")
            location = payload.get("location", f"Camera {camera_id} ({feed_title})")
            accident_type = payload.get("accident_type", "Vehicle Collision")
            confidence_pct = int(payload.get("confidence_pct", 90))
            vehicle_ids = payload.get("vehicle_ids", "")
            alert_message = payload.get("alert_message", "")
            evidence_details = payload.get("evidence_details", "")
            incident_id = payload.get("incident_id")

            # Store in database AccidentEvents table with deduplication
            rec_id, is_new, notif_sent = database.save_or_update_accident_event(
                feed_title=feed_title,
                location=location,
                accident_type=accident_type,
                confidence_pct=confidence_pct,
                vehicle_ids=vehicle_ids,
                frame_number=0,
                alert_message=alert_message,
                incident_id=incident_id,
                feed_id=feed_id,
                evidence_details=evidence_details,
            )

            # Log audit trail
            database.log_activity(
                user_id=user["id"],
                username=user["username"],
                action="ACCIDENT_ALERT_RECEIVED",
                details=f"Local Agent reported accident {incident_id} at {location} ({confidence_pct}% conf)",
            )

            self._send_json(200, {
                "success": True,
                "incident_id": incident_id,
                "record_id": rec_id,
                "is_new_incident": is_new,
                "notification_triggered": notif_sent,
            })
            return

        self._send_json(404, {"success": False, "error": f"Endpoint not found: {path}"})

    def log_message(self, format, *args):
        # Quiet server logging for normal heartbeats
        pass


_API_SERVER_INSTANCE = None
_API_SERVER_THREAD = None


def start_server_api(host: str = "0.0.0.0", port: int = 8502) -> ThreadingHTTPServer:
    """Start the central REST API server in a background thread."""
    global _API_SERVER_INSTANCE, _API_SERVER_THREAD
    if _API_SERVER_INSTANCE is not None:
        return _API_SERVER_INSTANCE

    database.init_database()
    server = ThreadingHTTPServer((host, port), ATCCAPIHandler)
    _API_SERVER_INSTANCE = server

    def _serve():
        logger.info(f"ATCC Central REST API running on http://{host}:{port}")
        server.serve_forever()

    _API_SERVER_THREAD = threading.Thread(target=_serve, daemon=True, name="ATCC-ServerAPI")
    _API_SERVER_THREAD.start()
    return server


def stop_server_api():
    """Shut down the API server."""
    global _API_SERVER_INSTANCE
    if _API_SERVER_INSTANCE is not None:
        _API_SERVER_INSTANCE.shutdown()
        _API_SERVER_INSTANCE = None


def main():
    parser = argparse.ArgumentParser(description="ATCC Central REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8502, help="Port (default 8502)")
    args = parser.parse_args()

    database.init_database()
    logger.info(f"Starting standalone ATCC Central REST API on {args.host}:{args.port}...")
    server = ThreadingHTTPServer((args.host, args.port), ATCCAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server API shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
