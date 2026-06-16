"""
VigiShield AI Frame Server

Serves the latest YOLO-annotated JPEG frame per camera over HTTP.
Used by the Flutter app's "AI View" mode to show real-time detection overlays.

Endpoint:
  GET /frame/{camera_id}  →  image/jpeg  (latest annotated frame)
  GET /status/{camera_id} →  application/json  (latest detection status)
  GET /health             →  200 OK

Start with start_server(port=5050). Store frames with store_frame(camera_id, jpeg_bytes).
"""

import http.server
import json
import logging
import os
import re
import threading
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Event snapshot filenames are "YYYYMMDD_<12 hex>.jpg" — validate to block any
# path traversal on the /event/{name} route.
_SNAPSHOT_NAME_RE = re.compile(r"^[0-9]{8}_[0-9a-f]{12}\.jpg$")

_frames: dict[str, bytes] = {}
_status: dict[str, dict] = {}
_last_request: dict[str, float] = {}
_lock = threading.Lock()

# If the AI view hasn't been polled for this long, stop spending CPU on
# annotating + JPEG-encoding frames for it.
_ACTIVE_WINDOW_SECONDS = 6.0


def store_frame(camera_id: str, jpeg_bytes: bytes) -> None:
    """Store the latest annotated JPEG frame for a camera (thread-safe)."""
    with _lock:
        _frames[camera_id] = jpeg_bytes


def set_status(camera_id: str, status: dict) -> None:
    """Store the latest detection status for a camera (thread-safe).

    The Flutter AI view polls /status/{camera_id} for this so it can show a
    live banner (current activity, suspicious flag, detected objects/faces)
    without the operator having to read pixels off the annotated frame.
    """
    status = dict(status)
    status["ts"] = time.time()
    with _lock:
        _status[camera_id] = status


def is_watched(camera_id: str) -> bool:
    """True if a client requested this camera's AI frames recently.

    Lets the pipeline skip the expensive plot()+encode work when nobody is
    looking at the AI view, keeping the RTSP reader from falling behind.
    """
    with _lock:
        last = _last_request.get(camera_id, 0.0)
    return (time.monotonic() - last) < _ACTIVE_WINDOW_SECONDS


def _get_frame(camera_id: str) -> Optional[bytes]:
    with _lock:
        _last_request[camera_id] = time.monotonic()
        return _frames.get(camera_id)


class _FrameHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.strip("/").split("/")

        if path == ["health"]:
            self._ok_text("VigiShield AI Frame Server OK")
            return

        if len(path) == 2 and path[0] == "frame":
            camera_id = path[1]
            frame = _get_frame(camera_id)
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_error(404, "No frame available yet for this camera")
            return

        if len(path) == 2 and path[0] == "event":
            name = path[1]
            if not _SNAPSHOT_NAME_RE.match(name):
                self.send_error(404, "Not found")
                return
            fpath = os.path.join(config.EVENT_SNAPSHOT_DIR, name)
            try:
                with open(fpath, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_error(404, "Snapshot not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            # Snapshots are immutable once written — let the app/browser cache them.
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
            return

        if len(path) == 2 and path[0] == "status":
            camera_id = path[1]
            # Touch the request clock too, so polling /status keeps the pipeline
            # "watched" (annotating) even before the first /frame arrives.
            with _lock:
                _last_request[camera_id] = time.monotonic()
                payload = _status.get(camera_id, {"state": "starting"})
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not found")

    def _ok_text(self, text: str) -> None:
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress per-request noise from access log


def start_server(port: int = 5050) -> http.server.HTTPServer:
    """Start the frame HTTP server in a background daemon thread."""
    # ThreadingHTTPServer: each request gets its own thread, so a slow client
    # (e.g. the phone fetching a 1080p JPEG over the internet) can't block the
    # whole server / other pollers. Single-threaded HTTPServer head-of-line
    # blocks under the app's /frame + /status polling and hangs all clients.
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _FrameHandler)
    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="vigishield-frame-server",
    )
    thread.start()
    logger.info("AI Frame server → http://0.0.0.0:%d/frame/{camera_id}", port)
    return server
