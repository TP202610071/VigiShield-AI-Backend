"""
OpenCV-based RTSP stream reader with automatic reconnection.

Key design decisions:
- TCP transport: avoids 'error while decoding MB bytestream' UDP packet-loss errors.
- grab() + retrieve() pattern: grab() continuously drains the MediaMTX RTSP output buffer
  without decoding every frame. This prevents the upstream (MediaMTX/camera) from backing
  up and producing 'render is too slow' warnings. retrieve() decodes only the frames we
  actually want to process.
"""

import os
import cv2
import logging
import time
from typing import Generator

# Force TCP transport BEFORE any VideoCapture is created.
# UDP (the default) drops packets on busy networks → decode errors.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Connects to an RTSP stream and yields frames at a controlled interval.

    Reads from:
    - MediaMTX re-exposure: rtsp://localhost:8554/{stream-key}  (preferred)
    - Direct camera:        rtsp://user:pass@192.168.1.x:554/path  (fallback)
    """

    def __init__(self, rtsp_url: str, reconnect_delay: int = 5, max_reconnects: int = 10):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._cap: cv2.VideoCapture | None = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        logger.info("Connecting to RTSP: %s", self._sanitize_url())
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Minimal internal buffer — always get the most recent frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._cap.isOpened():
            logger.error("Could not open stream: %s", self._sanitize_url())
            return False
        logger.info("Stream connected: %s", self._sanitize_url())
        return True

    def disconnect(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Frame reading ─────────────────────────────────────────────────────────

    def frames(self, interval_seconds: float = 1.0) -> Generator:
        """
        Generator that yields one frame every `interval_seconds`.

        Uses grab() between yields to keep the upstream buffer drained —
        this prevents MediaMTX from backing up and discarding frames.
        Only calls retrieve() (which decodes the frame) when we actually
        want to process a frame.
        """
        reconnect_count = 0
        consecutive_grab_failures = 0

        while True:
            # ── Re-connect if needed ──────────────────────────────────────────
            if not self._cap or not self._cap.isOpened():
                if reconnect_count >= self.max_reconnects:
                    logger.error("Max reconnects (%d) reached. Giving up on %s.",
                                 self.max_reconnects, self._sanitize_url())
                    return
                logger.warning("Reconnecting in %ds (attempt %d/%d)…",
                               self.reconnect_delay, reconnect_count + 1,
                               self.max_reconnects)
                time.sleep(self.reconnect_delay)
                self.connect()
                reconnect_count += 1
                continue

            # ── Grab-and-drain loop until interval has elapsed ────────────────
            deadline = time.monotonic() + interval_seconds
            got_frame = False

            while time.monotonic() < deadline:
                grabbed = self._cap.grab()
                if not grabbed:
                    consecutive_grab_failures += 1
                    if consecutive_grab_failures >= 10:
                        logger.warning("Stream disconnected (%d grab failures) — %s",
                                       consecutive_grab_failures, self._sanitize_url())
                        self.disconnect()
                        consecutive_grab_failures = 0
                        break
                    time.sleep(0.02)
                    continue
                consecutive_grab_failures = 0
                got_frame = True  # at least one good grab in this window

            # ── Decode and yield only when we have a frame ────────────────────
            if got_frame and self._cap and self._cap.isOpened():
                ret, frame = self._cap.retrieve()
                if ret and frame is not None:
                    reconnect_count = 0
                    yield frame

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    def _sanitize_url(self) -> str:
        import re
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", self.rtsp_url)
