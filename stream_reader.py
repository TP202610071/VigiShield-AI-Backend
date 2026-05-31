"""
OpenCV-based RTSP stream reader with automatic reconnection.
Works with both direct RTSP cameras and MediaMTX RTSP re-exposure.
"""

import cv2
import logging
import time
from typing import Generator

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Connects to an RTSP stream and yields frames.

    Supports both:
    - Direct camera RTSP:      rtsp://user:pass@192.168.1.100:554/stream
    - MediaMTX local RTSP:     rtsp://localhost:8554/live/stream-key
    """

    def __init__(self, rtsp_url: str, reconnect_delay: int = 5, max_reconnects: int = 10):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._cap: cv2.VideoCapture | None = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open the RTSP connection. Returns True on success."""
        logger.info(f"Connecting to RTSP: {self._sanitize_url()}")

        # Use FFMPEG backend for better RTSP compatibility
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        # Minimize buffer to get the most recent frame
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.error("Could not open RTSP stream.")
            return False

        logger.info("RTSP stream connected ✅")
        return True

    def disconnect(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Frame reading ─────────────────────────────────────────────────────────

    def read_frame(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Read one frame. Returns (success, frame)."""
        if not self._cap or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def frames(self, interval_seconds: float = 1.0) -> Generator:
        """
        Generator that yields frames at the specified interval.
        Handles reconnection automatically.
        """
        reconnect_count = 0

        while True:
            if not self._cap or not self._cap.isOpened():
                if reconnect_count >= self.max_reconnects:
                    logger.error(f"Max reconnects ({self.max_reconnects}) reached. Giving up.")
                    return
                logger.warning(f"Reconnecting in {self.reconnect_delay}s (attempt {reconnect_count + 1})")
                time.sleep(self.reconnect_delay)
                self.connect()
                reconnect_count += 1
                continue

            ret, frame = self.read_frame()
            if not ret:
                logger.warning("Frame read failed — stream disconnected?")
                self.disconnect()
                continue

            reconnect_count = 0  # reset on successful read
            yield frame
            time.sleep(interval_seconds)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sanitize_url(self) -> str:
        """Return URL with password masked for safe logging."""
        import re
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", self.rtsp_url)
