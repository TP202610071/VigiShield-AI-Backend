"""
VigiShield AI Backend — Entry Point

Connects to the IP camera via RTSP, processes frames for security events,
and forwards detected events to the ASP.NET Core backend API.

Usage:
    cp .env.example .env        # fill in your values
    pip install -r requirements.txt
    python main.py
"""

import logging
import sys
import time

import config  # loads .env and configures logging
from api_client import get_stream_config, ingest_event
from stream_reader import StreamReader
from event_detector import EventDetector

logger = logging.getLogger(__name__)


# ── Stream URL resolution ────────────────────────────────────────────────────

def resolve_rtsp_url() -> str | None:
    """Get the RTSP URL — from override env var or from the backend API."""
    if config.RTSP_URL_OVERRIDE:
        logger.info(f"Using RTSP_URL override: {config.RTSP_URL_OVERRIDE}")
        return config.RTSP_URL_OVERRIDE

    if not config.HOUSEHOLD_ID:
        logger.error(
            "HOUSEHOLD_ID is not set.\n"
            "  → Copy .env.example to .env and fill in HOUSEHOLD_ID.\n"
            "  → You can find it by calling GET /api/auth/me after logging in."
        )
        return None

    logger.info("Fetching stream configuration from backend...")
    stream_config = get_stream_config()

    if stream_config is None:
        logger.error("Could not fetch stream config from backend.")
        return None

    if not stream_config.get("isConfigured"):
        logger.error(
            "Camera is not configured yet.\n"
            "  → Open the VigiShield app → Settings → Cámara → Configure your camera."
        )
        return None

    rtsp_url = stream_config.get("rtspUrl")
    if not rtsp_url:
        logger.error("Backend returned no RTSP URL. Check camera configuration in the app.")
        return None

    logger.info(f"Stream mode: {stream_config.get('streamMode')}")
    return rtsp_url


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  VigiShield AI Backend")
    logger.info(f"  Backend: {config.BACKEND_API_URL}")
    logger.info(f"  Household: {config.HOUSEHOLD_ID or '(not set — using RTSP_URL override)'}")
    logger.info(f"  Frame interval: {config.FRAME_INTERVAL_SECONDS}s")
    logger.info(f"  Simulation event interval: {config.EVENT_SIMULATION_INTERVAL_SECONDS}s")
    logger.info("=" * 60)

    rtsp_url = resolve_rtsp_url()
    if not rtsp_url:
        sys.exit(1)

    detector = EventDetector(
        simulation_interval_seconds=config.EVENT_SIMULATION_INTERVAL_SECONDS
    )

    while True:  # outer reconnect loop
        logger.info("Connecting to stream...")
        with StreamReader(rtsp_url) as reader:
            if not reader._cap or not reader._cap.isOpened():
                logger.error("Failed to open stream. Retrying in 15s...")
                time.sleep(15)
                continue

            logger.info(f"Stream open. Processing every {config.FRAME_INTERVAL_SECONDS}s — Ctrl+C to stop.")

            try:
                for frame in reader.frames(interval_seconds=config.FRAME_INTERVAL_SECONDS):
                    events = detector.process_frame(frame)
                    for event in events:
                        ingest_event(**event)

            except KeyboardInterrupt:
                logger.info("Shutdown requested. Stopping...")
                break
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
                logger.info("Restarting stream in 10s...")
                time.sleep(10)
                # continue outer loop → reconnect


if __name__ == "__main__":
    main()
