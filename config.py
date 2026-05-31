"""
Central configuration for the VigiShield AI Backend.
Values are loaded from environment variables (set in .env).
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Backend connection ────────────────────────────────────────────────────────
BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:5020")
INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "dev-internal-api-key-12345")
HOUSEHOLD_ID: str | None = os.getenv("HOUSEHOLD_ID")

# ── Stream override ───────────────────────────────────────────────────────────
# When set, skips fetching config from backend and uses this RTSP URL directly.
RTSP_URL_OVERRIDE: str | None = os.getenv("RTSP_URL")

# ── Processing intervals ──────────────────────────────────────────────────────
FRAME_INTERVAL_SECONDS: int = int(os.getenv("FRAME_INTERVAL_SECONDS", "5"))
EVENT_SIMULATION_INTERVAL_SECONDS: int = int(os.getenv("EVENT_SIMULATION_INTERVAL_SECONDS", "30"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
