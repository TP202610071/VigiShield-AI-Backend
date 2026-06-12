"""Central configuration for the VigiShield AI Backend. Loaded from .env."""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Backend connection ────────────────────────────────────────────────────────
BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:5020")
INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "dev-internal-api-key-12345")

# ── Model paths ───────────────────────────────────────────────────────────────
ACTIVITY_MODEL_PATH: str = os.getenv(
    "ACTIVITY_MODEL_PATH",
    r"G:\Codigo\VigilShield\AI Model\models\vigishield_activity_model",
)
ACTIVITY_MODEL_META: str = os.getenv(
    "ACTIVITY_MODEL_META",
    r"G:\Codigo\VigilShield\AI Model\models\model_meta.json",
)
YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
FACES_DIR: Path = Path(os.getenv(
    "FACES_DIR",
    str(Path(__file__).parent / "known_faces"),
))

# ── Detection thresholds ──────────────────────────────────────────────────────
ACTIVITY_CONFIDENCE_THRESHOLD: float = float(os.getenv("ACTIVITY_CONFIDENCE_THRESHOLD", "0.60"))

# Activity EVENTS (alarms persisted to the backend) are gated MUCH harder than the
# on-screen banner. The UCF-Crime-style activity model was trained on street/store
# CCTV, so on a domestic indoor scene it over-fires "Burglary/Robbery". To ingest an
# activity event we now require ALL of: a person actually in frame, this confidence,
# AND the suspicious prediction to persist across this many consecutive inferences.
ACTIVITY_EVENT_CONFIDENCE: float = float(os.getenv("ACTIVITY_EVENT_CONFIDENCE", "0.85"))
ACTIVITY_EVENT_MIN_STREAK: int = int(os.getenv("ACTIVITY_EVENT_MIN_STREAK", "2"))

YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.50"))
FACE_DISTANCE_THRESHOLD: float = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.45"))

# Activity classes to ignore entirely (never suspicious, never the displayed
# prediction). This app is domestic surveillance, so retail-oriented classes the
# model misfires on (Shoplifting) and traffic events are excluded by default.
ACTIVITY_EXCLUDED_LABELS: set[str] = {
    s.strip() for s in os.getenv("ACTIVITY_EXCLUDED_LABELS", "Shoplifting,Roadaccidents").split(",")
    if s.strip()
}

# ── Event de-duplication ──────────────────────────────────────────────────────
# Don't ingest the SAME event_type for the same camera more than once per this
# many seconds (prevents 10 alerts in a few seconds for one ongoing situation).
EVENT_TYPE_COOLDOWN_SECONDS: float = float(os.getenv("EVENT_TYPE_COOLDOWN_SECONDS", "60"))

# ── MediaMTX ──────────────────────────────────────────────────────────────────
# When set, the AI backend reads from MediaMTX RTSP re-exposure instead of the
# camera directly. This avoids dual RTSP connections when the camera only supports one.
MEDIAMTX_RTSP_URL: str | None = os.getenv("MEDIAMTX_RTSP_URL", "rtsp://localhost:8554") or None

# ── Processing tuning ─────────────────────────────────────────────────────────
FRAME_INTERVAL_SECONDS: float = float(os.getenv("FRAME_INTERVAL_SECONDS", "0.5"))
CAMERA_REFRESH_INTERVAL: int = int(os.getenv("CAMERA_REFRESH_INTERVAL", "300"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Windows CUDA DLL fix ──────────────────────────────────────────────────────
# Python 3.8+ no longer searches PATH for DLLs; must explicitly add CUDA dir.
_cuda_bin = r"G:\NVIDIA\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
if sys.platform == "win32" and os.path.isdir(_cuda_bin):
    os.add_dll_directory(_cuda_bin)
