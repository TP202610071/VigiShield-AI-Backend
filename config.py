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

# Shared HS256 secret with the main backend. Used by the frame server's /authcheck
# endpoint (consumed by nginx auth_request) to validate viewer JWTs before serving
# camera frames/HLS. Must equal the main backend's Jwt:Secret.
JWT_SECRET: str = os.getenv("JWT_SECRET", "")
JWT_ISSUER: str = os.getenv("JWT_ISSUER", "VigiShield")
JWT_AUDIENCE: str = os.getenv("JWT_AUDIENCE", "VigiShieldApp")

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

# ── Person gating for EVENTS (kills false person / loiter / unknown-face spam) ──
# A YOLO 'person' box is still DRAWN at a low conf, but it only counts as a real
# person for event logic (loitering, activity, face recognition) when it clears
# BOTH this confidence and this minimum height (fraction of the frame height).
# Distant noise blobs across a street stay below these and are ignored.
PERSON_EVENT_CONFIDENCE: float = float(os.getenv("PERSON_EVENT_CONFIDENCE", "0.55"))
PERSON_MIN_HEIGHT_FRAC: float = float(os.getenv("PERSON_MIN_HEIGHT_FRAC", "0.12"))

# ── Face recognition (crop + upscale the person region) ───────────────────────
# Faces are ONLY looked for inside a confident person box (the person crop is
# upscaled so far/small faces are big enough for ArcFace). 'yunet' is a fast,
# accurate CPU face detector (far better than Haar 'opencv' on textured/night
# scenes); set FACE_DETECTOR_BACKEND=opencv to fall back. An UnknownFace event
# needs the unknown to persist across this many recognition passes.
FACE_DETECTOR_BACKEND: str = os.getenv("FACE_DETECTOR_BACKEND", "yunet")
# Upscale the person crop to at least this height (px) before face detection, so
# far/small faces are big enough for ArcFace to recognize without eye contact.
FACE_CROP_UPSCALE_TO: int = int(os.getenv("FACE_CROP_UPSCALE_TO", "480"))
FACE_MIN_DETECT_CONFIDENCE: float = float(os.getenv("FACE_MIN_DETECT_CONFIDENCE", "0.70"))
UNKNOWN_FACE_MIN_STREAK: int = int(os.getenv("UNKNOWN_FACE_MIN_STREAK", "2"))

# ── Person tracking + unknown-person grace ────────────────────────────────────
# Each person is tracked across frames (IoU). A NEW person gets this grace window
# to have their face recognized before an UnknownFace alert fires (so you can walk
# up to the camera). Once a track is recognized it STAYS that identity even if the
# face is no longer visible; each track alerts at most once (no WhatsApp/app spam).
# A track expires after this idle TTL — a person who leaves and returns is "new".
UNKNOWN_ALERT_GRACE_SECONDS: float = float(os.getenv("UNKNOWN_ALERT_GRACE_SECONDS", "8"))
PERSON_TRACK_TTL_SECONDS: float = float(os.getenv("PERSON_TRACK_TTL_SECONDS", "4"))
PERSON_TRACK_IOU: float = float(os.getenv("PERSON_TRACK_IOU", "0.25"))

# ── Activity model ────────────────────────────────────────────────────────────
# The UCF-Crime activity model is mis-domained (odd labels like "Meet_and_Split")
# and CPU-heavy. Disabled by default — rule-based behaviors + face/person tracking
# drive events now. Set ACTIVITY_ENABLED=true to re-enable.
ACTIVITY_ENABLED: bool = os.getenv("ACTIVITY_ENABLED", "false").lower() in ("1", "true", "yes")

# ── Loitering / prowling ("Merodeador") ───────────────────────────────────────
LOITER_SECONDS: float = float(os.getenv("LOITER_SECONDS", "25"))

# Objects kept in the annotated view + status (COCO class names). Everything else
# (furniture, appliances, vehicles, etc.) is dropped so YOLO misclassifications
# like door→refrigerator or bench don't clutter the view. persons/weapons/faces
# are always kept regardless of this list.
OBJECT_ALLOWLIST: set[str] = {
    s.strip() for s in os.getenv("OBJECT_ALLOWLIST", "backpack,handbag,suitcase").split(",")
    if s.strip()
}

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

# ── Event snapshots ───────────────────────────────────────────────────────────
# When an event fires we save a JPEG of the moment so the app's history can show
# a real photo. Files are written here and served by frame_server at /event/{name},
# which nginx exposes over HTTPS at EVENT_SNAPSHOT_BASE_URL.
EVENT_SNAPSHOT_DIR: str = os.getenv(
    "EVENT_SNAPSHOT_DIR",
    str(Path(__file__).parent / "event_snapshots"),
)
EVENT_SNAPSHOT_BASE_URL: str = os.getenv(
    "EVENT_SNAPSHOT_BASE_URL", "https://api-ai.vigishield.app/ai/event"
).rstrip("/")
EVENT_SNAPSHOT_RETENTION_DAYS: int = int(os.getenv("EVENT_SNAPSHOT_RETENTION_DAYS", "14"))

# How often (seconds) each camera worker re-reads its household's alert toggles
# from the backend so disabling an alert in the app takes effect without a restart.
ALERT_CONFIG_REFRESH_SECONDS: int = int(os.getenv("ALERT_CONFIG_REFRESH_SECONDS", "60"))

# Seconds of video to record from the camera when an event fires (0 = disabled).
EVENT_CLIP_SECONDS: int = int(os.getenv("EVENT_CLIP_SECONDS", "6"))

# ── Cloudflare R2 (S3-compatible) media storage ───────────────────────────────
# Event snapshots and clips are uploaded here and served from the public domain.
# Credentials come from .env (never commit them). When unset, media falls back to
# being served locally by the frame server at /ai/event.
R2_ENDPOINT: str = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY_ID: str = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET: str = os.getenv("R2_BUCKET", "vigishield-bucket")
R2_PUBLIC_BASE_URL: str = os.getenv("R2_PUBLIC_BASE_URL", "https://bucket.vigishield.app").rstrip("/")

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
