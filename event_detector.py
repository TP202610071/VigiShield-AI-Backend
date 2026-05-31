"""
VigiShield Event Detector

CURRENT STATE: Simulation mode.
Real AI model inference is stubbed — random events are emitted at a configured interval.

═══════════════════════════════════════════════════════════════════════════════
REAL IMPLEMENTATION GUIDE (when models are ready):
═══════════════════════════════════════════════════════════════════════════════

1. Person & object detection:
   → Use YOLOv8 (ultralytics) or YOLO-NAS
   → pip install ultralytics
   → model = YOLO("yolov8n.pt")
   → results = model(frame)

2. Face recognition:
   → pip install face-recognition (or deepface)
   → Compare detected faces against AuthorizedFace photos from the backend
   → Use api_client.get_authorized_faces() (endpoint to be added)

3. Pose estimation (aggression, climbing):
   → Use MediaPipe Pose or YOLOv8-pose
   → Detect skeleton keypoints and classify posture

4. Tailgating:
   → Track detected persons over time (SORT/ByteTrack)
   → Alert if person remains in zone > threshold seconds

5. Forced access:
   → Detect hands near door/lock region for > 3 seconds

Replace process_frame() body with actual model calls.
Remove the _simulation_interval guard.
═══════════════════════════════════════════════════════════════════════════════
"""

import random
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Event type → (risk_level, confidence_range) ──────────────────────────────
_EVENT_CATALOG: dict[str, tuple[str, float, float]] = {
    "FaceRecognized":       ("None",     0.78, 0.99),
    "UnknownFace":          ("Medium",   0.55, 0.85),
    "LowConfidenceFace":    ("Low",      0.30, 0.54),
    "RecurrentUnknownFace": ("High",     0.62, 0.90),
    "ForcedAccessAttempt":  ("Critical", 0.68, 0.96),
    "Tailgating":           ("High",     0.58, 0.83),
    "Climbing":             ("Critical", 0.72, 0.93),
    "PhysicalAggression":   ("Critical", 0.66, 0.89),
}

# Weighted probability for each event type (higher = more common)
_EVENT_WEIGHTS = [25, 20, 12, 8, 6, 10, 6, 6]

# Known person names for FaceRecognized simulation
_KNOWN_PERSONS = ["Diego García", "María García", "Carlos López"]


class EventDetector:
    def __init__(self, simulation_interval_seconds: int = 30):
        self._sim_interval = simulation_interval_seconds
        self._last_event_time: float = 0.0
        logger.warning(
            "⚠️  EventDetector is running in SIMULATION MODE.\n"
            "    Random events will be generated every %ds.\n"
            "    Replace process_frame() with real model inference when ready.",
            simulation_interval_seconds,
        )

    def process_frame(self, frame) -> list[dict]:
        """
        Analyze a video frame and return a list of detected events.

        Each event dict contains:
          event_type, confidence_score, risk_level, is_nighttime, person_name (optional)

        STUB: returns a random event at the configured interval.
        """
        # ── Real implementation goes here ─────────────────────────────────────
        # detections = self._run_yolo(frame)
        # face_matches = self._run_face_recognition(frame)
        # pose_events = self._run_pose_estimation(frame)
        # return self._classify_events(detections, face_matches, pose_events)
        # ─────────────────────────────────────────────────────────────────────

        return self._simulate_event()

    # ── Private simulation logic ──────────────────────────────────────────────

    def _simulate_event(self) -> list[dict]:
        now = time.monotonic()
        if now - self._last_event_time < self._sim_interval:
            return []

        self._last_event_time = now

        event_types = list(_EVENT_CATALOG.keys())
        chosen = random.choices(event_types, weights=_EVENT_WEIGHTS, k=1)[0]
        risk, conf_min, conf_max = _EVENT_CATALOG[chosen]
        confidence = round(random.uniform(conf_min, conf_max), 4)
        is_nighttime = _is_nighttime()

        person_name = None
        if chosen == "FaceRecognized":
            person_name = random.choice(_KNOWN_PERSONS)

        logger.info(
            "[SIMULATION] Event → %s | risk=%s | confidence=%.2f | nighttime=%s",
            chosen, risk, confidence, is_nighttime,
        )

        return [{
            "event_type": chosen,
            "confidence_score": confidence,
            "risk_level": risk,
            "is_nighttime": is_nighttime,
            "person_name": person_name,
        }]


def _is_nighttime() -> bool:
    """True between 22:00 and 06:00 local time."""
    h = datetime.now().hour
    return h >= 22 or h < 6
