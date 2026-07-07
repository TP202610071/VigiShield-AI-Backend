"""
VigiShield Event Detector — Three-model pipeline + behavior fusion

Models run on each frame:
  1. YOLOv8           — object detection every frame (persons, weapons, objects)
  2. FaceRecognizer   — ArcFace (DeepFace) every N frames (authorized vs unknown)
  3. ActivityDetector — MobileNetV2+LSTM on a 16-frame sliding window
  4. BehaviorAnalyzer — rule-based fusion of the above over time
                        (loitering, crouching, weapon-in-hand, facing camera)

When the AI view is being watched, every model's output is drawn onto a single
annotated frame (boxes + names + an activity banner + behavior alerts) and a
compact JSON status is published via frame_server.set_status() so the Flutter
app can show a live "suspicious" banner without reading pixels.
"""

import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


# ── Event snapshots ───────────────────────────────────────────────────────────
# Saved when an event fires so the app's history shows a real photo of the moment.

_snapshot_dir_ready = False


def _ensure_snapshot_dir() -> None:
    global _snapshot_dir_ready
    if not _snapshot_dir_ready:
        os.makedirs(config.EVENT_SNAPSHOT_DIR, exist_ok=True)
        _snapshot_dir_ready = True


def _prune_old_snapshots() -> None:
    """Delete snapshots older than the retention window so the disk can't fill."""
    try:
        cutoff = time.time() - config.EVENT_SNAPSHOT_RETENTION_DAYS * 86400
        for fn in os.listdir(config.EVENT_SNAPSHOT_DIR):
            if not fn.endswith(".jpg"):
                continue
            p = os.path.join(config.EVENT_SNAPSHOT_DIR, fn)
            try:
                if os.path.getmtime(p) < cutoff:
                    os.remove(p)
            except OSError:
                pass
    except FileNotFoundError:
        pass


def _draw_detections(img: np.ndarray, draw: dict | None) -> None:
    """Draw the boxes that triggered/were present in the event onto [img]."""
    if not draw:
        return
    for o in draw.get("objects", []):
        _draw_box(img, o["xyxy"], _COLOR_OBJECT, f"{o['name']} {o['conf']*100:.0f}%")
    for p in draw.get("persons", []):
        _draw_box(img, p["xyxy"], _COLOR_PERSON, f"person {p['conf']*100:.0f}%")
    for f in draw.get("faces", []):
        col = _COLOR_FACE_KNOWN if f.get("known") else _COLOR_FACE_UNKNOWN
        _draw_box(img, f["box"], col, f.get("name", "?"))
    for w in draw.get("weapons", []):  # weapons last/on top, thicker
        _draw_box(img, w["xyxy"], _COLOR_WEAPON, f"{w['name']} {w['conf']*100:.0f}%", thick=3)


def capture_event_snapshot(frame_bgr: np.ndarray, camera_name: str,
                           label: str = "", draw: dict | None = None) -> str | None:
    """Build a captioned JPEG of the event moment (with detection rectangles),
    upload it to R2 and return its public URL. Falls back to local serving
    (frame server /ai/event) when R2 isn't configured."""
    try:
        import r2_client

        img = frame_bgr.copy()
        h, w = img.shape[:2]
        # Detection rectangles first, then the caption bar on top.
        _draw_detections(img, draw)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        caption = f"{ts}  |  {camera_name}"
        if label:
            caption += f"  |  {label}"
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, caption, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        name = f"{datetime.now():%Y%m%d}_{uuid.uuid4().hex[:12]}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        data = buf.tobytes()

        # Prefer R2 (cloud storage); fall back to local file served by frame_server.
        url = r2_client.upload_bytes(data, f"events/{name}", "image/jpeg")
        if url:
            return url
        _ensure_snapshot_dir()
        with open(os.path.join(config.EVENT_SNAPSHOT_DIR, name), "wb") as f:
            f.write(data)
        return f"{config.EVENT_SNAPSHOT_BASE_URL}/{name}"
    except Exception as e:
        logger.warning("Failed to save event snapshot: %s", e)
        return None


def record_event_clip(rtsp_url: str, seconds: int) -> str | None:
    """Record a short clip from the camera's RTSP feed (no re-encode) and upload
    it to R2. Returns the public URL, or None on failure. Blocking — call from a
    background thread so detection keeps running."""
    import subprocess

    import r2_client

    try:
        _ensure_snapshot_dir()
        name = f"{datetime.now():%Y%m%d}_{uuid.uuid4().hex[:12]}.mp4"
        path = os.path.join(config.EVENT_SNAPSHOT_DIR, name)
        cmd = [
            "ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", rtsp_url,
            "-t", str(seconds), "-c", "copy", "-an",
            "-movflags", "+faststart", path,
        ]
        subprocess.run(cmd, timeout=seconds + 25,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            logger.warning("Event clip recording produced no usable file")
            return None
        url = r2_client.upload_file(path, f"clips/{name}", "video/mp4")
        try:
            os.remove(path)  # R2 is the source of truth; don't keep clips locally
        except OSError:
            pass
        return url
    except Exception as e:
        logger.warning("Failed to record/upload event clip: %s", e)
        return None


# ── Alert config → suppressed event types ─────────────────────────────────────
# Maps the 5 app alert toggles onto the backend EventType values they govern.
# FaceRecognized (a positive event) and WeaponDetected (always safety-critical)
# are never suppressed.
_ALERT_TOGGLE_EVENTS: dict[str, set[str]] = {
    "unknownPersonEnabled": {"UnknownFace", "RecurrentUnknownFace", "LowConfidenceFace"},
    "forcedAccessEnabled":  {"ForcedAccessAttempt", "LockpickingAttempt"},
    "tailgatingEnabled":    {"Tailgating"},
    "climbingEnabled":      {"Climbing"},
    "aggressionEnabled":    {"PhysicalAggression", "Assault", "Abuse", "Robbery",
                             "Stealing", "Vandalism", "Burglary", "Arson", "Arrest"},
}


def disabled_event_types(alert_config: dict | None) -> set[str]:
    """Event types the household has turned OFF. None/empty → suppress nothing."""
    if not alert_config:
        return set()
    disabled: set[str] = set()
    for toggle, events in _ALERT_TOGGLE_EVENTS.items():
        if alert_config.get(toggle, True) is False:
            disabled |= events
    return disabled

# ── COCO class IDs of security-relevant objects ───────────────────────────────
# yolov8n is trained on COCO 80 classes. Guns are NOT in COCO so we use
# knife/baseball bat/scissors as a proxy for close-range weapon detection.
_WEAPON_CLASSES = {
    43: "knife",
    34: "baseball bat",
    76: "scissors",
}
_PERSON_CLASS = 0

# Visibility threshold: draw boxes at a low conf so held/occluded objects show.
# Alarm threshold: only raise a WeaponDetected EVENT at a higher conf.
_YOLO_DRAW_CONF = 0.35
_WEAPON_ALARM_CONF = 0.45


def _confident_persons(persons: list[dict], frame_shape) -> list[dict]:
    """Subset of person detections solid enough to drive EVENT logic (loitering,
    activity, face recognition). Filters out low-confidence / tiny (far) blobs
    that cause false 'unknown person' and 'prowler' events."""
    h = frame_shape[0]
    out = []
    for p in persons:
        if p["conf"] < config.PERSON_EVENT_CONFIDENCE:
            continue
        _, y1, _, y2 = p["xyxy"]
        if (y2 - y1) < config.PERSON_MIN_HEIGHT_FRAC * h:
            continue
        out.append(p)
    return out

# Colors are BGR (OpenCV).
_COLOR_PERSON = (90, 220, 90)
_COLOR_WEAPON = (40, 40, 235)
_COLOR_OBJECT = (235, 200, 70)
_COLOR_FACE_KNOWN = (90, 220, 90)
_COLOR_FACE_UNKNOWN = (40, 40, 235)
_COLOR_OK = (90, 200, 90)
_COLOR_WARN = (40, 40, 235)

# ── Activity model output → backend EventType mapping ────────────────────────
_ACTIVITY_EVENT_MAP = {
    "Vandalism":     ("ForcedAccessAttempt", "High"),
    "Stealing":      ("Stealing",            "High"),
    "Shoplifting":   ("Shoplifting",         "Medium"),
    "Shooting":      ("PhysicalAggression",  "Critical"),
    "Robbery":       ("Robbery",             "Critical"),
    "Roadaccidents": ("Roadaccidents",       "High"),
    "Fighting":      ("PhysicalAggression",  "High"),
    "Explosion":     ("Explosion",           "Critical"),
    "Burglary":      ("Burglary",            "Critical"),
    "Assault":       ("Assault",             "Critical"),
    "Arson":         ("Arson",               "Critical"),
    "Arrest":        ("Arrest",              "Medium"),
    "Abuse":         ("Abuse",               "High"),
}

# ── Risk level ranking (for combining multiple signals) ───────────────────────
_RISK_RANK = {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}


def _is_nighttime() -> bool:
    h = datetime.now().hour
    return h >= 22 or h < 6


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _center_inside(inner: tuple, outer: tuple) -> bool:
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


# ── Drawing helpers (ASCII-only labels — cv2 Hershey fonts can't render tildes) ─

def _draw_box(img, xyxy, color, label, thick=2):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, ly), (x1 + tw + 6, ly + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 3, ly + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_activity_banner(img, status: dict):
    """Top banner: current activity prediction; red when suspicious."""
    label = status.get("label", "—")
    conf = status.get("confidence", 0.0)
    suspicious = status.get("suspicious", False)
    color = _COLOR_WARN if suspicious else _COLOR_OK
    text = f"ACTIVIDAD: {label} {conf * 100:.0f}%"
    if suspicious:
        text += "  SOSPECHOSO"
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (w, 34), color, 2)
    cv2.putText(img, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def _draw_alerts(img, alerts: list[tuple[str, str]]):
    """Bottom-left stack of behavior alerts (text, risk)."""
    y = img.shape[0] - 12
    for text, risk in reversed(alerts):
        color = _COLOR_WARN if risk in ("High", "Critical") else (60, 180, 235)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (8, y - th - 6), (8 + tw + 10, y + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y -= th + 14


class ActivityDetector:
    """MobileNetV2 + LSTM suspicious activity classifier."""

    WINDOW_FRAMES = 16
    STEP_FRAMES = 8
    IMG_SIZE = 224

    def __init__(self):
        # The trained model is a Keras-2 SavedModel (saved_model.pb +
        # keras_metadata.pb + variables/). The TensorFlow bundled here uses
        # Keras 3, whose load_model() refuses SavedModel directories. The
        # tf_keras (Keras 2) compatibility package IS installed and loads it
        # as a full Model with .predict(), so prefer it.
        try:
            import tf_keras as keras
            from tf_keras.applications.mobilenet_v2 import preprocess_input
        except ImportError:  # fallback if tf_keras isn't present
            from tensorflow import keras  # type: ignore
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

        self._preprocess = preprocess_input
        meta_path = Path(config.ACTIVITY_MODEL_META)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {meta_path}\n"
                "Run the training notebook (suspicious_activity_training.ipynb) first."
            )
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self._class_names: dict[int, str] = {int(k): v for k, v in meta["class_names"].items()}
        self._suspicious_ids: set[int] = set(meta["suspicious_ids"])
        # Drop excluded labels (e.g. Shoplifting) from BOTH the suspicious set and
        # the pool of predictable classes, so they never alarm nor get displayed.
        excluded = getattr(config, "ACTIVITY_EXCLUDED_LABELS", set())
        self._excluded_ids: set[int] = {i for i, n in self._class_names.items() if n in excluded}
        self._suspicious_ids -= self._excluded_ids
        self._model = keras.models.load_model(config.ACTIVITY_MODEL_PATH)
        logger.info("ActivityDetector loaded — %d classes, %d suspicious, %d excluded (%s)",
                    len(self._class_names), len(self._suspicious_ids),
                    len(self._excluded_ids), ", ".join(sorted(excluded)) or "none")

        self._buffer: deque[np.ndarray] = deque(maxlen=self.WINDOW_FRAMES)
        self._since_last = 0
        # Consecutive inferences that looked like a real alarm (person in frame +
        # high confidence). An activity event is only emitted once this clears the
        # configured minimum, which kills the model's transient false spikes.
        self._streak = 0
        # Latest prediction, exposed for the on-frame banner / app status. Holds
        # between inferences (which only happen every STEP_FRAMES frames).
        self.last_status: dict = {"label": "—", "confidence": 0.0, "suspicious": False}

    def push_frame(self, bgr_frame: np.ndarray, persons_present: bool = False) -> dict | None:
        """Feed one BGR frame; returns event dict when a suspicious clip alarms."""
        rgb = cv2.resize(
            cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB),
            (self.IMG_SIZE, self.IMG_SIZE),
        )
        self._buffer.append(rgb.astype(np.float32))
        self._since_last += 1

        if len(self._buffer) < self.WINDOW_FRAMES or self._since_last < self.STEP_FRAMES:
            return None

        self._since_last = 0
        return self._infer(persons_present)

    def _infer(self, persons_present: bool = False) -> dict | None:
        frames = self._preprocess(np.array(self._buffer).copy())
        probs = self._model.predict(np.expand_dims(frames, 0), verbose=0)[0]
        # Pick the highest-probability class that ISN'T excluded, so an excluded
        # label (e.g. Shoplifting) never becomes the prediction.
        order = np.argsort(probs)[::-1]
        top_class = next((int(c) for c in order if int(c) not in self._excluded_ids),
                         int(order[0]))
        confidence = float(probs[top_class])
        raw_name = self._class_names.get(top_class, str(top_class))
        is_suspicious = top_class in self._suspicious_ids

        # Always update the live status so the app/banner reflects the model.
        self.last_status = {
            "label": raw_name,
            "confidence": round(confidence, 3),
            "suspicious": bool(is_suspicious),
        }

        logger.info(
            "ActivityModel: '%s' (%.0f%%)%s",
            raw_name, confidence * 100,
            "  ⚠ SUSPICIOUS" if is_suspicious else "",
        )

        # ── Hard event gate (config.ACTIVITY_EVENT_*) ─────────────────────────
        # The banner above still shows whatever the model predicts, but to PERSIST
        # an event we require a person in frame + high confidence + persistence
        # across consecutive inferences. This is the fix for the domestic-scene
        # false "Burglary/Robbery" spam (the model is noisy on empty/indoor scenes).
        alarm = (is_suspicious
                 and persons_present
                 and confidence >= config.ACTIVITY_EVENT_CONFIDENCE)
        self._streak = self._streak + 1 if alarm else 0
        if not alarm or self._streak < config.ACTIVITY_EVENT_MIN_STREAK:
            if is_suspicious:
                logger.info("  → suppressed (conf=%.0f%% person=%s streak=%d/%d)",
                            confidence * 100, persons_present,
                            self._streak, config.ACTIVITY_EVENT_MIN_STREAK)
            return None

        event_type, risk_level = _ACTIVITY_EVENT_MAP.get(raw_name, ("PhysicalAggression", "High"))

        return {
            "event_type": event_type,
            "confidence_score": confidence,
            "risk_level": risk_level,
            "is_nighttime": _is_nighttime(),
            "person_name": None,
            "source": "activity",
        }


class YoloDetector:
    """YOLOv8 object detector — returns structured boxes for persons/weapons/objects."""

    def __init__(self):
        from ultralytics import YOLO
        logger.info("Loading YOLO model: %s", config.YOLO_MODEL)
        self._model = YOLO(config.YOLO_MODEL)
        self._names = self._model.names
        logger.info("YoloDetector ready")

    def detect(self, bgr_frame: np.ndarray) -> list[dict]:
        """Return a list of detection dicts:
           {xyxy:(x1,y1,x2,y2), name, conf, kind:'person'|'weapon'|'object'}.
        Low draw-conf so held/occluded objects (e.g. a knife) still appear.
        """
        results = self._model(bgr_frame, verbose=False, conf=_YOLO_DRAW_CONF)
        out: list[dict] = []
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            if cls_id == _PERSON_CLASS:
                kind, name = "person", "person"
            elif cls_id in _WEAPON_CLASSES:
                kind, name = "weapon", _WEAPON_CLASSES[cls_id]
            else:
                name = self._names.get(cls_id, str(cls_id))
                # Drop generic objects not on the allowlist so misclassifications
                # (door→refrigerator, street→bench, cars, etc.) don't clutter.
                if name not in config.OBJECT_ALLOWLIST:
                    continue
                kind = "object"
            out.append({"xyxy": (x1, y1, x2, y2), "name": name, "conf": conf, "kind": kind})
        return out

    @staticmethod
    def weapon_event(boxes: list[dict]) -> dict | None:
        """Raise a WeaponDetected event if a weapon box is confident enough."""
        for box in boxes:
            if box["kind"] == "weapon" and box["conf"] >= _WEAPON_ALARM_CONF:
                logger.info("YOLO: %s detected (conf=%.2f)", box["name"], box["conf"])
                return {
                    "event_type": "WeaponDetected",
                    "confidence_score": box["conf"],
                    "risk_level": "Critical",
                    "is_nighttime": _is_nighttime(),
                    "person_name": None,
                    "source": "yolo",
                }
        return None


class FaceRecognizer:
    """DeepFace/ArcFace face recognizer.

    Only looks for faces INSIDE confident YOLO person boxes: the person region is
    cropped and upscaled before running the face detector, which (a) lets ArcFace
    work on far/small faces and (b) eliminates the false 'unknown person' events
    the old whole-frame Haar detector produced on street/night texture. An
    UnknownFace event is only raised after the unknown persists across a couple of
    recognition passes.
    """

    DEEPFACE_MODEL = "ArcFace"

    def __init__(self, household_id: str, faces_dir: Path):
        self._household_id = household_id
        self._faces_dir = faces_dir / household_id
        self._faces_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._run_every = 5  # run recognition every Nth frame (only while a person is present)
        self._cached_faces: list[dict] = []
        self._unknown_streak = 0
        self._backend = config.FACE_DETECTOR_BACKEND
        logger.info("FaceRecognizer (ArcFace / %s) ready for household %s (faces: %s)",
                    self._backend, household_id, self._faces_dir)

    @property
    def faces(self) -> list[dict]:
        """Last known faces: list of {box:(x1,y1,x2,y2), name, known:bool}."""
        return self._cached_faces

    def push_frame(self, bgr_frame: np.ndarray, persons: list[dict]) -> None:
        """Refresh recognized faces (throttled) inside the given person boxes. The
        EventDetector reads .faces and drives known/unknown events via the person
        tracker — this method no longer decides events itself."""
        if not persons:
            self._cached_faces = []
            return
        self._frame_count += 1
        if self._frame_count % self._run_every != 0:
            return
        self._recognize(bgr_frame, persons)

    def _has_known(self) -> bool:
        return any(self._faces_dir.rglob("*.jpg")) or any(self._faces_dir.rglob("*.png"))

    def _crop_person(self, frame: np.ndarray, person: dict):
        """Crop the person region (padded) and upscale small crops. Returns
        (crop, ox, oy, scale) or None."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in person["xyxy"])
        pad_x = int(0.08 * (x2 - x1)); pad_y = int(0.08 * (y2 - y1))
        x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        scale = 1.0
        if 0 < ch < config.FACE_CROP_UPSCALE_TO:
            scale = config.FACE_CROP_UPSCALE_TO / ch
            crop = cv2.resize(crop, (max(1, int(cw * scale)), int(ch * scale)),
                              interpolation=cv2.INTER_CUBIC)
        return crop, x1, y1, scale

    def _faces_in_crop(self, crop: np.ndarray, has_known: bool) -> list[dict]:
        """Detect (and, if enrolled, recognize) faces within one person crop.
        Boxes are in crop-pixel coordinates."""
        from deepface import DeepFace
        found: list[dict] = []
        if has_known:
            try:
                dfs = DeepFace.find(
                    img_path=crop, db_path=str(self._faces_dir),
                    model_name=self.DEEPFACE_MODEL, detector_backend=self._backend,
                    enforce_detection=False, silent=True,
                )
            except Exception:
                dfs = []
            for df in dfs:
                if df is None or getattr(df, "empty", True):
                    continue
                row = df.iloc[0]
                sx = int(row.get("source_x", 0)); sy = int(row.get("source_y", 0))
                sw = int(row.get("source_w", 0)); sh = int(row.get("source_h", 0))
                if sw <= 0 or sh <= 0:
                    continue
                distance = float(row.get("distance", 1.0))
                name = Path(str(row.get("identity", ""))).parent.name or "?"
                known = distance <= config.FACE_DISTANCE_THRESHOLD
                found.append({"box": (sx, sy, sx + sw, sy + sh),
                              "name": name if known else "Desconocido",
                              "known": known, "distance": distance})
        # Detect any (unmatched → unknown) faces.
        try:
            extracted = DeepFace.extract_faces(
                img_path=crop, detector_backend=self._backend, enforce_detection=False)
        except Exception:
            extracted = []
        for fa in extracted:
            if fa.get("confidence", 1.0) < config.FACE_MIN_DETECT_CONFIDENCE:
                continue
            area = fa.get("facial_area", {})
            box = (int(area.get("x", 0)), int(area.get("y", 0)),
                   int(area.get("x", 0) + area.get("w", 0)),
                   int(area.get("y", 0) + area.get("h", 0)))
            if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
                continue
            if any(_iou(box, f["box"]) > 0.3 for f in found):
                continue  # already covered
            found.append({"box": box, "name": "Desconocido", "known": False, "distance": 1.0})
        return found

    def _recognize(self, bgr_frame: np.ndarray, persons: list[dict]) -> None:
        """Detect + recognize faces inside each person crop; update self.faces."""
        faces: list[dict] = []
        has_known = self._has_known()
        for person in persons:
            cropped = self._crop_person(bgr_frame, person)
            if cropped is None:
                continue
            crop, ox, oy, scale = cropped
            try:
                for f in self._faces_in_crop(crop, has_known):
                    bx1, by1, bx2, by2 = f["box"]
                    fb = (int(ox + bx1 / scale), int(oy + by1 / scale),
                          int(ox + bx2 / scale), int(oy + by2 / scale))
                    faces.append({"box": fb, "name": f["name"],
                                  "known": f["known"], "distance": f["distance"]})
            except Exception as exc:
                logger.debug("FaceRecognizer crop error: %s", exc)
        self._cached_faces = faces


class BehaviorAnalyzer:
    """Rule-based fusion of YOLO + face signals over time → suspicious behavior.

    Receives only CONFIDENT persons (see _confident_persons), so the loitering /
    posture rules no longer fire on distant noise blobs.
    """

    PERSON_GAP_RESET = 6.0   # seconds with no person → reset loiter timer

    def __init__(self):
        self._person_since: float | None = None
        self._last_person_seen = 0.0

    def analyze(self, persons, weapons, faces, frame_shape) -> tuple[list[tuple[str, str]], list[str]]:
        """Returns (alerts[(text, risk)], armed_weapon_names)."""
        alerts: list[tuple[str, str]] = []
        now = time.monotonic()
        H, W = frame_shape[:2]

        # ── Loitering: a person continuously present beyond a dwell threshold ──
        if persons:
            self._last_person_seen = now
            if self._person_since is None:
                self._person_since = now
            dwell = now - self._person_since
            if dwell >= config.LOITER_SECONDS:
                alerts.append((f"Merodeo {int(dwell)}s", "Medium"))
        elif now - self._last_person_seen > self.PERSON_GAP_RESET:
            self._person_since = None

        # ── Group: several people at once (possible forced entry / crowd) ──────
        if len(persons) >= 3:
            alerts.append((f"Grupo de {len(persons)} personas", "Medium"))

        # ── Crouching / lying (clearly wider-than-tall person bbox) ───────────
        for p in persons:
            x1, y1, x2, y2 = p["xyxy"]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0 and (h / w) < 0.9:
                alerts.append(("Postura agachada", "Medium"))
                break

        # ── Weapon-in-hand: a weapon box overlapping a person ─────────────────
        armed: list[str] = []
        for wb in weapons:
            for p in persons:
                if _center_inside(wb["xyxy"], p["xyxy"]) or _iou(wb["xyxy"], p["xyxy"]) > 0.0:
                    armed.append(wb["name"])
                    break
        for nm in sorted(set(armed)):
            alerts.append((f"Persona armada: {nm}", "Critical"))

        # ── Facing camera: a large, roughly centered face ─────────────────────
        for f in faces:
            fx1, fy1, fx2, fy2 = f["box"]
            fw = fx2 - fx1
            cx = (fx1 + fx2) / 2
            if fw > 0.18 * W and 0.25 * W < cx < 0.75 * W:
                alerts.append(("Mirando a la camara", "Low"))
                break

        return alerts, sorted(set(armed))


class PersonTracker:
    """Tracks confident persons across frames (greedy IoU matching) so that:
      - a NEW person gets a grace window to be recognized before an UnknownFace
        alert fires (config.UNKNOWN_ALERT_GRACE_SECONDS) — time to walk up to the cam,
      - once recognized, a track KEEPS that identity even if the face is no longer
        visible,
      - each track alerts at most once (no alert/WhatsApp spam),
      - a person who leaves for > TTL and returns counts as a new person.
    """

    def __init__(self):
        self._tracks: dict[int, dict] = {}
        self._next_id = 1

    def update(self, persons: list[dict], now: float) -> dict[int, int]:
        """Match person boxes to tracks. Returns {person_index: track_id}."""
        for tid in [t for t, tr in self._tracks.items()
                    if now - tr["last_seen"] > config.PERSON_TRACK_TTL_SECONDS]:
            del self._tracks[tid]
        assign: dict[int, int] = {}
        used: set[int] = set()
        for i, p in enumerate(persons):
            best_iou, best_tid = 0.0, None
            for tid, tr in self._tracks.items():
                if tid in used:
                    continue
                iou = _iou(p["xyxy"], tr["bbox"])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None and best_iou >= config.PERSON_TRACK_IOU:
                tr = self._tracks[best_tid]
                tr["bbox"] = p["xyxy"]; tr["last_seen"] = now
                assign[i] = best_tid; used.add(best_tid)
            else:
                tid = self._next_id; self._next_id += 1
                self._tracks[tid] = {"bbox": p["xyxy"], "first_seen": now,
                                     "last_seen": now, "state": "pending",
                                     "name": None, "alerted": False}
                assign[i] = tid
        return assign

    def mark_known(self, tid: int, name: str) -> bool:
        """Mark a track recognized. Returns True only the first time (→ log once)."""
        tr = self._tracks.get(tid)
        if tr is None or tr["state"] == "known":
            return False
        tr["state"] = "known"; tr["name"] = name; tr["alerted"] = True
        return True

    def pending_unknown(self, now: float) -> list[int]:
        """Track ids unrecognized past the grace window → fire UnknownFace once."""
        out = []
        for tid, tr in self._tracks.items():
            if (tr["state"] == "pending" and not tr["alerted"]
                    and now - tr["first_seen"] >= config.UNKNOWN_ALERT_GRACE_SECONDS):
                tr["alerted"] = True; tr["state"] = "unknown"
                out.append(tid)
        return out

    def get(self, tid) -> dict | None:
        return self._tracks.get(tid)

    def any_unknown(self) -> bool:
        return any(tr["state"] == "unknown" for tr in self._tracks.values())


class EventDetector:
    """
    Orchestrates the pipeline for one camera stream.

    A per-camera, per-event-type cooldown (config.EVENT_TYPE_COOLDOWN_SECONDS,
    default 60s) prevents the same ongoing situation from producing duplicate
    alerts in quick succession.
    """

    def __init__(self, household_id: str, camera_id: str, camera_name: str):
        self.household_id = household_id
        self.camera_id = camera_id
        self.camera_name = camera_name

        logger.info("Initializing pipeline for camera '%s' (%s)", camera_name, camera_id)

        self._activity: ActivityDetector | None = None
        if config.ACTIVITY_ENABLED:
            try:
                self._activity = ActivityDetector()
            except Exception as e:
                logger.error("ActivityDetector failed to load: %s", e)
                logger.warning("Activity detection disabled for camera '%s'", camera_name)
        else:
            logger.info("Activity model disabled (ACTIVITY_ENABLED=false)")

        self._yolo: YoloDetector | None = None
        try:
            self._yolo = YoloDetector()
        except Exception as e:
            logger.error("YoloDetector failed to load: %s", e)
            logger.warning("YOLO detection disabled for camera '%s'", camera_name)

        self._face: FaceRecognizer | None = None
        try:
            self._face = FaceRecognizer(household_id, config.FACES_DIR)
        except Exception as e:
            logger.error("FaceRecognizer failed to load: %s", e)
            logger.warning("Face recognition disabled for camera '%s'", camera_name)

        self._behavior = BehaviorAnalyzer()
        self._tracker = PersonTracker()
        self._last_event_time: dict[str, float] = {}
        # Latest detections, kept so an event snapshot can be drawn with the boxes
        # that triggered it (persons/weapons/objects/faces).
        self.last_draw: dict | None = None

        logger.info(
            "[%s] Pipeline ready — Activity:%s  YOLO:%s  Face:%s  Behavior:ON",
            camera_name,
            "ON" if self._activity else "OFF",
            "ON" if self._yolo else "OFF",
            "ON" if self._face else "OFF",
        )

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Run all models on one frame; returns list of events to ingest."""
        import frame_server as _fs
        watched = _fs.is_watched(self.camera_id)
        candidates: list[dict] = []

        # ── YOLO (always — needed for alarms + behavior) ──────────────────────
        boxes: list[dict] = self._yolo.detect(frame) if self._yolo else []
        persons = [b for b in boxes if b["kind"] == "person"]
        weapons = [b for b in boxes if b["kind"] == "weapon"]
        objects = [b for b in boxes if b["kind"] == "object"]
        # Confident, close-enough persons drive ALL event logic (loitering,
        # activity, face recognition). Low-conf / far blobs are drawn but ignored.
        event_persons = _confident_persons(persons, frame.shape)
        if self._yolo:
            ev = self._yolo.weapon_event(boxes)
            if ev:
                candidates.append(ev)

        # ── Person tracking (grace window + persistent identity + no spam) ────
        now = time.monotonic()
        assign = self._tracker.update(event_persons, now)

        # ── Face recognition (throttled; only inside confident person boxes) ──
        if self._face:
            self._face.push_frame(frame, event_persons)
        faces = self._face.faces if self._face else []

        # A known face inside a person box makes that track that identity (once →
        # FaceRecognized). The identity PERSISTS on the track even if the face is
        # later not visible (person stays "known" while on screen).
        for f in faces:
            if not f.get("known"):
                continue
            fcx = (f["box"][0] + f["box"][2]) / 2
            fcy = (f["box"][1] + f["box"][3]) / 2
            for i, p in enumerate(event_persons):
                x1, y1, x2, y2 = p["xyxy"]
                if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                    tid = assign.get(i)
                    if tid is not None and self._tracker.mark_known(tid, f["name"]):
                        candidates.append({
                            "event_type": "FaceRecognized",
                            "confidence_score": round(1.0 - f.get("distance", 0.0), 3),
                            "risk_level": "None", "is_nighttime": _is_nighttime(),
                            "person_name": f["name"], "source": "face"})
                    break

        # Unknown-person alert: a NEW person still unrecognized after the grace
        # window fires exactly ONE UnknownFace (no spam; re-fires only for a
        # genuinely new person).
        for _tid in self._tracker.pending_unknown(now):
            candidates.append({
                "event_type": "UnknownFace", "confidence_score": 0.85,
                "risk_level": "Medium", "is_nighttime": _is_nighttime(),
                "person_name": None, "source": "track-unknown"})

        # Per-person label from the track (drawn even when no face is visible).
        person_labels: dict[int, tuple] = {}
        for i, p in enumerate(event_persons):
            tr = self._tracker.get(assign.get(i))
            if tr and tr.get("name"):
                person_labels[i] = (tr["name"], _COLOR_FACE_KNOWN)
            elif tr and tr["state"] == "unknown":
                person_labels[i] = ("Desconocido", _COLOR_FACE_UNKNOWN)
            else:
                person_labels[i] = ("Identificando...", _COLOR_OBJECT)

        # Keep the current detections for event snapshots (drawn with boxes).
        self.last_draw = {
            "persons": persons, "weapons": weapons, "objects": objects, "faces": faces,
        }

        # ── Behavior fusion ───────────────────────────────────────────────────
        alerts, armed = self._behavior.analyze(event_persons, weapons, faces, frame.shape)
        if any(a[0].startswith("Merodeo") for a in alerts):
            candidates.append({
                "event_type": "Tailgating", "confidence_score": 0.75,
                "risk_level": "Medium", "is_nighttime": _is_nighttime(),
                "person_name": None, "source": "behavior-loiter"})

        overall_suspicious = bool(
            weapons or armed
            or self._tracker.any_unknown()
            or any(a[1] in ("High", "Critical") for a in alerts))

        # ── Publish status for the app's live banner ──────────────────────────
        _fs.set_status(self.camera_id, {
            "state": "ok",
            "persons": len(event_persons),
            "objects": sorted({o["name"] for o in objects} | {w["name"] for w in weapons}),
            "faces": [{"name": f["name"], "known": f["known"]} for f in faces],
            "alerts": [a[0] for a in alerts],
            "suspicious": overall_suspicious,
        })

        # ── Draw the annotated frame only when someone is watching ────────────
        if watched:
            annotated = frame.copy()
            for i, p in enumerate(event_persons):
                text, col = person_labels.get(i, ("persona", _COLOR_PERSON))
                _draw_box(annotated, p["xyxy"], col, text)
            for o in objects:
                _draw_box(annotated, o["xyxy"], _COLOR_OBJECT, f"{o['name']} {o['conf']*100:.0f}%")
            for w in weapons:  # weapons last so their red box sits on top
                _draw_box(annotated, w["xyxy"], _COLOR_WEAPON, f"{w['name']} {w['conf']*100:.0f}%", thick=3)
            for f in faces:
                col = _COLOR_FACE_KNOWN if f["known"] else _COLOR_FACE_UNKNOWN
                _draw_box(annotated, f["box"], col, f["name"], thick=2)
            _draw_alerts(annotated, alerts)
            try:
                h, w = annotated.shape[:2]
                if w > 800:
                    annotated = cv2.resize(annotated, (800, int(h * 800 / w)),
                                           interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
                _fs.store_frame(self.camera_id, buf.tobytes())
            except Exception:
                pass

        # ── Cooldown + emit ───────────────────────────────────────────────────
        # De-dup on event_type (not source): the SAME ongoing situation must not
        # fire repeatedly. One ongoing fight = one Fighting event per cooldown.
        events = []
        for ev in candidates:
            ev.pop("source", None)
            etype = ev["event_type"]
            if self._in_cooldown(etype):
                continue
            self._reset_cooldown(etype)
            ev["household_id"] = self.household_id
            ev["camera_id"] = self.camera_id
            ev["camera_name"] = self.camera_name
            events.append(ev)
            logger.info(
                "[%s] Event: %s | risk=%s | conf=%.2f",
                self.camera_name, ev["event_type"], ev["risk_level"], ev["confidence_score"],
            )

        return events

    def _in_cooldown(self, key: str) -> bool:
        last = self._last_event_time.get(key, 0.0)
        return time.monotonic() - last < config.EVENT_TYPE_COOLDOWN_SECONDS

    def _reset_cooldown(self, key: str):
        self._last_event_time[key] = time.monotonic()
