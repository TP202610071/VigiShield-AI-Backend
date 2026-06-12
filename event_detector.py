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
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

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
_YOLO_DRAW_CONF = 0.30
_WEAPON_ALARM_CONF = 0.45

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
                kind, name = "object", self._names.get(cls_id, str(cls_id))
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
    """DeepFace/ArcFace face recognizer — returns recognized faces with boxes."""

    DEEPFACE_MODEL = "ArcFace"
    DEEPFACE_DETECTOR = "opencv"

    def __init__(self, household_id: str, faces_dir: Path):
        self._household_id = household_id
        self._faces_dir = faces_dir / household_id
        self._faces_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._run_every = 5  # run recognition every Nth frame
        # Cache of last recognized faces so we can keep drawing boxes between runs.
        self._cached_faces: list[dict] = []
        logger.info("FaceRecognizer (ArcFace) ready for household %s (faces: %s)",
                    household_id, self._faces_dir)

    @property
    def faces(self) -> list[dict]:
        """Last known faces: list of {box:(x1,y1,x2,y2), name, known:bool}."""
        return self._cached_faces

    def push_frame(self, bgr_frame: np.ndarray) -> dict | None:
        """Run recognition periodically; updates self.faces. Returns an event."""
        self._frame_count += 1
        if self._frame_count % self._run_every != 0:
            return None
        return self._recognize(bgr_frame)

    def _has_known(self) -> bool:
        return any(self._faces_dir.rglob("*.jpg")) or any(self._faces_dir.rglob("*.png"))

    def _recognize(self, bgr_frame: np.ndarray) -> dict | None:
        try:
            from deepface import DeepFace

            faces: list[dict] = []
            event: dict | None = None
            has_known = self._has_known()

            if has_known:
                dfs = DeepFace.find(
                    img_path=bgr_frame,
                    db_path=str(self._faces_dir),
                    model_name=self.DEEPFACE_MODEL,
                    detector_backend=self.DEEPFACE_DETECTOR,
                    enforce_detection=False,
                    silent=True,
                )
                matched_regions: list[tuple] = []
                for df in dfs:
                    if df.empty:
                        continue
                    row = df.iloc[0]
                    sx = int(row.get("source_x", 0)); sy = int(row.get("source_y", 0))
                    sw = int(row.get("source_w", 0)); sh = int(row.get("source_h", 0))
                    box = (sx, sy, sx + sw, sy + sh)
                    distance = float(row.get("distance", 1.0))
                    name = Path(row.get("identity", "")).parent.name or "?"
                    known = distance <= config.FACE_DISTANCE_THRESHOLD
                    faces.append({"box": box, "name": name if known else "Desconocido", "known": known})
                    matched_regions.append(box)
                    if known:
                        event = {
                            "event_type": "FaceRecognized",
                            "confidence_score": round(1.0 - distance, 3),
                            "risk_level": "None",
                            "is_nighttime": _is_nighttime(),
                            "person_name": name,
                            "source": "face",
                        }
                    elif event is None:
                        event = {
                            "event_type": "LowConfidenceFace",
                            "confidence_score": round(max(0.0, 1.0 - distance), 3),
                            "risk_level": "Low",
                            "is_nighttime": _is_nighttime(),
                            "person_name": name,
                            "source": "face",
                        }

            # Add boxes for any face present that wasn't matched above (unknowns).
            extracted = DeepFace.extract_faces(
                img_path=bgr_frame,
                detector_backend=self.DEEPFACE_DETECTOR,
                enforce_detection=False,
            )
            for fa in extracted:
                area = fa.get("facial_area", {})
                if fa.get("confidence", 1.0) < 0.5:
                    continue
                box = (int(area.get("x", 0)), int(area.get("y", 0)),
                       int(area.get("x", 0) + area.get("w", 0)),
                       int(area.get("y", 0) + area.get("h", 0)))
                if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
                    continue
                if any(_iou(box, b["box"]) > 0.3 for b in faces):
                    continue  # already covered by a matched face
                faces.append({"box": box, "name": "Desconocido", "known": False})
                if event is None:
                    event = {
                        "event_type": "UnknownFace",
                        "confidence_score": 0.80,
                        "risk_level": "Medium",
                        "is_nighttime": _is_nighttime(),
                        "person_name": None,
                        "source": "face",
                    }

            self._cached_faces = faces
            return event

        except Exception as exc:
            logger.debug("FaceRecognizer error: %s", exc)
            return None


class BehaviorAnalyzer:
    """Rule-based fusion of YOLO + face signals over time → suspicious behavior."""

    LOITER_SECONDS = 20.0
    PERSON_GAP_RESET = 5.0   # seconds with no person → reset loiter timer

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
            if dwell >= self.LOITER_SECONDS:
                alerts.append((f"Merodeo {int(dwell)}s", "Medium"))
        elif now - self._last_person_seen > self.PERSON_GAP_RESET:
            self._person_since = None

        # ── Crouching / unusual low posture (wide-ish person bbox) ────────────
        for p in persons:
            x1, y1, x2, y2 = p["xyxy"]
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0 and (h / w) < 1.15:
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
        try:
            self._activity = ActivityDetector()
        except Exception as e:
            logger.error("ActivityDetector failed to load: %s", e)
            logger.warning("Activity detection disabled for camera '%s'", camera_name)

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
        self._last_event_time: dict[str, float] = {}

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
        if self._yolo:
            ev = self._yolo.weapon_event(boxes)
            if ev:
                candidates.append(ev)

        # ── Face recognition (updates self._face.faces) ───────────────────────
        if self._face:
            ev = self._face.push_frame(frame)
            if ev:
                candidates.append(ev)
        faces = self._face.faces if self._face else []

        # ── Activity model ────────────────────────────────────────────────────
        if self._activity:
            ev = self._activity.push_frame(frame, persons_present=len(persons) > 0)
            if ev:
                candidates.append(ev)
        if self._activity:
            activity_status = dict(self._activity.last_status)
        else:
            activity_status = {"label": "n/a", "confidence": 0.0, "suspicious": False}
        # Gate the VISUAL suspicious flag so the red banner doesn't fire on an
        # idle scene: require decent confidence AND a person in frame (suspicious
        # *human* activity needs a human; the model is noisy on empty scenes).
        # The formal alarm/event still uses the stricter threshold in ActivityDetector.
        activity_status["suspicious"] = bool(
            activity_status.get("suspicious")
            and activity_status.get("confidence", 0.0) >= 0.50
            and len(persons) > 0
        )

        # ── Behavior fusion ───────────────────────────────────────────────────
        alerts, armed = self._behavior.analyze(persons, weapons, faces, frame.shape)
        # Loitering → a real (Tailgating) event so it logs + notifies.
        if any(a[0].startswith("Merodeo") for a in alerts):
            candidates.append({
                "event_type": "Tailgating",
                "confidence_score": 0.75,
                "risk_level": "Medium",
                "is_nighttime": _is_nighttime(),
                "person_name": None,
                "source": "behavior-loiter",
            })

        overall_suspicious = bool(
            activity_status.get("suspicious")
            or weapons or armed
            or any(a[1] in ("High", "Critical") for a in alerts)
        )

        # ── Publish status for the app's live banner ──────────────────────────
        _fs.set_status(self.camera_id, {
            "state": "ok",
            "activity": activity_status,
            "persons": len(persons),
            "objects": sorted({o["name"] for o in objects} | {w["name"] for w in weapons}),
            "faces": [{"name": f["name"], "known": f["known"]} for f in faces],
            "alerts": [a[0] for a in alerts],
            "suspicious": overall_suspicious,
        })

        # ── Draw the annotated frame only when someone is watching ────────────
        if watched:
            annotated = frame.copy()
            for p in persons:
                _draw_box(annotated, p["xyxy"], _COLOR_PERSON, f"persona {p['conf']*100:.0f}%")
            for o in objects:
                _draw_box(annotated, o["xyxy"], _COLOR_OBJECT, f"{o['name']} {o['conf']*100:.0f}%")
            for w in weapons:  # weapons last so their red box sits on top
                _draw_box(annotated, w["xyxy"], _COLOR_WEAPON, f"{w['name']} {w['conf']*100:.0f}%", thick=3)
            for f in faces:
                col = _COLOR_FACE_KNOWN if f["known"] else _COLOR_FACE_UNKNOWN
                _draw_box(annotated, f["box"], col, f["name"], thick=2)
            _draw_activity_banner(annotated, activity_status)
            _draw_alerts(annotated, alerts)
            try:
                # Downscale + compress so the phone can fetch frames reliably over
                # WiFi (a full 1080p JPEG is ~95 KB and stutters/drops; ~800px @ q55
                # is ~25 KB). The phone screen doesn't need more than this.
                h, w = annotated.shape[:2]
                if w > 800:
                    annotated = cv2.resize(annotated, (800, int(h * 800 / w)),
                                           interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 55])
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
