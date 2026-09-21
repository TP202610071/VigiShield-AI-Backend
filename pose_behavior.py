"""
VigiShield — Detector de comportamiento por pose (YOLOv8-Pose + Random Forest).

Corre YOLOv8-Pose sobre el frame, toma los 17 keypoints de la persona más
confiable, mantiene una ventana deslizante y clasifica Normal/Sospechoso con el
modelo Random Forest entrenado (behavior_rf_binary.pkl). Su salida (label + prob)
la consume el EventDetector como una señal más para el CAIEE.

Diseñado para CPU: se ejecuta cada BEHAVIOR_RUN_EVERY frames y solo cuando hay una
persona confiable en escena, para no saturar la VM.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)


def _best_person_kpts(result, W: int, H: int):
    """17 keypoints (x,y,conf) normalizados de la persona más confiable, o None."""
    kp = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if kp is None or kp.data is None or len(kp.data) == 0:
        return None
    data = kp.data.cpu().numpy()  # [n,17,3]
    if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(data):
        confs = boxes.conf.cpu().numpy()
    else:
        confs = data[:, :, 2].mean(axis=1)
    idx = int(np.argmax(confs))
    if confs[idx] < 0.35:
        return None
    person = data[idx].copy()
    person[:, 0] /= max(W, 1); person[:, 1] /= max(H, 1)
    return person.astype(np.float32)  # [17,3]


class PoseBehaviorDetector:
    """Un detector por cámara. update(frame, persons_present) -> (label, prob)."""

    def __init__(self):
        import joblib
        from ultralytics import YOLO
        self._model = YOLO(config.BEHAVIOR_POSE_MODEL)
        self._clf = joblib.load(config.BEHAVIOR_MODEL_PATH)
        meta = json.loads(Path(config.BEHAVIOR_MODEL_META).read_text(encoding="utf-8"))
        self._window = int(meta.get("window", 16))
        self._classes = list(self._clf.classes_)
        self._buf: deque = deque(maxlen=self._window)
        self._n = 0
        self._last = ("Normal", 0.0)
        logger.info("PoseBehaviorDetector listo — clases=%s, ventana=%d, cada %d frames",
                    self._classes, self._window, config.BEHAVIOR_RUN_EVERY)

    def update(self, frame, persons_present: bool):
        # Sin persona: limpia la ventana y no gasta CPU.
        if not persons_present:
            self._buf.clear(); self._last = ("Normal", 0.0); return self._last
        self._n += 1
        if self._n % config.BEHAVIOR_RUN_EVERY != 0:
            return self._last
        try:
            H, W = frame.shape[:2]
            res = self._model.predict(frame, verbose=False)[0]
            k = _best_person_kpts(res, W, H)
            if k is None:
                return self._last
            self._buf.append(k.reshape(-1))  # 51 valores
            if len(self._buf) < self._window:
                return self._last
            X = np.asarray(self._buf, dtype=np.float32).reshape(1, -1)  # 1 x (window*51)
            proba = self._clf.predict_proba(X)[0]
            if "Sospechoso" in self._classes:
                p = float(proba[self._classes.index("Sospechoso")])
                label = "Sospechoso" if p >= config.BEHAVIOR_SUSPICIOUS_THRESHOLD else "Normal"
                self._last = (label, p)
            else:  # multiclase: reporta la clase top y su prob
                j = int(np.argmax(proba))
                self._last = (self._classes[j], float(proba[j]))
        except Exception as e:
            logger.debug("PoseBehavior update error: %s", e)
        return self._last
