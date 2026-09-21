"""
VigiShield — Detector de comportamiento por pose (YOLOv8-Pose + Random Forest).

Corre YOLOv8-Pose sobre el frame, asocia cada esqueleto detectado a la persona
rastreada que le corresponde y mantiene UNA ventana deslizante POR PERSONA. Cada
ventana completa se clasifica como Normal/Sospechoso con el Random Forest
entrenado (behavior_rf_binary.pkl) y su salida (label + prob) la consume el
EventDetector como una señal más para el CAIEE.

Por qué por persona y no por cuadro: antes se tomaba sólo el esqueleto más
confiable del cuadro y su veredicto se aplicaba a TODA la escena. Con varias
personas eso mezclaba los keypoints de unas y otras dentro de la misma ventana
(basura para el clasificador) y contagiaba el veredicto: si una persona corría,
todas quedaban marcadas. Ahora cada persona acumula su propia ventana y recibe su
propio veredicto.

Diseñado para CPU: la pose se ejecuta una sola vez cada BEHAVIOR_RUN_EVERY frames
y sólo cuando hay personas en escena, para no saturar la VM.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config

logger = logging.getLogger(__name__)

# Un estado por persona se descarta si no se le asocia ningún esqueleto en este
# tiempo (alineado con la memoria del tracker).
_STATE_TTL_SECONDS = 30.0


def _persons_kpts(result, W: int, H: int):
    """[(bbox_px, kpts[17,3] normalizados)] de cada persona con pose confiable."""
    kp = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if kp is None or kp.data is None or len(kp.data) == 0:
        return []
    data = kp.data.cpu().numpy()  # [n,17,3]
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) == len(data):
        xyxy = boxes.xyxy.cpu().numpy()
        confs = (boxes.conf.cpu().numpy() if boxes.conf is not None
                 else data[:, :, 2].mean(axis=1))
    else:
        return []
    out = []
    for i in range(len(data)):
        if confs[i] < 0.35:
            continue
        person = data[i].copy()
        person[:, 0] /= max(W, 1); person[:, 1] /= max(H, 1)
        out.append((tuple(float(v) for v in xyxy[i]), person.astype(np.float32)))
    return out


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


@dataclass
class _PersonState:
    buf: deque = field(default_factory=deque)
    p_smooth: float = 0.0
    on_streak: int = 0
    off_streak: int = 0
    suspicious: bool = False
    last_seen: float = 0.0
    verdict: tuple = ("Normal", 0.0)


class PoseBehaviorDetector:
    """Un detector por cámara. update(frame, {track_id: bbox}) -> {track_id: (label, prob)}."""

    def __init__(self):
        import joblib
        from ultralytics import YOLO
        self._model = YOLO(config.BEHAVIOR_POSE_MODEL)
        self._clf = joblib.load(config.BEHAVIOR_MODEL_PATH)
        meta = json.loads(Path(config.BEHAVIOR_MODEL_META).read_text(encoding="utf-8"))
        self._window = int(meta.get("window", 16))
        self._classes = list(self._clf.classes_)
        self._states: dict[int, _PersonState] = {}
        self._n = 0
        self._last: dict[int, tuple] = {}
        logger.info("PoseBehaviorDetector listo — clases=%s, ventana=%d, cada %d frames, "
                    "umbral on=%.2f off=%.2f (on=%d ventanas, off=%d)",
                    self._classes, self._window, config.BEHAVIOR_RUN_EVERY,
                    config.BEHAVIOR_SUSPICIOUS_THRESHOLD, config.BEHAVIOR_NORMAL_THRESHOLD,
                    config.BEHAVIOR_ON_WINDOWS, config.BEHAVIOR_OFF_WINDOWS)

    def update(self, frame, track_boxes: dict[int, tuple], now: float) -> dict[int, tuple]:
        """
        track_boxes: {track_id: bbox en píxeles} de las personas presentes.
        Devuelve {track_id: (label, prob)} — el veredicto vigente de cada persona.
        """
        # Sin personas: olvida todo y no gasta CPU.
        if not track_boxes:
            self._states.clear(); self._last = {}; return self._last
        self._n += 1
        if self._n % config.BEHAVIOR_RUN_EVERY != 0:
            return self._last
        try:
            H, W = frame.shape[:2]
            res = self._model.predict(frame, verbose=False)[0]
            poses = _persons_kpts(res, W, H)

            # Asocia cada esqueleto a la persona rastreada con mayor solape.
            used: set[int] = set()
            for bbox, k in poses:
                best_tid, best_iou = None, 0.25
                for tid, tbox in track_boxes.items():
                    if tid in used:
                        continue
                    v = _iou(bbox, tbox)
                    if v > best_iou:
                        best_tid, best_iou = tid, v
                if best_tid is None:
                    continue
                used.add(best_tid)
                st = self._states.get(best_tid)
                if st is None:
                    st = self._states[best_tid] = _PersonState(buf=deque(maxlen=self._window))
                st.last_seen = now
                st.buf.append(k.reshape(-1))  # 51 valores
                if len(st.buf) >= self._window:
                    X = np.asarray(st.buf, dtype=np.float32).reshape(1, -1)
                    proba = self._clf.predict_proba(X)[0]
                    if "Sospechoso" in self._classes:
                        p = float(proba[self._classes.index("Sospechoso")])
                        st.verdict = self._debounce(st, p, best_tid)
                    else:  # multiclase: clase top y su probabilidad
                        j = int(np.argmax(proba))
                        st.verdict = (self._classes[j], float(proba[j]))

            for tid in [t for t, st in self._states.items()
                        if now - st.last_seen > _STATE_TTL_SECONDS]:
                del self._states[tid]
            self._last = {tid: st.verdict for tid, st in self._states.items()}
        except Exception as e:
            logger.debug("PoseBehavior update error: %s", e)
        return self._last

    def _debounce(self, st: _PersonState, p: float, tid: int) -> tuple[str, float]:
        """
        Convierte la probabilidad cruda de UNA ventana en un veredicto estable.

        El RF clasifica ventanas de 16 cuadros de forma independiente, así que su
        salida parpadea: una ventana dice "Sospechoso" y la siguiente "Normal"
        aunque la escena sea la misma. Aquí:
          - se suaviza la probabilidad (EMA) para que un pico aislado no decida,
          - encender exige BEHAVIOR_ON_WINDOWS ventanas seguidas sobre el umbral
            alto (menos falsos positivos: niños jugando, gestos bruscos),
          - apagar exige BEHAVIOR_OFF_WINDOWS ventanas seguidas bajo el umbral de
            salida (el veredicto no se cae en mitad de una intrusión).
        """
        st.p_smooth = p if st.p_smooth == 0.0 else st.p_smooth + (p - st.p_smooth) * 0.5
        ps = st.p_smooth
        if ps >= config.BEHAVIOR_SUSPICIOUS_THRESHOLD:
            st.on_streak += 1; st.off_streak = 0
        elif ps < config.BEHAVIOR_NORMAL_THRESHOLD:
            st.off_streak += 1; st.on_streak = 0
        else:  # zona intermedia: ni confirma ni desmiente, congela el veredicto
            st.on_streak = st.off_streak = 0

        if not st.suspicious and st.on_streak >= config.BEHAVIOR_ON_WINDOWS:
            st.suspicious = True
            logger.info("[behavior] track %s: Sospechoso CONFIRMADO (p=%.2f)", tid, ps)
        elif st.suspicious and st.off_streak >= config.BEHAVIOR_OFF_WINDOWS:
            st.suspicious = False
            logger.info("[behavior] track %s: vuelve a Normal (p=%.2f)", tid, ps)
        return ("Sospechoso" if st.suspicious else "Normal", ps)
