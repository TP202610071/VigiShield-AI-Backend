"""
CAIEE — Context-Aware Intent Estimation Engine
Motor de Estimación de Intención con Contexto.

NO es un modelo de IA (no se entrena, no usa dataset ni GPU). Es un ALGORITMO:
una máquina de estados + scoring ponderado con acumulación temporal e histéresis.

Idea: un delito no ocurre de golpe, ocurre por ETAPAS. El CAIEE mantiene, por
persona rastreada, un puntaje de riesgo que SUBE cuando la evidencia se acumula
en el tiempo y en el contexto (zona, permanencia, identidad, arma, hora…), y
avisa ANTES de la culminación cuando el riesgo se sostiene alto.

Prioridad de diseño: CERTEZA y MÍNIMOS FALSOS POSITIVOS. Por eso:
  - el puntaje se INTEGRA en el tiempo (un pico aislado no dispara),
  - se exige CORROBORACIÓN de varias señales para llegar a riesgo alto,
  - una persona CONOCIDA (rostro autorizado) queda fuertemente atenuada,
  - hay HISTÉRESIS (umbral de entrada alto, de salida más bajo) para no parpadear,
  - dispara a lo más UNA alerta anticipatoria por persona (con cooldown).

El motor recibe, por cuadro y por track, un dict de EVIDENCIA (booleans/valores
ya calculados por el pipeline) y devuelve un evento anticipatorio cuando procede.
Es independiente del modelo de violencia: si algún día existe, entra como una
señal de evidencia más (`violence`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Pesos de evidencia (escala 0..100) ────────────────────────────────────────
# Cada señal aporta al "score instantáneo" E del cuadro. Están calibrados para que
# ninguna señal sola llegue al umbral de alerta: hace falta combinación sostenida.
W_UNKNOWN = 22.0          # persona desconocida (rostro no autorizado)
W_ENTRY_ZONE = 24.0       # está en zona de entrada (puerta/reja/ventana)
W_YARD_ZONE = 14.0        # está en jardín/patio (dentro del predio)
W_STREET_ZONE = -12.0     # está en la calle → tránsito esperado, atenúa
W_LOITER = 22.0           # merodeo (permanencia prolongada)
W_LOITER_IN_ENTRY = 12.0  # bonus si el merodeo es JUSTO en la entrada
W_WEAPON = 45.0           # arma cerca de la persona
W_GROUP = 10.0            # varias personas a la vez
W_NIGHT = 9.0             # horario nocturno
W_POSTURE = 15.0          # postura anómala (agachado/trepando)
W_APPROACH = 12.0         # se aproxima / cruzó hacia la entrada
W_MODEL = 40.0            # el modelo entrenado (pose+RF) clasificó "Sospechoso" (alta precisión)

# Atenuación cuando la persona está reconocida como autorizada.
KNOWN_ATTENUATION = 0.15

# Suavizado temporal: cuánto pesa el cuadro nuevo vs el histórico acumulado.
# Bajo => sube lento (más certeza, menos falsos positivos). Con FRAME_INTERVAL de
# 0.5s, ~0.18 tarda ~10-12s en acercarse a un E sostenido.
RISE_ALPHA = 0.18
FALL_ALPHA = 0.08   # baja más lento aún (evita perder una amenaza por un cuadro)

# Umbrales de estado con histéresis (entrar/salir).
TH_WATCH = 22.0
TH_SUSPECT = 42.0
TH_HIGH_ENTER = 68.0
TH_HIGH_EXIT = 52.0

# Cuántos cuadros consecutivos por encima del umbral alto para CONFIRMAR (sostén).
HIGH_CONFIRM_FRAMES = 3

# No repetir la alerta anticipatoria de un mismo track más seguido que esto.
INTENT_COOLDOWN_SECONDS = 120.0

# Estados de intención (orden = severidad).
S_CALM = "calm"
S_WATCH = "watch"
S_SUSPECT = "suspect"
S_HIGH = "high_risk"


@dataclass
class Evidence:
    """Señales ya calculadas por el pipeline para UNA persona en un cuadro."""
    is_unknown: bool = False
    is_known: bool = False
    zone_type: str | None = None      # 'door'|'gate'|'window'|'yard'|'street'|None
    in_entry_zone: bool = False
    loitering: bool = False
    weapon_near: bool = False
    group: bool = False
    night: bool = False
    anomalous_posture: bool = False
    approaching: bool = False
    violence: bool = False            # (compat) señal de violencia directa
    model_suspicious: bool = False    # el modelo entrenado (pose+RF) marcó "Sospechoso"
    model_conf: float = 0.0           # probabilidad del modelo (0..1)


@dataclass
class _TrackRisk:
    score: float = 0.0
    state: str = S_CALM
    high_streak: int = 0
    last_alert: float = 0.0
    peak_reasons: list[str] = field(default_factory=list)


class IntentEngine:
    """CAIEE para UNA cámara: mantiene el riesgo por track y decide alertas."""

    def __init__(self):
        self._tracks: dict[int, _TrackRisk] = {}

    def forget(self, active_track_ids: set[int]) -> None:
        """Olvida el estado de tracks que ya expiraron en el tracker."""
        for tid in [t for t in self._tracks if t not in active_track_ids]:
            del self._tracks[tid]

    # ── Scoring instantáneo ───────────────────────────────────────────────────
    def _instant_score(self, ev: Evidence) -> tuple[float, list[str]]:
        e = 0.0
        reasons: list[str] = []

        if ev.weapon_near:
            e += W_WEAPON; reasons.append("arma")
        if ev.is_unknown:
            e += W_UNKNOWN; reasons.append("desconocido")

        if ev.in_entry_zone:
            e += W_ENTRY_ZONE; reasons.append("en entrada")
        elif ev.zone_type == "yard":
            e += W_YARD_ZONE; reasons.append("en predio")
        elif ev.zone_type == "street":
            e += W_STREET_ZONE

        if ev.loitering:
            e += W_LOITER; reasons.append("merodeo")
            if ev.in_entry_zone:
                e += W_LOITER_IN_ENTRY
        if ev.anomalous_posture:
            e += W_POSTURE; reasons.append("postura")
        if ev.approaching:
            e += W_APPROACH; reasons.append("aproximación")
        if ev.group:
            e += W_GROUP; reasons.append("grupo")
        if ev.violence:
            e += W_WEAPON; reasons.append("violencia")  # trata violencia como crítica
        if ev.model_suspicious:
            # peso proporcional a la confianza del modelo (0.5..1.0 del peso base)
            e += W_MODEL * (0.5 + 0.5 * max(0.0, min(1.0, ev.model_conf)))
            reasons.append("modelo:sospechoso")
        if ev.night:
            e += W_NIGHT

        # Una persona autorizada casi nunca es amenaza: atenúa fuerte, SALVO que
        # haya un arma o el modelo detecte comportamiento sospechoso (una acción
        # violenta/peligrosa importa aunque la persona sea conocida).
        if ev.is_known and not ev.weapon_near and not ev.model_suspicious:
            e *= KNOWN_ATTENUATION
            reasons = [r for r in reasons if r != "arma"]

        return max(0.0, e), reasons

    # ── Actualización por cuadro ──────────────────────────────────────────────
    def update(self, track_id: int, ev: Evidence, now: float | None = None) -> dict | None:
        """
        Integra la evidencia del cuadro para un track. Devuelve un evento
        anticipatorio (dict) cuando el riesgo se CONFIRMA alto, o None.
        """
        now = time.monotonic() if now is None else now
        tr = self._tracks.setdefault(track_id, _TrackRisk())

        e, reasons = self._instant_score(ev)
        # Integración temporal asimétrica (sube lento, baja más lento).
        alpha = RISE_ALPHA if e >= tr.score else FALL_ALPHA
        tr.score += (e - tr.score) * alpha
        tr.score = max(0.0, min(100.0, tr.score))

        prev_state = tr.state
        tr.state = self._state_for(tr.score, prev_state)

        if reasons and tr.score >= TH_SUSPECT:
            tr.peak_reasons = reasons  # guarda las razones del momento de mayor riesgo

        # Confirmación por sostén: cuenta cuadros seguidos por encima del umbral alto.
        if tr.score >= TH_HIGH_ENTER:
            tr.high_streak += 1
        elif tr.score < TH_HIGH_EXIT:
            tr.high_streak = 0

        confirmed = tr.high_streak == HIGH_CONFIRM_FRAMES  # dispara UNA vez al confirmar
        cooled = now - tr.last_alert >= INTENT_COOLDOWN_SECONDS
        if confirmed and cooled:
            tr.last_alert = now
            reason_txt = ", ".join(tr.peak_reasons or reasons) or "escalamiento sostenido"
            risk = "Critical" if (ev.weapon_near or ev.violence or tr.score >= 82) else "High"
            logger.info("[CAIEE] Intento sospechoso confirmado (track %s, score=%.0f): %s",
                        track_id, tr.score, reason_txt)
            return {
                "event_type": "SuspiciousIntent",
                "confidence_score": round(min(0.99, tr.score / 100.0), 3),
                "risk_level": risk,
                "is_nighttime": ev.night,
                "person_name": None,
                "source": "caiee",
            }
        return None

    def _state_for(self, score: float, prev: str) -> str:
        # Histéresis en el salto a/desde HIGH.
        if prev == S_HIGH:
            if score >= TH_HIGH_EXIT:
                return S_HIGH
        elif score >= TH_HIGH_ENTER:
            return S_HIGH
        if score >= TH_SUSPECT:
            return S_SUSPECT
        if score >= TH_WATCH:
            return S_WATCH
        return S_CALM

    def state_of(self, track_id: int) -> tuple[str, float]:
        tr = self._tracks.get(track_id)
        return (tr.state, tr.score) if tr else (S_CALM, 0.0)

    def max_state(self) -> tuple[str, float]:
        """Estado/score más alto entre los tracks activos (para el banner)."""
        if not self._tracks:
            return (S_CALM, 0.0)
        tr = max(self._tracks.values(), key=lambda t: t.score)
        return (tr.state, tr.score)
