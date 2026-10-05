"""
La captura y el clip del evento deben mostrar el momento que lo explica.

El riesgo se acumula durante segundos. Cuando la alerta salta, la persona
puede haber salido ya del cuadro: la foto salia vacia y sin recuadros, y el
clip -codificado suponiendo una cadencia que la VM no alcanza- salia
acelerado y cortado justo en el instante de la alerta.

    py -3 -m pytest tests/test_evidencia_evento.py
"""
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from event_detector import (  # noqa: E402
    _COLOR_RISK_HIGH, EventDetector, puntuar_evidencia, retime_frames)

AHORA = 1_800_000_000.0


def _detector():
    """EventDetector sin modelos: solo se ejercita la memoria de evidencia."""
    d = EventDetector.__new__(EventDetector)
    d.last_draw = None
    d.last_frame_time = AHORA
    d._evidencias = deque(maxlen=48)
    d._clip_lock = threading.Lock()
    d._clip_buffer = deque(maxlen=100)
    return d


def _cuadro(valor, h=4):
    return np.full((h, h, 3), valor, dtype=np.uint8)


def _persona(color=(190, 190, 190), alto=2):
    return {"labeled": [{"xyxy": (0, 0, 2, alto), "text": "x", "color": color}]}


def _guardar(d, t, cuadro, draw):
    d._evidencias.append((t, cuadro, draw, puntuar_evidencia(draw, cuadro.shape[0])))


def test_sin_evidencia_usa_el_cuadro_actual():
    d = _detector()
    actual = _cuadro(7)
    cuadro, draw, momento = d.evidencia_del_evento(actual)
    assert cuadro is actual and draw is None and momento == AHORA


def test_prefiere_el_cuadro_donde_si_habia_alguien():
    d = _detector()
    bueno = _cuadro(1)
    _guardar(d, AHORA - 1, bueno, _persona())
    cuadro, draw, momento = d.evidencia_del_evento(_cuadro(9))
    assert cuadro is bueno
    assert draw["labeled"], "la captura debe llevar los recuadros que explican la alerta"
    assert momento == AHORA - 1, "la foto lleva la hora de SU cuadro"


def test_gana_el_cuadro_mas_explicativo_no_el_ultimo():
    """El último suele ser una espalda saliendo de plano."""
    d = _detector()
    riesgo = _cuadro(1)
    _guardar(d, AHORA - 3, riesgo, _persona(color=_COLOR_RISK_HIGH))
    _guardar(d, AHORA - 1, _cuadro(2), _persona())
    cuadro, _, _ = d.evidencia_del_evento(_cuadro(9))
    assert cuadro is riesgo


def test_una_cara_desconocida_gana_a_una_persona_sin_cara():
    d = _detector()
    cara = _cuadro(1)
    _guardar(d, AHORA - 4, cara, {**_persona(), "faces": [{"known": False}]})
    _guardar(d, AHORA - 1, _cuadro(2), _persona(color=_COLOR_RISK_HIGH))
    cuadro, _, _ = d.evidencia_del_evento(_cuadro(9))
    assert cuadro is cara


def test_una_evidencia_vieja_no_se_usa():
    """Más atrás que el clip ya no corresponde a lo que el usuario verá."""
    d = _detector()
    _guardar(d, AHORA - config.EVENT_CLIP_PRE_SECONDS - 5, _cuadro(1), _persona())
    actual = _cuadro(9)
    cuadro, _, _ = d.evidencia_del_evento(actual)
    assert cuadro is actual


def test_clip_frames_respeta_la_ventana():
    d = _detector()
    for i in range(20):
        d._clip_buffer.append((AHORA - 10 + i, _cuadro(i)))
    dentro = d.clip_frames(AHORA - 3, AHORA + 2)
    assert [t for t, _ in dentro] == [AHORA - 3 + k for k in range(6)]


def test_retime_conserva_la_duracion_real():
    """3 cuadros en 6 s a 2 fps reales salen como 6 s a la cadencia pedida,
    no como 0.75 s a 4 fps."""
    timed = [(AHORA, _cuadro(10, 40)), (AHORA + 3, _cuadro(20, 40)), (AHORA + 6, _cuadro(30, 40))]
    out = retime_frames(timed, fps=4, event_time=AHORA + 3)
    assert len(out) == 6 * 4 + 1
    # Cada instante muestra el último cuadro disponible en ese momento.
    assert out[0][8, 30, 0] == 10 and out[11][8, 30, 0] == 10
    assert out[12][8, 30, 0] == 20


def test_retime_marca_el_instante_del_evento():
    timed = [(AHORA, _cuadro(0, 40)), (AHORA + 2, _cuadro(0, 40))]
    out = retime_frames(timed, fps=2, event_time=AHORA + 1)
    # Borde rojo (BGR) a partir del evento, no antes.
    assert tuple(out[0][0, 0]) != (0, 0, 255)
    assert tuple(out[2][0, 0]) == (0, 0, 255)
