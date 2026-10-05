"""
La captura del evento debe mostrar el momento que lo explica.

El riesgo se acumula durante segundos. Cuando la alerta salta, la persona
puede haber salido ya del cuadro: la foto salia vacia y sin recuadros
mientras el clip -que si tiene pre-grabacion- mostraba el incidente.

    py -3 -m pytest tests/test_evidencia_evento.py
"""
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from event_detector import EventDetector  # noqa: E402


def _detector():
    """EventDetector sin modelos: solo se ejercita la memoria de evidencia."""
    d = EventDetector.__new__(EventDetector)
    d.last_draw = None
    d._evidencia = None
    return d


def _cuadro(valor):
    return np.full((4, 4, 3), valor, dtype=np.uint8)


def test_sin_evidencia_usa_el_cuadro_actual():
    d = _detector()
    actual = _cuadro(7)

    cuadro, draw = d.evidencia_del_evento(actual)

    assert cuadro is actual
    assert draw is None


def test_prefiere_el_cuadro_donde_si_habia_alguien():
    d = _detector()
    bueno = _cuadro(1)
    cajas = {"labeled": [{"xyxy": (0, 0, 2, 2), "text": "Desconocido", "color": (0, 0, 255)}]}
    d._evidencia = (time.monotonic(), bueno, cajas)

    # El cuadro del instante de la alerta ya esta vacio.
    cuadro, draw = d.evidencia_del_evento(_cuadro(9))

    assert cuadro is bueno
    assert draw["labeled"], "la captura debe llevar los recuadros que explican la alerta"


def test_una_evidencia_vieja_no_se_usa():
    """Mas atras que el clip ya no corresponde a lo que el usuario vera."""
    d = _detector()
    viejo = _cuadro(1)
    antiguedad = max(2.0, float(config.EVENT_CLIP_SECONDS)) + 5
    d._evidencia = (time.monotonic() - antiguedad, viejo, {"labeled": [{"x": 1}]})
    actual = _cuadro(9)

    cuadro, _ = d.evidencia_del_evento(actual)

    assert cuadro is actual


def test_la_ventana_coincide_con_la_del_clip():
    """Justo dentro de la ventana todavia vale."""
    d = _detector()
    bueno = _cuadro(1)
    dentro = max(2.0, float(config.EVENT_CLIP_SECONDS)) - 0.5
    d._evidencia = (time.monotonic() - dentro, bueno, {"labeled": [{"x": 1}]})

    cuadro, _ = d.evidencia_del_evento(_cuadro(9))

    assert cuadro is bueno
