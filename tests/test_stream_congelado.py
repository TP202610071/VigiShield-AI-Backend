"""
Si el stream cambia de resolución a mitad (el teléfono por WebRTC sube o baja
la calidad según la red), OpenCV deja de decodificar y entrega el mismo cuadro
una y otra vez: la vista de IA se congelaba. El lector debe notarlo y
reconectar.

    py -3 -m pytest tests/test_stream_congelado.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import stream_reader  # noqa: E402


class _Captura:
    """VideoCapture falso: el primero entrega siempre el mismo cuadro."""
    abiertas = 0

    def __init__(self, *_):
        _Captura.abiertas += 1
        self.n = 0
        self.congelada = _Captura.abiertas == 1

    def set(self, *_):
        return True

    def isOpened(self):
        return True

    def grab(self):
        return True

    def retrieve(self):
        self.n += 1
        valor = 7 if self.congelada else self.n % 250
        return True, np.full((72, 128, 3), valor, dtype=np.uint8)

    def release(self):
        pass


def test_cuadro_congelado_reconecta(monkeypatch):
    reloj = {"t": 0.0}
    monkeypatch.setattr(stream_reader.cv2, "VideoCapture", _Captura)
    monkeypatch.setattr(stream_reader.time, "monotonic", lambda: reloj.__setitem__("t", reloj["t"] + 0.05) or reloj["t"])
    monkeypatch.setattr(stream_reader.time, "sleep", lambda s: None)
    _Captura.abiertas = 0

    lector = stream_reader.StreamReader("rtsp://localhost:8554/prueba")
    lector.connect()
    cuadros = lector.frames(interval_seconds=0.2)
    distintos = set()
    for _ in range(80):
        distintos.add(int(next(cuadros)[0, 0, 0]))

    # Se reconectó una vez y, desde entonces, los cuadros vuelven a cambiar.
    assert _Captura.abiertas == 2
    assert len(distintos) > 10


def test_una_escena_normal_no_reconecta(monkeypatch):
    reloj = {"t": 0.0}

    class _Viva(_Captura):
        def __init__(self, *_):
            super().__init__()
            self.congelada = False

    monkeypatch.setattr(stream_reader.cv2, "VideoCapture", _Viva)
    monkeypatch.setattr(stream_reader.time, "monotonic", lambda: reloj.__setitem__("t", reloj["t"] + 0.05) or reloj["t"])
    _Captura.abiertas = 0

    lector = stream_reader.StreamReader("rtsp://localhost:8554/prueba")
    lector.connect()
    cuadros = lector.frames(interval_seconds=0.2)
    for _ in range(200):
        next(cuadros)
    assert _Captura.abiertas == 1
