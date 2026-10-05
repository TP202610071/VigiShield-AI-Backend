"""
La IA debe quedarse con toda camara que pueda leer, venga de donde venga.

Una camara de telefono publica por WebRTC y no tiene IP, asi que el backend
le manda `rtspUrl` nulo y solo `mediaMtxRtspUrl`. Filtrar por `rtspUrl` la
descartaba: publicaba correctamente y nadie la leia nunca.

    py -3 -m pytest tests/test_seleccion_camaras.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_client  # noqa: E402


class _Respuesta:
    def __init__(self, datos):
        self._datos = datos

    def raise_for_status(self):
        pass

    def json(self):
        return self._datos


def _camaras(datos):
    with patch.object(api_client.requests, "get", return_value=_Respuesta(datos)):
        return api_client.get_all_cameras()


def test_acepta_la_camara_del_telefono():
    """Sin IP pero con ruta en MediaMTX: hay que procesarla."""
    movil = {"id": "1", "name": "Telefono", "rtspUrl": None,
             "mediaMtxRtspUrl": "rtsp://localhost:8554/8002373a2067"}

    assert _camaras([movil]) == [movil]


def test_acepta_la_camara_ip_de_siempre():
    ip = {"id": "2", "name": "Escritorio",
          "rtspUrl": "rtsp://192.168.1.82:554/11",
          "mediaMtxRtspUrl": "rtsp://localhost:8554/d68f5fae5a77"}

    assert _camaras([ip]) == [ip]


def test_descarta_la_que_no_se_puede_leer_por_ningun_lado():
    """Sin ninguna de las dos URLs no hay nada que abrir."""
    sin_fuente = {"id": "3", "name": "A medio configurar",
                  "rtspUrl": None, "mediaMtxRtspUrl": None}

    assert _camaras([sin_fuente]) == []


def test_conserva_las_utiles_y_descarta_el_resto():
    utiles = [
        {"id": "1", "rtspUrl": None, "mediaMtxRtspUrl": "rtsp://localhost:8554/a"},
        {"id": "2", "rtspUrl": "rtsp://192.168.1.82:554/11", "mediaMtxRtspUrl": None},
    ]
    inutil = {"id": "3", "rtspUrl": None, "mediaMtxRtspUrl": None}

    assert _camaras(utiles + [inutil]) == utiles
