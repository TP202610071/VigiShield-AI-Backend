"""
La lista de cámaras se relee cada pocos segundos (para que el video de ejemplo
o una cámara nueva empiecen a analizarse enseguida). Eso exige que:

- un fallo del backend no detenga los workers que ya corren,
- la misma cámara con otro stream reinicie su worker,
- los rostros no se descarguen en cada lectura.

    py -3 -m pytest tests/test_refresco_camaras.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api_client  # noqa: E402
import main  # noqa: E402


class _Worker:
    def __init__(self, camera):
        self.camera = camera
        self.detenido = False

    def start(self):
        pass

    def stop(self):
        self.detenido = True

    def update_zones(self, zones):
        pass


def _cam(cid, key, hogar="h1"):
    return {"id": cid, "householdId": hogar, "name": cid,
            "mediaMtxRtspUrl": f"rtsp://localhost:8554/{key}", "rtspUrl": None}


def test_fallo_del_backend_no_es_lista_vacia():
    with patch.object(api_client.requests, "get",
                      side_effect=requests.exceptions.ConnectionError()):
        assert api_client.get_all_cameras() is None


def test_otro_stream_reinicia_el_worker_y_los_rostros_no_se_repiten():
    sincronizados = []
    with patch.object(main, "CameraWorker", _Worker), \
         patch.object(main, "sync_faces", lambda hid, _dir: sincronizados.append(hid)):
        m = main.CameraManager()
        m.refresh([_cam("c1", "ejemplo01")])
        primero = m._workers["c1"]

        m.refresh([_cam("c1", "ejemplo01")], sync_all_faces=False)
        assert m._workers["c1"] is primero
        assert sincronizados == ["h1"]  # el hogar ya estaba sincronizado

        m.refresh([_cam("c1", "ejemplo07")], sync_all_faces=False)
        assert primero.detenido
        assert m._workers["c1"].camera["mediaMtxRtspUrl"].endswith("ejemplo07")

        # Hogar nuevo: se sincroniza aunque no toque la ronda completa.
        m.refresh([_cam("c1", "ejemplo07"), _cam("c2", "abc", hogar="h2")], sync_all_faces=False)
        assert sincronizados == ["h1", "h2"]

        # Lista vacía de verdad (p. ej. terminó el video de ejemplo): se detienen.
        m.refresh([], sync_all_faces=False)
        assert m.count == 0
