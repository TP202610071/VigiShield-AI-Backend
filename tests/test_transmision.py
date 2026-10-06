"""
La app pregunta si la cámara de un teléfono está transmitiendo antes de
esperar su video.

    py -3 -m pytest tests/test_transmision.py
"""
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frame_server  # noqa: E402
import transmision  # noqa: E402


class _Resp:
    def __init__(self, cuerpo: dict):
        self._cuerpo = json.dumps(cuerpo).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self._cuerpo


def _http_error(codigo):
    return urllib.error.HTTPError("http://x", codigo, "x", {}, io.BytesIO(b""))


def test_publicando_si_y_no():
    with patch.object(transmision.urllib.request, "urlopen", return_value=_Resp({"ready": True})) as u:
        assert transmision.en_vivo("abc123") is True
    assert u.call_args[0][0].endswith("/v3/paths/get/abc123")
    with patch.object(transmision.urllib.request, "urlopen", return_value=_Resp({"ready": False})):
        assert transmision.en_vivo("abc123") is False
    # Nadie publica: MediaMTX ni siquiera tiene el path.
    with patch.object(transmision.urllib.request, "urlopen", side_effect=_http_error(404)):
        assert transmision.en_vivo("abc123") is False


def test_sin_respuesta_de_mediamtx_no_se_sabe():
    with patch.object(transmision.urllib.request, "urlopen", side_effect=_http_error(500)):
        assert transmision.en_vivo("abc123") is None
    with patch.object(transmision.urllib.request, "urlopen", side_effect=TimeoutError()):
        assert transmision.en_vivo("abc123") is None


@pytest.mark.parametrize("clave", ["", "../config", "a/b", "a" * 65, "abc%20"])
def test_clave_mal_formada_no_llega_a_mediamtx(clave):
    with patch.object(transmision.urllib.request, "urlopen") as u, pytest.raises(ValueError):
        transmision.en_vivo(clave)
    u.assert_not_called()


def test_ruta_del_servidor():
    servidor = frame_server.start_server(port=0)
    puerto = servidor.server_address[1]
    try:
        def pedir(ruta):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{puerto}{ruta}", timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, None

        with patch.object(transmision, "en_vivo", return_value=True):
            assert pedir("/en-vivo/abc123") == (200, {"enVivo": True})
        with patch.object(transmision, "en_vivo", return_value=False):
            assert pedir("/en-vivo/abc123") == (200, {"enVivo": False})
        with patch.object(transmision, "en_vivo", return_value=None):
            assert pedir("/en-vivo/abc123")[0] == 503
        assert pedir("/en-vivo/..%2Fconfig")[0] == 404
    finally:
        servidor.shutdown()
