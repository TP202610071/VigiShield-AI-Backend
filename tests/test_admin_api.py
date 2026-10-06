"""
Las rutas del panel de administración solo responden a un JWT con rol Admin.

    py -3 -m pytest tests/test_admin_api.py
"""
import base64
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import admin_api  # noqa: E402
import config  # noqa: E402


def _jwt(payload: dict, secret: str) -> str:
    b64 = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    cabecera, cuerpo = b64({"alg": "HS256", "typ": "JWT"}), b64(payload)
    firma = hmac.new(secret.encode(), f"{cabecera}.{cuerpo}".encode(), hashlib.sha256).digest()
    return f"{cabecera}.{cuerpo}." + base64.urlsafe_b64encode(firma).rstrip(b"=").decode()


def _base(rol):
    return {"role": rol, "exp": time.time() + 600, "iss": config.JWT_ISSUER, "aud": config.JWT_AUDIENCE}


def test_solo_el_rol_admin_entra(monkeypatch):
    monkeypatch.setattr(config, "JWT_SECRET", "secreto-de-prueba")
    assert admin_api.es_admin("Bearer " + _jwt(_base("Admin"), "secreto-de-prueba"))
    assert not admin_api.es_admin("Bearer " + _jwt(_base("Primary"), "secreto-de-prueba"))
    assert not admin_api.es_admin("Bearer " + _jwt(_base("Admin"), "otro-secreto"))
    vencido = dict(_base("Admin"), exp=time.time() - 5)
    assert not admin_api.es_admin("Bearer " + _jwt(vencido, "secreto-de-prueba"))
    assert not admin_api.es_admin(None)


def test_metricas_y_streams_tienen_lo_que_grafica_el_panel():
    m = admin_api.metricas()
    for clave in ("cpu", "ram", "ramTotalMb", "ramUsadaMb", "nucleos", "procesos"):
        assert clave in m

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"items": [{"name": "abc123", "ready": True, "readers": [1, 2],
                                          "source": {"type": "webRTCSession"}}]}).encode()

    with patch.object(admin_api.urllib.request, "urlopen", return_value=_Resp()):
        s = admin_api.streams({"cam-1": {"ts": time.time() - 2, "persons": 1,
                                         "intent": {"state": "suspect"}}})
    assert s["paths"][0] == {"clave": "abc123", "tipo": "webRTCSession", "listo": True,
                             "desde": None, "lectores": 2, "bytes": 0}
    assert s["ia"]["cam-1"]["personas"] == 1 and s["ia"]["cam-1"]["intencion"] == "suspect"
