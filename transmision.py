"""
¿Alguien está transmitiendo esta cámara ahora mismo?

La cámara de un teléfono solo transmite mientras VigiShield está abierta en
ese teléfono: al salir de la app, el sistema le quita la cámara y la
transmisión se corta. Su configuración no cambia (sigue «Activa»), así que la
app lo pregunta aquí antes de quedarse esperando un video que no va a llegar.

Solo responde sí o no: la API de MediaMTX sigue siendo privada.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Optional

_CLAVE_RE = re.compile(r"^[A-Za-z0-9]{1,64}$")
_MEDIAMTX_PATH = "http://127.0.0.1:9997/v3/paths/get/"


def en_vivo(clave: str) -> Optional[bool]:
    """True si alguien publica en el path `clave`; False si nadie; None si no
    se pudo saber (MediaMTX no responde). Una clave mal formada da ValueError."""
    if not _CLAVE_RE.match(clave or ""):
        raise ValueError("clave inválida")
    try:
        with urllib.request.urlopen(_MEDIAMTX_PATH + clave, timeout=2) as r:
            return bool(json.loads(r.read()).get("ready"))
    except urllib.error.HTTPError as e:
        # 404: el path no existe porque nadie lo publica.
        return False if e.code == 404 else None
    except Exception:
        return None
