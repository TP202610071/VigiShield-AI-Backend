"""
Rutas del panel de administración (https://vigishield.app/admin/).

Llegan por nginx /ai/admin/..., que ya exige un JWT válido de VigiShield; aquí
se exige además el rol Admin: la lista de streams incluye las claves de todas
las cámaras y no debe verla un usuario cualquiera.

Nada de esto toca la ruta de los cuadros ni la de estado que usa la app.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

import psutil

_ROLE_CLAIMS = ("role", "http://schemas.microsoft.com/ws/2008/06/identity/claims/role")
_MEDIAMTX_API = "http://127.0.0.1:9997/v3/paths/list"

# psutil mide la CPU desde la llamada anterior: la primera devuelve 0.
psutil.cpu_percent(interval=None)


def _payload(token: str | None) -> dict | None:
    """Contenido del JWT si su firma, vigencia, emisor y audiencia son válidos."""
    from frame_server import validate_jwt  # aquí, para no importar en círculo
    if not validate_jwt(token):
        return None
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    segmento = token.split(".")[1]
    try:
        return json.loads(base64.urlsafe_b64decode(segmento + "=" * (-len(segmento) % 4)))
    except Exception:
        return None


def es_admin(token: str | None) -> bool:
    """El JWT es válido y su rol es Admin."""
    payload = _payload(token)
    if not payload:
        return False
    for clave in _ROLE_CLAIMS:
        rol = payload.get(clave)
        if rol == "Admin" or (isinstance(rol, list) and "Admin" in rol):
            return True
    return False


def metricas() -> dict:
    """CPU, RAM, disco y los procesos que más consumen en esta VM."""
    mem = psutil.virtual_memory()
    disco = psutil.disk_usage("/")
    procesos = []
    for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            cpu = p.cpu_percent(interval=None)
            nombre = p.info["name"] or ""
            cmd = " ".join(p.info["cmdline"] or [])[:80]
            procesos.append({
                "pid": p.pid, "nombre": nombre, "cmd": cmd, "cpu": round(cpu, 1),
                "ramMb": round((p.info["memory_info"].rss if p.info["memory_info"] else 0) / 2**20),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    procesos.sort(key=lambda x: (x["cpu"], x["ramMb"]), reverse=True)
    return {
        "ts": time.time(),
        "nucleos": psutil.cpu_count() or 1,
        "cpu": round(psutil.cpu_percent(interval=None), 1),
        "carga": list(psutil.getloadavg()),
        "ramTotalMb": round(mem.total / 2**20),
        "ramUsadaMb": round((mem.total - mem.available) / 2**20),
        "ram": round(mem.percent, 1),
        "discoTotalGb": round(disco.total / 2**30, 1),
        "discoUsadoGb": round(disco.used / 2**30, 1),
        "encendidaDesde": psutil.boot_time(),
        "procesos": procesos[:8],
    }


def streams(estados: dict[str, dict]) -> dict:
    """Paths de MediaMTX con publicador y el último estado de cada worker de IA.

    `estados` es camera_id → último estado publicado por el pipeline (con su
    ts), para saber si la IA está procesando esa cámara ahora mismo.
    """
    try:
        with urllib.request.urlopen(_MEDIAMTX_API, timeout=3) as r:
            items = json.loads(r.read()).get("items", [])
    except Exception:
        items = []
    paths = []
    for i in items:
        fuente = i.get("source") or {}
        paths.append({
            "clave": i.get("name"),
            "tipo": fuente.get("type"),
            "listo": bool(i.get("ready")),
            "desde": i.get("readyTime"),
            "lectores": len(i.get("readers") or []),
            "bytes": i.get("bytesReceived", 0),
        })
    ahora = time.time()
    ia = {cid: {"hace": round(ahora - s.get("ts", 0), 1), "personas": s.get("persons", 0),
                "intencion": (s.get("intent") or {}).get("state")}
          for cid, s in estados.items()}
    return {"ts": ahora, "paths": paths, "ia": ia}
