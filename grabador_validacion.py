"""
Grabación de evidencia para la validación (objetivo 4).

Cuando la cámara de un celular empieza a transmitir, se guarda su primer minuto
como evidencia del estudio. SOLO de los hogares con «Grabar evidencia»
encendido: lo decide el backend (interruptor del panel de administración y la
regla por defecto, ver EvidenciaService) y se suman los de `consentimiento.txt`.
Grabar a quien no lo aceptó incumpliría la política de privacidad.

Cuando un hogar deja de existir (cuenta eliminada), se borran sus grabaciones.

Es un servicio aparte (vigishield-grabaciones.service). No toca el pipeline de
IA, MediaMTX ni el backend: solo lee. Si falla, todo lo demás sigue igual.

- Pregunta a MediaMTX (API local) qué paths publican por WebRTC, que es como
  transmite la cámara del celular.
- Copia el video sin recodificar (ffmpeg -c copy): casi no usa CPU.
- Como mucho una grabación por cámara al día, de 60 s, 6 a la vez y 3 GB en
  total. Si la transmisión se corta antes, queda lo que se alcanzó a grabar.
- Se guarda en el disco de la VM (no en el bucket, que es público por URL),
  con un índice CSV para saber de qué hogar y cámara es cada archivo.

    python grabador_validacion.py     # lo arranca systemd
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger("grabador_validacion")

DIRECTORIO = Path(os.getenv("GRAB_DIR", "/opt/vigishield-ai/grabaciones-validacion"))
SEGUNDOS = int(os.getenv("GRAB_SEGUNDOS", "60"))
POR_DIA = int(os.getenv("GRAB_POR_DIA", "1"))
MAX_BYTES = int(float(os.getenv("GRAB_MAX_GB", "3")) * 2**30)
SIMULTANEAS = int(os.getenv("GRAB_SIMULTANEAS", "6"))
INTERVALO = float(os.getenv("GRAB_INTERVALO", "5"))
REFRESCO_CAMARAS = 60.0
REFRESCO_HOGARES = 60.0
MEDIAMTX_PATHS = "http://127.0.0.1:9997/v3/paths/list"
RTSP_LOCAL = "rtsp://127.0.0.1:8554/"
ZONA = ZoneInfo("America/Lima")
COLUMNAS = ["fecha_utc", "dia_lima", "hogar", "camara_id", "camara_nombre", "archivo", "segundos", "bytes"]
_CLAVE_RE = re.compile(r"^[A-Za-z0-9]{1,64}$")
# Un archivo más chico que esto no tiene video útil (la sesión murió al empezar).
_MINIMO_BYTES = 50_000


# ── Piezas sueltas (sin estado, fáciles de probar) ──────────────────────────────

def leer_consentimiento(ruta: Path) -> set[str]:
    """IDs de hogar que aceptaron, uno por línea. Lo que va tras «#» es un
    comentario (por ejemplo, el correo del tester). Sin archivo: nadie."""
    try:
        texto = ruta.read_text(encoding="utf-8")
    except OSError:
        return set()
    hogares = set()
    for linea in texto.splitlines():
        valor = linea.split("#", 1)[0].strip().lower()
        if valor:
            hogares.add(valor)
    return hogares


def publicando_por_webrtc(items: list[dict]) -> list[str]:
    """Paths que están transmitiendo desde un celular (WebRTC/WHIP)."""
    claves = []
    for i in items:
        fuente = i.get("source") or {}
        nombre = i.get("name") or ""
        if i.get("ready") and fuente.get("type") == "webRTCSession" and _CLAVE_RE.match(nombre):
            claves.append(nombre)
    return claves


def clave_de_camara(cam: dict) -> Optional[str]:
    """La clave del stream es el último tramo de su URL en MediaMTX."""
    url = cam.get("mediaMtxRtspUrl") or ""
    clave = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    return clave if _CLAVE_RE.match(clave) else None


def camaras_de_celular(camaras: list[dict]) -> dict[str, dict]:
    """clave del stream → cámara, solo las cámaras de celular."""
    mapa = {}
    for c in camaras:
        clave = clave_de_camara(c)
        if clave and c.get("streamMode") == "MobileWebRtc":
            mapa[clave] = c
    return mapa


def hogares_del_backend() -> Optional[tuple[set[str], set[str]]]:
    """(hogares a grabar, hogares que existen) según el backend, o None si no
    respondió. Solo lectura, con la clave interna."""
    import requests
    from config import BACKEND_API_URL, INTERNAL_API_KEY
    try:
        r = requests.get(f"{BACKEND_API_URL}/api/internal/evidencia/hogares",
                         headers={"X-Api-Key": INTERNAL_API_KEY}, timeout=10)
        r.raise_for_status()
        d = r.json()
        return ({str(h).lower() for h in d.get("grabar", [])},
                {str(h).lower() for h in d.get("existentes", [])})
    except Exception as e:
        logger.warning("No se pudo leer «Grabar evidencia» del backend: %s", e)
        return None


def dia_lima(momento: datetime) -> str:
    return momento.astimezone(ZONA).strftime("%Y-%m-%d")


def comando_ffmpeg(clave: str, destino: Path, segundos: int) -> list[str]:
    # MP4 fragmentado: si la transmisión se corta a la mitad, el archivo
    # igual se puede reproducir (un MP4 normal quedaría sin índice).
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-timeout", "15000000",
        "-i", RTSP_LOCAL + clave,
        "-t", str(segundos), "-an", "-c:v", "copy",
        "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4", "-y", str(destino),
    ]


def duracion(archivo: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(archivo)],
            capture_output=True, text=True, timeout=20)
        return round(float(r.stdout.strip()), 1)
    except Exception:
        return None


def grabar(clave: str, destino: Path, segundos: int) -> int:
    """Graba hasta `segundos` del stream en `destino`. Devuelve los bytes
    escritos (0 si no se grabó nada)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(".part")
    proceso = subprocess.Popen(comando_ffmpeg(clave, parcial, segundos),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, err = proceso.communicate(timeout=segundos + 45)
    except subprocess.TimeoutExpired:
        # Se pide a ffmpeg que cierre el archivo bien; si no responde, se mata.
        proceso.send_signal(signal.SIGINT)
        try:
            _, err = proceso.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proceso.kill()
            _, err = proceso.communicate()
    if err:
        logger.info("ffmpeg (%s): %s", clave[:4] + "…", err.decode(errors="replace").strip()[:200])
    tamano = parcial.stat().st_size if parcial.exists() else 0
    if tamano < _MINIMO_BYTES:
        parcial.unlink(missing_ok=True)
        return 0
    parcial.rename(destino)
    return tamano


def uso_disco(directorio: Path) -> int:
    return sum(p.stat().st_size for p in directorio.rglob("*.mp4") if p.is_file())


# ── El servicio ─────────────────────────────────────────────────────────────────

class Grabador:
    def __init__(self, directorio: Path = DIRECTORIO,
                 leer_paths: Optional[Callable[[], list[dict]]] = None,
                 leer_camaras: Optional[Callable[[], Optional[list[dict]]]] = None,
                 leer_hogares: Optional[Callable[[], Optional[tuple[set[str], set[str]]]]] = None,
                 grabar_fn: Callable[[str, Path, int], int] = grabar,
                 segundos: int = SEGUNDOS, por_dia: int = POR_DIA,
                 max_bytes: int = MAX_BYTES, simultaneas: int = SIMULTANEAS):
        self.directorio = directorio
        self.indice = directorio / "indice.csv"
        self.consentimiento = directorio / "consentimiento.txt"
        self._leer_paths = leer_paths or self._paths_de_mediamtx
        self._leer_camaras = leer_camaras or self._camaras_del_backend
        self._leer_hogares = leer_hogares or hogares_del_backend
        self._hogares_grabar: set[str] = set()
        self._hogares_en = 0.0
        self._grabar = grabar_fn
        self.segundos, self.por_dia = segundos, por_dia
        self.max_bytes, self.simultaneas = max_bytes, simultaneas
        self._camaras: dict[str, dict] = {}
        self._camaras_en = 0.0
        self._en_curso: set[str] = set()
        self._hilos: list[threading.Thread] = []
        self._lock = threading.Lock()

    @staticmethod
    def _paths_de_mediamtx() -> list[dict]:
        with urllib.request.urlopen(MEDIAMTX_PATHS, timeout=3) as r:
            return json.loads(r.read()).get("items", [])

    @staticmethod
    def _camaras_del_backend() -> Optional[list[dict]]:
        from api_client import get_all_cameras  # solo lectura
        return get_all_cameras()

    def _grabadas_hoy(self, camara_id: str, dia: str) -> int:
        try:
            with self.indice.open(encoding="utf-8", newline="") as f:
                return sum(1 for fila in csv.DictReader(f)
                           if fila.get("camara_id") == camara_id and fila.get("dia_lima") == dia)
        except OSError:
            return 0

    def _anotar(self, fila: dict) -> None:
        with self._lock:
            nuevo = not self.indice.exists()
            with self.indice.open("a", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNAS)
                if nuevo:
                    w.writeheader()
                w.writerow(fila)

    def _refrescar_camaras(self) -> None:
        if time.monotonic() - self._camaras_en < REFRESCO_CAMARAS and self._camaras:
            return
        camaras = self._leer_camaras()
        if camaras is not None:  # None = backend caído: se usa la lista anterior
            self._camaras = camaras_de_celular(camaras)
            self._camaras_en = time.monotonic()

    def _refrescar_hogares(self) -> None:
        if self._hogares_en and time.monotonic() - self._hogares_en < REFRESCO_HOGARES:
            return
        self._hogares_en = time.monotonic()
        respuesta = self._leer_hogares()
        if respuesta is None:  # backend caído: se sigue con lo último que respondió
            return
        grabar, existentes = respuesta
        self._hogares_grabar = grabar
        self.limpiar_borrados(existentes)

    def limpiar_borrados(self, existentes: set[str]) -> int:
        """Borra las grabaciones de hogares que ya no existen (cuenta eliminada).
        Con una lista vacía no borra nada: sería un error del backend, no que
        se borraran todas las cuentas."""
        if not existentes:
            return 0
        with self._lock:
            try:
                with self.indice.open(encoding="utf-8", newline="") as f:
                    filas = list(csv.DictReader(f))
            except OSError:
                return 0
            quedan, borradas = [], 0
            for fila in filas:
                if fila.get("hogar", "").lower() in existentes:
                    quedan.append(fila)
                    continue
                (self.directorio / fila.get("archivo", "")).unlink(missing_ok=True)
                borradas += 1
            if borradas:
                temporal = self.indice.with_suffix(".tmp")
                with temporal.open("w", encoding="utf-8", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=COLUMNAS)
                    w.writeheader()
                    w.writerows(quedan)
                temporal.replace(self.indice)
                logger.info("Se borraron %d grabaciones de cuentas eliminadas", borradas)
            return borradas

    def ciclo(self) -> list[str]:
        """Una vuelta: arranca las grabaciones que tocan. Devuelve sus claves."""
        self._hilos = [h for h in self._hilos if h.is_alive()]
        self._refrescar_hogares()
        hogares = leer_consentimiento(self.consentimiento) | self._hogares_grabar
        if not hogares:
            return []
        claves = publicando_por_webrtc(self._leer_paths())
        if not claves:
            return []
        self._refrescar_camaras()
        ahora = datetime.now(timezone.utc)
        dia = dia_lima(ahora)
        iniciadas = []
        for clave in claves:
            cam = self._camaras.get(clave)
            if cam is None or str(cam.get("householdId", "")).lower() not in hogares:
                continue
            camara_id = str(cam["id"])
            if camara_id in self._en_curso or self._grabadas_hoy(camara_id, dia) >= self.por_dia:
                continue
            if len(self._en_curso) >= self.simultaneas:
                logger.warning("Ya hay %d grabaciones en curso: se omite una", self.simultaneas)
                break
            if uso_disco(self.directorio) >= self.max_bytes:
                logger.warning("Se alcanzó el límite de %.1f GB: no se graba más", self.max_bytes / 2**30)
                break
            self._en_curso.add(camara_id)
            hilo = threading.Thread(target=self._grabar_camara, args=(clave, cam, ahora), daemon=True,
                                    name=f"grab-{camara_id[:8]}")
            self._hilos.append(hilo)
            hilo.start()
            iniciadas.append(clave)
        return iniciadas

    def _grabar_camara(self, clave: str, cam: dict, inicio: datetime) -> None:
        camara_id, hogar = str(cam["id"]), str(cam["householdId"])
        dia = dia_lima(inicio)
        nombre = f"{inicio.astimezone(ZONA):%H%M%S}_{hogar[:8]}_{camara_id[:8]}.mp4"
        destino = self.directorio / dia / nombre
        try:
            logger.info("Grabando %s s de la cámara %s (hogar %s)", self.segundos, camara_id[:8], hogar[:8])
            tamano = self._grabar(clave, destino, self.segundos)
            if tamano:
                self._anotar({
                    "fecha_utc": inicio.isoformat(timespec="seconds"), "dia_lima": dia,
                    "hogar": hogar, "camara_id": camara_id, "camara_nombre": cam.get("name", ""),
                    "archivo": f"{dia}/{nombre}", "segundos": duracion(destino) or "",
                    "bytes": tamano,
                })
                logger.info("Grabación lista: %s (%d KB)", nombre, tamano // 1024)
            else:
                logger.info("La cámara %s no dejó video (se cortó al empezar)", camara_id[:8])
        except Exception:
            logger.exception("No se pudo grabar la cámara %s", camara_id[:8])
        finally:
            self._en_curso.discard(camara_id)

    def correr(self) -> None:
        self.directorio.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            logger.error("No está ffmpeg: no se puede grabar")
        logger.info("Grabador de validación listo (%s s, %d por cámara al día, máx %.1f GB)",
                    self.segundos, self.por_dia, self.max_bytes / 2**30)
        while True:
            try:
                self.ciclo()
            except Exception:
                logger.exception("Fallo en una vuelta del grabador; se reintenta")
            time.sleep(INTERVALO)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    Grabador().correr()
