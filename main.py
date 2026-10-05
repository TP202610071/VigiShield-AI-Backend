"""
VigiShield AI Backend — Multi-Camera Entry Point

On startup:
  1. Fetches ALL configured cameras from the backend (all households)
  2. Syncs authorized face photos for each household
  3. Starts one worker thread per camera
  4. Each thread runs the full detection pipeline (YOLO + DeepFace + ActivityModel)
  5. Re-reads the camera list every CAMERA_POLL_SECONDS (cheap: one GET) and
     re-syncs faces every CAMERA_REFRESH_INTERVAL seconds

Usage:
    cp .env.example .env   # fill in values (defaults work for local dev)
    pip install -r requirements.txt
    python main.py
"""

import logging
import sys
import threading
import time
from collections import defaultdict

import requests as _requests

import config  # loads .env, configures logging, adds CUDA DLL dir
import frame_server
from api_client import (
    attach_clip,
    get_alert_config,
    get_all_cameras,
    ingest_event,
    sync_faces,
)
from event_detector import (
    EventDetector,
    build_annotated_clip,
    retime_frames,
    capture_event_snapshot,
    disabled_event_types,
)
from stream_reader import StreamReader

logger = logging.getLogger(__name__)


# ── HLS muxer warmer ──────────────────────────────────────────────────────────

def _hls_warmer(hls_url: str, stop_event: threading.Event) -> None:
    """
    Keeps the MediaMTX HLS muxer alive by fetching the manifest every 10 s.

    Why: MediaMTX creates the HLS muxer on first client request and destroys it
    when unused. The first client after destruction waits up to 16 s for the
    camera's next keyframe before the muxer serves anything. By fetching the
    manifest every 10 s we prevent destruction — all real clients (app, browser)
    connect instantly to a pre-warmed muxer with existing segments.
    """
    session = _requests.Session()
    while not stop_event.wait(10):
        try:
            session.get(hls_url, timeout=25)
        except Exception:
            pass  # MediaMTX might not be up yet — retry next tick


# ── Per-camera worker ─────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    """Reads one RTSP stream and runs the full AI pipeline on it."""

    def __init__(self, camera: dict):
        super().__init__(name=f"cam-{camera['id'][:8]}", daemon=True)
        self.camera = camera
        self._stop_event = threading.Event()
        self.detector = None            # se asigna al crear el pipeline en run()
        self._zones_raw = camera.get("zones")

    def stop(self):
        self._stop_event.set()

    def update_zones(self, zones_raw) -> None:
        """Recarga en caliente las zonas de esta cámara (llamado desde el manager).
        No reinicia el worker ni el stream — solo reemplaza el polígono en memoria."""
        if zones_raw == self._zones_raw:
            return  # sin cambios
        self._zones_raw = zones_raw
        d = self.detector
        if d is not None:
            d.set_zones(zones_raw)

    def run(self):
        cam = self.camera
        mediamtx_url = cam.get("mediaMtxRtspUrl")
        direct_url = cam.get("rtspUrl")
        rtsp_url = mediamtx_url or direct_url
        hls_url = cam.get("hlsLocalUrl")
        camera_id = str(cam["id"])
        camera_name = cam.get("name", camera_id[:8])
        household_id = str(cam["householdId"])

        logger.info("[%s] Worker started → %s", camera_name, _mask_url(rtsp_url))

        # Keep the MediaMTX HLS muxer warm so app/browser connects instantly
        if hls_url:
            threading.Thread(
                target=_hls_warmer,
                args=(hls_url, self._stop_event),
                daemon=True,
                name=f"hls-warm-{camera_id[:8]}",
            ).start()
            logger.info("[%s] HLS warmer started → %s", camera_name, hls_url)

        detector = EventDetector(
            household_id=household_id,
            camera_id=camera_id,
            camera_name=camera_name,
            zones_raw=self._zones_raw,
        )
        self.detector = detector  # expuesto para recarga en caliente de zonas

        # Household alert toggles — refreshed lazily so disabling an alert in the
        # app suppresses those event types within ALERT_CONFIG_REFRESH_SECONDS.
        disabled_types: set[str] = set()
        last_alert_fetch = 0.0

        # Espera creciente entre reintentos (3 y luego 6 s). Un celular que
        # reinicia su transmisión al girarlo vuelve en dos segundos: con esperas
        # de 15 s la vista de IA quedaba congelada mucho más que el corte real.
        espera = 3
        while not self._stop_event.is_set():
            try:
                with StreamReader(rtsp_url) as reader:
                    if not reader._cap or not reader._cap.isOpened():
                        logger.warning("[%s] Stream not available. Retry in %ds...", camera_name, espera)
                        self._stop_event.wait(espera)
                        espera = min(6, espera * 2)
                        continue

                    espera = 3
                    logger.info("[%s] Stream connected. Processing every %.1fs.",
                                camera_name, config.FRAME_INTERVAL_SECONDS)

                    for frame in reader.frames(interval_seconds=config.FRAME_INTERVAL_SECONDS):
                        if self._stop_event.is_set():
                            break

                        events = detector.process_frame(frame)
                        if not events:
                            continue

                        # Refresh the household's alert toggles, then drop disabled types.
                        now = time.monotonic()
                        if now - last_alert_fetch > config.ALERT_CONFIG_REFRESH_SECONDS:
                            disabled_types = disabled_event_types(get_alert_config(household_id))
                            last_alert_fetch = now
                        events = [e for e in events if e["event_type"] not in disabled_types]
                        if not events:
                            logger.info("[%s] All %d event(s) suppressed by alert config",
                                        camera_name, len(disabled_types))
                            continue

                        # One snapshot of the moment (with detection boxes drawn),
                        # shared by every event in this batch.
                        # No se usa el cuadro de ESTE instante: cuando el riesgo
                        # acumulado hace saltar la alerta, la persona suele haber
                        # salido ya y la foto quedaba vacia. Se pide el ultimo
                        # cuadro reciente con detecciones etiquetadas.
                        # Instante del evento = hora del cuadro que lo hizo saltar.
                        momento_evento = detector.last_frame_time
                        cuadro_evidencia, draw_evidencia, momento_foto = detector.evidencia_del_evento(frame)
                        snap_url = capture_event_snapshot(
                            cuadro_evidencia, camera_name,
                            label=events[0]["event_type"], draw=draw_evidencia,
                            moment=momento_foto)
                        first_event_id = None
                        for ev in events:
                            if snap_url:
                                ev["image_capture_path"] = snap_url
                            result = ingest_event(**ev)
                            if first_event_id is None and isinstance(result, dict):
                                first_event_id = result.get("id")

                        # Record a short clip in the background and attach it to the
                        # event once uploaded (so the live alert isn't delayed).
                        if first_event_id and config.EVENT_CLIP_SECONDS > 0:
                            # El hilo espera los segundos POSTERIORES (el worker
                            # sigue llenando el búfer) y luego arma el clip con
                            # la ventana [evento - PRE, evento + POST], para no
                            # retrasar la alerta.
                            threading.Thread(
                                target=_build_and_attach_clip,
                                args=(detector, momento_evento, first_event_id, self._stop_event),
                                daemon=True,
                                name=f"clip-{first_event_id[:8]}",
                            ).start()

            except Exception as e:
                logger.error("[%s] Pipeline error: %s — restarting in 10s", camera_name, e, exc_info=True)
                self._stop_event.wait(10)

        logger.info("[%s] Worker stopped", camera_name)


# ── Camera list manager ───────────────────────────────────────────────────────

class CameraManager:
    """Tracks running workers; adds/removes them as cameras change."""

    def __init__(self):
        self._workers: dict[str, CameraWorker] = {}
        self._faces_synced: set[str] = set()

    @staticmethod
    def _stream_of(cam: dict) -> tuple:
        return (cam.get("mediaMtxRtspUrl"), cam.get("rtspUrl"))

    def refresh(self, cameras: list[dict], sync_all_faces: bool = True):
        current_ids = {str(c["id"]) for c in cameras}
        running_ids = set(self._workers.keys())

        # Stop workers for cameras that disappeared
        for cid in running_ids - current_ids:
            logger.info("Camera %s removed — stopping worker", cid[:8])
            self._workers.pop(cid).stop()

        # La misma cámara con otro stream (el video de ejemplo cambia de path al
        # pedir otro): el worker seguía leyendo el anterior. Se reinicia.
        for cam in cameras:
            cid = str(cam["id"])
            w = self._workers.get(cid)
            if w is not None and self._stream_of(w.camera) != self._stream_of(cam):
                logger.info("Camera %s changed stream — restarting worker", cid[:8])
                self._workers.pop(cid).stop()

        # Sync faces per household (deduplicated). Cada CAMERA_REFRESH_INTERVAL
        # todos; entre medias solo los hogares nuevos, para que la lista de
        # cámaras se pueda leer seguido sin descargar rostros cada vez.
        household_ids_seen = set()
        for cam in cameras:
            hid = str(cam["householdId"])
            if hid in household_ids_seen:
                continue
            household_ids_seen.add(hid)
            if not sync_all_faces and hid in self._faces_synced:
                continue
            try:
                sync_faces(hid, config.FACES_DIR)
                self._faces_synced.add(hid)
            except Exception as e:
                logger.warning("Face sync failed for household %s: %s", hid[:8], e)

        # Start workers for new cameras; hot-reload zones on existing ones.
        for cam in cameras:
            cid = str(cam["id"])
            if cid not in self._workers:
                worker = CameraWorker(cam)
                worker.start()
                self._workers[cid] = worker
                logger.info("Camera '%s' (%s) → worker started", cam.get("name"), cid[:8])
            else:
                # Recarga de zonas sin reiniciar el stream (aplica lo que el
                # usuario dibuje en la app en el siguiente refresco).
                self._workers[cid].update_zones(cam.get("zones"))

    def stop_all(self):
        for w in self._workers.values():
            w.stop()
        self._workers.clear()

    @property
    def count(self) -> int:
        return len(self._workers)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  VigiShield AI Backend — Multi-Camera")
    logger.info("  Backend : %s", config.BACKEND_API_URL)
    logger.info("  Models  : Activity=%s | YOLO=%s | DeepFace=VGG-Face",
                config.ACTIVITY_MODEL_PATH, config.YOLO_MODEL)
    logger.info("  Refresh : cameras every %ds, faces every %ds",
                config.CAMERA_POLL_SECONDS, config.CAMERA_REFRESH_INTERVAL)
    logger.info("=" * 60)

    # Start AI frame server — Flutter app polls this for annotated video frames.
    frame_server.start_server(port=5050)

    manager = CameraManager()

    # La lista se lee cada CAMERA_POLL_SECONDS: una cámara nueva (o el video de
    # ejemplo que alguien acaba de pedir) empieza a analizarse en segundos y no
    # en hasta cinco minutos. Solo se registra en el log cuando algo cambia.
    last_faces = float("-inf")
    last_ids: set[str] | None = None
    try:
        while True:
            cameras = get_all_cameras()
            if cameras is None:
                # Backend caído: se mantiene lo que ya corre y se reintenta.
                time.sleep(config.CAMERA_POLL_SECONDS)
                continue
            ids = {str(c["id"]) for c in cameras}
            faces_due = time.monotonic() - last_faces >= config.CAMERA_REFRESH_INTERVAL

            if ids != last_ids:
                if not cameras:
                    logger.warning(
                        "No configured cameras found in backend.\n"
                        "  → Open the VigiShield app → Settings → Cameras → Add a camera."
                    )
                else:
                    logger.info("Camera list: %d camera(s)", len(cameras))
            manager.refresh(cameras, sync_all_faces=faces_due)
            if faces_due:
                last_faces = time.monotonic()
            if ids != last_ids:
                logger.info("Active workers: %d", manager.count)
                last_ids = ids

            time.sleep(config.CAMERA_POLL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Shutdown requested — stopping all workers...")
        manager.stop_all()
        logger.info("Goodbye.")


def _build_and_attach_clip(detector, momento: float, event_id: str,
                           stop: threading.Event) -> None:
    """Segundo plano: espera a tener los segundos posteriores al evento, arma
    el clip con los cuadros anotados de [momento - PRE, momento + POST], lo sube
    a R2 y lo adjunta al evento."""
    stop.wait(max(0.0, config.EVENT_CLIP_POST_SECONDS))
    timed = detector.clip_frames(momento - config.EVENT_CLIP_PRE_SECONDS,
                                 momento + config.EVENT_CLIP_POST_SECONDS)
    fps = max(1.0, config.EVENT_CLIP_OUTPUT_FPS)
    frames = retime_frames(timed, fps, event_time=momento)
    url = build_annotated_clip(frames, fps=fps)
    if url:
        attach_clip(event_id, url)


def _mask_url(url: str) -> str:
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


if __name__ == "__main__":
    main()
