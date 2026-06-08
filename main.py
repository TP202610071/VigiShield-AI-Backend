"""
VigiShield AI Backend — Multi-Camera Entry Point

On startup:
  1. Fetches ALL configured cameras from the backend (all households)
  2. Syncs authorized face photos for each household
  3. Starts one worker thread per camera
  4. Each thread runs the full detection pipeline (YOLO + DeepFace + ActivityModel)
  5. Refreshes the camera list every CAMERA_REFRESH_INTERVAL seconds

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
from api_client import get_all_cameras, ingest_event, sync_faces
from event_detector import EventDetector
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

    def stop(self):
        self._stop_event.set()

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
        )

        while not self._stop_event.is_set():
            try:
                with StreamReader(rtsp_url) as reader:
                    if not reader._cap or not reader._cap.isOpened():
                        logger.warning("[%s] Stream not available. Retry in 15s...", camera_name)
                        self._stop_event.wait(15)
                        continue

                    logger.info("[%s] Stream connected. Processing every %.1fs.",
                                camera_name, config.FRAME_INTERVAL_SECONDS)

                    for frame in reader.frames(interval_seconds=config.FRAME_INTERVAL_SECONDS):
                        if self._stop_event.is_set():
                            break

                        events = detector.process_frame(frame)
                        for ev in events:
                            ingest_event(**ev)

            except Exception as e:
                logger.error("[%s] Pipeline error: %s — restarting in 10s", camera_name, e, exc_info=True)
                self._stop_event.wait(10)

        logger.info("[%s] Worker stopped", camera_name)


# ── Camera list manager ───────────────────────────────────────────────────────

class CameraManager:
    """Tracks running workers; adds/removes them as cameras change."""

    def __init__(self):
        self._workers: dict[str, CameraWorker] = {}

    def refresh(self, cameras: list[dict]):
        current_ids = {str(c["id"]) for c in cameras}
        running_ids = set(self._workers.keys())

        # Stop workers for cameras that disappeared
        for cid in running_ids - current_ids:
            logger.info("Camera %s removed — stopping worker", cid[:8])
            self._workers.pop(cid).stop()

        # Sync faces per household (deduplicated)
        household_ids_seen = set()
        for cam in cameras:
            hid = str(cam["householdId"])
            if hid not in household_ids_seen:
                household_ids_seen.add(hid)
                try:
                    sync_faces(hid, config.FACES_DIR)
                except Exception as e:
                    logger.warning("Face sync failed for household %s: %s", hid[:8], e)

        # Start workers for new cameras
        for cam in cameras:
            cid = str(cam["id"])
            if cid not in self._workers:
                worker = CameraWorker(cam)
                worker.start()
                self._workers[cid] = worker
                logger.info("Camera '%s' (%s) → worker started", cam.get("name"), cid[:8])

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
    logger.info("  Refresh : every %ds", config.CAMERA_REFRESH_INTERVAL)
    logger.info("=" * 60)

    # Start AI frame server — Flutter app polls this for annotated video frames.
    frame_server.start_server(port=5050)

    manager = CameraManager()

    try:
        while True:
            cameras = get_all_cameras()

            if not cameras:
                logger.warning(
                    "No configured cameras found in backend.\n"
                    "  → Open the VigiShield app → Settings → Cameras → Add a camera."
                )
            else:
                logger.info("Camera list: %d camera(s)", len(cameras))
                manager.refresh(cameras)
                logger.info("Active workers: %d", manager.count)

            logger.info("Next camera refresh in %ds...", config.CAMERA_REFRESH_INTERVAL)
            time.sleep(config.CAMERA_REFRESH_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Shutdown requested — stopping all workers...")
        manager.stop_all()
        logger.info("Goodbye.")


def _mask_url(url: str) -> str:
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


if __name__ == "__main__":
    main()
