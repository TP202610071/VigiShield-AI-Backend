"""HTTP client for communicating with the VigiShield ASP.NET Core backend."""

import logging
import requests
from config import BACKEND_API_URL, INTERNAL_API_KEY

logger = logging.getLogger(__name__)

_HEADERS = {
    "X-Api-Key": INTERNAL_API_KEY,
    "Content-Type": "application/json",
}


def get_all_cameras() -> list[dict]:
    """
    Fetch all configured cameras across all households.
    Returns list of dicts: {id, householdId, name, rtspUrl, streamMode, isDefault}
    """
    try:
        resp = requests.get(
            f"{BACKEND_API_URL}/api/stream/ai-config/all",
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        cameras = resp.json()
        configured = [c for c in cameras if c.get("rtspUrl")]
        logger.info("Fetched %d configured camera(s) from backend", len(configured))
        return configured
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to backend at %s — is it running?", BACKEND_API_URL)
    except requests.exceptions.HTTPError as e:
        logger.error("Backend error fetching cameras: %s — %s", e.response.status_code, e.response.text)
    except Exception as e:
        logger.error("Unexpected error fetching cameras: %s", e)
    return []


def get_alert_config(household_id: str) -> dict | None:
    """
    Fetch a household's alert toggle config so the pipeline can suppress events
    the user has disabled in the app. Returns the dict
    {unknownPersonEnabled, forcedAccessEnabled, tailgatingEnabled,
     climbingEnabled, aggressionEnabled, ...} or None on failure (caller then
     treats all alerts as enabled — fail-open, never silently drop everything).
    """
    try:
        resp = requests.get(
            f"{BACKEND_API_URL}/api/stream/ai-alert-config",
            headers=_HEADERS,
            params={"householdId": household_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Could not fetch alert config for %s: %s", household_id, e)
    return None


def get_authorized_faces(household_id: str) -> list[dict]:
    """
    Fetch authorized face profiles for a household.
    Returns list of dicts: {id, personName, photoPaths, createdAt}
    """
    try:
        resp = requests.get(
            f"{BACKEND_API_URL}/api/stream/ai-faces",
            headers=_HEADERS,
            params={"householdId": household_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("Could not fetch authorized faces for %s: %s", household_id, e)
    return []


def sync_faces(household_id: str, faces_dir) -> None:
    """
    Mirror the household's authorized faces into the local faces directory.
    Directory structure: faces_dir / household_id / person_name / photo.jpg

    IMPORTANT: this is a two-way MIRROR, not just a download:
      - descarga las fotos nuevas que falten localmente,
      - ELIMINA localmente las personas/fotos que ya NO están autorizadas en el
        backend (antes solo descargaba → una cara borrada en la app seguía
        reconociéndose porque su copia local quedaba huérfana),
      - si algo cambió, BORRA el caché de embeddings de DeepFace (*.pkl) para que
        se regenere sin la persona eliminada.
    """
    import shutil
    import requests as req_module
    from pathlib import Path

    faces = get_authorized_faces(household_id)  # [] si no hay ninguna autorizada
    base_dir = Path(faces_dir) / household_id
    base_dir.mkdir(parents=True, exist_ok=True)

    changed = False
    authorized_dirs: set[str] = set()

    for face in faces:
        person_name = face.get("personName", "unknown").strip()
        photo_paths = face.get("photoPaths", [])
        authorized_dirs.add(person_name)
        person_dir = base_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        authorized_files = {Path(u).name for u in photo_paths}

        # 1) descargar las que falten
        for url_path in photo_paths:
            local_path = person_dir / Path(url_path).name
            if local_path.exists():
                continue
            try:
                img_resp = req_module.get(f"{BACKEND_API_URL}{url_path}", timeout=30)
                img_resp.raise_for_status()
                local_path.write_bytes(img_resp.content)
                changed = True
                logger.debug("Downloaded face photo: %s", local_path)
            except Exception as e:
                logger.warning("Failed to download face photo %s: %s", url_path, e)

        # 2) borrar fotos locales de esta persona que ya no están autorizadas
        for local in list(person_dir.glob("*")):
            if local.is_file() and local.suffix.lower() in (".jpg", ".jpeg", ".png") \
                    and local.name not in authorized_files:
                try:
                    local.unlink(); changed = True
                    logger.info("Removed stale face photo: %s", local)
                except OSError:
                    pass

    # 3) borrar carpetas de personas que ya NO están autorizadas (o todas si faces=[])
    if base_dir.exists():
        for person_dir in list(base_dir.iterdir()):
            if person_dir.is_dir() and person_dir.name not in authorized_dirs:
                try:
                    shutil.rmtree(person_dir); changed = True
                    logger.info("Removed unauthorized face profile: %s", person_dir.name)
                except OSError:
                    pass

    # 4) si cambió algo, invalidar el caché de embeddings de DeepFace
    if changed:
        for pkl in base_dir.glob("*.pkl"):
            try:
                pkl.unlink()
                logger.info("Invalidated DeepFace cache: %s", pkl.name)
            except OSError:
                pass

    logger.info("Face sync complete for household %s (%d perfiles autorizados%s)",
                household_id, len(faces), ", cambios aplicados" if changed else "")


def ingest_event(
    event_type: str,
    confidence_score: float,
    risk_level: str,
    household_id: str,
    camera_id: str | None = None,
    camera_name: str | None = None,
    is_nighttime: bool = False,
    person_name: str | None = None,
    image_capture_path: str | None = None,
    video_clip_path: str | None = None,
) -> dict | None:
    """
    Send a detected security event to the backend.
    event_type must match the EventType enum in the backend:
        FaceRecognized, UnknownFace, LowConfidenceFace, RecurrentUnknownFace,
        ForcedAccessAttempt, Tailgating, Climbing, Burglary,
        PhysicalAggression, Assault, Abuse, Arrest,
        Stealing, Shoplifting, Vandalism, Robbery, Arson,
        Explosion, Roadaccidents, WeaponDetected
    """
    payload = {
        "householdId": household_id,
        "eventType": event_type,
        "cameraId": camera_id,
        "cameraName": camera_name,
        "confidenceScore": round(confidence_score, 4),
        "riskLevel": risk_level,
        "isNighttime": is_nighttime,
        "personName": person_name,
        "imageCapturePath": image_capture_path,
        "videoClipPath": video_clip_path,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        resp = requests.post(
            f"{BACKEND_API_URL}/api/events/ingest",
            json=payload,
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        # 204 = el backend descartó el evento a propósito (la monitorización del
        # hogar está en pausa). No es un error y no trae cuerpo: intentar leerlo
        # como JSON llenaba el log de "Expecting value: line 1 column 1".
        if resp.status_code == 204 or not resp.content:
            logger.info("Evento %s descartado por el backend (monitorización en pausa)",
                        event_type)
            return None
        logger.info("Event ingested: %s | risk=%s | conf=%.2f | cam=%s",
                    event_type, risk_level, confidence_score, camera_name or camera_id)
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error("Failed to ingest event: %s — %s", e.response.status_code, e.response.text)
    except Exception as e:
        logger.error("Unexpected error ingesting event: %s", e)
    return None


def attach_clip(event_id: str, video_clip_path: str) -> bool:
    """Attach a recorded video clip URL to an already-ingested event."""
    try:
        resp = requests.post(
            f"{BACKEND_API_URL}/api/events/{event_id}/clip",
            json={"videoClipPath": video_clip_path},
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        logger.info("Clip attached to event %s", event_id)
        return True
    except Exception as e:
        logger.error("Failed to attach clip to event %s: %s", event_id, e)
        return False
