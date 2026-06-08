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
    Download face photos from the backend and save them to the local faces directory.
    Directory structure: faces_dir / household_id / person_name / photo.jpg
    Only downloads files that do not already exist locally.
    """
    import requests as req_module
    from pathlib import Path

    faces = get_authorized_faces(household_id)
    if not faces:
        return

    base_dir = Path(faces_dir) / household_id

    for face in faces:
        person_name = face.get("personName", "unknown").strip()
        photo_paths = face.get("photoPaths", [])
        person_dir = base_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)

        for url_path in photo_paths:
            filename = Path(url_path).name
            local_path = person_dir / filename
            if local_path.exists():
                continue

            try:
                full_url = f"{BACKEND_API_URL}{url_path}"
                img_resp = req_module.get(full_url, timeout=30)
                img_resp.raise_for_status()
                local_path.write_bytes(img_resp.content)
                logger.debug("Downloaded face photo: %s → %s", url_path, local_path)
            except Exception as e:
                logger.warning("Failed to download face photo %s: %s", url_path, e)

    logger.info("Face sync complete for household %s (%d profiles)", household_id, len(faces))


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
        logger.info("Event ingested: %s | risk=%s | conf=%.2f | cam=%s",
                    event_type, risk_level, confidence_score, camera_name or camera_id)
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error("Failed to ingest event: %s — %s", e.response.status_code, e.response.text)
    except Exception as e:
        logger.error("Unexpected error ingesting event: %s", e)
    return None
