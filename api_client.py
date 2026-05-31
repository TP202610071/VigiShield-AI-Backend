"""
HTTP client for communicating with the VigiShield ASP.NET Core backend.
"""

import logging
import requests
from config import BACKEND_API_URL, INTERNAL_API_KEY, HOUSEHOLD_ID

logger = logging.getLogger(__name__)

_HEADERS = {
    "X-Api-Key": INTERNAL_API_KEY,
    "Content-Type": "application/json",
}


def get_stream_config() -> dict | None:
    """
    Fetch camera RTSP URL and stream mode from the backend.
    Returns dict with keys: rtspUrl, streamMode, streamKey, isConfigured
    """
    if not HOUSEHOLD_ID:
        logger.error("HOUSEHOLD_ID is not set in .env — cannot fetch stream config.")
        return None
    try:
        resp = requests.get(
            f"{BACKEND_API_URL}/api/stream/ai-config",
            headers=_HEADERS,
            params={"householdId": HOUSEHOLD_ID},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to backend at {BACKEND_API_URL}. Is it running?")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Backend returned error: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error fetching stream config: {e}")
    return None


def ingest_event(
    event_type: str,
    confidence_score: float,
    risk_level: str,
    is_nighttime: bool = False,
    person_name: str | None = None,
    image_capture_path: str | None = None,
    video_clip_path: str | None = None,
) -> dict | None:
    """
    Send a detected security event to the backend.
    event_type must match the EventType enum:
        FaceRecognized, UnknownFace, LowConfidenceFace, RecurrentUnknownFace,
        ForcedAccessAttempt, Tailgating, Climbing, PhysicalAggression
    risk_level: None, Low, Medium, High, Critical
    """
    payload = {
        "householdId": HOUSEHOLD_ID,
        "eventType": event_type,
        "confidenceScore": round(confidence_score, 4),
        "riskLevel": risk_level,
        "isNighttime": is_nighttime,
        "personName": person_name,
        "imageCapturePath": image_capture_path,
        "videoClipPath": video_clip_path,
    }
    # Remove None values to keep payload clean
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        resp = requests.post(
            f"{BACKEND_API_URL}/api/events/ingest",
            json=payload,
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            f"✅ Event ingested → {event_type} | confidence={confidence_score:.2f} | risk={risk_level}"
        )
        return data
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to ingest event: {e.response.status_code} — {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error ingesting event: {e}")
    return None
