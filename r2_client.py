"""Cloudflare R2 (S3-compatible) upload helper for event media.

Uploads snapshots and clips to the R2 bucket and returns their public URL
(served from R2_PUBLIC_BASE_URL). If R2 isn't configured or a call fails it
returns None so the caller can fall back to local serving.
"""

import logging
import mimetypes
import threading

import config

logger = logging.getLogger(__name__)

_client = None
_lock = threading.Lock()
_unavailable = False


def is_configured() -> bool:
    return bool(
        config.R2_ENDPOINT
        and config.R2_ACCESS_KEY_ID
        and config.R2_SECRET_ACCESS_KEY
        and config.R2_BUCKET
    )


def _get_client():
    global _client, _unavailable
    if _client is not None or _unavailable:
        return _client
    if not is_configured():
        _unavailable = True
        return None
    with _lock:
        if _client is None and not _unavailable:
            try:
                import boto3
                from botocore.config import Config as BotoConfig

                _client = boto3.client(
                    "s3",
                    endpoint_url=config.R2_ENDPOINT,
                    aws_access_key_id=config.R2_ACCESS_KEY_ID,
                    aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
                    region_name="auto",
                    config=BotoConfig(signature_version="s3v4"),
                )
                logger.info("R2 client ready → %s / %s", config.R2_ENDPOINT, config.R2_BUCKET)
            except Exception as e:
                logger.error("Could not init R2 client: %s", e)
                _unavailable = True
    return _client


def upload_bytes(data: bytes, key: str, content_type: str) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        client.put_object(
            Bucket=config.R2_BUCKET, Key=key, Body=data, ContentType=content_type
        )
        return f"{config.R2_PUBLIC_BASE_URL}/{key}"
    except Exception as e:
        logger.error("R2 upload failed for %s: %s", key, e)
        return None


def upload_file(path: str, key: str, content_type: str | None = None) -> str | None:
    if content_type is None:
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    try:
        with open(path, "rb") as f:
            return upload_bytes(f.read(), key, content_type)
    except Exception as e:
        logger.error("R2 read+upload failed for %s: %s", path, e)
        return None
