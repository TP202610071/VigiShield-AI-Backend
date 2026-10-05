"""
La captura cruda para el editor de zonas rechaza claves raras y responde 404
cuando la cámara no tiene video (en vez de colgarse o devolver basura).

    py -3 -m pytest tests/test_snapshot.py
"""
import http.server
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import frame_server  # noqa: E402


def _servidor():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), frame_server._FrameHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _codigo(url):
    try:
        return urllib.request.urlopen(url, timeout=20).status
    except urllib.error.HTTPError as e:
        return e.code


def test_clave_invalida_404():
    srv = _servidor()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assert _codigo(f"{base}/snapshot/..%2Fetc") == 404
        assert _codigo(f"{base}/snapshot/a-b") == 404
    finally:
        srv.shutdown()


def test_sin_video_404():
    srv = _servidor()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        assert _codigo(f"{base}/snapshot/noexiste123") == 404
    finally:
        srv.shutdown()
