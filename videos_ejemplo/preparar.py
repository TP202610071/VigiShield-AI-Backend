"""
Descarga de Pexels los videos de ejemplo y los deja listos para MediaMTX.

    py -3 videos_ejemplo/preparar.py <carpeta_salida>

Cada clip se recorta a 16:9 y se recodifica a 1280x720, 25 fps, H.264 sin
B-frames y con un keyframe por segundo, sin audio. Así MediaMTX lo puede
reenviar con `-c copy` en bucle y la app engancha el video enseguida.

Las zonas de cada video NO están aquí: viven en el backend
(SampleVideoCatalog.cs), que es quien se las pasa a la IA con la cámara.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0"}
FILTRO = "crop='min(iw,ih*16/9)':'min(ih,iw*9/16)',scale=1280:720,fps=25,format=yuv420p"


def url_original(pexels_id: str) -> str:
    """El enlace de descarga de Pexels redirige al archivo en su CDN."""
    req = urllib.request.Request(f"https://www.pexels.com/download/video/{pexels_id}/", headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.geturl()


def main(salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    for v in json.loads((AQUI / "manifiesto.json").read_text(encoding="utf-8")):
        crudo = salida / f"{v['key']}.orig.mp4"
        final = salida / f"{v['key']}.mp4"
        if not crudo.exists():
            req = urllib.request.Request(url_original(v["pexelsId"]), headers=UA)
            with urllib.request.urlopen(req) as r:
                crudo.write_bytes(r.read())
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-i", str(crudo), "-an", "-vf", FILTRO,
            "-c:v", "libx264", "-preset", "slow", "-profile:v", "main", "-crf", "23",
            "-maxrate", "2500k", "-bufsize", "5000k", "-g", "25", "-keyint_min", "25",
            "-sc_threshold", "0", "-bf", "0", "-movflags", "+faststart", str(final),
        ], check=True)
        crudo.unlink()
        print(f"{v['key']}  {final.stat().st_size // 1024} KB  {v['titulo']}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "ejemplos"))
