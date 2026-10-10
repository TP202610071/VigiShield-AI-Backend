"""
Imagen del rostro en el detalle del evento.

- Acceso reconocido: el rostro capturado por la cámara, ampliado, al lado de
  la foto registrada de esa persona. En ambos se marcan el rostro y sus cinco
  puntos de referencia (ojos, nariz y comisuras de la boca) y se unen los
  puntos correspondientes, con la similitud que calculó el reconocimiento.
  Las líneas ayudan a comparar a simple vista; la decisión la toma ArcFace con
  la similitud del rostro completo, no contando puntos.
- Persona desconocida: el rostro capturado, ampliado. Si hay suficientes fotos
  buenas de su cara, también se suben para que el residente pueda registrarla
  desde la app ("Es alguien conocido").

Todo se sube a R2 junto a la foto del evento, con el mismo nombre más un
sufijo por tipo (`<foto>_reconocido.jpg`; `<foto>_desconocido.jpg` y
`<foto>_desconocido_1.jpg`…), en un hilo aparte para no retrasar la alerta.
Un lote puede traer los dos tipos (comparten la foto): por eso cada uno tiene
su sufijo. La app lo busca por ese nombre; si no existe, no muestra la
sección. El backend borra estos archivos junto con la foto del evento.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

import config
from identidad_rostro import recortar_rostro

logger = logging.getLogger(__name__)

_LADO = 360
_FONDO = (30, 15, 10)           # BGR de #0A0F1E, el fondo de la app
_SUPERFICIE = (48, 30, 22)
_TEXTO = (245, 245, 245)
_SECUNDARIO = (180, 170, 160)
_ACENTO = (255, 194, 0)         # cian de la app
_VERDE = (90, 220, 90)
_AMBAR = (40, 170, 255)
# Ojo derecho, ojo izquierdo, nariz, comisura derecha, comisura izquierda.
_COLORES_PUNTOS = [_ACENTO, _ACENTO, _VERDE, _AMBAR, _AMBAR]


# ── Puntos de referencia del rostro (YuNet de OpenCV, el mismo de DeepFace) ──

def _modelo_yunet() -> str | None:
    candidatos = [
        Path(os.getenv("DEEPFACE_HOME", str(Path.home()))) / ".deepface" / "weights" / "face_detection_yunet_2023mar.onnx",
        Path.home() / ".deepface" / "weights" / "face_detection_yunet_2023mar.onnx",
    ]
    for c in candidatos:
        if c.is_file():
            return str(c)
    return None


def puntos_del_rostro(img) -> tuple | None:
    """(caja x1,y1,x2,y2, puntos 5x2) del rostro más grande, o None."""
    modelo = _modelo_yunet()
    if modelo is None or img is None or img.size == 0:
        return None
    try:
        h, w = img.shape[:2]
        det = cv2.FaceDetectorYN.create(modelo, "", (w, h), 0.6, 0.3, 5000)
        _, caras = det.detect(img)
    except Exception as e:  # noqa: BLE001 — una imagen rara no debe tumbar nada
        logger.debug("YuNet no pudo analizar el rostro: %s", e)
        return None
    if caras is None or len(caras) == 0:
        return None
    c = max(caras, key=lambda f: f[2] * f[3])
    x, y, cw, ch = c[:4]
    puntos = np.array(c[4:14], dtype=np.float32).reshape(5, 2)
    return (x, y, x + cw, y + ch), puntos


# ── Dibujo ───────────────────────────────────────────────────────────────────

def _fuente(tam: int, negrita: bool = False):
    from PIL import ImageFont
    nombres = (["DejaVuSans-Bold.ttf", "segoeuib.ttf", "arialbd.ttf"] if negrita
               else ["DejaVuSans.ttf", "segoeui.ttf", "arial.ttf"])
    carpetas = ["/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"]
    for carpeta in carpetas:
        for n in nombres:
            ruta = Path(carpeta) / n
            if ruta.is_file():
                return ImageFont.truetype(str(ruta), tam)
    return None


def _textos(img, textos: list[tuple]) -> np.ndarray:
    """Escribe [(x, y, texto, tamaño, color_bgr, negrita, centrado)] con tildes."""
    try:
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, t, tam, color, negrita, centrado in textos:
            f = _fuente(tam, negrita)
            if f is None:
                raise RuntimeError("sin fuente")
            if centrado:
                x0, _, x1, _ = d.textbbox((0, 0), t, font=f)
                x = x - (x1 - x0) / 2
            d.text((x, y), t, font=f, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        import unicodedata
        for x, y, t, tam, color, _, centrado in textos:
            a = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
            escala = tam / 30
            if centrado:
                (tw, _), _ = cv2.getTextSize(a, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
                x = x - tw / 2
            cv2.putText(img, a, (int(x), int(y + tam)), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)
        return img


def _panel(img, lado: int = _LADO):
    """Ajusta la imagen a un cuadrado (sin deformarla). -> (panel, escala, dx, dy)."""
    panel = np.full((lado, lado, 3), _SUPERFICIE, dtype=np.uint8)
    h, w = img.shape[:2]
    escala = lado / max(h, w)
    nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
    interp = cv2.INTER_CUBIC if escala > 1 else cv2.INTER_AREA
    dx, dy = (lado - nw) // 2, (lado - nh) // 2
    panel[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return panel, escala, dx, dy


def _marcar(lienzo, rostro, escala, ox, oy, color_caja):
    """Dibuja la caja y los puntos del rostro en el lienzo. -> puntos en el lienzo."""
    if rostro is None:
        return None
    caja, puntos = rostro
    x1, x2 = int(ox + caja[0] * escala), int(ox + caja[2] * escala)
    y1, y2 = int(oy + caja[1] * escala), int(oy + caja[3] * escala)
    cv2.rectangle(lienzo, (x1, y1), (x2, y2), color_caja, 2, cv2.LINE_AA)
    en_lienzo = [(int(ox + px * escala), int(oy + py * escala)) for px, py in puntos]
    for (px, py), color in zip(en_lienzo, _COLORES_PUNTOS):
        cv2.circle(lienzo, (px, py), 5, color, -1, cv2.LINE_AA)
        cv2.circle(lienzo, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
    return en_lienzo


def componer_comparacion(captura, referencia, nombre: str, distancia: float | None):
    """Captura y foto registrada lado a lado, con rostro, puntos y similitud."""
    margen, separacion, titulo, etiqueta, pie = 20, 40, 56, 30, 70
    ancho = margen * 2 + separacion + _LADO * 2
    alto = titulo + etiqueta + _LADO + pie
    lienzo = np.full((alto, ancho, 3), _FONDO, dtype=np.uint8)
    xa, xb, y0 = margen, margen + _LADO + separacion, titulo + etiqueta

    pa, ea, dxa, dya = _panel(captura)
    lienzo[y0:y0 + _LADO, xa:xa + _LADO] = pa
    puntos_a = _marcar(lienzo, puntos_del_rostro(captura), ea, xa + dxa, y0 + dya, _VERDE)
    puntos_b = None
    if referencia is not None:
        pb, eb, dxb, dyb = _panel(referencia)
        lienzo[y0:y0 + _LADO, xb:xb + _LADO] = pb
        puntos_b = _marcar(lienzo, puntos_del_rostro(referencia), eb, xb + dxb, y0 + dyb, _VERDE)
    if puntos_a and puntos_b:
        for pa_, pb_, color in zip(puntos_a, puntos_b, _COLORES_PUNTOS):
            cv2.line(lienzo, pa_, pb_, color, 1, cv2.LINE_AA)

    textos = [
        (ancho / 2, 14, f"Acceso reconocido: {nombre}", 22, _TEXTO, True, True),
        (xa + _LADO / 2, titulo + 4, "Captura de la cámara", 16, _SECUNDARIO, False, True),
        (xb + _LADO / 2, titulo + 4, "Foto registrada", 16, _SECUNDARIO, False, True),
    ]
    if distancia is not None:
        similitud = max(0.0, 1.0 - float(distancia))
        minimo = max(0.0, 1.0 - config.FACE_DISTANCE_THRESHOLD)
        textos.append((ancho / 2, y0 + _LADO + 12,
                       f"Similitud del rostro: {similitud * 100:.0f} %   ·   mínimo para reconocer: {minimo * 100:.0f} %",
                       17, _VERDE, True, True))
    textos.append((ancho / 2, y0 + _LADO + 40,
                   "Puntos de referencia: ojos, nariz y comisuras de la boca",
                   14, _SECUNDARIO, False, True))
    return _textos(lienzo, textos)


def componer_desconocido(captura):
    """El rostro de una persona desconocida, ampliado."""
    margen, titulo, etiqueta, pie = 20, 56, 30, 44
    ancho = margen * 2 + _LADO
    alto = titulo + etiqueta + _LADO + pie
    lienzo = np.full((alto, ancho, 3), _FONDO, dtype=np.uint8)
    y0 = titulo + etiqueta
    p, e, dx, dy = _panel(captura)
    lienzo[y0:y0 + _LADO, margen:margen + _LADO] = p
    _marcar(lienzo, puntos_del_rostro(captura), e, margen + dx, y0 + dy, _AMBAR)
    return _textos(lienzo, [
        (ancho / 2, 14, "Persona desconocida", 22, _TEXTO, True, True),
        (ancho / 2, titulo + 4, "Rostro capturado por la cámara", 16, _SECUNDARIO, False, True),
        (ancho / 2, y0 + _LADO + 12, "No coincide con ningún residente registrado", 15, _AMBAR, False, True),
    ])


# ── Datos pendientes por cámara ──────────────────────────────────────────────

class RostrosDelEvento:
    """Guarda, por tipo de evento, el rostro del último evento candidato hasta
    que el worker sabe si el evento salió (cooldown, interruptores) y tiene la
    foto del evento."""

    def __init__(self):
        self._pendiente: dict[str, dict] = {}
        self._cerrojo = threading.Lock()

    def al_reconocer(self, cuadro, rostro: dict) -> None:
        recorte = recortar_rostro(cuadro, rostro["box"])
        if recorte is not None:
            with self._cerrojo:
                self._pendiente["FaceRecognized"] = {
                    "recorte": recorte, "nombre": rostro.get("name") or "",
                    "distancia": rostro.get("distance")}

    def al_desconocido(self, fotos: list) -> None:
        if fotos:
            with self._cerrojo:
                self._pendiente["UnknownFace"] = {"fotos": list(fotos)}

    def tomar(self, tipos: set[str]) -> dict:
        with self._cerrojo:
            datos = {t: self._pendiente.pop(t) for t in list(self._pendiente) if t in tipos}
        return datos


def _foto_registrada(household_id: str, nombre: str):
    """Una foto registrada de esa persona (la primera en la que se ve un rostro)."""
    carpeta = Path(config.FACES_DIR) / household_id / nombre
    if not nombre or not carpeta.is_dir():
        return None
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        rostro = puntos_del_rostro(img)
        if rostro is None:
            continue
        # Se recorta alrededor del rostro, como la captura, para que se comparen
        # a la misma escala.
        return recortar_rostro(img, rostro[0], margen=0.40) if rostro else img
    return None


def _jpg(img, calidad: int = 88) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    return buf.tobytes() if ok else None


def clave_base(url_foto: str | None) -> str | None:
    """`events/xxx` a partir de la URL pública de la foto del evento (o None si
    la foto no está en R2)."""
    base = config.R2_PUBLIC_BASE_URL.rstrip("/") + "/"
    if not url_foto or not url_foto.startswith(base):
        return None
    clave = url_foto[len(base):].split("?")[0]
    if not clave.lower().endswith(".jpg"):
        return None
    return clave[:-4]


def _procesar(datos: dict, base: str, household_id: str) -> None:
    import r2_client
    try:
        if "FaceRecognized" in datos:
            d = datos["FaceRecognized"]
            ref = _foto_registrada(household_id, d["nombre"])
            img = componer_comparacion(d["recorte"], ref, d["nombre"], d["distancia"])
            data = _jpg(img)
            if data and r2_client.upload_bytes(data, f"{base}_reconocido.jpg", "image/jpeg"):
                logger.info("Comparación de rostro subida (%s)", base)
        if "UnknownFace" in datos:
            fotos = datos["UnknownFace"]["fotos"]
            mejor = max(fotos, key=lambda f: f[0])[1]
            data = _jpg(componer_desconocido(mejor))
            if data and r2_client.upload_bytes(data, f"{base}_desconocido.jpg", "image/jpeg"):
                logger.info("Rostro desconocido subido (%s)", base)
            buenas = [rec for alto, rec in fotos if alto >= config.ROSTRO_REGISTRO_MIN_PX]
            if len(buenas) >= config.ROSTRO_REGISTRO_FOTOS:
                for k, rec in enumerate(buenas[:config.ROSTRO_REGISTRO_FOTOS], start=1):
                    data = _jpg(rec, 92)
                    if data:
                        r2_client.upload_bytes(data, f"{base}_desconocido_{k}.jpg", "image/jpeg")
                logger.info("Fotos para registrar el rostro subidas (%s)", base)
    except Exception as e:  # noqa: BLE001 — la alerta ya salió; esto es un extra
        logger.warning("No se pudo armar la imagen del rostro: %s", e)


def subir_en_segundo_plano(detector, url_foto: str | None, eventos: list[dict]) -> None:
    """Si en este lote salió un acceso reconocido o una persona desconocida,
    arma su imagen del rostro y la sube junto a la foto del evento."""
    rostros = getattr(detector, "_rostros", None)
    if not config.ROSTRO_EN_EVENTO or rostros is None:
        return
    tipos = {e.get("event_type") for e in eventos}
    datos = rostros.tomar(tipos & {"FaceRecognized", "UnknownFace"})
    base = clave_base(url_foto)
    if not datos or base is None:
        return
    threading.Thread(target=_procesar, args=(datos, base, detector.household_id),
                     daemon=True, name=f"rostro-{base[-8:]}").start()
