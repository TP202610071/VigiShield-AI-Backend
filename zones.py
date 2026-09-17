"""
VigiShield — Zonas de interés (ROI) definidas por el usuario.

El usuario dibuja polígonos sobre un cuadro de su cámara en la app (puerta, reja,
calle, jardín…). Se guardan en el backend por cámara con coordenadas NORMALIZADAS
(0..1) — así son independientes de la resolución: la app dibuja sobre un frame de
un tamaño y el backend de IA procesa a otro, y el polígono sigue calzando.

Este módulo:
  - parsea la config de zonas que llega con la cámara (JSON o lista ya parseada),
  - resuelve en qué zona cae una persona (usa el punto de "pies": centro-inferior
    del bounding box), con prioridad puerta > reja > jardín > calle,
  - es tolerante a ausencia de zonas (si el usuario no dibujó nada, todo funciona
    igual que antes: `zone_for_bbox` devuelve None y el CAIEE opera sin contexto).

Diseño a propósito SIN dependencias pesadas (solo stdlib) para que sea trivial de
testear y no afecte el arranque del pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Tipos de zona reconocidos y su prioridad (mayor = gana cuando un punto cae en
# varias zonas solapadas). La calle es tránsito esperado → prioridad baja.
ZONE_PRIORITY: dict[str, int] = {
    "door": 40,    # puerta / entrada
    "gate": 30,    # reja / portón
    "yard": 20,    # jardín / patio
    "window": 25,  # ventana
    "street": 10,  # calle / vereda
    "custom": 15,
}
DEFAULT_PRIORITY = 15

# Zonas consideradas "de entrada" (donde merodear/aproximarse es más sospechoso).
ENTRY_ZONE_TYPES = {"door", "gate", "window"}


@dataclass
class Zone:
    """Un polígono etiquetado, en coordenadas normalizadas (0..1)."""
    zone_type: str
    name: str
    polygon: list[tuple[float, float]]
    zone_id: str = ""
    priority: int = field(default=DEFAULT_PRIORITY)

    def contains(self, nx: float, ny: float) -> bool:
        """Ray casting: ¿el punto normalizado (nx, ny) está dentro del polígono?"""
        pts = self.polygon
        n = len(pts)
        if n < 3:
            return False
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > ny) != (yj > ny)) and (
                nx < (xj - xi) * (ny - yi) / ((yj - yi) or 1e-12) + xi
            ):
                inside = not inside
            j = i
        return inside

    @property
    def is_entry(self) -> bool:
        return self.zone_type in ENTRY_ZONE_TYPES


class CameraZones:
    """Colección de zonas de UNA cámara + resolución de zona por persona."""

    def __init__(self, zones: list[Zone]):
        # Ordenadas por prioridad descendente para que la primera que contiene el
        # punto sea la de mayor prioridad.
        self._zones = sorted(zones, key=lambda z: z.priority, reverse=True)

    def __bool__(self) -> bool:
        return bool(self._zones)

    def __len__(self) -> int:
        return len(self._zones)

    @property
    def zones(self) -> list[Zone]:
        return self._zones

    def zone_for_point(self, nx: float, ny: float) -> Zone | None:
        """Zona de mayor prioridad que contiene el punto normalizado, o None."""
        for z in self._zones:
            if z.contains(nx, ny):
                return z
        return None

    def zone_for_bbox(self, xyxy, frame_w: int, frame_h: int) -> Zone | None:
        """
        Zona donde 'pisa' una persona. Usamos el centro-inferior del bounding box
        (los pies) como punto de referencia — es lo que define en qué zona del
        suelo está parada, no su cabeza.
        """
        if not self._zones or frame_w <= 0 or frame_h <= 0:
            return None
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        by = y2  # base (pies)
        return self.zone_for_point(cx / frame_w, by / frame_h)


def _coerce_polygon(raw) -> list[tuple[float, float]]:
    """Acepta [[x,y],...] o [{'x':..,'y':..},...] y devuelve [(x,y),...] en 0..1."""
    poly: list[tuple[float, float]] = []
    for p in raw or []:
        try:
            if isinstance(p, dict):
                x, y = float(p.get("x")), float(p.get("y"))
            else:
                x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        # Clampeo defensivo a [0,1] por si vienen coords ligeramente fuera.
        poly.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return poly


def parse_zones(raw) -> CameraZones:
    """
    Construye CameraZones desde lo que mande el backend en el campo `zones` de la
    cámara. `raw` puede ser: None, un string JSON, un dict {version, zones:[...]},
    o directamente una lista de zonas. Nunca lanza — ante error devuelve vacío.
    """
    if not raw:
        return CameraZones([])
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            items = raw.get("zones", [])
        elif isinstance(raw, list):
            items = raw
        else:
            items = []

        zones: list[Zone] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            ztype = str(it.get("type", "custom")).lower().strip() or "custom"
            polygon = _coerce_polygon(it.get("polygon") or it.get("points"))
            if len(polygon) < 3:
                continue  # un polígono necesita al menos 3 vértices
            zones.append(Zone(
                zone_type=ztype,
                name=str(it.get("name", ztype)),
                polygon=polygon,
                zone_id=str(it.get("id", "")),
                priority=ZONE_PRIORITY.get(ztype, DEFAULT_PRIORITY),
            ))
        if zones:
            logger.info("Parsed %d zone(s): %s", len(zones),
                        ", ".join(f"{z.zone_type}" for z in zones))
        return CameraZones(zones)
    except Exception as e:  # nunca romper el pipeline por una config de zonas mala
        logger.warning("Could not parse zones (%s) — continuing without zones", e)
        return CameraZones([])
