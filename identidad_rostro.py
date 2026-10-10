"""
"Persona desconocida" solo con rostro.

Antes, cualquier persona que no fuera reconocida en los primeros segundos
pasaba a "Desconocido" y disparaba el evento, aunque el sistema nunca hubiera
visto su cara: alguien de espaldas, con la cara tapada o lejos de la cámara
quedaba como desconocido, sumaba riesgo en el motor y avisaba.

Ahora una persona solo es desconocida si su rostro se vio (con un tamaño
suficiente, en al menos UNKNOWN_FACE_MIN_STREAK pasadas de reconocimiento) y
no coincide con ningún residente. Quien no mostró la cara queda
"Sin identificar": no genera el evento de persona desconocida ni suma ese
peso en el motor. El resto de señales (zona, merodeo, postura, arma, modelo de
pose) siguen funcionando igual para esa persona.

De paso se guardan los mejores recortes del rostro de cada persona
desconocida: se muestran en el detalle del evento y permiten registrarla.
"""

from __future__ import annotations

import config

_MARGEN = 0.40        # margen alrededor del rostro en el recorte (por lado)
_MAX_LADO = 320       # los recortes se guardan reducidos: es evidencia, no archivo
_ENTRE_FOTOS = 1.0    # segundos mínimos entre recortes guardados


def recortar_rostro(cuadro, caja, margen: float = _MARGEN):
    """Recorte del rostro con margen, en píxeles del cuadro original (o None)."""
    import cv2
    alto_img, ancho_img = cuadro.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in caja)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    mx, my = int(w * margen), int(h * margen)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(ancho_img, x2 + mx), min(alto_img, y2 + my)
    recorte = cuadro[y1:y2, x1:x2]
    if recorte.size == 0:
        return None
    lado = max(recorte.shape[:2])
    if lado > _MAX_LADO:
        f = _MAX_LADO / lado
        recorte = cv2.resize(recorte, (max(1, int(recorte.shape[1] * f)), max(1, int(recorte.shape[0] * f))),
                             interpolation=cv2.INTER_AREA)
    return recorte.copy()


def _persona_del_rostro(caja_rostro, personas: list[dict]) -> int | None:
    """Índice de la persona cuya caja contiene el centro del rostro (la más
    chica si hay varias: es la que está delante)."""
    cx = (caja_rostro[0] + caja_rostro[2]) / 2
    cy = (caja_rostro[1] + caja_rostro[3]) / 2
    mejor, area_mejor = None, None
    for i, p in enumerate(personas):
        x1, y1, x2, y2 = p["xyxy"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            if area_mejor is None or area < area_mejor:
                mejor, area_mejor = i, area
    return mejor


class IdentidadPorRostro:
    def __init__(self):
        # tid -> {"vistas": pasadas con rostro desconocido, "fotos": [(t, alto, recorte)]}
        self._por_track: dict[int, dict] = {}

    def observar(self, rostros: list[dict], personas: list[dict], asignacion: dict[int, int],
                 cuadro, fresco: bool, ahora: float) -> None:
        """Registra los rostros NO reconocidos de una pasada nueva del
        reconocedor (los rostros quedan en caché entre pasadas: contarlos en
        cada cuadro inflaría las vistas)."""
        if not fresco:
            return
        vistos: set[int] = set()
        for r in rostros:
            if r.get("known"):
                continue
            x1, y1, x2, y2 = r["box"]
            alto = y2 - y1
            if alto < config.ROSTRO_MIN_PX:
                continue
            i = _persona_del_rostro(r["box"], personas)
            tid = asignacion.get(i) if i is not None else None
            if tid is None or tid in vistos:
                continue
            vistos.add(tid)
            st = self._por_track.setdefault(tid, {"vistas": 0, "fotos": []})
            st["vistas"] += 1
            recorte = recortar_rostro(cuadro, r["box"])
            if recorte is not None:
                self._guardar_foto(st, ahora, alto, recorte)

    @staticmethod
    def _guardar_foto(st: dict, ahora: float, alto: int, recorte) -> None:
        fotos = st["fotos"]
        if fotos and ahora - fotos[-1][0] < _ENTRE_FOTOS:
            if alto > fotos[-1][1]:
                fotos[-1] = (ahora, alto, recorte)
            return
        fotos.append((ahora, alto, recorte))
        if len(fotos) > config.ROSTRO_REGISTRO_FOTOS:
            # Se quedan las más grandes, en orden de llegada.
            fotos.sort(key=lambda f: f[1], reverse=True)
            del fotos[config.ROSTRO_REGISTRO_FOTOS:]
            fotos.sort(key=lambda f: f[0])

    def pendientes_desconocidos(self, tracker, ahora: float) -> list[int]:
        """Como PersonTracker.pending_unknown, pero exige que el rostro se haya
        visto. Marca el track como desconocido (una sola vez)."""
        salida = []
        for tid in tracker.active_ids():
            tr = tracker.get(tid)
            if tr is None or tr["state"] != "pending" or tr["alerted"]:
                continue
            if ahora - tr["last_seen"] > config.PERSON_TRACK_TTL_SECONDS:
                continue
            if ahora - tr["first_seen"] < config.UNKNOWN_ALERT_GRACE_SECONDS:
                continue
            if self._por_track.get(tid, {}).get("vistas", 0) < config.UNKNOWN_FACE_MIN_STREAK:
                continue
            tr["alerted"] = True
            tr["state"] = "unknown"
            salida.append(tid)
        return salida

    def sin_rostro(self, tid: int | None, tracker, ahora: float) -> bool:
        """Persona que pasó el tiempo de gracia sin mostrar el rostro."""
        tr = tracker.get(tid) if tid is not None else None
        return bool(tr and tr["state"] == "pending"
                    and ahora - tr["first_seen"] >= config.UNKNOWN_ALERT_GRACE_SECONDS
                    and self._por_track.get(tid, {}).get("vistas", 0) == 0)

    def fotos(self, tid: int) -> list:
        """Recortes del rostro de la persona, del más antiguo al más reciente:
        [(alto_px, recorte)]."""
        return [(alto, rec) for _, alto, rec in self._por_track.get(tid, {}).get("fotos", [])]

    def limpiar(self, tracker) -> None:
        vivos = tracker.active_ids()
        for tid in list(self._por_track):
            if tid not in vivos:
                del self._por_track[tid]
