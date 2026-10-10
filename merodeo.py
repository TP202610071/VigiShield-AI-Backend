"""
Merodeo por persona.

Antes el merodeo era un solo reloj para toda la escena: arrancaba cuando
aparecía cualquier persona y solo volvía a cero tras 6 s seguidos sin nadie.
Con gente entrando y saliendo, un residente moviéndose por el patio o una
detección fija (un abrigo colgado), el contador crecía sin parar
("Merodeo 900s") y el evento se repetía cada vez que vencía la espera entre
alertas.

Aquí cada persona seguida lleva su propio contador:
  - mide cuánto lleva AQUÍ sin irse; si sale de la zona o deja de verse más de
    MERODEO_PAUSA_SEGUNDOS, vuelve a cero;
  - un residente reconocido no merodea en su casa;
  - si la cámara tiene zonas dibujadas, solo cuenta dentro de ellas y nunca en
    la calle (pasar o esperar en la vereda es tránsito esperado). Sin zonas,
    cuenta en toda la imagen, como antes;
  - cada persona genera a lo más un evento de merodeo.
"""

from __future__ import annotations

import config


class MerodeoPorPersona:
    def __init__(self):
        # tid -> {"desde": inicio del tramo continuo, "ultimo": última vez que
        #         contó, "alertado": ya generó su evento}
        self._estado: dict[int, dict] = {}

    @staticmethod
    def _cuenta_aqui(xyxy, zonas, ancho: int, alto: int) -> bool:
        """¿Estar en esta posición cuenta como merodear?"""
        if not zonas:
            return True
        zona = zonas.zone_for_bbox(xyxy, ancho, alto)
        return zona is not None and zona.zone_type != "street"

    def actualizar(self, personas: list[dict], asignacion: dict[int, int], tracker,
                   zonas, forma_cuadro, ahora: float) -> list[int]:
        """Actualiza los contadores con las personas de este cuadro. Devuelve los
        tracks que ACABAN de cumplir el tiempo de merodeo (evento una sola vez)."""
        alto, ancho = forma_cuadro[:2]
        for i, p in enumerate(personas):
            tid = asignacion.get(i)
            track = tracker.get(tid) if tid is not None else None
            if track is None:
                continue
            st = self._estado.setdefault(tid, {"desde": None, "ultimo": None, "alertado": False})
            if track.get("state") == "known" or not self._cuenta_aqui(p["xyxy"], zonas, ancho, alto):
                continue  # no suma; si sigue así más de la pausa, se reinicia
            if st["ultimo"] is None or ahora - st["ultimo"] > config.MERODEO_PAUSA_SEGUNDOS:
                st["desde"] = ahora
            st["ultimo"] = ahora

        # Residentes reconocidos y tracks que ya no existen: fuera.
        vivos = tracker.active_ids()
        for tid in list(self._estado):
            track = tracker.get(tid)
            if tid not in vivos or (track is not None and track.get("state") == "known"):
                del self._estado[tid]

        nuevos = []
        for tid, st in self._estado.items():
            if not st["alertado"] and self.segundos(tid, ahora) >= config.LOITER_SECONDS:
                st["alertado"] = True
                nuevos.append(tid)
        return nuevos

    def segundos(self, tid: int, ahora: float) -> float:
        """Permanencia continua de la persona; 0 si se fue o salió de la zona."""
        st = self._estado.get(tid)
        if not st or st["desde"] is None or st["ultimo"] is None:
            return 0.0
        if ahora - st["ultimo"] > config.MERODEO_PAUSA_SEGUNDOS:
            return 0.0
        return st["ultimo"] - st["desde"]

    def merodeando(self, tid: int, ahora: float) -> bool:
        return self.segundos(tid, ahora) >= config.LOITER_SECONDS

    def maximo(self, ahora: float) -> float:
        """La permanencia más larga entre quienes ya merodean (para el aviso en
        pantalla); 0 si nadie."""
        valores = [self.segundos(t, ahora) for t in self._estado]
        valores = [v for v in valores if v >= config.LOITER_SECONDS]
        return max(valores) if valores else 0.0
