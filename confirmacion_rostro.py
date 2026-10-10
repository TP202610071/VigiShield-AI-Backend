"""
Reconocimiento facial con confirmación.

Con el umbral estricto de producción (FACE_DISTANCE_THRESHOLD, 0.35) un
residente solo era reconocido con el rostro casi de frente: con la cabeza un
poco girada la distancia quedaba entre 0.35 y 0.45 y el sistema lo dejaba como
desconocido. Subir el umbral sin más arriesgaba lo contrario: que otra persona
saliera con el nombre de un residente.

Aquí, un rostro con distancia hasta FACE_DISTANCE_CONFIRMACION (0.45, el
umbral que documenta la tesis) cuenta como un VOTO por esa persona, y solo se
la da por reconocida si:
  - el mismo nombre gana en FACE_CONFIRMACION_PASADAS pasadas distintas del
    reconocedor para la misma persona seguida (y más votos que cualquier otro
    nombre), y
  - en cada pasada ese residente le gana al siguiente residente más parecido
    por al menos FACE_MARGEN_MIN (si se parece casi igual a dos residentes, no
    se nombra a ninguno).
El camino estricto de antes (≤ 0.35 en una sola pasada) no cambia.
"""

from __future__ import annotations

import config
from identidad_rostro import _persona_del_rostro


class ConfirmacionRostro:
    def __init__(self):
        self._votos: dict[int, dict[str, int]] = {}

    def observar(self, rostros: list[dict], personas: list[dict], asignacion: dict[int, int],
                 tracker, fresco: bool) -> list[tuple[int, dict]]:
        """Suma los votos de una pasada nueva del reconocedor. Devuelve
        [(track, rostro)] de las personas que quedan confirmadas en esta pasada."""
        if not fresco:
            return []
        confirmados = []
        for r in rostros:
            if r.get("known"):
                continue
            nombre = r.get("candidato")
            if not nombre or r.get("margen", 0.0) < config.FACE_MARGEN_MIN:
                continue
            i = _persona_del_rostro(r["box"], personas)
            tid = asignacion.get(i) if i is not None else None
            track = tracker.get(tid) if tid is not None else None
            if track is None or track.get("state") == "known":
                continue
            votos = self._votos.setdefault(tid, {})
            votos[nombre] = votos.get(nombre, 0) + 1
            otros = max((v for n, v in votos.items() if n != nombre), default=0)
            if votos[nombre] >= config.FACE_CONFIRMACION_PASADAS and votos[nombre] > otros:
                confirmados.append((tid, r))
        return confirmados

    def limpiar(self, tracker) -> None:
        vivos = tracker.active_ids()
        for tid in list(self._votos):
            track = tracker.get(tid)
            if tid not in vivos or (track is not None and track.get("state") == "known"):
                del self._votos[tid]
