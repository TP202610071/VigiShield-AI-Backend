"""
Mejoras de detección del 2026-10-10.

  1. El merodeo era un reloj de escena que no se reiniciaba mientras hubiera
     alguien en la imagen; ahora es por persona y vuelve a cero al irse.
  2. Una persona que nunca mostró el rostro ya no pasa a "Desconocido".
  3. La imagen del rostro del evento se arma aunque no se encuentren los
     puntos de la cara, y su nombre en R2 se deriva de la foto del evento.

Sin cámara ni modelos: `py -3 tests/test_mejoras_2026_10.py`
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from event_detector import PersonTracker  # noqa: E402
from identidad_rostro import IdentidadPorRostro  # noqa: E402
from merodeo import MerodeoPorPersona  # noqa: E402
from rostros_evento import (RostrosDelEvento, clave_base, componer_comparacion,  # noqa: E402
                            componer_desconocido)
from zones import parse_zones  # noqa: E402

CUADRO = (720, 1280, 3)
CAJA = (500.0, 200.0, 620.0, 600.0)   # pies en (560, 600) → (0.44, 0.83)


def _quieta(tracker, merodeo, desde, hasta, dt=0.5, caja=CAJA, zonas=None):
    """Una persona quieta entre `desde` y `hasta`. -> (tid, [tracks nuevos de merodeo])."""
    t, tid, eventos = desde, None, []
    while t <= hasta:
        asignacion = tracker.update([{"xyxy": caja}], t)
        tid = asignacion[0]
        eventos += merodeo.actualizar([{"xyxy": caja}], asignacion, tracker, zonas, CUADRO, t)
        t += dt
    return tid, eventos


def test_merodeo_cuenta_por_persona_y_avisa_una_vez():
    tr, m = PersonTracker(), MerodeoPorPersona()
    tid, eventos = _quieta(tr, m, 0.0, config.LOITER_SECONDS + 15)
    assert eventos == [tid], eventos
    assert m.merodeando(tid, config.LOITER_SECONDS + 15)


def test_merodeo_vuelve_a_cero_al_irse():
    """El fallo reportado: el contador nunca se reiniciaba."""
    tr, m = PersonTracker(), MerodeoPorPersona()
    fin = config.LOITER_SECONDS + 5
    tid, _ = _quieta(tr, m, 0.0, fin)
    assert m.maximo(fin) >= config.LOITER_SECONDS
    # Se va más que la pausa y vuelve: su permanencia arranca de nuevo.
    vuelta = fin + config.MERODEO_PAUSA_SEGUNDOS + 3
    tr.update([], fin + 1)
    _quieta(tr, m, vuelta, vuelta + 2)
    assert m.segundos(tid, vuelta + 2) <= 2.5, m.segundos(tid, vuelta + 2)
    assert m.maximo(vuelta + 2) == 0.0


def test_otra_persona_no_hereda_el_tiempo():
    """Antes: A se quedaba un rato, B llegaba y el reloj seguía contando."""
    tr, m = PersonTracker(), MerodeoPorPersona()
    _quieta(tr, m, 0.0, 15.0, caja=(100.0, 200.0, 220.0, 600.0))
    tr.update([], 16.0)
    t_b = 16.0 + config.PERSON_TRACK_LOST_SECONDS + 1   # A ya no se puede reasociar
    tid_b, eventos = _quieta(tr, m, t_b, t_b + 10, caja=(900.0, 200.0, 1020.0, 600.0))
    assert eventos == [] and m.segundos(tid_b, t_b + 10) <= 10.5


def test_residente_reconocido_no_merodea():
    tr, m = PersonTracker(), MerodeoPorPersona()
    tid = tr.update([{"xyxy": CAJA}], 0.0)[0]
    tr.mark_known(tid, "Diego")
    _, eventos = _quieta(tr, m, 0.5, config.LOITER_SECONDS + 10)
    assert eventos == [] and m.maximo(config.LOITER_SECONDS + 10) == 0.0


def test_con_zonas_la_calle_no_cuenta_y_la_puerta_si():
    calle = parse_zones([{"type": "street", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}])
    puerta = parse_zones([{"type": "door", "polygon": [[0, 0.5], [1, 0.5], [1, 1], [0, 1]]}])
    fuera = parse_zones([{"type": "door", "polygon": [[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]]}])
    fin = config.LOITER_SECONDS + 10
    for zonas, esperado in ((calle, False), (puerta, True), (fuera, False)):
        tr, m = PersonTracker(), MerodeoPorPersona()
        _, eventos = _quieta(tr, m, 0.0, fin, zonas=zonas)
        assert bool(eventos) == esperado, (zonas.zones[0].zone_type, eventos)


def _rostro(alto=60, conocido=False):
    """Un rostro dentro de CAJA (cabeza de la persona)."""
    return {"box": (540, 210, 540 + int(alto * 0.8), 210 + alto), "name": "Desconocido",
            "known": conocido, "distance": 1.0}


def test_sin_rostro_no_pasa_a_desconocido():
    tr, ident = PersonTracker(), IdentidadPorRostro()
    cuadro = np.zeros(CUADRO, dtype=np.uint8)
    t, salieron = 0.0, []
    while t < config.UNKNOWN_ALERT_GRACE_SECONDS + 20:
        asig = tr.update([{"xyxy": CAJA}], t)
        ident.observar([], [{"xyxy": CAJA}], asig, cuadro, True, t)
        salieron += ident.pendientes_desconocidos(tr, t)
        t += 0.5
    tid = asig[0]
    assert salieron == [] and tr.get(tid)["state"] == "pending"
    assert ident.sin_rostro(tid, tr, t)


def test_con_rostro_visto_si_pasa_a_desconocido_una_vez():
    tr, ident = PersonTracker(), IdentidadPorRostro()
    cuadro = np.zeros(CUADRO, dtype=np.uint8)
    t, salieron, k = 0.0, [], 0
    while t < config.UNKNOWN_ALERT_GRACE_SECONDS + 5:
        asig = tr.update([{"xyxy": CAJA}], t)
        fresco = (k % 4 == 0)       # el reconocedor corre cada pocos cuadros
        ident.observar([_rostro()], [{"xyxy": CAJA}], asig, cuadro, fresco, t)
        salieron += ident.pendientes_desconocidos(tr, t)
        t += 0.5; k += 1
    assert salieron == [asig[0]] and tr.get(asig[0])["state"] == "unknown"
    fotos = ident.fotos(asig[0])
    assert 1 <= len(fotos) <= config.ROSTRO_REGISTRO_FOTOS


def test_rostro_chico_o_repetido_no_cuenta():
    tr, ident = PersonTracker(), IdentidadPorRostro()
    cuadro = np.zeros(CUADRO, dtype=np.uint8)
    t = 0.0
    while t < config.UNKNOWN_ALERT_GRACE_SECONDS + 5:
        asig = tr.update([{"xyxy": CAJA}], t)
        ident.observar([_rostro(alto=config.ROSTRO_MIN_PX - 5)], [{"xyxy": CAJA}], asig, cuadro, True, t)
        ident.observar([_rostro()], [{"xyxy": CAJA}], asig, cuadro, False, t)  # caché, no cuenta
        assert ident.pendientes_desconocidos(tr, t) == []
        t += 0.5


def test_imagenes_del_rostro_sin_puntos():
    cara = np.full((120, 100, 3), 128, dtype=np.uint8)
    comp = componer_comparacion(cara, cara.copy(), "Diego", 0.20)
    assert comp.ndim == 3 and comp.shape[1] > comp.shape[0] > 300
    sola = componer_desconocido(cara)
    assert sola.ndim == 3 and sola.shape[0] > sola.shape[1]
    sin_ref = componer_comparacion(cara, None, "Diego", None)
    assert sin_ref.shape == comp.shape


def test_clave_y_datos_pendientes():
    base = config.R2_PUBLIC_BASE_URL
    assert clave_base(f"{base}/events/20261010_abc123.jpg") == "events/20261010_abc123"
    assert clave_base("http://localhost:5050/ai/event/x.jpg") is None
    assert clave_base(None) is None
    r = RostrosDelEvento()
    cuadro = np.zeros(CUADRO, dtype=np.uint8)
    r.al_reconocer(cuadro, {"box": (540, 210, 590, 270), "name": "Diego", "distance": 0.2})
    r.al_desconocido([(70, np.zeros((80, 70, 3), dtype=np.uint8))])
    assert set(r.tomar({"UnknownFace"})) == {"UnknownFace"}
    assert set(r.tomar({"FaceRecognized", "UnknownFace"})) == {"FaceRecognized"}
    assert r.tomar({"FaceRecognized"}) == {}


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            try:
                fn(); print("OK  ", nombre)
            except AssertionError as e:
                fallos += 1; print("FALLA", nombre, e)
    sys.exit(1 if fallos else 0)
