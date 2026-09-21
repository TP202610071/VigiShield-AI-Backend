"""
Pruebas de fiabilidad del seguimiento de personas y del motor de intención.

Cubren los fallos que reportó el uso real:
  1. una persona que se MUEVE recibía un id nuevo en cada cuadro y perdía su
     identidad ("Identificando…") y el riesgo acumulado,
  2. durante una intrusión el estado caía de "sospechoso" a "normal" porque el
     modelo dudaba unas ventanas,
  3. una niña jugando quedaba marcada como sospechosa.

Se ejecutan sin cámara ni modelos: `py -3 tests/test_tracking_and_intent.py`
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caiee import Evidence, IntentEngine  # noqa: E402
from event_detector import PersonTracker  # noqa: E402


def _walk(tracker, start_x, step, frames, dt=0.5, t0=0.0):
    """Una persona de 80x200 px avanzando `step` px por cuadro. -> ids asignados."""
    t, x, ids = t0, float(start_x), []
    for _ in range(frames):
        a = tracker.update([{"xyxy": (x, 300.0, x + 80, 500.0)}], t)
        ids.append(a[0]); x += step; t += dt
    return ids


def test_track_survives_walking():
    ids = _walk(PersonTracker(), 100, 45, 20)
    assert len(set(ids)) == 1, f"el track se rompio al caminar: {ids}"


def test_track_survives_running_without_overlap():
    """Zancadas tan largas que el IoU entre cuadros es 0 → salva la cercanía."""
    ids = _walk(PersonTracker(), 100, 95, 15)
    assert len(set(ids)) == 1, f"el track se rompio al correr: {ids}"


def test_identity_survives_occlusion():
    tr = PersonTracker()
    tid = tr.update([{"xyxy": (100.0, 300.0, 180.0, 500.0)}], 0.0)[0]
    tr.mark_known(tid, "Diego")
    tr.update([], 3.0)  # 3 s sin nadie en escena
    again = tr.update([{"xyxy": (140.0, 300.0, 220.0, 500.0)}], 3.5)[0]
    assert again == tid and tr.get(again)["name"] == "Diego"


def test_crossing_people_keep_their_ids():
    """Al cruzarse, la última caja de un track cae sobre la OTRA persona."""
    tr, t = PersonTracker(), 0.0
    ids_a, ids_b = [], []
    for k in range(12):
        a = tr.update([{"xyxy": (100.0 + k * 20, 300.0, 180.0 + k * 20, 500.0)},
                       {"xyxy": (400.0 - k * 20, 300.0, 480.0 - k * 20, 500.0)}], t)
        ids_a.append(a[0]); ids_b.append(a[1]); t += 0.5
    assert len(set(ids_a)) == 1 and len(set(ids_b)) == 1, (ids_a, ids_b)


def test_new_person_far_away_gets_new_id():
    tr = PersonTracker()
    a = tr.update([{"xyxy": (100.0, 300.0, 180.0, 500.0)}], 0.0)[0]
    b = tr.update([{"xyxy": (900.0, 100.0, 980.0, 300.0)}], 1.0)[0]
    assert a != b


def test_risk_holds_while_model_hesitates():
    eng, t = IntentEngine(), 0.0
    forcing = Evidence(is_unknown=True, in_entry_zone=True, zone_type="window",
                       model_suspicious=True, model_conf=0.92, anomalous_posture=True)
    for _ in range(40):
        eng.update(1, forcing, t); t += 0.5
    assert eng.state_of(1)[0] in ("suspect", "high_risk")

    doubting = Evidence(is_unknown=True, in_entry_zone=True, zone_type="window")
    for _ in range(12):  # 6 s sin veredicto del modelo
        eng.update(1, doubting, t); t += 0.5
    assert eng.max_state(t)[0] in ("suspect", "high_risk"), "el incidente se apagó"


def test_risk_survives_disappearing_and_is_forgotten_later():
    eng, t = IntentEngine(risk_memory_seconds=45.0), 0.0
    bad = Evidence(is_unknown=True, in_entry_zone=True, zone_type="door",
                   model_suspicious=True, model_conf=0.95)
    for _ in range(40):
        eng.update(1, bad, t); t += 0.5
    eng.forget(set(), t + 10.0)
    assert eng.state_of(1)[1] > 30, "el riesgo se borró al desaparecer 10 s"
    eng.forget(set(), t + 90.0)
    assert eng.state_of(1)[1] == 0.0, "el riesgo no se olvidó nunca"


def test_child_playing_is_not_high_risk():
    """Modelo apenas sobre el umbral + patio, sin ninguna otra señal."""
    eng, t = IntentEngine(model_min_conf=0.78), 0.0
    kid = Evidence(is_unknown=True, zone_type="yard",
                   model_suspicious=True, model_conf=0.80)
    for _ in range(60):
        eng.update(9, kid, t); t += 0.5
    assert eng.state_of(9)[0] != "high_risk", "falso positivo: riesgo alto por un niño"


def test_real_intruder_fires_an_alert():
    eng, t, fired = IntentEngine(model_min_conf=0.78), 0.0, None
    thief = Evidence(is_unknown=True, in_entry_zone=True, zone_type="door",
                     loitering=True, model_suspicious=True, model_conf=0.95, night=True)
    for _ in range(60):
        fired = fired or eng.update(7, thief, t)
        t += 0.5
    assert eng.state_of(7)[0] == "high_risk" and fired is not None
    assert fired["event_type"] == "SuspiciousIntent"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn(); print(f"  OK   {name}")
        except AssertionError as e:
            failed += 1; print(f"  FALLA {name}: {e}")
    print("\nTODO OK" if not failed else f"\n{failed} prueba(s) fallaron")
    sys.exit(1 if failed else 0)
