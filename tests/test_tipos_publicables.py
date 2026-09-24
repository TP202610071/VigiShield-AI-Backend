"""
El catalogo de eventos que publica el sistema debe corresponder a lo que una
camara de vivienda puede sostener de verdad.

El modelo de actividad viene de UCF-Crime y distingue clases como "Explosion",
"Arson" o "Roadaccidents". Publicarlas deja al producto ofreciendo alertas que
nunca va a poder demostrar, asi que se descartan o se doblan sobre el evento
vecino que si tiene sentido.

    py -3 -m pytest tests/test_tipos_publicables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_detector import (  # noqa: E402
    _ACTIVITY_EVENT_MAP,
    _ACTIVITY_IGNORED,
    _EVENT_ES,
)

# Lo que el backend deja gobernar (Common/Alerts/AlertableEvents.cs). Si cambia
# alli, esta lista tiene que cambiar aqui: son las dos caras del mismo contrato.
GOBERNABLES = {
    "UnknownFace", "Tailgating", "SuspiciousIntent", "WeaponDetected",
    "ForcedAccessAttempt", "PhysicalAggression", "Burglary", "Robbery",
    "Stealing",
}


def test_todo_lo_que_publica_la_actividad_es_gobernable():
    """Ningun evento del modelo puede salir sin interruptor en la app."""
    publicados = {evento for evento, _ in _ACTIVITY_EVENT_MAP.values()}
    assert publicados <= GOBERNABLES, publicados - GOBERNABLES


def test_las_clases_no_defendibles_no_se_publican():
    """Explosion, incendio, arresto y accidentes quedan fuera del catalogo."""
    for clase in ("Explosion", "Arson", "Arrest", "Roadaccidents"):
        assert clase in _ACTIVITY_IGNORED
        assert clase not in _ACTIVITY_EVENT_MAP


def test_las_clases_vecinas_se_doblan_en_vez_de_perderse():
    """Hurto en tienda y asalto/abuso no se tiran: alimentan el evento util."""
    assert _ACTIVITY_EVENT_MAP["Shoplifting"][0] == "Stealing"
    assert _ACTIVITY_EVENT_MAP["Assault"][0] == "PhysicalAggression"
    assert _ACTIVITY_EVENT_MAP["Abuse"][0] == "PhysicalAggression"


def test_ninguna_clase_esta_mapeada_e_ignorada_a_la_vez():
    assert not (_ACTIVITY_IGNORED & set(_ACTIVITY_EVENT_MAP))


def test_los_eventos_gobernables_tienen_nombre_en_espanol():
    """Sin traduccion, la alerta llegaria con el nombre interno en ingles."""
    faltan = [e for e in GOBERNABLES if e not in _EVENT_ES]
    assert not faltan, faltan
