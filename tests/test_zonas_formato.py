"""
Las zonas deben leerse aunque vengan en PascalCase.

El backend las guardó un tiempo como {"Id","Type","Polygon"}; el motor las
buscaba en minúsculas y las ignoraba sin avisar.

    py -3 -m pytest tests/test_zonas_formato.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zones import parse_zones  # noqa: E402

POLI = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6]]


def test_pascal_case_del_backend():
    raw = json.dumps({"Version": 1, "Zones": [{"Id": "z1", "Type": "door", "Name": "Puerta", "Polygon": POLI}]})
    zonas = parse_zones(raw)
    assert len(zonas.zones) == 1
    assert zonas.zones[0].zone_type == "door"


def test_camel_case_sigue_funcionando():
    raw = json.dumps({"version": 1, "zones": [{"id": "z1", "type": "street", "name": "Calle", "polygon": POLI}]})
    assert parse_zones(raw).zones[0].zone_type == "street"
