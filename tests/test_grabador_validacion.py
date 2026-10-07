"""
El grabador de evidencia solo graba celulares de hogares que aceptaron, una vez
al día por cámara, y no pasa del límite de disco.

    py -3 -m pytest tests/test_grabador_validacion.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grabador_validacion as g  # noqa: E402

HOGAR = "11111111-aaaa-bbbb-cccc-000000000001"
OTRO = "22222222-aaaa-bbbb-cccc-000000000002"


def _cam(clave, hogar=HOGAR, modo="MobileWebRtc", cid=None):
    return {"id": cid or f"cam-{clave}", "householdId": hogar, "name": "Celular",
            "streamMode": modo, "mediaMtxRtspUrl": f"rtsp://localhost:8554/{clave}"}


def _path(clave, tipo="webRTCSession", listo=True):
    return {"name": clave, "ready": listo, "source": {"type": tipo}}


class _Grabaciones:
    """Sustituye a ffmpeg: escribe un archivo y anota qué se grabó."""

    def __init__(self):
        self.claves = []

    def __call__(self, clave, destino, segundos):
        self.claves.append((clave, segundos))
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"x" * 100_000)
        return 100_000


def _grabador(tmp_path, paths, camaras, consentidos=(HOGAR,), **kw):
    if consentidos is not None:
        (tmp_path / "consentimiento.txt").write_text(
            "# hogares que aceptaron\n" + "\n".join(f"{h}   # tester@x.com" for h in consentidos) + "\n",
            encoding="utf-8")
    grabaciones = _Grabaciones()
    kw.setdefault("leer_hogares", lambda: None)
    grab = g.Grabador(tmp_path, leer_paths=lambda: paths, leer_camaras=lambda: camaras,
                      grabar_fn=grabaciones, **kw)
    return grab, grabaciones


def _esperar(grab):
    for h in grab._hilos:
        h.join(timeout=5)


def test_lista_de_consentimiento_admite_comentarios(tmp_path):
    ruta = tmp_path / "c.txt"
    ruta.write_text(f"# cabecera\n\n{HOGAR.upper()}  # ana@x.com\n  {OTRO}\n", encoding="utf-8")
    assert g.leer_consentimiento(ruta) == {HOGAR, OTRO}
    assert g.leer_consentimiento(tmp_path / "no-existe.txt") == set()


def test_solo_cuentan_los_celulares_que_ya_transmiten():
    items = [_path("cel1"), _path("ip1", tipo="rtmpConn"), _path("cel2", listo=False),
             _path("../x"), {"name": "sin_fuente", "ready": True}]
    assert g.publicando_por_webrtc(items) == ["cel1"]
    mapa = g.camaras_de_celular([_cam("cel1"), _cam("ip1", modo="RtmpRelay"), {"id": "x", "streamMode": "MobileWebRtc"}])
    assert list(mapa) == ["cel1"]


def test_sin_consentimiento_no_graba_ni_pregunta_a_mediamtx(tmp_path):
    def no_llamar():
        raise AssertionError("no debía consultar MediaMTX")
    grab = g.Grabador(tmp_path, leer_paths=no_llamar, leer_camaras=lambda: [], grabar_fn=_Grabaciones(),
                      leer_hogares=lambda: (set(), {HOGAR}))
    assert grab.ciclo() == []


def test_graba_el_celular_de_un_hogar_que_acepto_una_vez_al_dia(tmp_path):
    grab, grabaciones = _grabador(tmp_path, [_path("cel1")], [_cam("cel1")], segundos=60)
    assert grab.ciclo() == ["cel1"]
    _esperar(grab)
    assert grabaciones.claves == [("cel1", 60)]
    filas = list(csv.DictReader((tmp_path / "indice.csv").open(encoding="utf-8")))
    assert len(filas) == 1 and filas[0]["hogar"] == HOGAR and filas[0]["camara_id"] == "cam-cel1"
    assert (tmp_path / filas[0]["archivo"]).exists()
    # Vuelve a transmitir el mismo día (o gira el celular): no se repite.
    assert grab.ciclo() == []


def test_no_graba_hogares_sin_permiso_ni_camaras_ip(tmp_path):
    paths = [_path("cel1"), _path("cel2"), _path("ip1", tipo="rtmpConn")]
    camaras = [_cam("cel1", hogar=OTRO), _cam("cel2"), _cam("ip1", modo="RtmpRelay")]
    grab, grabaciones = _grabador(tmp_path, paths, camaras)
    assert grab.ciclo() == ["cel2"]
    _esperar(grab)
    assert [c for c, _ in grabaciones.claves] == ["cel2"]


def test_respeta_el_limite_de_disco_y_de_simultaneas(tmp_path):
    grab, _ = _grabador(tmp_path, [_path("cel1")], [_cam("cel1")], max_bytes=10)
    (tmp_path / "viejo.mp4").write_bytes(b"x" * 20)
    assert grab.ciclo() == []

    paths = [_path(f"cel{i}") for i in range(4)]
    camaras = [_cam(f"cel{i}") for i in range(4)]
    otro_dir = tmp_path / "b"
    otro_dir.mkdir()
    grab, _ = _grabador(otro_dir, paths, camaras, simultaneas=2)
    grab._en_curso.update({"ocupada-1"})  # ya hay una en curso
    assert len(grab.ciclo()) == 1


def test_backend_caido_usa_la_lista_anterior(tmp_path):
    respuestas = [[_cam("cel1")], None]
    grab, grabaciones = _grabador(tmp_path, [_path("cel1")], None)
    grab._leer_camaras = lambda: respuestas.pop(0)
    grab._refrescar_camaras()
    grab._camaras_en = 0  # fuerza otra lectura, que falla
    assert grab.ciclo() == ["cel1"]
    _esperar(grab)


def test_el_comando_copia_sin_recodificar_desde_mediamtx_local(tmp_path):
    cmd = g.comando_ffmpeg("abc123", tmp_path / "x.part", 60)
    assert "rtsp://127.0.0.1:8554/abc123" in cmd
    assert cmd[cmd.index("-c:v") + 1] == "copy" and "-an" in cmd
    assert cmd[cmd.index("-t") + 1] == "60"
    assert "frag_keyframe" in cmd[cmd.index("-movflags") + 1]


def test_el_interruptor_del_backend_se_suma_a_la_lista(tmp_path):
    grab, _ = _grabador(tmp_path, [_path("cel1"), _path("cel2")],
                        [_cam("cel1"), _cam("cel2", hogar=OTRO)], consentidos=(),
                        leer_hogares=lambda: ({OTRO}, {HOGAR, OTRO}))
    assert grab.ciclo() == ["cel2"]
    _esperar(grab)


def test_si_el_backend_no_responde_se_queda_con_lo_ultimo(tmp_path):
    respuestas = [({OTRO}, {OTRO}), None]
    grab, _ = _grabador(tmp_path, [_path("cel2")], [_cam("cel2", hogar=OTRO)], consentidos=(),
                        leer_hogares=lambda: respuestas.pop(0))
    grab._refrescar_hogares()
    grab._hogares_en = 0  # fuerza otra lectura, que falla
    assert grab.ciclo() == ["cel2"]
    assert respuestas == []
    _esperar(grab)


def test_borra_las_grabaciones_de_cuentas_eliminadas_y_nunca_con_lista_vacia(tmp_path):
    grab, _ = _grabador(tmp_path, [_path("cel1"), _path("cel2")],
                        [_cam("cel1"), _cam("cel2", hogar=OTRO)], consentidos=(HOGAR, OTRO))
    grab.ciclo()
    _esperar(grab)
    filas = list(csv.DictReader((tmp_path / "indice.csv").open(encoding="utf-8")))
    archivos = {f["hogar"]: tmp_path / f["archivo"] for f in filas}
    assert len(archivos) == 2

    assert grab.limpiar_borrados(set()) == 0          # lista vacía: no se toca nada
    assert all(a.exists() for a in archivos.values())

    assert grab.limpiar_borrados({HOGAR}) == 1        # OTRO eliminó su cuenta
    assert archivos[HOGAR].exists() and not archivos[OTRO].exists()
    filas = list(csv.DictReader((tmp_path / "indice.csv").open(encoding="utf-8")))
    assert [f["hogar"] for f in filas] == [HOGAR]
