# Videos de ejemplo

Para quien prueba VigiShield y no puede provocar una escena en su casa, la app
ofrece **«Ver un video de ejemplo»**:

1. El backend elige un video del catálogo. Primero van los que ese hogar aún no
   vio y, entre ellos, los menos usados por todos.
2. Lo asigna a la cámara interna del hogar (`IsSample`) por 3 minutos. Durante
   ese tiempo la IA la analiza como a cualquier cámara.
3. Los eventos quedan en el historial. No llegan por WhatsApp ni disparan la
   alerta de emergencia, porque la cámara tiene los avisos apagados.

## Videos

Todos son de [Pexels](https://www.pexels.com/license/): uso libre, sin
atribución obligatoria. Las fuentes están en `manifiesto.json`.

Cada clip se probó con el detector de producción en bucle, con reloj simulado,
a una cadencia de 0,65 a 1 s:

- Los 10 dan «Persona desconocida» en unos 10 s.
- La mayoría pasa a «Merodeo» (~25 s) y a «Riesgo de intrusión» (30–60 s).
- `ejemplo08` (faroles, de noche) y `ejemplo09` (auto, de noche) pueden quedarse
  en «Persona desconocida» o «Merodeo».
- `ejemplo10` (mirilla) llega a «Riesgo de intrusión» hacia los 2 min.

Las zonas de cada escena están en el backend, en `SampleVideoCatalog.cs`. Sin
zona de entrada el motor nunca pasa de «persona desconocida».

## Despliegue en la VM de IA

```bash
py -3 videos_ejemplo/preparar.py ejemplos          # en el PC: descarga y recodifica
scp ejemplos/ejemplo*.mp4 vigishield@api-ai.vigishield.app:/opt/vigishield-ai/ejemplos/
```

En `/opt/vigishield-ai/mediamtx.yml`, dentro de `paths:`, poner esto antes de
`all_others`:

```yaml
  '~^ejemplo(\d{2})$':
    runOnDemand: ffmpeg -hide_banner -loglevel error -re -stream_loop -1 -i /opt/vigishield-ai/ejemplos/ejemplo$G1.mp4 -c copy -f rtsp rtsp://localhost:$RTSP_PORT/$MTX_PATH
    runOnDemandRestart: yes
    runOnDemandCloseAfter: 20s
```

Con `runOnDemand`, ffmpeg solo corre mientras alguien mira o la IA analiza.
Cada sesión arranca el video desde el principio.
