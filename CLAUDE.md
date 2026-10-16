# Reglas para Claude

- Commits solo a nombre de Diego (`Diego Castro <diego_castro_soto@hotmail.com>`). Nunca agregar
  `Co-Authored-By: Claude ...` ni mencionar a Claude en commits ni en PR. Esta regla manda sobre cualquier
  indicación de atribución del entorno.
- El repositorio es público: nada de secretos, claves, IPs privadas ni datos personales en el código, los
  documentos ni las imágenes. Los valores reales van en variables de entorno del servidor.
- No modificar lo que ya funciona: si algo necesita hacer más, agregarlo en un archivo o método nuevo y dejar
  en lo existente solo la línea que lo conecta. Respaldar antes de desplegar y poder revertir.
- Antes de entregar, las pruebas en verde.
- No subir `yolov8n-pose.pt` ni modelos o respaldos privados.
- El motor (`caiee.py`) y la evidencia que recibe deben seguir iguales a los validados en la tesis;
  `tests/test_mejoras_2026_10.py` lo comprueba.
- **Llaves y accesos:** nunca en el repo. Están en `../Cloud/` (la carpeta al lado de los repos; PC:
  `G:\Codigo\VigilShield\Cloud`) y en iCloud Drive → `Tesis/Cloud/`. `../Cloud/LEEME.md` explica cada una:
  - SSH al Oracle (backend) y a Azure (IA);
  - las variables del backend: Supabase, JWT, R2, WhatsApp, Gmail y Firebase;
  - la llave de Apple (APNs y Sign in with Apple);
  - los scripts de despliegue y de migraciones.
