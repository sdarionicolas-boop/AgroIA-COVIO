"""
deploy_hf.py — Despliegue automatizado de AgroIA-COVIO v2.1 en Hugging Face Spaces
=================================================================================

Crea (o actualiza) un Space de tipo Streamlit y sube los archivos necesarios
usando la API oficial `huggingface_hub`.

Uso
---
    pip install huggingface_hub
    export HF_TOKEN=hf_xxx              # token con permiso de escritura
    python deploy_hf.py --repo tu_usuario/agroia-covio

Opcional:
    --private        crea el Space como privado
    --token hf_xxx   pasa el token por línea de comandos (en vez de HF_TOKEN)

Desarrollado por Darío Sánchez Leguizamón · GNU GPLv3.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Archivos que componen la app web (se ignora todo lo del escritorio).
# El Space corre sobre SDK "docker" (Dockerfile propio): pinea torch/numpy CPU
# y evita el bloat de CUDA, algo que el SDK nativo de streamlit no permite.
FILES_TO_UPLOAD = [
    "streamlit_app.py",
    "requirements-web.txt",  # se publica como requirements.txt en el Space
    "Dockerfile",
    ".streamlit/config.toml",
]

# README del Space con la cabecera YAML que exige Hugging Face Spaces.
# sdk: docker + app_port deben coincidir con el Dockerfile (EXPOSE 7860).
SPACE_README = """\
---
title: AgroIA-COVIO v2.1
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: gpl-3.0
---

# 🌱 AgroIA-COVIO v2.1 — Conteo de plantas por dron

**Desarrollado por Darío Sánchez Leguizamón.** Procesamiento local, offline y
democratizado para el productor de LATAM.

App web que cuenta plantas en imágenes de dron / ortofotos usando **MobileSAM**
(~40 MB, optimizado para CPU), genera un **mapa de calor de densidad** y estima
métricas agronómicas (**plantas/ha** y **rinde tn/ha**) en segundos.

> ⚠️ **Versión liviana (demo de bolsillo).** Usa un modelo chico optimizado para
> velocidad en CPU, **no para máxima precisión**: el conteo es aproximado y puede
> sub- o sobre-detectar con plantas muy chicas, baja resolución o imágenes poco
> nítidas. Para análisis de campo serio existe una versión más potente
> (SAM pesado + YOLO) bajo contacto.

## Uso

1. Elegí el cultivo en la barra lateral (Maíz, Lechuga, Palma Aceitera, Girasol).
2. Ajustá el **GSD (cm/píxel)** de tu vuelo — define la escala real y las métricas.
3. Subí una imagen (o usá una de prueba) y presioná **Detectar plantas**.
4. Revisá las cajas de detección vs. el mapa de calor y descargá el **CSV** con
   las coordenadas de cada planta.

## Notas técnicas

- La imagen se redimensiona al lado largo de 1024 px para mantener la respuesta
  por debajo de ~3 s en CPU; el GSD se reescala en consecuencia.
- El filtro de tamaño usa la fórmula agronómica del pipeline COVIO:
  `área_px = π · (diámetro_cm / (2·gsd))²`.
- El **rinde es una estimación orientativa** (densidad × kg/planta), editable, y
  no reemplaza el muestreo de campo ni el juicio del agrónomo.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Licencia

GNU GPL v3.0 con **Términos Adicionales (Sección 7b)**: toda interfaz derivada
debe mostrar de forma prominente la atribución
*"AgroIA-COVIO — desarrollado por Darío Sánchez Leguizamón"*.
Licenciamiento dual disponible para integración comercial propietaria mediante
contacto directo con el autor.
"""


def main():
    ap = argparse.ArgumentParser(description="Desplegar AgroIA-COVIO en HF Spaces")
    ap.add_argument("--repo", required=True,
                    help="ID del Space, ej: usuario/agroia-covio")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="Token HF (o variable de entorno HF_TOKEN)")
    ap.add_argument("--private", action="store_true", help="Space privado")
    args = ap.parse_args()

    if not args.token:
        sys.exit("ERROR: falta el token. Definí HF_TOKEN o pasá --token.")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("ERROR: instalá huggingface_hub → pip install huggingface_hub")

    api = HfApi(token=args.token)

    # repo_exists() es una consulta liviana; create_repo() en cambio revalida
    # el tier del SDK (Docker) incluso con exist_ok=True, y eso exige PRO en
    # la cuenta aunque el Space ya esté corriendo gratis. Evitamos llamarlo
    # si el Space ya existe.
    if api.repo_exists(repo_id=args.repo, repo_type="space"):
        print(f"→ Space existente: {args.repo} (se omite creación)")
    else:
        print(f"→ Creando Space: {args.repo}")
        api.create_repo(
            repo_id=args.repo,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
        )

    # README del Space (cabecera YAML obligatoria).
    print("→ Subiendo README.md del Space")
    api.upload_file(
        path_or_fileobj=SPACE_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="space",
    )

    for fname in FILES_TO_UPLOAD:
        path = os.path.join(BASE_DIR, fname)
        remote = "requirements.txt" if fname == "requirements-web.txt" else fname
        if not os.path.exists(path):
            print(f"  ⚠ omitido (no existe): {fname}")
            continue
        print(f"→ Subiendo {fname}")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=remote,
            repo_id=args.repo,
            repo_type="space",
        )

    url = f"https://huggingface.co/spaces/{args.repo}"
    print(f"\n✅ Listo. El Space se está construyendo en:\n   {url}")


if __name__ == "__main__":
    main()
