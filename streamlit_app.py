"""
AgroIA-COVIO v2.1 — Web App (Streamlit / Hugging Face Spaces)
Conteo de plantas, zonificación de densidad y métricas agronómicas con MobileSAM.

Copyright (c) 2026 Darío Sánchez Leguizamón

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See <https://www.gnu.org/licenses/>.

Additional Terms (Section 7b of GNU GPL v3.0):
Any interactive user interface derived from or using this Program must
prominently display the attribution:
"AgroIA-COVIO — desarrollado por Darío Sánchez Leguizamón"
"""

import os
import io
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Configuración global
# ──────────────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MOBILE_SAM_PATH = os.path.join(MODELS_DIR, "mobile_sam.pt")
MOBILE_SAM_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)
DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_images")

# Tamaño máximo del lado largo de la imagen procesada (px).
# Mantiene la inferencia en CPU por debajo de ~3 s y la RAM acotada.
MAX_DIM = 1024

ATTRIBUTION = "AgroIA-COVIO — desarrollado por Darío Sánchez Leguizamón"
CONTACT_EMAIL = "sdarionicolas@gmail.com"
CONTACT_LINKEDIN = "https://www.linkedin.com/in/darionicolas/"

# ──────────────────────────────────────────────────────────────────────────────
# Presets de cultivo
# Parámetros físicos calcados del pipeline de escritorio (app.py / pipeline.py).
# `yield_kg_planta` es una ESTIMACIÓN agronómica orientativa, editable por el
# usuario; nunca un dato medido en campo.
# ──────────────────────────────────────────────────────────────────────────────
CROP_PRESETS = {
    "🌽 Maíz (V2-V3 · Temprano)": {
        "gsd_cm": 1.5,
        "min_cm": 3.0,
        "max_cm": 120.0,
        "nms_cm": 37.5,
        "conf": 0.15,
        "yield_kg_planta": 0.16,
        "yield_label": "grano seco / planta",
    },
    "🥬 Lechuga (Cosecha)": {
        "gsd_cm": 0.6,
        "min_cm": 10.0,
        "max_cm": 40.0,
        "nms_cm": 15.0,
        "conf": 0.20,
        "yield_kg_planta": 0.45,
        "yield_label": "peso cabeza comercial",
    },
    "🌴 Palma Aceitera (Adulta)": {
        "gsd_cm": 13.0,
        "min_cm": 300.0,
        "max_cm": 1200.0,
        "nms_cm": 650.0,
        "conf": 0.15,
        "yield_kg_planta": 150.0,
        "yield_label": "racimos FFB / planta·año",
    },
    "🌻 Girasol (Canopea)": {
        "gsd_cm": 1.5,
        "min_cm": 3.0,
        "max_cm": 120.0,
        "nms_cm": 37.5,
        "conf": 0.12,
        "yield_kg_planta": 0.06,
        "yield_label": "grano / planta",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Tema oscuro (CSS) — estética "Valorant-style" del proyecto
# ──────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    .stApp { background-color: #0F1923; color: #E8EDF2; }
    section[data-testid="stSidebar"] { background-color: #1A2535; }
    h1, h2, h3, h4 { color: #E8EDF2 !important; font-family: 'Trebuchet MS', sans-serif; }
    .covio-attrib {
        background: linear-gradient(90deg, #1A2535 0%, #1F2D40 100%);
        border-left: 4px solid #4CAF82;
        padding: 10px 16px; border-radius: 6px; margin-bottom: 14px;
        font-size: 0.85rem; color: #B7C4D2; letter-spacing: .3px;
    }
    .covio-attrib b { color: #4CAF82; }
    .demo-note {
        background: #2A2210; border-left: 4px solid #F4A261;
        padding: 12px 16px; border-radius: 6px; margin: 4px 0 14px 0;
        font-size: .9rem; color: #E8D9C4;
    }
    .demo-note b { color: #F4A261; }
    .metric-card {
        background-color: #1A2535; border: 1px solid #2A3F58;
        border-radius: 10px; padding: 16px 18px; text-align: center;
    }
    .metric-card .v { font-size: 1.9rem; font-weight: 700; color: #4CAF82; }
    .metric-card .l { font-size: .78rem; color: #7A8FA6; text-transform: uppercase; letter-spacing: .5px; }
    .stButton>button {
        background-color: #4CAF82; color: #0F1923; font-weight: 700;
        border: none; border-radius: 8px; padding: 10px 18px;
    }
    .stButton>button:hover { background-color: #5fcf99; color: #0F1923; }
    .footer-note { color: #5C6E82; font-size: .78rem; margin-top: 18px; }
    /* ── Llamado a la acción (contacto) ── */
    .cta-card {
        background: linear-gradient(135deg, #15212F 0%, #1F2D40 100%);
        border: 1px solid #2A3F58; border-radius: 12px;
        padding: 22px 26px; margin-top: 30px;
    }
    .cta-card h3 { margin: 0 0 4px 0; color: #4CAF82 !important; }
    .cta-card p { color: #B7C4D2; font-size: .92rem; margin: 6px 0 16px 0; }
    .cta-card .who { color: #7A8FA6; font-size: .82rem; }
    .cta-card .who b { color: #E8EDF2; }
    .cta-btn {
        display: inline-block; text-decoration: none; font-weight: 700;
        border-radius: 8px; padding: 11px 20px; margin: 4px 10px 4px 0;
        font-size: .92rem;
    }
    .cta-btn.mail { background-color: #4CAF82; color: #0F1923 !important; }
    .cta-btn.mail:hover { background-color: #5fcf99; }
    .cta-btn.li {
        background-color: transparent; color: #4CAF82 !important;
        border: 1.5px solid #4CAF82;
    }
    .cta-btn.li:hover { background-color: #4CAF82; color: #0F1923 !important; }
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Carga del modelo (cacheada — una sola vez por sesión del Space)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando MobileSAM (~40 MB)…")
def load_mobile_sam():
    """Descarga (si falta) y carga MobileSAM en CPU. Devuelve el modelo SAM."""
    from mobile_sam import sam_model_registry

    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(MOBILE_SAM_PATH):
        urllib.request.urlretrieve(MOBILE_SAM_URL, MOBILE_SAM_PATH)

    sam = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_PATH)
    sam.to("cpu")
    sam.eval()
    return sam


def build_mask_generator(sam, min_area_px):
    """Generador automático de máscaras con los mismos umbrales que el pipeline."""
    from mobile_sam import SamAutomaticMaskGenerator

    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,           # 4x más rápido que el default (32), buena precisión
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=int(max(20, min_area_px * 0.5)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Procesamiento de imagen
# ──────────────────────────────────────────────────────────────────────────────
def prepare_image(img_rgb):
    """Redimensiona la imagen al lado largo MAX_DIM. Devuelve (img, scale)."""
    h, w = img_rgb.shape[:2]
    longest = max(h, w)
    scale = MAX_DIM / longest if longest > MAX_DIM else 1.0
    if scale < 1.0:
        img_rgb = cv2.resize(
            img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return img_rgb, scale


def detect_plants(img_proc, gsd_proc_cm, min_cm, max_cm):
    """
    Ejecuta MobileSAM y filtra máscaras por área agronómica.
    Filtro de área (idéntico a pipeline.py):
        area_px = π · (diámetro_cm / (2 · gsd))²
    Devuelve lista de detecciones: dict(cx, cy, w, h, area).
    """
    sam = load_mobile_sam()

    # Piso de 50 px²: evita ruido de textura en imágenes pequeñas.
    min_area = max(50, int(np.pi * (min_cm / (2 * gsd_proc_cm)) ** 2 * 0.5))
    max_area = int(np.pi * (max_cm / (2 * gsd_proc_cm)) ** 2 * 3.0)
    # Acota el máximo a la mitad del frame para descartar la máscara "fondo".
    max_area = min(max_area, int(img_proc.shape[0] * img_proc.shape[1] * 0.5))

    mask_gen = build_mask_generator(sam, min_area)
    masks = mask_gen.generate(img_proc)

    dets = []
    for m in masks:
        if not (min_area < m["area"] < max_area):
            continue
        bx, by, bw, bh = m["bbox"]
        dets.append(
            {
                "cx": bx + bw / 2.0,
                "cy": by + bh / 2.0,
                "x": bx,
                "y": by,
                "w": bw,
                "h": bh,
                "area": m["area"],
            }
        )
    return dets, (min_area, max_area)


def draw_detections(img_proc, dets):
    """Dibuja cajas verdes sobre una copia de la imagen (RGB)."""
    out = img_proc.copy()
    for d in dets:
        x1, y1 = int(d["x"]), int(d["y"])
        x2, y2 = int(d["x"] + d["w"]), int(d["y"] + d["h"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (76, 175, 130), 2)
    return out


def density_heatmap(img_proc, dets):
    """Mapa de calor de densidad: histograma 2D → suavizado Gaussiano → JET."""
    h, w = img_proc.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    for d in dets:
        cx, cy = int(d["cx"]), int(d["cy"])
        if 0 <= cy < h and 0 <= cx < w:
            acc[cy, cx] += 1.0

    if acc.max() > 0:
        sigma = max(8, int(min(h, w) / 30))
        ksize = sigma * 6 + 1
        acc = cv2.GaussianBlur(acc, (ksize, ksize), sigma)
        acc = acc / acc.max()

    heat = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_proc, 0.55, heat, 0.45, 0)
    return blended


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AgroIA-COVIO v2.1", page_icon="🌱", layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Atribución legal obligatoria (GPLv3 §7b) — visible en toda vista.
    st.markdown(
        f'<div class="covio-attrib">🌱 <b>AgroIA-COVIO v2.1</b> — '
        f"desarrollado por <b>Darío Sánchez Leguizamón</b> · "
        f"GNU GPL v3.0 · Procesamiento local, offline y democratizado para el productor."
        f"</div>",
        unsafe_allow_html=True,
    )

    st.title("🌱 AgroIA-COVIO · Conteo de plantas por dron")
    st.caption(
        "Subí una ortofoto o imagen de dron, elegí el cultivo y obtené conteo, "
        "densidad (plantas/ha), rinde estimado y mapa de calor — en segundos, sobre CPU."
    )

    # Aviso honesto de alcance: esta web es una demo liviana.
    st.markdown(
        '<div class="demo-note">⚠️ <b>Versión liviana (demo de bolsillo).</b> '
        "Usa un modelo chico (~40 MB) optimizado para correr rápido en CPU, "
        "<b>no para máxima precisión</b>: el conteo es aproximado y puede sub- o "
        "sobre-detectar, sobre todo con plantas muy chicas, poca resolución o "
        "imágenes poco nítidas. Sirve para probar el concepto y una zonificación "
        "express. Para análisis de campo serio existe una versión más potente "
        "(SAM pesado + YOLO) — escribime abajo. 👇</div>",
        unsafe_allow_html=True,
    )

    with st.expander("💡 Cómo mejorar la detección"):
        st.markdown(
            "- **Ajustá el GSD** (cm/píxel) al de tu vuelo real: es lo que más "
            "afecta el conteo y las métricas. Un GSD mal cargado descalibra todo.\n"
            "- **Imagen nítida y con contraste** planta vs. suelo (evitá fotos "
            "movidas, con sombra fuerte o sobreexpuestas).\n"
            "- La imagen se reduce a 1024 px de lado para ir rápido: si el lote es "
            "muy grande, **subí un recorte** de una zona en vez de todo el "
            "ortomosaico — vas a detectar mucho mejor.\n"
            "- Probá el cultivo correcto: cada preset cambia el rango de tamaño "
            "esperado de la planta.\n"
            "- En **Ajustes avanzados** podés afinar el tamaño mín./máx. de objeto "
            "si ves que ignora o inventa plantas."
        )

    # ── Sidebar: parámetros ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parámetros")
        crop = st.selectbox("Cultivo (preset oficial)", list(CROP_PRESETS.keys()))
        preset = CROP_PRESETS[crop]

        st.markdown("**Resolución de la imagen**")
        gsd_cm = st.number_input(
            "GSD (cm/píxel)",
            min_value=0.05,
            max_value=100.0,
            value=float(preset["gsd_cm"]),
            step=0.1,
            help="Tamaño de terreno por píxel. Determina densidad y filtros de tamaño.",
        )

        with st.expander("Ajustes avanzados"):
            min_cm = st.number_input(
                "Tamaño mín. objeto (cm)", 0.5, 5000.0, float(preset["min_cm"]), 0.5
            )
            max_cm = st.number_input(
                "Tamaño máx. objeto (cm)", 1.0, 5000.0, float(preset["max_cm"]), 1.0
            )
            yield_kg = st.number_input(
                f"Rinde estimado ({preset['yield_label']}) [kg/planta]",
                0.0,
                10000.0,
                float(preset["yield_kg_planta"]),
                0.01,
                help="Estimación orientativa, editable. No es un dato medido en campo.",
            )

        st.markdown("---")
        st.markdown(
            f"<span style='color:#7A8FA6;font-size:.8rem'>Modelo: "
            f"<b>MobileSAM</b> (vit_t · ~40 MB) en CPU</span>",
            unsafe_allow_html=True,
        )

    # ── Entrada de imagen ────────────────────────────────────────────────────
    col_up, col_demo = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "📤 Arrastrá tu imagen (JPG / PNG / TIFF)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
        )
    with col_demo:
        demo_files = (
            sorted(f for f in os.listdir(DEMO_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
            if os.path.isdir(DEMO_DIR)
            else []
        )
        demo_choice = st.selectbox(
            "…o usá una imagen de prueba",
            ["(ninguna)"] + demo_files,
            help="Colocá imágenes en la carpeta demo_images/ para tenerlas acá.",
        )

    img_rgb = None
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is not None:
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif demo_choice != "(ninguna)":
        bgr = cv2.imread(os.path.join(DEMO_DIR, demo_choice))
        if bgr is not None:
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if img_rgb is None:
        st.info("Esperando una imagen para analizar.")
        _footer()
        return

    h0, w0 = img_rgb.shape[:2]
    st.write(f"Imagen original: **{w0}×{h0}px** · GSD **{gsd_cm} cm/px**")

    if st.button("🔍 Detectar plantas"):
        t0 = time.time()
        with st.spinner("Segmentando con MobileSAM…"):
            img_proc, scale = prepare_image(img_rgb)
            # GSD efectivo tras el redimensionado: cada píxel cubre más terreno.
            gsd_proc_cm = gsd_cm / scale
            dets, (min_a, max_a) = detect_plants(
                img_proc, gsd_proc_cm, min_cm, max_cm
            )
        elapsed = time.time() - t0

        n = len(dets)

        # ── Métricas agronómicas ─────────────────────────────────────────────
        # Área de terreno desde las dimensiones ORIGINALES (no depende del resize).
        area_m2 = (w0 * gsd_cm / 100.0) * (h0 * gsd_cm / 100.0)
        area_ha = area_m2 / 10000.0
        density = n / area_ha if area_ha > 0 else 0.0          # plantas/ha
        yield_tn_ha = density * yield_kg / 1000.0               # tn/ha

        st.success(
            f"✅ {n} objetos detectados en {elapsed:.1f}s "
            f"(procesado a {img_proc.shape[1]}×{img_proc.shape[0]}px · "
            f"filtro área {min_a}–{max_a}px²)"
        )

        c1, c2, c3, c4 = st.columns(4)
        _metric(c1, f"{n:,}".replace(",", "."), "Plantas detectadas")
        _metric(c2, f"{density:,.0f}".replace(",", "."), "Densidad · pl/ha")
        _metric(c3, f"{yield_tn_ha:,.1f}".replace(",", "."), "Rinde est. · tn/ha")
        _metric(c4, f"{area_ha:,.2f}".replace(",", "."), "Área · ha")

        # ── Visualización lado a lado ────────────────────────────────────────
        st.markdown("### Resultados")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Detecciones**")
            st.image(draw_detections(img_proc, dets), use_container_width=True)
        with v2:
            st.markdown("**Mapa de calor de densidad**")
            st.image(density_heatmap(img_proc, dets), use_container_width=True)

        # ── Exportación CSV ──────────────────────────────────────────────────
        df = pd.DataFrame(
            {
                "id": range(1, n + 1),
                # Coordenadas relativas (0–1), independientes de la resolución
                "x_rel": [d["cx"] / img_proc.shape[1] for d in dets],
                "y_rel": [d["cy"] / img_proc.shape[0] for d in dets],
                "px_x_proc": [round(d["cx"], 1) for d in dets],
                "px_y_proc": [round(d["cy"], 1) for d in dets],
                "area_px": [int(d["area"]) for d in dets],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Descargar CSV de plantas detectadas",
            buf.getvalue(),
            file_name="plantas_detectadas_covio.csv",
            mime="text/csv",
        )

        st.caption(
            "ℹ️ El rinde es una estimación orientativa (densidad × kg/planta) y "
            "no reemplaza el juicio del agrónomo ni un muestreo de campo."
        )

    _footer()


def _metric(col, value, label):
    col.markdown(
        f'<div class="metric-card"><div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def _contact_cta():
    """Llamado a la acción de contacto — productor y aseguradora/empresa."""
    subject = "Consulta AgroIA-COVIO"
    mailto = f"mailto:{CONTACT_EMAIL}?subject={subject.replace(' ', '%20')}"
    st.markdown(
        f"""
        <div class="cta-card">
            <h3>📩 ¿Querés llevar esto a tu proyecto?</h3>
            <p>Hablemos. Adapto, integro y licencio AgroIA-COVIO para cualquier
            actor del agro que necesite contar plantas y zonificar a escala.</p>
            <a class="cta-btn mail" href="{mailto}">✉️ Escribime un mail</a>
            <a class="cta-btn li" href="{CONTACT_LINKEDIN}" target="_blank"
               rel="noopener">in · Conectá en LinkedIn</a>
            <p class="who" style="margin-top:14px">
                🌱 <b>Productores y cooperativas</b> · 🛰️ <b>AgTechs y software</b> ·
                🏢 <b>Aseguradoras y empresas</b> · 🎓 <b>Instituciones e investigación</b>
                <br>Desde analizar un lote hasta integración a medida y licenciamiento.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _footer():
    _contact_cta()
    st.markdown(
        f'<div class="footer-note">{ATTRIBUTION} · '
        f"Licenciado bajo GNU GPL v3.0 con Términos Adicionales (§7b). "
        f"Para integración comercial propietaria, contactá al autor.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
