"""
AgroIA-COVIO v2.1 - Detección automática de plantas con YOLOv8 + SAM
Autor: Darío Sánchez Leguizamón

Copyright (c) 2026 Darío Sánchez Leguizamón

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional Terms (Section 7b of GNU GPL v3.0):
Any interactive user interface (including GUIs, CLIs, and web apps)
derived from or using this Program must prominently display the
attribution: "AgroIA-COVIO — desarrollado por Darío Sánchez Leguizamón"
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import re

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SAM_MODELS = {
    "SAM ViT-H (Pesado)": {
        "url":  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "path": os.path.join(MODELS_DIR, "sam_vit_h_4b8939.pth"),
        "size": 2.4,
        "type": "vit_h"
    },
    "MobileSAM (Ligero)": {
        "url":  "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "path": os.path.join(MODELS_DIR, "mobile_sam.pt"),
        "size": 0.04,
        "type": "mobile"
    }
}

CROP_PRESETS = {
    "Personalizado": None,
    "Maíz (V2-V3 - Temprano)": {
        "gsd": "1.5",
        "tile": "1024",
        "sam_fraction": "70",
        "nms_cm": "37.5",
        "min_cm": "3.0",
        "max_cm": "120.0",
        "conf": "0.25",
        "epochs": "10",
        "grid": "10",
        "sam_model": "SAM ViT-H (Pesado)"
    },
    "Lechuga (Cosecha)": {
        "gsd": "0.6",
        "tile": "1024",
        "sam_fraction": "70",
        "nms_cm": "15.0",
        "min_cm": "10.0",
        "max_cm": "40.0",
        "conf": "0.20",
        "epochs": "10",
        "grid": "1",
        "sam_model": "SAM ViT-H (Pesado)"
    },
    "Palma Aceitera (Adulta)": {
        "gsd": "13.0",
        "tile": "1024",
        "sam_fraction": "100",
        "nms_cm": "650.0",
        "min_cm": "300.0",
        "max_cm": "1200.0",
        "conf": "0.25",
        "epochs": "10",
        "grid": "10",
        "sam_model": "SAM ViT-H (Pesado)"
    },
    "Girasol (Canopea - Zonificación)": {
        "gsd": "1.5",
        "tile": "1024",
        "sam_fraction": "100",
        "nms_cm": "37.5",
        "min_cm": "3.0",
        "max_cm": "120.0",
        "conf": "0.12",
        "epochs": "10",
        "grid": "10",
        "sam_model": "SAM ViT-H (Pesado)"
    }
}

# ── Colors & Fonts ─────────────────────────────────────────────────────────────
BG_DARK   = "#0F1923"
BG_CARD   = "#1A2535"
BG_CARD2  = "#1F2D40"
ACCENT    = "#4CAF82"
ACCENT2   = "#2E86AB"
TEXT_MAIN = "#E8EDF2"
TEXT_MUTED= "#7A8FA6"
TEXT_WARN = "#F4A261"
TEXT_ERR  = "#E76F51"
BORDER    = "#2A3F58"

FONT_TITLE = ("Trebuchet MS", 20, "bold")
FONT_SUB   = ("Trebuchet MS", 10)
FONT_LABEL = ("Calibri", 10, "bold")
FONT_BODY  = ("Calibri", 10)
FONT_MONO  = ("Consolas", 9)
FONT_BIG   = ("Trebuchet MS", 22, "bold")


def sanitize_folder_name(name):
    """Convierte 'Maiz H12 lote 15' en nombre válido de carpeta."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    return name if name else "sin_nombre"


class CovioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AgroIA-COVIO v2.1 · Detección de Plantas por Dron")
        self.geometry("960x760")
        self.minsize(880, 640)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.ortho_path   = tk.StringVar(value="")
        self.output_path  = tk.StringVar(value=os.path.join(BASE_DIR, "output"))
        self.status_text  = tk.StringVar(value="Listo para comenzar.")
        self.progress_val = tk.DoubleVar(value=0.0)
        self.export_shp   = tk.BooleanVar(value=False)
        self._pipeline_running = False
        self._stop_event = threading.Event()

        self._build_ui()
        self._check_sam_status()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG_DARK, pady=14)
        hdr.pack(fill="x", padx=28)
        tk.Label(hdr, text="AgroIA-COVIO", font=FONT_TITLE,
                 bg=BG_DARK, fg=ACCENT).pack(side="left")
        tk.Label(hdr, text="  v2.1 · Detección de Plantas por Dron",
                 font=FONT_SUB, bg=BG_DARK, fg=TEXT_MUTED).pack(side="left", pady=4)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=28)

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=28, pady=12)
        body.columnconfigure(0, weight=1, minsize=360)
        body.columnconfigure(1, weight=1, minsize=360)
        body.rowconfigure(0, weight=1)

        left  = tk.Frame(body, bg=BG_DARK)
        right = tk.Frame(body, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_lote_card(left)
        self._build_input_card(left)
        self._build_output_card(left)
        self._build_params_card(left)
        self._build_sam_card(left)
        self._build_run_button(left)

        self._build_log_card(right)
        self._build_results_card(right)

        self._build_statusbar()

    # ── LEFT CARDS ─────────────────────────────────────────────────────────────

    def _build_lote_card(self, parent):
        card = self._card(parent, "🌿  Identificación del lote")
        card.pack(fill="x", pady=(0, 8))

        tk.Label(card, text="Nombre del cultivo / lote",
                 font=FONT_BODY, bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w")
        self.lote_entry = tk.Entry(card, font=FONT_BODY,
                                   bg=BG_CARD2, fg=TEXT_MAIN,
                                   relief="flat", bd=0,
                                   insertbackground=TEXT_MAIN)
        self.lote_entry.pack(fill="x", ipady=7, ipadx=6, pady=(2, 0))
        self.lote_entry.insert(0, "Ej: Maiz H12 lote 15")
        self.lote_entry.bind("<FocusIn>",  self._clear_lote_placeholder)
        self.lote_entry.bind("<FocusOut>", self._restore_lote_placeholder)
        self._lote_placeholder = True

        tk.Label(card,
                 text="Se creará una subcarpeta con este nombre dentro de la carpeta de resultados.",
                 font=("Calibri", 9), bg=BG_CARD, fg=TEXT_MUTED,
                 wraplength=320, justify="left").pack(anchor="w", pady=(4, 0))

    def _clear_lote_placeholder(self, e):
        if self._lote_placeholder:
            self.lote_entry.delete(0, "end")
            self.lote_entry.config(fg=TEXT_MAIN)
            self._lote_placeholder = False

    def _restore_lote_placeholder(self, e):
        if not self.lote_entry.get().strip():
            self.lote_entry.insert(0, "Ej: Maiz H12 lote 15")
            self.lote_entry.config(fg=TEXT_MUTED)
            self._lote_placeholder = True

    def _get_lote_name(self):
        if self._lote_placeholder or not self.lote_entry.get().strip():
            return "sin_nombre"
        return sanitize_folder_name(self.lote_entry.get())

    def _build_input_card(self, parent):
        card = self._card(parent, "📂  Ortomosaico de entrada")
        card.pack(fill="x", pady=(0, 8))
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x")
        tk.Entry(row, textvariable=self.ortho_path,
                 bg=BG_CARD2, fg=TEXT_MAIN, font=FONT_BODY,
                 relief="flat", bd=0, insertbackground=TEXT_MAIN
                 ).pack(side="left", fill="x", expand=True, ipady=6, ipadx=6)
        tk.Button(row, text="Examinar", font=FONT_BODY,
                  bg=ACCENT2, fg="white", relief="flat", bd=0,
                  activebackground="#236f8a", cursor="hand2", padx=12,
                  command=self._browse_ortho).pack(side="right", ipady=6)
        tk.Label(card, text="Archivo GeoTIFF (.tif / .tiff) georreferenciado",
                 font=("Calibri", 9), bg=BG_CARD, fg=TEXT_MUTED
                 ).pack(anchor="w", pady=(4, 0))

    def _build_output_card(self, parent):
        card = self._card(parent, "💾  Carpeta de resultados")
        card.pack(fill="x", pady=(0, 8))
        row = tk.Frame(card, bg=BG_CARD)
        row.pack(fill="x")
        tk.Entry(row, textvariable=self.output_path,
                 bg=BG_CARD2, fg=TEXT_MAIN, font=FONT_BODY,
                 relief="flat", bd=0, insertbackground=TEXT_MAIN
                 ).pack(side="left", fill="x", expand=True, ipady=6, ipadx=6)
        tk.Button(row, text="Cambiar", font=FONT_BODY,
                  bg=ACCENT2, fg="white", relief="flat", bd=0,
                  activebackground="#236f8a", cursor="hand2", padx=12,
                  command=self._browse_output).pack(side="right", ipady=6)
        tk.Label(card,
                 text="Se creará una subcarpeta con el nombre del lote dentro de esta carpeta.",
                 font=("Calibri", 9), bg=BG_CARD, fg=TEXT_MUTED
                 ).pack(anchor="w", pady=(4, 0))

    def _build_params_card(self, parent):
        card = self._card(parent, "⚙️  Parámetros del pipeline")
        card.pack(fill="x", pady=(0, 8))

        # Preset row
        preset_row = tk.Frame(card, bg=BG_CARD)
        preset_row.pack(fill="x", pady=(0, 6))
        tk.Label(preset_row, text="Preajuste Cultivo:", font=FONT_LABEL,
                 bg=BG_CARD, fg=TEXT_MUTED).pack(side="left")
        
        self.preset_var = tk.StringVar(value="Personalizado")
        self.preset_menu = ttk.OptionMenu(preset_row, self.preset_var, "Personalizado",
                                          *CROP_PRESETS.keys(), command=self._on_preset_change)
        self.preset_menu.pack(side="right", fill="x", expand=True, padx=(10, 0))

        # Parámetros: (label, key, default, unidad, tooltip)
        params = [
            ("GSD",                      "gsd",          "1.5",  "cm/px",
             "Resolución del ortomosaico. Revisá el reporte de tu software de fotogrametría."),
            ("Tile size",                "tile",         "1024", "px",
             "Ventana de procesamiento. 1024 es el valor recomendado. Bajar a 512 si RAM < 16 GB."),
            ("SAM cobertura",            "sam_fraction", "70",   "%",
             "% de tiles analizados por SAM. 70% = balance velocidad/precisión. 100% = máxima precisión."),
            ("Dist. mín. entre plantas", "nms_cm",       "37.5", "cm",
             "Distancia mínima real entre plantas. Evita contar la misma planta dos veces."),
            ("Tamaño mínimo de objeto",  "min_cm",       "3.0",  "cm",
             "Descarta objetos más chicos que este valor. Filtra artefactos de suelo."),
            ("Tamaño máximo de objeto",  "max_cm",       "120.0","cm",
             "Descarta objetos más grandes que este valor. Filtra malezas de gran porte."),
            ("Umbral confianza",         "conf",         "0.25", "",
             "Score mínimo para aceptar una detección. Bajar = más detecciones, más ruido."),
            ("Épocas entrenamiento",     "epochs",       "10",   "",
             "Iteraciones de entrenamiento YOLO. 10 es óptimo para CPU."),
            ("Grilla SIG",               "grid",         "10",   "m",
             "Resolución del mapa de zonificación. 10×10 m = celdas de 100 m²."),
        ]

        self.param_vars = {}
        for label, key, default, unit, tooltip in params:
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill="x", pady=2)

            tk.Label(row, text=label, font=FONT_BODY,
                     bg=BG_CARD, fg=TEXT_MUTED, anchor="w"
                     ).pack(side="left", fill="x", expand=True)

            var = tk.StringVar(value=default)
            var.trace_add("write", lambda *args, k=key: self._on_param_write(k))
            self.param_vars[key] = var
            entry = tk.Entry(row, textvariable=var, width=7,
                             bg=BG_CARD2, fg=TEXT_MAIN, font=FONT_MONO,
                             relief="flat", bd=0, insertbackground=TEXT_MAIN)
            entry.pack(side="right", ipady=4, ipadx=6)
            self._add_tooltip(entry, tooltip)

            if unit:
                tk.Label(row, text=unit, font=("Calibri", 9),
                         bg=BG_CARD, fg=TEXT_MUTED, width=5, anchor="w"
                         ).pack(side="right", padx=(4, 2))

        # Checkbox SHP
        sep = tk.Frame(card, bg=BORDER, height=1)
        sep.pack(fill="x", pady=(8, 4))
        shp_row = tk.Frame(card, bg=BG_CARD)
        shp_row.pack(fill="x")
        tk.Checkbutton(shp_row, text="Exportar también como Shapefile (.shp)",
                       variable=self.export_shp,
                       bg=BG_CARD, fg=TEXT_MUTED, font=FONT_BODY,
                       activebackground=BG_CARD, selectcolor=BG_CARD2,
                       cursor="hand2").pack(side="left")

    def _build_sam_card(self, parent):
        card = self._card(parent, "🤖  Modelo SAM (Segment Anything)")
        card.pack(fill="x", pady=(0, 8))
        
        tk.Label(card, text="Seleccionar arquitectura",
                 font=("Calibri", 9), bg=BG_CARD, fg=TEXT_MUTED).pack(anchor="w")
        
        self.sam_model_var = tk.StringVar(value=list(SAM_MODELS.keys())[0])
        self.sam_menu = ttk.OptionMenu(card, self.sam_model_var, self.sam_model_var.get(), 
                                       *SAM_MODELS.keys(), command=self._on_sam_model_change)
        self.sam_menu.pack(fill="x", pady=(2, 6))
        
        self.sam_status_lbl = tk.Label(card, text="Verificando...",
                                       font=FONT_BODY, bg=BG_CARD,
                                       fg=TEXT_MUTED, anchor="w")
        self.sam_status_lbl.pack(fill="x")
        
        self.sam_progress_var = tk.DoubleVar()
        self.sam_progress = ttk.Progressbar(card, variable=self.sam_progress_var,
                                            maximum=100, mode="determinate")
        
        self.dl_label = tk.Label(card, text="", font=("Calibri", 9),
                                 bg=BG_CARD, fg=TEXT_MUTED)
        
        self.sam_dl_btn = tk.Button(card, text="⬇  Descargar modelo",
                                    font=FONT_BODY, bg=TEXT_WARN, fg=BG_DARK,
                                    relief="flat", bd=0, cursor="hand2",
                                    activebackground="#d4883a",
                                    command=self._download_sam_thread)

    def _on_sam_model_change(self, *args):
        self._check_sam_status()
        if not getattr(self, "_updating_from_preset", False):
            self.preset_var.set("Personalizado")

    def _on_preset_change(self, preset_name):
        preset = CROP_PRESETS.get(preset_name)
        if not preset:
            return
        
        self._updating_from_preset = True
        try:
            for key, val in preset.items():
                if key == "sam_model":
                    self.sam_model_var.set(val)
                    self._check_sam_status()
                elif key in self.param_vars:
                    self.param_vars[key].set(val)
        finally:
            self._updating_from_preset = False

    def _on_param_write(self, key):
        if not getattr(self, "_updating_from_preset", False):
            self.preset_var.set("Personalizado")

    def _build_run_button(self, parent):
        frame = tk.Frame(parent, bg=BG_DARK)
        frame.pack(fill="x", pady=(0, 4))
        
        btns = tk.Frame(frame, bg=BG_DARK)
        btns.pack(fill="x")
        
        self.run_btn = tk.Button(
            btns, text="▶  INICIAR ANÁLISIS",
            font=("Trebuchet MS", 12, "bold"),
            bg=ACCENT, fg=BG_DARK, relief="flat", bd=0,
            activebackground="#3d9e6b", cursor="hand2", pady=10,
            command=self._run_pipeline_thread)
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 2))
        
        self.cancel_btn = tk.Button(
            btns, text="✕",
            font=("Trebuchet MS", 12, "bold"),
            bg=TEXT_ERR, fg=TEXT_MAIN, relief="flat", bd=0,
            activebackground="#c65a41", cursor="hand2", pady=10,
            state="disabled", command=self._abort_pipeline)
        self.cancel_btn.pack(side="right", padx=(2, 0))
        self._add_tooltip(self.cancel_btn, "Detener el análisis actual")

        self.preview_btn = tk.Button(
            frame, text="👁  Previsualizar Parámetros (1 Tile)",
            font=FONT_BODY, bg=BG_CARD2, fg=TEXT_MUTED, relief="flat", bd=0,
            activebackground=BORDER, cursor="hand2", pady=6,
            command=self._run_preview_thread)
        self.preview_btn.pack(fill="x", pady=(4, 0))
        self._add_tooltip(self.preview_btn, "Procesa solo 1 tile para validar si el GSD y SAM están bien configurados.")

        self.main_progress = ttk.Progressbar(frame, variable=self.progress_val,
                                             maximum=100, mode="determinate")
        self.main_progress.pack(fill="x", pady=(4, 0))

    # ── RIGHT CARDS ────────────────────────────────────────────────────────────

    def _build_log_card(self, parent):
        card = self._card(parent, "📋  Log de ejecución")
        card.pack(fill="both", expand=True, pady=(0, 8))
        self.log_text = tk.Text(card, bg=BG_CARD2, fg=TEXT_MAIN,
                                font=FONT_MONO, relief="flat", bd=0,
                                wrap="word", state="disabled",
                                selectbackground=ACCENT2)
        sb = ttk.Scrollbar(card, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.tag_config("ok",    foreground=ACCENT)
        self.log_text.tag_config("info",  foreground=TEXT_MAIN)
        self.log_text.tag_config("warn",  foreground=TEXT_WARN)
        self.log_text.tag_config("err",   foreground=TEXT_ERR)
        self.log_text.tag_config("muted", foreground=TEXT_MUTED)

    def _build_results_card(self, parent):
        card = self._card(parent, "📊  Resultados")
        card.pack(fill="x")

        mf = tk.Frame(card, bg=BG_CARD)
        mf.pack(fill="x")
        mf.columnconfigure((0, 1), weight=1)

        self.metric_vars = {}
        metrics = [
            ("plantas",  "Objetos detectados",      ACCENT),
            ("low",      "Zona baja (rel. lote)",    TEXT_ERR),
            ("mid",      "Zona media (rel. lote)",   TEXT_WARN),
            ("high",     "Zona alta (rel. lote)",    "#7BC67E"),
        ]
        for i, (key, label, color) in enumerate(metrics):
            f = tk.Frame(mf, bg=BG_CARD2, padx=10, pady=8)
            f.grid(row=i//2, column=i%2, padx=3, pady=3, sticky="nsew")
            var = tk.StringVar(value="—")
            self.metric_vars[key] = var
            tk.Label(f, textvariable=var, font=FONT_BIG,
                     bg=BG_CARD2, fg=color).pack()
            tk.Label(f, text=label, font=("Calibri", 8),
                     bg=BG_CARD2, fg=TEXT_MUTED).pack()

        # Nota zonificación relativa
        tk.Label(card,
                 text="⚠ La zonificación es relativa al lote analizado (percentiles 33%-66%).\n"
                      "Las densidades pl/ha son orientativas según el cultivo.",
                 font=("Calibri", 8), bg=BG_CARD, fg=TEXT_MUTED,
                 justify="left", wraplength=320).pack(anchor="w", pady=(6, 4))

        # Botones exportación
        exports = [
            ("🗺️  Mapa de densidad (.png)",          "export_map",    "mapa_densidad.png"),
            ("📦  Zonificación GeoPackage (.gpkg)",   "export_gpkg",   "zonificacion_productiva.gpkg"),
            ("📍  Plantas individuales (.gpkg)",      "export_puntos", "plantas_puntos.gpkg"),
            ("📄  CSV plantas georreferenciadas",     "export_csv",    "plantas_detectadas.csv"),
            ("📋  Log de ejecución (.txt)",           "export_log",    None),
        ]
        self.export_btns = {}
        for label, key, filename in exports:
            btn = tk.Button(card, text=label, font=FONT_BODY,
                            bg=BG_CARD2, fg=TEXT_MUTED, relief="flat", bd=0,
                            cursor="hand2", state="disabled", anchor="w",
                            padx=10, pady=4,
                            command=lambda fn=filename: self._open_output(fn))
            btn.pack(fill="x", pady=1)
            self.export_btns[key] = btn

        tk.Button(card, text="📁  Abrir carpeta de resultados", font=FONT_BODY,
                  bg=BG_CARD, fg=ACCENT2, relief="flat", bd=0,
                  cursor="hand2", anchor="w", padx=10, pady=4,
                  command=self._open_output_folder).pack(fill="x", pady=(4, 0))

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_CARD, pady=4)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self.status_text,
                 font=("Calibri", 9), bg=BG_CARD, fg=TEXT_MUTED,
                 anchor="w").pack(side="left", padx=14)
        tk.Label(bar, text="AgroIA-COVIO v2.1 — desarrollado por Darío Sánchez Leguizamón",
                 font=("Calibri", 9, "italic"), bg=BG_CARD, fg=TEXT_MUTED,
                 anchor="e").pack(side="right", padx=14)

    def _abort_pipeline(self):
        if self._pipeline_running:
            if messagebox.askyesno("Confirmar", "¿Seguro que quieres detener el análisis?"):
                self._stop_event.set()
                self._log("Petición de parada enviada. Esperando cierre de procesos...", "warn")
                self.cancel_btn.config(state="disabled")

    def _run_preview_thread(self):
        if self._pipeline_running: return
        if not self.ortho_path.get() or not os.path.exists(self.ortho_path.get()):
            messagebox.showerror("Error", "Seleccioná un archivo GeoTIFF válido.")
            return
        
        selected_name = self.sam_model_var.get()
        if not os.path.exists(SAM_MODELS[selected_name]["path"]):
            messagebox.showerror("Modelo no disponible", f"Descargá {selected_name} primero.")
            return

        self._pipeline_running = True
        self.preview_btn.config(state="disabled", text="⏳ Procesando tile...")
        threading.Thread(target=self._run_preview, daemon=True).start()

    def _run_preview(self):
        try:
            from pipeline import preview_tile
            from PIL import Image, ImageTk
            
            params = self._get_params()
            self._log("Iniciando previsualización rápida...", "warn")
            
            img_arr, count = preview_tile(
                self.ortho_path.get(), MODELS_DIR, params, self._log
            )
            
            # Mostrar en popup
            popup = tk.Toplevel(self)
            popup.title(f"Previsualización - {count} instancias detectadas")
            popup.configure(bg=BG_DARK)
            
            # Redimensionar si es muy grande para la pantalla
            img_pil = Image.fromarray(img_arr)
            max_size = 800
            if max(img_pil.size) > max_size:
                img_pil.thumbnail((max_size, max_size))
            
            img_tk = ImageTk.PhotoImage(img_pil)
            lbl = tk.Label(popup, image=img_tk, bg=BG_DARK)
            lbl.image = img_tk
            lbl.pack(padx=20, pady=20)
            
            tk.Label(popup, text=f"Se detectaron {count} objetos con los parámetros actuales.\n"
                                 f"Verde: Máscara SAM + BBox filtrado.",
                     font=FONT_BODY, bg=BG_DARK, fg=TEXT_MAIN).pack(pady=(0, 20))
            
            self._log(f"Previsualización completada: {count} objetos en el tile central.", "ok")
            
        except Exception as e:
            self._log(f"Error en preview: {e}", "err")
            messagebox.showerror("Error", str(e))
        finally:
            self._pipeline_running = False
            self.preview_btn.config(state="normal", text="👁  Previsualizar Parámetros (1 Tile)")

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=BG_CARD, padx=12, pady=8,
                         highlightbackground=BORDER, highlightthickness=1)
        tk.Label(outer, text=title, font=FONT_LABEL,
                 bg=BG_CARD, fg=ACCENT).pack(anchor="w", pady=(0, 6))
        return outer

    def _add_tooltip(self, widget, text):
        tip = None
        def show(e):
            nonlocal tip
            tip = tk.Toplevel(self)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root+14}+{e.y_root-28}")
            tk.Label(tip, text=text, font=("Calibri", 9),
                     bg="#2A3F58", fg=TEXT_MAIN, padx=8, pady=4,
                     relief="flat", wraplength=300).pack()
        def hide(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _log(self, msg, tag="info"):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, msg):
        self.status_text.set(msg)
        self.update_idletasks()

    def _set_progress(self, val):
        self.progress_val.set(val)
        self.update_idletasks()

    def _browse_ortho(self):
        path = filedialog.askopenfilename(
            title="Seleccionar ortomosaico GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Todos", "*.*")])
        if path:
            self.ortho_path.set(path)
            self._log(f"Ortomosaico: {os.path.basename(path)}", "ok")
            try:
                import rasterio
                with rasterio.open(path) as src:
                    crs = src.crs
                    if crs and not crs.is_geographic:
                        actual_gsd_cm = abs(src.transform.a) * 100
                        self.param_vars["gsd"].set(f"{actual_gsd_cm:.2f}")
                        self._log(f"  GSD real detectado del raster: {actual_gsd_cm:.3f} cm/px (CRS: {crs})", "ok")
                    else:
                        self._log(f"  CRS geográfico o indefinido detectado. Por favor, verificar el GSD manualmente.", "warn")
            except Exception as e:
                self._log(f"  No se pudo auto-detectar el GSD: {e}", "muted")

    def _browse_output(self):
        path = filedialog.askdirectory(title="Seleccionar carpeta de resultados")
        if path:
            self.output_path.set(path)
            self._log(f"Carpeta base: {path}", "muted")

    def _get_output_dir(self):
        base = self.output_path.get()
        lote = self._get_lote_name()
        full = os.path.join(base, lote)
        os.makedirs(full, exist_ok=True)
        return full

    def _open_output(self, filename):
        folder = self._get_output_dir()
        if filename is None:
            logs = sorted([f for f in os.listdir(folder)
                           if f.startswith("pipeline_log")])
            filename = logs[-1] if logs else None
            if not filename:
                messagebox.showinfo("Sin log", "Todavía no se generó ningún log.")
                return
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            # Buscar variante .geojson
            alt = path.replace(".gpkg", ".geojson")
            if os.path.exists(alt):
                path = alt
            else:
                messagebox.showwarning("Archivo no encontrado", f"No se encontró:\n{path}")
                return
        import subprocess, platform
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    def _open_output_folder(self):
        folder = self._get_output_dir()
        import subprocess, platform
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.call(["open", folder])
        else:
            subprocess.call(["xdg-open", folder])

    # ── SAM ────────────────────────────────────────────────────────────────────

    def _check_sam_status(self):
        selected_name = self.sam_model_var.get()
        model_info = SAM_MODELS[selected_name]
        path = model_info["path"]
        
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            # ViT-H > 2000MB, MobileSAM > 35MB
            min_size = 2000 if "ViT-H" in selected_name else 35
            if size_mb > min_size:
                self.sam_status_lbl.config(
                    text=f"✅  Modelo disponible  ({size_mb/1000:.2f} GB)" if size_mb > 1000 else f"✅  Modelo disponible  ({size_mb:.1f} MB)", 
                    fg=ACCENT)
                self.sam_dl_btn.pack_forget()
                return
        
        self.sam_status_lbl.config(
            text=f"❌  Modelo no encontrado — descarga requerida (~{model_info['size']} GB)" if model_info['size'] > 0.1 else f"❌  Modelo no encontrado — descarga requerida (~{model_info['size']*1000:.0f} MB)",
            fg=TEXT_WARN)
        self.sam_dl_btn.config(text=f"⬇  Descargar {selected_name.split(' ')[0]} (~{model_info['size']} GB)" if model_info['size'] > 0.1 else f"⬇  Descargar {selected_name.split(' ')[0]} (~{model_info['size']*1000:.0f} MB)")
        self.sam_dl_btn.pack(fill="x", pady=(6, 0))

    def _download_sam_thread(self):
        threading.Thread(target=self._download_sam, daemon=True).start()

    def _download_sam(self):
        import urllib.request
        selected_name = self.sam_model_var.get()
        model_info = SAM_MODELS[selected_name]
        url = model_info["url"]
        path = model_info["path"]

        self.sam_dl_btn.config(state="disabled")
        self.sam_progress.pack(fill="x", pady=(4, 0))
        self.dl_label.pack(anchor="w")
        self._log(f"Descargando {selected_name}...", "warn")
        self._set_status(f"Descargando {selected_name}...")
        try:
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    pct = min(100, count * block_size / total_size * 100)
                    self.sam_progress_var.set(pct)
                    mb = count * block_size / 1e6
                    self.dl_label.config(
                        text=f"{mb:.1f} MB / {total_size/1e6:.1f} MB  ({pct:.1f}%)")
                self.update_idletasks()
            urllib.request.urlretrieve(url, path, reporthook)
            self._log(f"✓ {selected_name} descargado.", "ok")
            self._check_sam_status()
            self.sam_progress.pack_forget()
            self.dl_label.pack_forget()
        except Exception as e:
            self._log(f"Error en descarga: {e}", "err")
            if os.path.exists(path):
                os.remove(path)
            self.sam_dl_btn.config(state="normal")
        finally:
            self._set_status("Listo.")

    # ── PIPELINE ───────────────────────────────────────────────────────────────

    def _run_pipeline_thread(self):
        if self._pipeline_running:
            return
        if not self.ortho_path.get() or not os.path.exists(self.ortho_path.get()):
            messagebox.showerror("Error", "Seleccioná un archivo GeoTIFF válido.")
            return
        
        selected_name = self.sam_model_var.get()
        model_info = SAM_MODELS[selected_name]
        if not os.path.exists(model_info["path"]):
            messagebox.showerror("Modelo no disponible",
                                 f"Descargá el modelo {selected_name} antes de iniciar.")
            return

        if self._lote_placeholder or not self.lote_entry.get().strip():
            messagebox.showerror("Nombre requerido",
                                 "Ingresá el nombre del cultivo / lote antes de continuar.")
            return

        output_dir = self._get_output_dir()
        self._pipeline_running = True
        self._stop_event.clear()
        self.run_btn.config(state="disabled", text="⏳  Analizando...")
        self.cancel_btn.config(state="normal")
        
        threading.Thread(target=self._run_pipeline,
                         args=(self.ortho_path.get(), output_dir),
                         daemon=True).start()

    def _run_pipeline(self, ortho_path, output_dir):
        try:
            self._log("=" * 50, "muted")
            self._log("AgroIA-COVIO v2.1 — INICIANDO ANÁLISIS", "ok")
            self._log("=" * 50, "muted")
            params = self._get_params()
            from pipeline import run_full_pipeline
            result = run_full_pipeline(
                ortho_path=ortho_path,
                models_dir=MODELS_DIR,
                output_dir=output_dir,
                params=params,
                log_fn=self._log,
                progress_fn=self._set_progress,
                status_fn=self._set_status,
                export_shp=self.export_shp.get(),
                stop_event=self._stop_event
            )
            if result:
                self.update_metrics(
                    result["n_plants"],
                    result["density_low"],
                    result["density_mid"],
                    result["density_high"],
                )
        except KeyboardInterrupt:
            self._log("\nANÁLISIS CANCELADO POR EL USUARIO", "warn")
            self._set_status("Cancelado.")
        except ImportError as e:
            self._log(f"Dependencia faltante: {e}", "err")
        except Exception as e:
            self._log(f"Error: {e}", "err")
            import traceback
            self._log(traceback.format_exc(), "err")
        finally:
            self._pipeline_running = False
            self.run_btn.config(state="normal", text="▶  INICIAR ANÁLISIS")
            self.cancel_btn.config(state="disabled")
            self._set_progress(0)

    def _get_params(self):
        def f(k): return float(self.param_vars[k].get())
        def i(k): return int(float(self.param_vars[k].get()))
        gsd_cm = f("gsd")
        selected_name = self.sam_model_var.get()
        return {
            "gsd":          gsd_cm / 100,           # cm → m
            "gsd_cm":       gsd_cm,                  # para log
            "tile":         i("tile"),
            "conf":         f("conf"),
            "nms_dist":     f("nms_cm") / gsd_cm,   # cm → px
            "nms_cm":       f("nms_cm"),             # para log
            "min_size":     f("min_cm") / gsd_cm,   # cm → px
            "min_cm":       f("min_cm"),             # para log
            "max_size":     f("max_cm") / gsd_cm,   # cm → px
            "max_cm":       f("max_cm"),             # para log
            "epochs":       i("epochs"),
            "grid_size":    i("grid"),
            "sam_fraction": f("sam_fraction") / 100,
            "sam_type":     SAM_MODELS[selected_name]["type"],
            "lote":         self._get_lote_name(),
        }

    def update_metrics(self, n_plants, density_low, density_mid, density_high):
        self.metric_vars["plantas"].set(str(n_plants))
        self.metric_vars["low"].set(f"~{int(density_low):,}")
        self.metric_vars["mid"].set(f"~{int(density_mid):,}")
        self.metric_vars["high"].set(f"~{int(density_high):,}")
        for btn in self.export_btns.values():
            btn.config(state="normal", fg=TEXT_MAIN)


def main():
    root = CovioApp()
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Horizontal.TProgressbar",
                    troughcolor=BG_CARD2, background=ACCENT,
                    bordercolor=BG_CARD2, lightcolor=ACCENT, darkcolor=ACCENT)
    style.configure("TScrollbar",
                    troughcolor=BG_CARD2, background=BORDER, bordercolor=BG_CARD)
    root.mainloop()


if __name__ == "__main__":
    main()
