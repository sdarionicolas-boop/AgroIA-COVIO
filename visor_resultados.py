"""Visor de resultados de indices dentro de la interfaz (Tkinter).

Lee la carpeta de salida (la de lote.py / indices.py) y muestra pestañas: Mapa del lote (si se
analizaron lotes), Indices del ortomosaico, Tabla y Notas. No necesita los objetos en memoria: sirve
tambien para reabrir resultados anteriores.
"""
import base64
import csv
import os
import tkinter as tk
from tkinter import ttk

import cv2

from indices import INDICES, aviso_para
from mapa_lote import MapaLote

_ETIQUETAS = {
    "area_ha": "Área del polígono (ha)", "area_con_datos_ha": "Área con datos (ha)",
    "indice_vegetacion": "Índice de vegetación", "indice_vigor": "Índice de vigor",
    "cobertura_veg_pct": "Cobertura vegetal (%)", "n_fallas": "Nº de fallas",
    "umbral_separacion": "Separación del umbral (0–1; <0,7 = dudoso)",
    "pct_area_baja": "Celdas de vigor bajo (% del área)", "pct_area_normal": "Celdas de vigor normal (% del área)",
    "pct_area_alta": "Celdas de vigor alto (% del área)", "desvio_min_abs": "Desvío mínimo para clasificar (valor)", "area_fallas_m2": "Área de fallas (m²)", "fallas_pct": "Fallas (% del lote)", "n_celdas": "Nº de celdas",
}


def _etiqueta(k):
    if k in _ETIQUETAS:
        return _ETIQUETAS[k]
    for nom, (tit, _b) in INDICES.items():
        if k.startswith("umbral_" + nom):
            return f"Umbral {tit}"
        if k.startswith(nom + "_"):
            suf = {"medio": "media", "std": "desvío", "p10": "percentil 10", "p90": "percentil 90",
                   "mediana_celdas": "mediana entre celdas"}.get(k[len(nom) + 1:], k[len(nom) + 1:])
            return f"{tit} {suf}"
        if k == f"cv_{nom}_entre_celdas_pct":
            return f"{tit}: variación entre celdas (%)"
    return k


def cargar_carpeta(out_dir):
    """Descubre lo que hay en la carpeta de resultados. Devuelve un dict con rutas y tablas (puede estar vacio)."""
    ind_dir = os.path.join(out_dir, "indices") if os.path.isdir(os.path.join(out_dir, "indices")) else out_dir
    r = dict(out_dir=out_dir, mapa=None, indices_png=None, gpkg=None, notas="", tabla=None, titulo="")
    f = os.path.join(out_dir, "vista.png")
    r["mapa"] = f if os.path.exists(f) else None
    f = os.path.join(ind_dir, "indices.png")
    r["indices_png"] = f if os.path.exists(f) else None
    f = os.path.join(out_dir, "lote_analizado.gpkg")
    r["gpkg"] = f if os.path.exists(f) else None
    usados = []
    fs = os.path.join(ind_dir, "indices_stats.csv")
    stats = []
    if os.path.exists(fs):
        with open(fs, encoding="utf-8") as fh:
            stats = list(csv.DictReader(fh))
        usados = [x["indice"] for x in stats]
    fr = os.path.join(out_dir, "resumen.csv")
    if os.path.exists(fr):
        with open(fr, encoding="utf-8") as fh:
            filas = list(csv.DictReader(fh))
        if filas:
            claves = list(filas[0].keys())
            # tabla transpuesta: una fila por metrica, una columna por lote
            cols = ["Métrica"] + [x["lote"] for x in filas]
            datos = [[_etiqueta(k)] + [(x[k] if x[k] != "" else "—") for x in filas] for k in claves if k != "lote"]
            r["tabla"] = (cols, datos)
            r["titulo"] = f"{len(filas)} lote(s) analizado(s)"
    elif stats:
        cols = ["Índice", "Media", "P2", "P98"]
        datos = [[INDICES.get(x["indice"], (x["indice"],))[0], x["media"], x["p2"], x["p98"]] for x in stats]
        r["tabla"] = (cols, datos)
        r["titulo"] = "Índices de todo el ortomosaico (sin lotes)"
    fl = os.path.join(out_dir, "LEEME.txt")
    if os.path.exists(fl):
        with open(fl, encoding="utf-8") as fh:
            r["notas"] = fh.read()
    elif usados:
        r["notas"] = aviso_para(usados)
    return r


class _ImagenAjustada(tk.Canvas):
    """Canvas que muestra una imagen escalada para entrar en el espacio disponible."""

    def __init__(self, parent, path, bg="#000"):
        super().__init__(parent, bg=bg, highlightthickness=0)
        self.img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) if path else None
        self._photo = None
        self._after = None
        self.bind("<Configure>", self._on_conf)

    def _on_conf(self, _e):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(60, self._dibujar)

    def _dibujar(self):
        self.delete("all")
        if self.img is None:
            return
        cw, ch = max(50, self.winfo_width()), max(50, self.winfo_height())
        h, w = self.img.shape[:2]
        k = min(cw / w, ch / h)
        show = cv2.resize(self.img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
        self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        self.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")


class VisorResultados(tk.Toplevel):
    def __init__(self, parent, out_dir, colors=None):
        super().__init__(parent)
        c = dict(bg="#0F1923", card="#1A2535", text="#E8EDF2", muted="#7A8FA6", accent="#4CAF82", accent2="#2E86AB",
                 warn="#F4A261")
        c.update(colors or {})
        self.c = c
        self.out_dir = out_dir
        self.res = cargar_carpeta(out_dir)
        self.title("Resultados de índices")
        self.configure(bg=c["bg"])
        self.geometry("1180x780")
        self.transient(parent)

        style = ttk.Style(self)
        style.configure("Visor.Treeview", background=c["card"], fieldbackground=c["card"], foreground=c["text"],
                        rowheight=24, borderwidth=0)
        style.configure("Visor.Treeview.Heading", background=c["bg"], foreground=c["accent"], relief="flat")
        style.map("Visor.Treeview", background=[("selected", c["accent2"])])
        style.configure("TNotebook", background=c["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=c["card"], foreground=c["muted"], padding=(14, 6), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", c["accent2"])], foreground=[("selected", "#ffffff")])

        tk.Label(self, text="Resultados de índices", font=("Trebuchet MS", 14, "bold"), bg=c["bg"], fg=c["accent"]
                 ).pack(anchor="w", padx=14, pady=(12, 0))
        tk.Label(self, text=self.res["titulo"], bg=c["bg"], fg=c["muted"], font=("Calibri", 10)
                 ).pack(anchor="w", padx=14, pady=(0, 6))

        adv = [ln[len("ADVERTENCIA:"):].strip() for ln in self.res["notas"].splitlines()
               if ln.startswith("ADVERTENCIA:")]
        for a in adv:
            tk.Label(self, text="⚠ " + a, bg=c["bg"], fg=c["warn"], font=("Calibri", 10), wraplength=1120,
                     justify="left", anchor="w").pack(fill="x", padx=14, pady=(0, 6))

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=14)
        if self.res["gpkg"]:
            f = tk.Frame(nb, bg=c["bg"])
            m = MapaLote(f, out_dir, colors=c)
            m.pack(fill="both", expand=True, padx=6, pady=6)
            if m.d is not None:
                nb.add(f, text="🗺  Mapa del lote")
            elif self.res["mapa"]:
                self._tab_imagen(nb, "🗺  Mapa del lote", self.res["mapa"])
        elif self.res["mapa"]:
            self._tab_imagen(nb, "🗺  Mapa del lote", self.res["mapa"])
        if self.res["indices_png"]:
            self._tab_imagen(nb, "🛰  Índices del ortomosaico", self.res["indices_png"])
        if self.res["tabla"]:
            self._tab_tabla(nb, self.res["tabla"])
        self._tab_notas(nb, self.res["notas"])
        if not nb.tabs():
            tk.Label(self, text="No hay resultados en esta carpeta.", bg=c["bg"], fg=c["muted"]).pack(pady=40)

        foot = tk.Frame(self, bg=c["bg"])
        foot.pack(fill="x", padx=14, pady=10)
        tk.Button(foot, text="📁  Abrir carpeta", command=self._abrir_carpeta, bg=c["card"], fg=c["text"], relief="flat",
                  bd=0, cursor="hand2", padx=12, pady=6).pack(side="left")
        if self.res["gpkg"]:
            tk.Button(foot, text="📦  Abrir GeoPackage (QGIS)", command=lambda: self._abrir(self.res["gpkg"]),
                      bg=c["card"], fg=c["text"], relief="flat", bd=0, cursor="hand2", padx=12, pady=6
                      ).pack(side="left", padx=6)
        tk.Button(foot, text="Cerrar", command=self.destroy, bg=c["accent"], fg=c["bg"], relief="flat", bd=0,
                  cursor="hand2", padx=16, pady=6).pack(side="right")

    def _tab_imagen(self, nb, titulo, path):
        f = tk.Frame(nb, bg="#000")
        _ImagenAjustada(f, path).pack(fill="both", expand=True)
        nb.add(f, text=titulo)

    def _tab_tabla(self, nb, tabla):
        cols, datos = tabla
        f = tk.Frame(nb, bg=self.c["card"])
        tv = ttk.Treeview(f, columns=cols, show="headings", style="Visor.Treeview")
        for i, col in enumerate(cols):
            tv.heading(col, text=col)
            tv.column(col, width=300 if i == 0 else 140, anchor="w" if i == 0 else "e", stretch=(i == 0))
        for fila in datos:
            tv.insert("", "end", values=fila)
        sy = ttk.Scrollbar(f, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        tv.pack(fill="both", expand=True)
        nb.add(f, text="📋  Tabla")

    def _tab_notas(self, nb, texto):
        f = tk.Frame(nb, bg=self.c["card"])
        t = tk.Text(f, wrap="word", bg=self.c["card"], fg=self.c["text"], relief="flat", bd=0, padx=14, pady=12,
                    font=("Calibri", 10))
        t.insert("1.0", texto or "Sin notas.")
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)
        nb.add(f, text="ℹ  Notas")

    def _abrir(self, path):
        import platform
        import subprocess
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", path])
        else:
            subprocess.call(["xdg-open", path])

    def _abrir_carpeta(self):
        self._abrir(self.out_dir)
