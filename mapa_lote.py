"""Mapa interactivo de los lotes analizados (Tkinter): zoom con la rueda, arrastrar para desplazar,
capas que se prenden y apagan (indice, celdas, clases, fallas, limite) y valores al pasar el mouse.

Lee la carpeta de salida de lote.py: lote_analizado.gpkg (capas lotes, grilla, fallas) y los .tif de
indices en <salida>/indices.
"""
import base64
import glob
import os
import tkinter as tk

import cv2
import numpy as np

import lotes_io
from indices import INDICES

# clase de vigor -> color RGB (None = no se pinta)
COLORES_CLASE = {"baja": (215, 48, 39), "alta": (26, 152, 80), "media": (254, 224, 139),
                 "normal": None, "n/d": (150, 150, 150)}


def _clave_indice(titulo):
    """'NDRE' -> 'ndre' (inverso de INDICES[...][0])."""
    for k, (t, _b) in INDICES.items():
        if t == titulo:
            return k
    return None


def cargar_datos(out_dir):
    """Lee el GeoPackage de lotes y los indices disponibles. Devuelve None si no hay lotes analizados."""
    import geopandas as gpd

    gpkg = os.path.join(out_dir, "lote_analizado.gpkg")
    if not os.path.exists(gpkg):
        return None

    def capa(nombre):
        try:
            return gpd.read_file(gpkg, layer=nombre)
        except Exception:
            return None

    lotes = capa("lotes")
    if lotes is None or lotes.empty:
        return None
    ind_dir = os.path.join(out_dir, "indices")
    tifs = {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(glob.glob(os.path.join(ind_dir, "*.tif")))
            if os.path.splitext(os.path.basename(p))[0] in INDICES}
    if not tifs:
        return None
    vigor = _clave_indice(str(lotes.iloc[0].get("indice_vigor", ""))) or next(iter(tifs))
    return dict(lotes=lotes, grilla=capa("grilla"), fallas=capa("fallas"), tifs=tifs, vigor=vigor,
                bounds=tuple(lotes.total_bounds))


class MapaLote(tk.Frame):
    AYUDA = ("Rueda: zoom | arrastrar: desplazar\n"
             "Pasá el mouse sobre el mapa para ver el valor exacto de cada índice en ese punto")

    def __init__(self, parent, out_dir, colors=None):
        c = dict(bg="#0F1923", card="#1A2535", text="#E8EDF2", muted="#7A8FA6", accent="#4CAF82", accent2="#2E86AB")
        c.update(colors or {})
        super().__init__(parent, bg=c["bg"])
        self.c = c
        self.d = cargar_datos(out_dir)
        self._base = {}          # indice -> vista_raster
        self._comp = {}          # (indice, celdas, clases) -> imagen compuesta
        self._photo = None
        self.z, self.ox, self.oy = 1.0, 0.0, 0.0
        self._drag = None
        if self.d is None:
            tk.Label(self, text="No hay lotes analizados en esta carpeta.", bg=c["bg"], fg=c["muted"]).pack(pady=40)
            return

        self.fondo = tk.StringVar(value=self.d["vigor"])
        self.v_celdas = tk.BooleanVar(value=True)
        self.v_clases = tk.BooleanVar(value=False)
        self.v_fallas = tk.BooleanVar(value=True)
        self.v_limite = tk.BooleanVar(value=True)
        self._ds = {}            # indice -> dataset rasterio abierto (lectura exacta de pixel)
        self._mov = None         # ultimo evento de mouse pendiente (se atiende con un pequeño retardo)
        self.info = tk.StringVar(value=self.AYUDA)
        self._build()
        self._preparar()
        self._primera = True
        self.canvas.bind("<Configure>", lambda e: self._ajustar() if self._primera else self._dibujar())

    # ── datos geometricos ────────────────────────────────────────────────
    def _preparar(self):
        b = self._vista(self.d["vigor"])
        self.tr, self.inv = b["transform"], ~b["transform"]
        self.size = b["size"]
        self._poly_px = {"lotes": [], "fallas": [], "celdas": []}
        for g in self.d["lotes"].geometry:
            self._poly_px["lotes"] += self._a_px(g)
        if self.d["fallas"] is not None:
            for g in self.d["fallas"].geometry:
                self._poly_px["fallas"] += self._a_px(g)
        g = self.d["grilla"]
        self._celdas = []
        if g is not None and not g.empty:
            for _, r in g.iterrows():
                self._celdas.append((r.geometry, r))

    def _a_px(self, geom):
        partes = geom.geoms if hasattr(geom, "geoms") else [geom]
        return [np.array([self.inv * (x, y) for x, y in p.exterior.coords], np.float32) for p in partes
                if p.geom_type == "Polygon"]

    def _vista(self, clave):
        if clave not in self._base:
            self._base[clave] = lotes_io.vista_raster(self.d["tifs"][clave], bounds=self.d["bounds"])
        return self._base[clave]

    # ── UI ───────────────────────────────────────────────────────────────
    def _build(self):
        c = self.c
        bar = tk.Frame(self, bg=c["bg"])
        bar.pack(fill="x", pady=(0, 4))
        tk.Label(bar, text="Índice:", bg=c["bg"], fg=c["muted"]).pack(side="left", padx=(0, 2))
        for k in self.d["tifs"]:
            tk.Radiobutton(bar, text=INDICES[k][0], value=k, variable=self.fondo, command=self._dibujar,
                           bg=c["bg"], fg=c["text"], selectcolor=c["card"], activebackground=c["bg"],
                           activeforeground=c["text"]).pack(side="left")
        tk.Frame(bar, width=14, bg=c["bg"]).pack(side="left")
        for txt, var in (("Celdas (valor)", self.v_celdas), ("Clases", self.v_clases),
                         ("Fallas", self.v_fallas), ("Límite", self.v_limite)):
            tk.Checkbutton(bar, text=txt, variable=var, command=self._dibujar, bg=c["bg"], fg=c["text"],
                           selectcolor=c["card"], activebackground=c["bg"], activeforeground=c["text"]
                           ).pack(side="left", padx=2)
        for txt, f in (("−", lambda: self._zoom_centro(1 / 1.3)), ("+", lambda: self._zoom_centro(1.3)),
                       ("Ajustar", self._ajustar)):
            tk.Button(bar, text=txt, command=f, bg=c["card"], fg=c["text"], relief="flat", bd=0, padx=10,
                      cursor="hand2").pack(side="right", padx=2)
        self.canvas = tk.Canvas(self, bg="#101820", highlightthickness=0, cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        tk.Label(self, textvariable=self.info, bg=c["bg"], fg=c["text"], anchor="w", justify="left",
                 font=("Consolas", 10), height=2).pack(fill="x", pady=(4, 0))
        cv = self.canvas
        cv.bind("<MouseWheel>", self._on_wheel)
        cv.bind("<ButtonPress-1>", self._on_press)
        cv.bind("<B1-Motion>", self._on_drag)
        cv.bind("<Motion>", self._on_move)
        cv.bind("<Leave>", lambda e: self.info.set(self.AYUDA))

    # ── composicion de la imagen ─────────────────────────────────────────
    def _compuesta(self):
        clave = (self.fondo.get(), self.v_celdas.get(), self.v_clases.get())
        if clave in self._comp:
            return self._comp[clave]
        base = self._vista(self.fondo.get())["img"].copy()
        if self._celdas and (clave[1] or clave[2]):
            h, w = base.shape[:2]
            if clave[1]:
                v = self._vista(self.d["vigor"])
                col = f"{self.d['vigor']}_medio"
                ov = base.copy()
                for g, r in self._celdas:
                    val = r.get(col)
                    if val is None or not np.isfinite(val):
                        continue
                    rgb = lotes_io.colorear(np.array([[val]], np.float32), v["vmin"], v["vmax"])[0, 0]
                    for p in self._a_px(g):
                        cv2.fillPoly(ov, [p.astype(np.int32)], tuple(int(x) for x in rgb))
                base = cv2.addWeighted(ov, 0.85, base, 0.15, 0)
            if clave[2]:
                ov = base.copy()
                hay = False
                for g, r in self._celdas:
                    rgb = COLORES_CLASE.get(str(r.get("vigor")))
                    if rgb is None:
                        continue
                    hay = True
                    for p in self._a_px(g):
                        cv2.fillPoly(ov, [p.astype(np.int32)], rgb)
                if hay:
                    base = cv2.addWeighted(ov, 0.6, base, 0.4, 0)
        if len(self._comp) > 6:
            self._comp.clear()
        self._comp[clave] = base
        return base

    def _dibujar(self):
        if self.d is None:
            return
        cv = self.canvas
        cw, ch = max(50, cv.winfo_width()), max(50, cv.winfo_height())
        img = self._compuesta()
        m = np.float32([[self.z, 0, -self.ox * self.z], [0, self.z, -self.oy * self.z]])
        interp = cv2.INTER_AREA if self.z < 1 else cv2.INTER_LINEAR
        show = cv2.warpAffine(img, m, (cw, ch), flags=interp, borderValue=(16, 24, 32))
        ok, png = cv2.imencode(".png", cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
        self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        cv.delete("all")
        cv.create_image(0, 0, image=self._photo, anchor="nw")
        sp = lambda p: [(x - self.ox) * self.z if i % 2 == 0 else (x - self.oy) * self.z
                        for pt in p for i, x in enumerate(pt)]
        if self.v_fallas.get():
            for p in self._poly_px["fallas"]:
                cv.create_polygon(*sp(p), outline="#ffffff", fill="", width=1.5)
        if self.v_limite.get():
            for p in self._poly_px["lotes"]:
                cv.create_polygon(*sp(p), outline="#ffd400", fill="", width=2)
        self._leyenda(cv, cw)

    def _leyenda(self, cv, cw):
        v = self._vista(self.d["vigor"])
        barra = lotes_io.colorear(np.linspace(v["vmax"], v["vmin"], 120, dtype=np.float32).reshape(-1, 1).repeat(14, 1),
                                  v["vmin"], v["vmax"])
        ok, png = cv2.imencode(".png", cv2.cvtColor(barra, cv2.COLOR_RGB2BGR))
        self._ph_barra = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        cv.create_rectangle(cw - 78, 8, cw - 6, 160, fill="#0F1923", outline="")
        cv.create_image(cw - 70, 20, image=self._ph_barra, anchor="nw")
        cv.create_text(cw - 52, 20, text=f"{v['vmax']:.2f}", anchor="w", fill="#E8EDF2", font=("Calibri", 9))
        cv.create_text(cw - 52, 134, text=f"{v['vmin']:.2f}", anchor="w", fill="#E8EDF2", font=("Calibri", 9))
        cv.create_text(cw - 42, 13, text=INDICES[self.d["vigor"]][0], fill="#7A8FA6", font=("Calibri", 8, "bold"))

    # ── zoom / desplazamiento ────────────────────────────────────────────
    def _ajustar(self):
        cw, ch = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        w, h = self.size
        self.z = min(cw / w, ch / h)
        self.ox = -(cw / self.z - w) / 2
        self.oy = -(ch / self.z - h) / 2
        self._primera = False
        self._dibujar()

    def _zoom_en(self, factor, cx, cy):
        z2 = max(0.1, min(self.z * factor, 12.0))
        # mantener fijo el punto bajo el cursor
        px, py = cx / self.z + self.ox, cy / self.z + self.oy
        self.z = z2
        self.ox, self.oy = px - cx / z2, py - cy / z2
        self._dibujar()

    def _zoom_centro(self, factor):
        self._zoom_en(factor, self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)

    def _on_wheel(self, e):
        self._zoom_en(1.2 if e.delta > 0 else 1 / 1.2, e.x, e.y)

    def _on_press(self, e):
        self._drag = (e.x, e.y, self.ox, self.oy)

    def _on_drag(self, e):
        if self._drag:
            x0, y0, ox, oy = self._drag
            self.ox, self.oy = ox - (e.x - x0) / self.z, oy - (e.y - y0) / self.z
            self._dibujar()

    # ── valores al pasar el mouse ────────────────────────────────────────
    def _valor_exacto(self, clave, x, y):
        """Valor del indice en el pixel del ortomosaico bajo (x, y) (coordenadas del mapa); None si no hay dato."""
        import rasterio
        ds = self._ds.get(clave)
        if ds is None:
            ds = self._ds[clave] = rasterio.open(self.d["tifs"][clave])
        try:
            v = next(ds.sample([(x, y)]))[0]
        except Exception:
            return None
        return float(v) if np.isfinite(v) else None

    def _on_move(self, e):
        # se atiende con un retardo corto para no leer el archivo en cada pixel de movimiento
        pendiente = self._mov is not None
        self._mov = (e.x, e.y)
        if not pendiente:
            self.after(45, self._atender_mov)

    def _atender_mov(self):
        if self._mov is None:
            return
        ex, ey = self._mov
        self._mov = None
        col, row = ex / self.z + self.ox, ey / self.z + self.oy
        x, y = self.tr * (col, row)
        self.info.set(self.texto_en(x, y))

    def texto_en(self, x, y):
        """Texto para el punto (x, y) del mapa: coordenadas, valor exacto de cada indice y datos de la celda."""
        from shapely.geometry import Point
        vals = []
        hay = False
        for k in self.d["tifs"]:
            v = self._valor_exacto(k, x, y)
            hay = hay or v is not None
            vals.append(f"{INDICES[k][0]} {v:.3f}" if v is not None else f"{INDICES[k][0]} s/d")
        l1 = f"X {x:,.1f}  Y {y:,.1f}   |   " + "   ".join(vals) if hay else f"X {x:,.1f}  Y {y:,.1f}   |   sin dato en este punto"
        l2 = "fuera de las celdas analizadas"
        pt = Point(x, y)
        for g, r in self._celdas:
            if g.contains(pt):
                l2 = (f"celda de {r.get('lote')}: cobertura {r.get('cobertura_pct')} %   falla {r.get('falla_m2')} m²"
                      + (f"   clase {r.get('vigor')}" if r.get("vigor") is not None else ""))
                break
        return l1 + "\n" + l2

    def destroy(self):
        for ds in self._ds.values():
            try:
                ds.close()
            except Exception:
                pass
        self._ds = {}
        super().destroy()
