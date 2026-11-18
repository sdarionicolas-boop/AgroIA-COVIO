"""Editor para dibujar lotes con el mouse sobre una vista previa del ortomosaico (Tkinter).

Uso: clic izquierdo = agregar vertice | clic derecho o Enter = cerrar poligono (pide un nombre) |
Retroceso = deshacer vertice | Esc = descartar el poligono en curso.
'Guardar y usar' escribe un GeoPackage con los lotes dibujados.
"""
import base64
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import cv2

import lotes_io

_COLORES = ["#ffd400", "#00e5ff", "#ff6bd6", "#9cff57", "#ff9f40", "#b388ff"]


class EditorLotes(tk.Toplevel):
    def __init__(self, parent, ortho_path, out_path, existing_path=None, on_saved=None, colors=None, modo=None):
        super().__init__(parent)
        c = dict(bg="#0F1923", card="#1A2535", text="#E8EDF2", muted="#7A8FA6", accent="#4CAF82", accent2="#2E86AB")
        c.update(colors or {})
        self.c = c
        self.ortho_path, self.out_path, self.existing_path = ortho_path, out_path, existing_path
        self.on_saved = on_saved
        self.modo_indices = modo
        self.title("Dibujar lotes")
        self.configure(bg=c["bg"])
        self.geometry("1100x780")
        self.transient(parent)

        self.modo = tk.StringVar(value="rgb")
        self.polys = []       # [{"nombre": str, "pts": [(col, fila), ...]}]
        self.cur = []         # vertices del poligono en curso
        self.zoom = 1.0
        self._photo = None
        self.prev = None

        self._build_ui()
        self._cargar_fondo()
        self._cargar_existentes()
        self._redraw()
        self.after(150, self._fit)   # cuando la ventana ya tiene su tamano real

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        c = self.c
        bar = tk.Frame(self, bg=c["bg"])
        bar.pack(fill="x", padx=10, pady=(8, 4))

        def btn(text, cmd, bg=None, **kw):
            b = tk.Button(bar, text=text, command=cmd, bg=bg or c["card"], fg=c["text"], relief="flat", bd=0,
                          cursor="hand2", padx=10, pady=5, activebackground=c["accent2"], **kw)
            b.pack(side="left", padx=2)
            return b

        tk.Label(bar, text="Fondo:", bg=c["bg"], fg=c["muted"]).pack(side="left", padx=(0, 2))
        for txt, val in self._fondos():
            tk.Radiobutton(bar, text=txt, value=val, variable=self.modo, command=self._cambiar_fondo,
                           bg=c["bg"], fg=c["text"], selectcolor=c["card"], activebackground=c["bg"],
                           activeforeground=c["text"]).pack(side="left")
        tk.Frame(bar, width=10, bg=c["bg"]).pack(side="left")
        btn("−", lambda: self._set_zoom(self.zoom / 1.25))
        btn("+", lambda: self._set_zoom(self.zoom * 1.25))
        btn("Ajustar", self._fit)
        tk.Frame(bar, width=10, bg=c["bg"]).pack(side="left")
        btn("✔ Cerrar polígono", self.cerrar_poligono, bg=c["accent2"])
        btn("↶ Deshacer vértice", self.deshacer_vertice)
        btn("🗑 Borrar último lote", self.borrar_ultimo)

        mid = tk.Frame(self, bg=c["bg"])
        mid.pack(fill="both", expand=True, padx=10)
        self.canvas = tk.Canvas(mid, bg="#000", highlightthickness=0, cursor="crosshair")
        sx = ttk.Scrollbar(mid, orient="horizontal", command=self.canvas.xview)
        sy = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", lambda e: self.cerrar_poligono())
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Control-MouseWheel>", lambda e: self._set_zoom(self.zoom * (1.15 if e.delta > 0 else 1 / 1.15)))
        for k, f in (("<Return>", lambda e: self.cerrar_poligono()), ("<BackSpace>", lambda e: self.deshacer_vertice()),
                     ("<Escape>", lambda e: self.descartar_actual())):
            self.bind(k, f)

        foot = tk.Frame(self, bg=c["bg"])
        foot.pack(fill="x", padx=10, pady=8)
        self.status = tk.StringVar(value="")
        tk.Label(foot, textvariable=self.status, bg=c["bg"], fg=c["muted"], anchor="w").pack(side="left")
        tk.Button(foot, text="💾  Guardar y usar", command=self.guardar, bg=c["accent"], fg=c["bg"], relief="flat", bd=0,
                  cursor="hand2", padx=16, pady=6, font=("Calibri", 10, "bold")).pack(side="right")
        tk.Button(foot, text="Cancelar", command=self.destroy, bg=c["card"], fg=c["text"], relief="flat", bd=0,
                  cursor="hand2", padx=12, pady=6).pack(side="right", padx=6)
        self._msg()

    def _fondos(self):
        """Fondos ofrecidos: RGB y los indices que permiten las bandas del ortomosaico."""
        import rasterio
        from indices import INDICES, detectar_bandas, indices_disponibles
        with rasterio.open(self.ortho_path) as src:
            disp = indices_disponibles(detectar_bandas(src), self.modo_indices)
        return [("RGB", "rgb")] + [(INDICES[i][0], i) for i in disp]

    def _msg(self, extra=""):
        n = len(self.polys)
        base = (f"Clic izquierdo: vértice | clic derecho o Enter: cerrar polígono | Retroceso: deshacer | "
                f"Ctrl+rueda: zoom.   Lotes: {n}" + (f" | vértices en curso: {len(self.cur)}" if self.cur else ""))
        self.status.set(base + (("   " + extra) if extra else ""))

    # ── fondo ──────────────────────────────────────────────────────────────
    def _cargar_fondo(self):
        try:
            self.prev = lotes_io.vista_previa(self.ortho_path, self.modo.get())
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar la vista previa:\n{e}", parent=self)
            self.after(10, self.destroy)
            raise

    def _cambiar_fondo(self):
        try:
            self.prev = lotes_io.vista_previa(self.ortho_path, self.modo.get())
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            self.modo.set("rgb")
            return
        self._redraw()

    def _cargar_existentes(self):
        p = self.existing_path
        if p and os.path.exists(p):
            try:
                nombres, polys = lotes_io.cargar_poligonos_px(p, self.prev["transform"], self.prev["crs"])
                self.polys = [dict(nombre=n, pts=pts) for n, pts in zip(nombres, polys)]
            except Exception:
                self.polys = []

    # ── zoom / dibujo ─────────────────────────────────────────────────────
    def _fit(self):
        self.update_idletasks()
        w, h = self.prev["size"]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 50 or ch < 50:                      # aun sin tamano real (ventana no mostrada)
            cw, ch = 1000, 650
        self._set_zoom(min(cw / w, ch / h, 1.0))

    def _set_zoom(self, z):
        self.zoom = max(0.1, min(z, 6.0))
        self._redraw()

    def _redraw(self):
        img = self.prev["img"]
        w, h = self.prev["size"]
        W, H = max(1, int(w * self.zoom)), max(1, int(h * self.zoom))
        show = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR)
        ok, png = cv2.imencode(".png", cv2.cvtColor(show, cv2.COLOR_RGB2BGR))
        self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        cv = self.canvas
        cv.delete("all")
        cv.create_image(0, 0, image=self._photo, anchor="nw")
        cv.configure(scrollregion=(0, 0, W, H))
        z = self.zoom
        for i, p in enumerate(self.polys):
            col = _COLORES[i % len(_COLORES)]
            flat = [v * z for pt in p["pts"] for v in pt]
            cv.create_polygon(*flat, outline=col, fill="", width=2)
            cx = sum(pt[0] for pt in p["pts"]) / len(p["pts"]) * z
            cy = sum(pt[1] for pt in p["pts"]) / len(p["pts"]) * z
            cv.create_text(cx, cy, text=p["nombre"], fill=col, font=("Calibri", 11, "bold"))
        if self.cur:
            col = "#ffffff"
            for a, b in zip(self.cur, self.cur[1:]):
                cv.create_line(a[0] * z, a[1] * z, b[0] * z, b[1] * z, fill=col, width=2)
            for x, y in self.cur:
                cv.create_oval(x * z - 3, y * z - 3, x * z + 3, y * z + 3, fill=col, outline="#000")
        self._msg()

    # ── interaccion ───────────────────────────────────────────────────────
    def _on_click(self, e):
        self.agregar_vertice(self.canvas.canvasx(e.x) / self.zoom, self.canvas.canvasy(e.y) / self.zoom)

    def _on_motion(self, e):
        if self.prev is None:
            return
        col, row = self.canvas.canvasx(e.x) / self.zoom, self.canvas.canvasy(e.y) / self.zoom
        x, y = self.prev["transform"] * (col, row)
        self._msg(f"[{x:,.1f}, {y:,.1f}]")

    def agregar_vertice(self, col, fila):
        w, h = self.prev["size"]
        self.cur.append((min(max(col, 0), w - 1), min(max(fila, 0), h - 1)))
        self._redraw()

    def deshacer_vertice(self):
        if self.cur:
            self.cur.pop()
            self._redraw()

    def descartar_actual(self):
        self.cur = []
        self._redraw()

    def borrar_ultimo(self):
        if self.polys:
            self.polys.pop()
            self._redraw()

    def _pedir_nombre(self, defecto):
        return simpledialog.askstring("Nombre del lote", "Nombre del lote:", initialvalue=defecto, parent=self)

    def cerrar_poligono(self):
        if len(self.cur) < 3:
            self.bell()
            self._msg("(hacen falta al menos 3 vértices)")
            return False
        nombre = self._pedir_nombre(f"lote_{len(self.polys) + 1}")
        if nombre is None:
            return False
        nombre = nombre.strip() or f"lote_{len(self.polys) + 1}"
        self.polys.append(dict(nombre=nombre, pts=list(self.cur)))
        self.cur = []
        self._redraw()
        return True

    def guardar(self):
        if len(self.cur) >= 3 and not self.cerrar_poligono():
            return
        if not self.polys:
            messagebox.showinfo("Sin lotes", "Dibujá al menos un lote antes de guardar.", parent=self)
            return
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        if os.path.exists(self.out_path):
            os.remove(self.out_path)
        lotes_io.guardar_poligonos([p["pts"] for p in self.polys], [p["nombre"] for p in self.polys],
                                   self.prev["transform"], self.prev["crs"], self.out_path)
        if self.on_saved:
            self.on_saved(self.out_path)
        self.destroy()
