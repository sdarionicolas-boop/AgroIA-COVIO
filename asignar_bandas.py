"""Ventana para indicar que es cada banda de un ortomosaico que no trae nombres de banda (Tkinter).

Muchos ortomosaicos multiespectrales (p. ej. exportados por Pix4D o Metashape) son GeoTIFF de 5 bandas sin
descripcion: el programa no puede saber cual es el rojo y cual el NIR. Esta ventana propone una asignacion a
partir de los valores medios de cada banda y deja que el usuario la corrija. La asignacion se recuerda para
ese archivo (indices.registrar_bandas).
"""
import tkinter as tk
from tkinter import ttk, messagebox

import indices

IGNORAR = "(ignorar)"


class AsignarBandas(tk.Toplevel):
    def __init__(self, parent, path, colors=None):
        super().__init__(parent)
        c = dict(bg="#0F1923", card="#1A2535", text="#E8EDF2", muted="#7A8FA6", accent="#4CAF82", accent2="#2E86AB",
                 warn="#F4A261")
        c.update(colors or {})
        self.c = c
        self.path = path
        self.result = None
        self.title("Asignar bandas del ortomosaico")
        self.configure(bg=c["bg"])
        self.transient(parent)

        import rasterio
        with rasterio.open(path) as s:
            self.n = s.count
            self.nombre_archivo = path.replace("\\", "/").split("/")[-1]
        try:
            self.preset, self.sugerido, self.medias = indices.sugerir_asignacion(path)
        except Exception:
            self.preset, self.sugerido, self.medias = None, {}, [0.0] * self.n

        tk.Label(self, text="Asignar bandas", font=("Trebuchet MS", 14, "bold"), bg=c["bg"], fg=c["accent"]
                 ).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(self, text=(f"«{self.nombre_archivo}» tiene {self.n} bandas sin nombre, y el programa no puede saber "
                             "cuál es cada una. Indicalo acá (se recuerda para este archivo). "
                             "La sugerencia sale de los valores medios: en vegetación el NIR es la banda más alta y el "
                             "rojo es bajo. Si no estás seguro, confirmalo con quien procesó el ortomosaico."),
                 font=("Calibri", 10), bg=c["bg"], fg=c["muted"], wraplength=560, justify="left"
                 ).pack(anchor="w", padx=16, pady=(0, 8))

        fila = tk.Frame(self, bg=c["bg"])
        fila.pack(fill="x", padx=16, pady=(0, 6))
        tk.Label(fila, text="Sugerencias:", bg=c["bg"], fg=c["text"], font=("Calibri", 10, "bold")).pack(side="left")
        self.var_preset = tk.StringVar(value=self.preset or "")
        cb = ttk.Combobox(fila, textvariable=self.var_preset, values=list(indices.PRESETS), state="readonly", width=58)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.aplicar_preset(self.var_preset.get()))

        tabla = tk.Frame(self, bg=c["card"])
        tabla.pack(fill="x", padx=16, pady=4)
        for j, t in enumerate(("Banda", "Valor medio", "Es…")):
            tk.Label(tabla, text=t, bg=c["card"], fg=c["accent"], font=("Calibri", 10, "bold"), anchor="w"
                     ).grid(row=0, column=j, sticky="w", padx=10, pady=(6, 2))
        self.vars = []
        etiquetas = [IGNORAR] + [indices.ROLES_ES[r] for r in indices.ROLES]
        for i in range(self.n):
            tk.Label(tabla, text=f"Banda {i + 1}", bg=c["card"], fg=c["text"]).grid(row=i + 1, column=0, sticky="w", padx=10, pady=2)
            tk.Label(tabla, text=f"{self.medias[i]:,.0f}", bg=c["card"], fg=c["muted"]).grid(row=i + 1, column=1, sticky="w", padx=10)
            v = tk.StringVar(value=IGNORAR)
            ttk.Combobox(tabla, textvariable=v, values=etiquetas, state="readonly", width=22
                         ).grid(row=i + 1, column=2, sticky="w", padx=10, pady=2)
            self.vars.append(v)
        tk.Frame(tabla, height=6, bg=c["card"]).grid(row=self.n + 1, column=0)

        self.set_roles(self.sugerido)

        pie = tk.Frame(self, bg=c["bg"])
        pie.pack(fill="x", padx=16, pady=12)
        tk.Button(pie, text="Aceptar", command=self.aceptar, bg=c["accent"], fg=c["bg"], relief="flat", bd=0,
                  cursor="hand2", padx=18, pady=6, font=("Calibri", 10, "bold")).pack(side="right")
        tk.Button(pie, text="Cancelar", command=self.destroy, bg=c["card"], fg=c["text"], relief="flat", bd=0,
                  cursor="hand2", padx=12, pady=6).pack(side="right", padx=6)
        self.update_idletasks()
        try:
            self.grab_set()
        except tk.TclError:
            pass

    # ── logica (sin depender de la ventana visible) ─────────────────────
    def set_roles(self, mapeo):
        """mapeo: {rol: banda (1-based)}. Las bandas sin rol quedan en 'ignorar'."""
        for v in self.vars:
            v.set(IGNORAR)
        for rol, b in mapeo.items():
            if 1 <= b <= self.n and rol in indices.ROLES:
                self.vars[b - 1].set(indices.ROLES_ES[rol])

    def aplicar_preset(self, nombre):
        self.set_roles(indices.PRESETS.get(nombre, {}))

    def roles(self):
        inv = {v: k for k, v in indices.ROLES_ES.items()}
        return {inv[v.get()]: i + 1 for i, v in enumerate(self.vars) if v.get() in inv}

    def aceptar(self):
        mapeo = self.roles()
        vistos = [v.get() for v in self.vars if v.get() != IGNORAR]
        if len(vistos) != len(set(vistos)):
            messagebox.showerror("Roles repetidos", "Un mismo rol está asignado a dos bandas.", parent=self)
            return None
        rgb = all(r in mapeo for r in ("red", "green", "blue"))
        multi = "red" in mapeo and "nir" in mapeo
        if not (rgb or multi):
            messagebox.showerror(
                "Faltan bandas",
                "Hace falta asignar como mínimo Rojo, Verde y Azul (índices RGB) o Rojo y NIR (NDVI).", parent=self)
            return None
        indices.registrar_bandas(self.path, mapeo)
        self.result = mapeo
        self.destroy()
        return mapeo
