"""Lectura de ortomosaicos multiespectrales e indices de vegetacion (NDVI, NDRE).

- detectar_bandas(): identifica las bandas por su descripcion (Red, Green, Blue, NIR, Rededge,
  alpha), como las escribe ODM/WebODM, sin asumir el orden.
- calcular_indices(): escribe ndvi.tif y ndre.tif (float32) y un PNG de vista previa.

Sin calibracion radiometrica (ODM con radiometric-calibration=none) los valores son RELATIVOS:
sirven para comparar zonas dentro de un mismo vuelo, no entre vuelos o fechas.
"""
import os
import numpy as np

AVISO_CALIBRACION = (
    "AVISO: NDVI/NDRE calculados sobre los valores del ortomosaico tal como vienen. Si WebODM se corrio "
    "sin calibracion radiometrica (radiometric-calibration=none) los valores son RELATIVOS: sirven para "
    "comparar zonas dentro de este vuelo, no contra otros vuelos o fechas ni contra valores de la "
    "bibliografia. El archivo no permite verificar como se calibro. Estos indices no estan validados "
    "con datos de campo.")

_ALIAS = {
    "red": "red", "r": "red",
    "green": "green", "g": "green",
    "blue": "blue", "b": "blue",
    "nir": "nir", "nearinfrared": "nir", "near infrared": "nir",
    "rededge": "rededge", "red edge": "rededge", "red_edge": "rededge", "re": "rededge",
    "alpha": "alpha",
}


def detectar_bandas(src):
    """Devuelve {'red': 1, 'green': 2, ...} (indices 1-based) segun las descripciones del raster.
    Sin descripciones: 3 bandas = RGB, 4 = RGB+alpha; con mas bandas no se asume nada (dict vacio)."""
    bandas = {}
    for i, d in enumerate(src.descriptions, start=1):
        if d:
            k = _ALIAS.get(d.strip().lower())
            if k and k not in bandas:
                bandas[k] = i
    if "alpha" not in bandas:   # ODM no nombra el alpha: solo lo marca en colorinterp
        for i, ci in enumerate(src.colorinterp, start=1):
            if ci.name == "alpha":
                bandas["alpha"] = i
                break
    if not any(k in bandas for k in ("red", "green", "blue", "nir", "rededge")):
        bandas = {k: v for k, v in bandas.items() if k == "alpha"}
        if src.count in (3, 4):
            bandas.update(red=1, green=2, blue=3)
            if src.count == 4:
                bandas.setdefault("alpha", 4)
    return bandas


def bandas_rgb(src):
    """Indices (R, G, B) a leer para el pipeline de conteo. Cae a (1, 2, 3) si no se pueden identificar."""
    b = detectar_bandas(src)
    if all(k in b for k in ("red", "green", "blue")):
        return (b["red"], b["green"], b["blue"])
    return (1, 2, 3)


def _nd(a, b, valido):
    s = a + b
    ok = valido & (s > 0)
    out = np.full(a.shape, np.nan, dtype="float32")
    out[ok] = ((a[ok] - b[ok]) / s[ok]).astype("float32")
    return out


def calcular_indices(ortho_path, out_dir, log=print, block=1024):
    """Calcula NDVI (NIR,Red) y NDRE (NIR,RedEdge). Procesa por bloques. Devuelve un dict de rutas
    y estadisticas; lanza ValueError si faltan las bandas necesarias."""
    import rasterio
    from rasterio.windows import Window

    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(ortho_path) as src:
        b = detectar_bandas(src)
        faltan = [k for k in ("red", "nir") if k not in b]
        if faltan:
            raise ValueError(
                "El raster no parece multiespectral: no se encontraron las bandas "
                f"{faltan} (descripciones: {src.descriptions}).")
        con_re = "rededge" in b
        if not con_re:
            log("  Sin banda RedEdge: solo se calcula NDVI.")
        prof = src.profile.copy()
        prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512)
        prof.pop("photometric", None)
        prof.pop("alpha", None)
        rutas = {"ndvi": os.path.join(out_dir, "ndvi.tif")}
        if con_re:
            rutas["ndre"] = os.path.join(out_dir, "ndre.tif")
        dst = {k: rasterio.open(p, "w", **prof) for k, p in rutas.items()}
        sm = {k: [] for k in rutas}
        try:
            for y in range(0, src.height, block):
                w = Window(0, y, src.width, min(block, src.height - y))
                R = src.read(b["red"], window=w).astype("float32")
                N = src.read(b["nir"], window=w).astype("float32")
                if "alpha" in b:
                    valido = src.read(b["alpha"], window=w) > 0
                else:
                    valido = (R > 0) | (N > 0)
                res = {"ndvi": _nd(N, R, valido)}
                if con_re:
                    E = src.read(b["rededge"], window=w).astype("float32")
                    res["ndre"] = _nd(N, E, valido)
                for k, v in res.items():
                    dst[k].write(v, 1, window=w)
                    f = v[np.isfinite(v)]
                    if f.size:
                        sm[k].append(f[:: max(1, f.size // 20000)])
        finally:
            for d in dst.values():
                d.close()
    stats = {}
    for k, l in sm.items():
        if l:
            a = np.concatenate(l)
            stats[k] = dict(media=float(a.mean()), p2=float(np.percentile(a, 2)), p98=float(np.percentile(a, 98)))
    _vista_previa(rutas, os.path.join(out_dir, "indices.png"), log)
    log("  Indices listos: " + ", ".join(os.path.basename(p) for p in rutas.values()))
    log("  " + AVISO_CALIBRACION)
    return dict(rutas=rutas, stats=stats, png=os.path.join(out_dir, "indices.png"), aviso=AVISO_CALIBRACION)


def _vista_previa(rutas, png, log):
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, len(rutas), figsize=(8 * len(rutas), 7), squeeze=False)
    for a, (k, p) in zip(ax[0], rutas.items()):
        with rasterio.open(p) as s:
            k_ = max(1, max(s.width, s.height) // 1500)
            v = s.read(1, out_shape=(max(1, s.height // k_), max(1, s.width // k_)))
        f = v[np.isfinite(v)]
        im = a.imshow(v, cmap="RdYlGn", vmin=np.percentile(f, 2), vmax=np.percentile(f, 98))
        a.set_title(k.upper() + " (valores relativos)"); a.axis("off")
        fig.colorbar(im, ax=a, shrink=0.7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.5, 0.012, "Relativo: comparable solo dentro de este vuelo; sin calibracion verificada ni validacion de campo.",
             ha="center", fontsize=9, color="#444")
    fig.savefig(png, dpi=100); plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("Uso: python indices.py ORTOMOSAICO.tif CARPETA_SALIDA")
    r = calcular_indices(sys.argv[1], sys.argv[2])
    for k, v in r["stats"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
