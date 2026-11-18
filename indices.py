"""Lectura de ortomosaicos e indices de vegetacion.

- Multiespectral (bandas NIR/RedEdge, p. ej. WebODM con MicaSense RedEdge-M): NDVI y NDRE.
- RGB comun (sin NIR): ExG, GLI y VARI, calculados sobre coordenadas cromaticas normalizadas.

Los dos grupos se mantienen separados: un ortomosaico multiespectral ofrece NDVI/NDRE y uno RGB
ofrece los indices RGB.

- detectar_bandas(): identifica las bandas por su descripcion (Red, Green, Blue, NIR, Rededge,
  alpha), como las escribe ODM/WebODM, sin asumir el orden.
- calcular_indices(): escribe un GeoTIFF float32 por indice y un PNG de vista previa.

Los valores son RELATIVOS: sirven para comparar zonas dentro de un mismo vuelo, no entre vuelos o
fechas. Los indices RGB ademas dependen de la camara, la hora y las sombras.
"""
import os
import numpy as np

AVISO_CALIBRACION = (
    "AVISO: NDVI/NDRE calculados sobre los valores del ortomosaico tal como vienen. Si WebODM se corrio "
    "sin calibracion radiometrica (radiometric-calibration=none) los valores son RELATIVOS: sirven para "
    "comparar zonas dentro de este vuelo, no contra otros vuelos o fechas ni contra valores de la "
    "bibliografia. El archivo no permite verificar como se calibro. Estos indices no estan validados "
    "con datos de campo.")

AVISO_RGB = (
    "AVISO: indices RGB (ExG, GLI, VARI) calculados sobre un ortomosaico de 3 bandas. Son RELATIVOS y "
    "mas sensibles que NDVI/NDRE a la camara, la hora del dia, las sombras y el estiramiento de brillo: "
    "sirven para comparar zonas dentro de este vuelo, no entre vuelos o fechas. No distinguen bien "
    "vegetacion verde de otras superficies verdes ni vegetacion senescente. No estan validados con "
    "datos de campo.")

# nombre -> (titulo, bandas necesarias)
INDICES = {
    "ndvi": ("NDVI", ("nir", "red")),
    "ndre": ("NDRE", ("nir", "rededge")),
    "exg": ("ExG", ("red", "green", "blue")),
    "gli": ("GLI", ("red", "green", "blue")),
    "vari": ("VARI", ("red", "green", "blue")),
}

_ALIAS = {
    "red": "red", "r": "red",
    "green": "green", "g": "green",
    "blue": "blue", "b": "blue",
    "nir": "nir", "nearinfrared": "nir", "near infrared": "nir",
    "rededge": "rededge", "red edge": "rededge", "red_edge": "rededge", "re": "rededge",
    "alpha": "alpha",
}


# ── asignacion manual de bandas (ortomosaicos sin nombres de banda) ──────────────────────────
ROLES = ("red", "green", "blue", "nir", "rededge", "alpha")
ROLES_ES = {"red": "Rojo", "green": "Verde", "blue": "Azul", "nir": "NIR", "rededge": "RedEdge", "alpha": "Alpha (transparencia)"}

# sugerencias para ortomosaicos sin nombres de banda: nombre -> {rol: banda}
PRESETS = {
    "Azul, Verde, Rojo, RedEdge, NIR (orden por longitud de onda)": dict(blue=1, green=2, red=3, rededge=4, nir=5),
    "Azul, Verde, Rojo, NIR, RedEdge (orden MicaSense)": dict(blue=1, green=2, red=3, nir=4, rededge=5),
    "Rojo, Verde, Azul, NIR": dict(red=1, green=2, blue=3, nir=4),
    "Rojo, Verde, Azul": dict(red=1, green=2, blue=3),
}

_ASIGNADAS = {}          # clave de archivo -> {rol: banda} (sesion actual)
_CONFIG_CARGADA = False


def _clave(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _config_path():
    base = os.environ.get("AGROIA_CONFIG_DIR") or os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                                                                 "AgroIA-COVIO")
    return os.path.join(base, "bandas.json")


def _leer_config():
    import json
    try:
        with open(_config_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def registrar_bandas(path, mapeo, guardar=True):
    """Fija que banda es cada rol ({'red': 3, 'nir': 5, ...}) para un ortomosaico y, si `guardar`, lo recuerda
    entre sesiones (en la carpeta de configuracion del usuario, no junto a los datos)."""
    import json
    mapeo = {k: int(v) for k, v in mapeo.items() if k in ROLES}
    if len(set(mapeo.values())) != len(mapeo):
        raise ValueError("Una misma banda no puede tener dos roles.")
    _ASIGNADAS[_clave(path)] = mapeo
    if guardar:
        cfg = _leer_config()
        try:
            cfg[_clave(path)] = dict(size=os.path.getsize(path), bandas=mapeo)
            os.makedirs(os.path.dirname(_config_path()), exist_ok=True)
            with open(_config_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1)
        except OSError:
            pass          # no poder guardar no impide usar la asignacion en esta sesion
    return mapeo


def bandas_registradas(path):
    """Asignacion guardada para este archivo (sesion o configuracion del usuario), o None."""
    k = _clave(path)
    if k in _ASIGNADAS:
        return dict(_ASIGNADAS[k])
    reg = _leer_config().get(k)
    try:
        if reg and int(reg.get("size", -1)) == os.path.getsize(path):      # si el archivo cambio, se descarta
            _ASIGNADAS[k] = {r: int(b) for r, b in reg["bandas"].items() if r in ROLES}
            return dict(_ASIGNADAS[k])
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def olvidar_bandas(path):
    import json
    _ASIGNADAS.pop(_clave(path), None)
    cfg = _leer_config()
    if cfg.pop(_clave(path), None) is not None:
        try:
            with open(_config_path(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=1)
        except OSError:
            pass


def _detectar_auto(src):
    """Deteccion automatica: por descripcion de banda (como las escribe ODM/WebODM) y colorinterp.
    Sin descripciones: 3 bandas = RGB, 4 = RGB+alpha; con mas bandas no se asume nada."""
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


def detectar_bandas(src):
    """Devuelve {'red': 1, 'green': 2, ...} (indices 1-based). Usa primero la asignacion manual guardada para
    ese archivo (registrar_bandas); si no hay, la deteccion automatica."""
    reg = bandas_registradas(src.name) if getattr(src, "name", None) else None
    return reg if reg else _detectar_auto(src)


def necesita_asignacion(src):
    """True si el ortomosaico tiene bandas sin nombre que no se pueden interpretar solas (mas de 4 bandas, o 4 sin
    alpha) y todavia no hay una asignacion guardada: hay que preguntarle al usuario cual es cada banda."""
    if getattr(src, "name", None) and bandas_registradas(src.name):
        return False
    nombradas = _detectar_auto(src)
    if any(k in nombradas for k in ("red", "green", "blue", "nir", "rededge")) and \
            any(d for d in src.descriptions):
        return False                                      # tiene nombres de banda reconocibles
    tiene_alpha = any(ci.name == "alpha" for ci in src.colorinterp)
    return src.count >= 5 or (src.count == 4 and not tiene_alpha)


def sugerir_asignacion(path, max_px=900):
    """(nombre_preset, mapeo, medias): propone una asignacion para un ortomosaico sin nombres de banda mirando
    el valor medio de cada banda (en vegetacion el NIR es la banda mas alta y el rojo es bajo)."""
    import rasterio
    with rasterio.open(path) as s:
        k = max(1.0, max(s.width, s.height) / max_px)
        d = s.read(out_shape=(s.count, max(1, int(s.height / k)), max(1, int(s.width / k)))).astype("float64")
        nd = s.nodata
        n = s.count
    ok = np.ones(d.shape[1:], bool)
    for i in range(n):
        ok &= d[i] > 0
        if nd is not None:
            ok &= d[i] != nd
    medias = [float(d[i][ok].mean()) if ok.any() else 0.0 for i in range(n)]
    nombres = list(PRESETS)
    if n == 5:
        # orden por longitud de onda si el NIR (banda 5) supera al borde rojo (banda 4); si no, orden MicaSense
        elegido = nombres[0] if medias[4] >= medias[3] else nombres[1]
    elif n == 4:
        elegido = nombres[2]
    else:
        elegido = nombres[3] if n == 3 else None
    return elegido, (dict(PRESETS[elegido]) if elegido else {}), medias


def bandas_rgb(src):
    """Indices (R, G, B) a leer para el pipeline de conteo. Cae a (1, 2, 3) si no se pueden identificar."""
    b = detectar_bandas(src)
    if all(k in b for k in ("red", "green", "blue")):
        return (b["red"], b["green"], b["blue"])
    return (1, 2, 3)


MODOS = ("multi", "rgb")


def indices_disponibles(bandas, modo=None):
    """Indices que se pueden calcular con las bandas dadas (dict de detectar_bandas).

    modo='multi': NDVI (y NDRE si hay RedEdge) -> exige NIR y Red.
    modo='rgb':   ExG, GLI y VARI -> exige R, G y B (sirve tambien en un multiespectral, usando sus bandas visibles).
    modo=None:    automatico: multiespectral si hay NIR + Red, si no RGB. Los dos grupos nunca se mezclan."""
    multi = ["ndvi"] + (["ndre"] if "rededge" in bandas else []) if ("nir" in bandas and "red" in bandas) else []
    rgb = ["exg", "gli", "vari"] if all(k in bandas for k in ("red", "green", "blue")) else []
    if modo == "multi":
        return multi
    if modo == "rgb":
        return rgb
    return multi or rgb


def roles(disponibles):
    """(indice_vegetacion, indice_vigor) para el analisis por lote: el primero separa vegetacion de
    suelo (umbral automatico) y el segundo zonifica el vigor relativo."""
    if "ndvi" in disponibles:
        return "ndvi", ("ndre" if "ndre" in disponibles else "ndvi")
    if "exg" in disponibles:
        return "exg", "gli"
    raise ValueError("No hay indices disponibles para analizar.")


def _sin_bandas(modo, descripciones=None):
    if modo == "multi":
        return ("Este ortomosaico no tiene las bandas NIR y Red (por nombre), necesarias para NDVI/NDRE. "
                "Si es un ortomosaico RGB, usa los indices RGB.")
    if modo == "rgb":
        return "No se encontraron las bandas R, G y B (3 bandas o por nombre) para los indices RGB."
    return ("No se pudieron identificar las bandas: hace falta RGB (3 bandas) o multiespectral con "
            f"Red y NIR por nombre (descripciones: {descripciones}).")


def indices_para_lotes(ortho_path, modo=None):
    """(indice_vegetacion, indice_vigor) que usa el analisis por lote para este ortomosaico y modo."""
    import rasterio
    with rasterio.open(ortho_path) as src:
        disp = indices_disponibles(detectar_bandas(src), modo)
    if not disp:
        raise ValueError(_sin_bandas(modo))
    return roles(disp)


def aviso_para(indices):
    return AVISO_CALIBRACION if any(i in ("ndvi", "ndre") for i in indices) else AVISO_RGB


def formula(nombre, B, valido):
    """Calcula un indice a partir de {'red': arr, ...} (float32). Devuelve float32 con NaN fuera de `valido`."""
    out = np.full(next(iter(B.values())).shape, np.nan, dtype="float32")
    if nombre in ("ndvi", "ndre"):
        a, b = B["nir"], B["red" if nombre == "ndvi" else "rededge"]
        s = a + b
        ok = valido & (s > 0)
        out[ok] = (a[ok] - b[ok]) / s[ok]
        return out
    R, G, Bl = B["red"], B["green"], B["blue"]
    s = R + G + Bl
    ok = valido & (s > 0)
    if nombre == "exg":                       # 2g - r - b sobre coordenadas cromaticas
        out[ok] = (2 * G[ok] - R[ok] - Bl[ok]) / s[ok]
    elif nombre == "gli":                     # (2G - R - B) / (2G + R + B)
        d = 2 * G + R + Bl
        ok &= d > 0
        out[ok] = (2 * G[ok] - R[ok] - Bl[ok]) / d[ok]
    elif nombre == "vari":                    # (G - R) / (G + R - B); inestable si el denominador ~ 0
        d = G + R - Bl
        ok &= np.abs(d) > 0.05 * s
        out[ok] = np.clip((G[ok] - R[ok]) / d[ok], -1, 1)
    else:
        raise ValueError(f"Indice desconocido: {nombre}")
    return out


def mascara_valida(arrs, alpha):
    """Pixeles con dato: alpha > 0 si hay banda alpha; si no, cualquier banda distinta de 0."""
    if alpha is not None:
        return alpha > 0
    v = np.zeros(next(iter(arrs.values())).shape, bool)
    for a in arrs.values():
        v |= a > 0
    return v


def calcular_indices(ortho_path, out_dir, log=print, block=1024, indices=None, modo=None):
    """Calcula los indices (por defecto los que corresponden a las bandas del raster) por bloques.
    Devuelve dict(rutas, stats, png, aviso, indices); lanza ValueError si no hay bandas suficientes."""
    import rasterio
    from rasterio.windows import Window

    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(ortho_path) as src:
        b = detectar_bandas(src)
        disp = indices_disponibles(b, modo)
        if not disp:
            raise ValueError(_sin_bandas(modo, src.descriptions))
        indices = list(indices) if indices else disp
        mal = [i for i in indices if i not in disp]
        if mal:
            raise ValueError(f"Indices no disponibles para este raster: {mal} (disponibles: {disp}).")
        if "ndvi" in disp and "ndre" not in disp:
            log("  Sin banda RedEdge: solo se calcula NDVI.")
        necesarias = sorted({k for i in indices for k in INDICES[i][1]})
        prof = src.profile.copy()
        prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512)
        prof.pop("photometric", None)
        prof.pop("alpha", None)
        rutas = {i: os.path.join(out_dir, f"{i}.tif") for i in indices}
        dst = {k: rasterio.open(p, "w", **prof) for k, p in rutas.items()}
        sm = {k: [] for k in rutas}
        try:
            nblk = (src.height + block - 1) // block
            for bi, y in enumerate(range(0, src.height, block)):
                if nblk > 6 and bi % max(1, nblk // 5) == 0 and bi:
                    log(f"  ... {int(bi / nblk * 100)} % del ortomosaico")
                w = Window(0, y, src.width, min(block, src.height - y))
                B = {k: src.read(b[k], window=w).astype("float32") for k in necesarias}
                alpha = src.read(b["alpha"], window=w) if "alpha" in b else None
                valido = mascara_valida(B, alpha)
                for i in indices:
                    v = formula(i, B, valido)
                    dst[i].write(v, 1, window=w)
                    f = v[np.isfinite(v)]
                    if f.size:
                        sm[i].append(f[:: max(1, f.size // 20000)])
        finally:
            for d in dst.values():
                d.close()
    stats = {}
    for k, l in sm.items():
        if l:
            a = np.concatenate(l)
            stats[k] = dict(media=float(a.mean()), p2=float(np.percentile(a, 2)), p98=float(np.percentile(a, 98)))
    with open(os.path.join(out_dir, "indices_stats.csv"), "w", encoding="utf-8") as f:
        f.write("indice,media,p2,p98" + chr(10))
        for k, d in stats.items():
            f.write(f"{k},{d['media']:.4f},{d['p2']:.4f},{d['p98']:.4f}" + chr(10))
    png = os.path.join(out_dir, "indices.png")
    _vista_previa(rutas, png, log)
    aviso = aviso_para(indices)
    log("  Indices listos: " + ", ".join(os.path.basename(p) for p in rutas.values()))
    log("  " + aviso)
    return dict(rutas=rutas, stats=stats, png=png, aviso=aviso, indices=indices)


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
        a.set_title(INDICES[k][0] + " (valores relativos)")
        a.axis("off")
        fig.colorbar(im, ax=a, shrink=0.7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(0.5, 0.012, "Relativo: comparable solo dentro de este vuelo; sin calibracion verificada ni validacion de campo.",
             ha="center", fontsize=9, color="#444")
    fig.savefig(png, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("Uso: python indices.py ORTOMOSAICO.tif CARPETA_SALIDA")
    r = calcular_indices(sys.argv[1], sys.argv[2])
    for k, v in r["stats"].items():
        print(k, {a: round(b, 3) for a, b in v.items()})
