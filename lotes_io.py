"""Vista previa de un ortomosaico y lectura/escritura de poligonos de lote.

Es la parte sin interfaz grafica del editor de lotes (editor_lotes.py), para poder probarla aparte.
Los poligonos se manejan en pixeles de la vista previa (col, fila) y se pasan a coordenadas del mapa
con la transformada de la vista previa.
"""
import numpy as np


def vista_previa(ortho_path, modo="rgb", max_px=1600):
    """Imagen reducida (uint8, HxWx3) del ortomosaico para dibujar sobre ella.

    modo: 'rgb' (bandas R,G,B con estiramiento 2-98) o el nombre de un indice de indices.INDICES
    (ndvi, ndre, exg, gli, vari; mapa de color).
    Devuelve dict(img, transform, crs, size): `transform` lleva pixeles de la vista previa a
    coordenadas del mapa (CRS del ortomosaico).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from indices import INDICES, detectar_bandas, bandas_rgb, formula, mascara_valida

    with rasterio.open(ortho_path) as src:
        k = max(1.0, max(src.width, src.height) / max_px)
        ow, oh = max(1, int(src.width / k)), max(1, int(src.height / k))

        def leer(i):
            return src.read(i, out_shape=(oh, ow), resampling=Resampling.average).astype("float32")

        b = detectar_bandas(src)
        if "alpha" in b:
            valido = src.read(b["alpha"], out_shape=(oh, ow), resampling=Resampling.nearest) > 0
        else:
            valido = None

        if modo == "rgb":
            rgb = np.dstack([leer(i) for i in bandas_rgb(src)])
            if valido is None:
                valido = rgb.any(axis=2)
            if rgb.max() <= 255 and src.dtypes[0] == "uint8":
                img = rgb.astype("uint8")
            else:
                v = rgb[valido] if valido.any() else rgb.reshape(-1, 3)
                lo, hi = np.percentile(v, (2, 98))
                if hi <= lo:
                    hi = lo + 1
                img = np.clip((rgb - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
        else:
            if modo not in INDICES:
                raise ValueError(f"Indice desconocido: {modo}")
            falta = [n for n in INDICES[modo][1] if n not in b]
            if falta:
                raise ValueError(f"Faltan bandas para {modo.upper()}: {falta}")
            B = {n: leer(b[n]) for n in INDICES[modo][1]}
            if valido is None:
                valido = mascara_valida(B, None)
            idx = formula(modo, B, valido)
            f = idx[np.isfinite(idx)]
            lo, hi = (np.percentile(f, (2, 98)) if f.size else (0.0, 1.0))
            if hi <= lo:
                hi = lo + 1e-6
            import matplotlib
            cmap = matplotlib.colormaps["RdYlGn"]
            norm = np.clip((np.nan_to_num(idx, nan=lo) - lo) / (hi - lo), 0, 1)
            img = (cmap(norm)[..., :3] * 255).astype("uint8")
            valido = valido & np.isfinite(idx)
        img[~valido] = 30   # sin datos: gris oscuro
        tr = src.transform * Affine.scale(src.width / ow, src.height / oh)
        return dict(img=img, transform=tr, crs=src.crs, size=(ow, oh))


def guardar_poligonos(polys_px, nombres, transform, crs, path):
    """Escribe los poligonos (listas de (col, fila) de la vista previa) como GeoPackage, capa 'lotes'."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    geoms = []
    for pts in polys_px:
        if len(pts) < 3:
            raise ValueError("Un poligono necesita al menos 3 vertices.")
        geoms.append(Polygon([transform * (c, r) for c, r in pts]))
    gdf = gpd.GeoDataFrame({"nombre": list(nombres)}, geometry=geoms, crs=crs)
    gdf.to_file(path, layer="lotes", driver="GPKG")
    return path


def cargar_poligonos_px(path, transform, crs):
    """Lee un archivo de poligonos y devuelve (nombres, [lista de (col, fila)]) en pixeles de la vista previa."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs(crs)
    namecol = next((c for c in ("nombre", "name", "Nombre", "NOMBRE") if c in gdf.columns), None)
    inv = ~transform
    nombres, polys = [], []
    for i, (_, r) in enumerate(gdf.iterrows()):
        g = r.geometry
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda p: p.area)
        if g.geom_type != "Polygon":
            continue
        nombres.append(str(r[namecol]) if namecol else f"lote_{i + 1}")
        polys.append([tuple(inv * (x, y)) for x, y in list(g.exterior.coords)[:-1]])
    return nombres, polys


def colorear(valores, lo, hi, cmap="RdYlGn"):
    """Valores float (NaN = sin dato) -> imagen RGB uint8 con el mapa de color; sin dato = gris oscuro."""
    import matplotlib
    cm = matplotlib.colormaps[cmap]
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((np.nan_to_num(valores, nan=lo) - lo) / (hi - lo), 0, 1)
    img = (cm(norm)[..., :3] * 255).astype("uint8")
    img[~np.isfinite(valores)] = 30
    return img


def vista_raster(tif_path, bounds=None, margin=15.0, max_px=1400, cmap="RdYlGn"):
    """Vista reducida y coloreada de un raster de un indice (ndvi.tif, ...), recortada a `bounds`
    (minx, miny, maxx, maxy) mas un margen en metros. Devuelve dict(img, transform, crs, size, vmin, vmax,
    valores): `transform` lleva pixeles de la vista a coordenadas del mapa."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.windows import Window, from_bounds

    with rasterio.open(tif_path) as s:
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            win = from_bounds(x0 - margin, y0 - margin, x1 + margin, y1 + margin, s.transform)
            win = win.round_offsets().round_lengths()
        else:
            win = Window(0, 0, s.width, s.height)
        k = max(1.0, max(win.width, win.height) / max_px)
        ow, oh = max(1, int(win.width / k)), max(1, int(win.height / k))
        v = s.read(1, window=win, out_shape=(oh, ow), boundless=True, fill_value=np.nan,
                   resampling=Resampling.average).astype("float32")
        tr = s.window_transform(win) * Affine.scale(win.width / ow, win.height / oh)
        crs = s.crs
    f = v[np.isfinite(v)]
    lo, hi = (np.percentile(f, (2, 98)) if f.size else (0.0, 1.0))
    return dict(img=colorear(v, lo, hi, cmap), transform=tr, crs=crs, size=(ow, oh),
                vmin=float(lo), vmax=float(hi), valores=v)
