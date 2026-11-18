"""Vista previa de un ortomosaico y lectura/escritura de poligonos de lote.

Es la parte sin interfaz grafica del editor de lotes (editor_lotes.py), para poder probarla aparte.
Los poligonos se manejan en pixeles de la vista previa (col, fila) y se pasan a coordenadas del mapa
con la transformada de la vista previa.
"""
import numpy as np


def vista_previa(ortho_path, modo="rgb", max_px=1600):
    """Imagen reducida (uint8, HxWx3) del ortomosaico para dibujar sobre ella.

    modo: 'rgb' (bandas R,G,B con estiramiento 2-98) | 'ndvi' | 'ndre' (mapa de color).
    Devuelve dict(img, transform, crs, size): `transform` lleva pixeles de la vista previa a
    coordenadas del mapa (CRS del ortomosaico).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from indices import detectar_bandas, bandas_rgb

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
            falta = [n for n in ("red", "nir") + (("rededge",) if modo == "ndre" else ()) if n not in b]
            if falta:
                raise ValueError(f"Faltan bandas para {modo.upper()}: {falta}")
            N = leer(b["nir"])
            D = leer(b["red" if modo == "ndvi" else "rededge"])
            if valido is None:
                valido = (N > 0) | (D > 0)
            s = N + D
            ok = valido & (s > 0)
            idx = np.full(N.shape, np.nan, "float32")
            idx[ok] = (N[ok] - D[ok]) / s[ok]
            f = idx[np.isfinite(idx)]
            lo, hi = (np.percentile(f, (2, 98)) if f.size else (0.0, 1.0))
            if hi <= lo:
                hi = lo + 1e-6
            import matplotlib
            cmap = matplotlib.colormaps["RdYlGn"]
            norm = np.clip((np.nan_to_num(idx, nan=lo) - lo) / (hi - lo), 0, 1)
            img = (cmap(norm)[..., :3] * 255).astype("uint8")
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
