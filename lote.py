"""Estadisticas, fallas y zonificacion de vigor dentro de poligonos de lote que dibuja el usuario.

Uso:
    python lote.py ORTOMOSAICO.tif POLIGONOS.gpkg|shp CARPETA_SALIDA [--capa NOMBRE]
                   [--grid 10] [--umbral-ndvi X] [--min-falla 2]

Por cada poligono: NDVI/NDRE (media, desvio, p10/p90), cobertura vegetal, fallas (manchas de suelo o
vegetacion muy baja >= --min-falla m2) y una grilla (--grid m) con el vigor relativo de cada celda
(terciles de NDRE dentro del lote: baja/media/alta).

Los valores son RELATIVOS si el ortomosaico no tiene calibracion radiometrica: sirven para comparar
zonas dentro del mismo vuelo, no entre fechas. El umbral de vegetacion por defecto es automatico (Otsu
sobre todo el mosaico); fijalo con --umbral-ndvi si el resultado no te convence.
"""
import argparse
import os
import numpy as np


def _otsu(v, bins=256):
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def _umbral_global(ndvi_path):
    import rasterio
    with rasterio.open(ndvi_path) as s:
        k = max(1, max(s.width, s.height) // 2000)
        v = s.read(1, out_shape=(max(1, s.height // k), max(1, s.width // k)))
    v = v[np.isfinite(v)]
    return _otsu(v) if v.size else 0.3


def analizar_lotes(ortho_path, poligonos_path, out_dir, capa=None, grid_m=10.0,
                   umbral_ndvi=None, min_falla_m2=2.0, log=print):
    """Analiza cada poligono del archivo. Devuelve dict con resumen (DataFrame), gpkg y png."""
    import cv2
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask, shapes
    from rasterio.windows import from_bounds, Window
    from shapely.geometry import shape, box
    from indices import calcular_indices

    os.makedirs(out_dir, exist_ok=True)
    idx = calcular_indices(ortho_path, os.path.join(out_dir, "indices"), log=log)
    rn = idx["rutas"]
    if "ndre" not in rn:
        raise ValueError("Hace falta la banda RedEdge para la zonificacion por NDRE.")
    th = float(umbral_ndvi) if umbral_ndvi is not None else _umbral_global(rn["ndvi"])
    log(f"  Umbral NDVI de vegetacion: {th:.3f} ({'manual' if umbral_ndvi is not None else 'automatico, Otsu'})")

    sn, se = rasterio.open(rn["ndvi"]), rasterio.open(rn["ndre"])
    if abs(sn.transform.b) > 1e-9 or abs(sn.transform.d) > 1e-9:
        raise ValueError("Ortomosaico rotado: se espera una grilla norte-arriba (como la de ODM).")
    if sn.crs is None or sn.crs.is_geographic:
        raise ValueError("Hace falta un CRS proyectado (metros) para medir areas.")
    px = abs(sn.res[0])

    pol = gpd.read_file(poligonos_path, layer=capa) if capa else gpd.read_file(poligonos_path)
    pol = pol[pol.geometry.notna() & ~pol.geometry.is_empty].to_crs(sn.crs).reset_index(drop=True)
    namecol = next((c for c in ("nombre", "name", "Nombre", "NOMBRE") if c in pol.columns), None)

    filas, fallas_l, grilla_l = [], [], []
    for i, r in pol.iterrows():
        nombre = str(r[namecol]) if namecol else f"lote_{i + 1}"
        g = r.geometry
        minx, miny, maxx, maxy = g.bounds
        win = from_bounds(minx, miny, maxx, maxy, sn.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(Window(0, 0, sn.width, sn.height))
        except Exception:          # el poligono no toca el ortomosaico
            win = Window(0, 0, 0, 0)
        if win.width < 2 or win.height < 2:
            log(f"  {nombre}: fuera del ortomosaico, se omite")
            continue
        tr = sn.window_transform(win)
        ndvi, ndre = sn.read(1, window=win), se.read(1, window=win)
        dentro = geometry_mask([g], out_shape=ndvi.shape, transform=tr, invert=True)
        ok = dentro & np.isfinite(ndvi) & np.isfinite(ndre)
        if ok.sum() * px ** 2 < 20:
            log(f"  {nombre}: sin datos suficientes dentro del poligono, se omite")
            continue
        veg = ok & (ndvi >= th)
        bare = ok & ~veg

        # fallas: manchas de suelo / vegetacion muy baja (se quitan las menores a min_falla_m2)
        k = max(3, int(round(0.15 / px)) * 2 + 1)
        b = cv2.morphologyEx(bare.astype("uint8"), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        keep_ids = [j for j in range(1, n) if st[j, cv2.CC_STAT_AREA] * px ** 2 >= min_falla_m2]
        keep = np.isin(lab, keep_ids)
        for geom, _v in shapes(lab.astype("int32"), mask=keep, transform=tr):
            pg = shape(geom)
            m = geometry_mask([pg], out_shape=ndvi.shape, transform=tr, invert=True)
            fallas_l.append(dict(lote=nombre, area_m2=round(pg.area, 1),
                                 ndvi_medio=round(float(np.nanmean(ndvi[m & ok])), 3) if (m & ok).any() else None,
                                 geometry=pg))
        n_f = len(keep_ids)
        area_f = float(sum(st[j, cv2.CC_STAT_AREA] for j in keep_ids) * px ** 2)

        # grilla de vigor: estadisticas por celda con bincount
        rows, cols = np.indices(ndvi.shape)
        xs = tr.c + (cols + 0.5) * tr.a
        ys = tr.f + (rows + 0.5) * tr.e
        gx = np.floor((xs - minx) / grid_m).astype(int)
        gy = np.floor((ys - miny) / grid_m).astype(int)
        nx = int((maxx - minx) // grid_m) + 1
        ncell = (int((maxy - miny) // grid_m) + 1) * nx
        cid = gy * nx + gx
        cnt = np.bincount(cid[ok], minlength=ncell)
        cvg = np.bincount(cid[veg], minlength=ncell)
        s_re = np.bincount(cid[veg], weights=ndre[veg], minlength=ncell)
        s_vi = np.bincount(cid[veg], weights=ndvi[veg], minlength=ncell)
        s_bf = np.bincount(cid[keep & ok], minlength=ncell)
        celdas = []
        for c in np.nonzero(cnt)[0]:
            cy_, cx_ = divmod(int(c), nx)
            cel = box(minx + cx_ * grid_m, miny + cy_ * grid_m,
                      minx + (cx_ + 1) * grid_m, miny + (cy_ + 1) * grid_m)
            inter = cel.intersection(g)
            if inter.area < 0.5 * grid_m ** 2 or cvg[c] == 0:
                continue
            celdas.append(dict(lote=nombre, ndvi_medio=round(float(s_vi[c] / cvg[c]), 3),
                               ndre_medio=round(float(s_re[c] / cvg[c]), 3),
                               cobertura_pct=round(float(cvg[c] / cnt[c] * 100), 1),
                               falla_m2=round(float(s_bf[c] * px ** 2), 1), geometry=inter))
        q1 = q2 = None
        if len(celdas) >= 3:
            v = np.array([c["ndre_medio"] for c in celdas])
            q1, q2 = np.percentile(v, [33.3, 66.7])
            for c in celdas:
                c["vigor"] = "baja" if c["ndre_medio"] < q1 else ("media" if c["ndre_medio"] < q2 else "alta")
        else:
            for c in celdas:
                c["vigor"] = "n/d"
        grilla_l += celdas
        cv_cel = None
        if len(celdas) >= 2:
            mu = float(np.mean([c["ndre_medio"] for c in celdas]))
            if mu != 0:
                cv_cel = float(np.std([c["ndre_medio"] for c in celdas]) / abs(mu) * 100)
        nv, ne = ndvi[veg], ndre[veg]
        r2 = lambda x, d=3: round(float(x), d) if x is not None else None
        filas.append(dict(
            lote=nombre, area_ha=round(g.area / 1e4, 3), area_con_datos_ha=round(ok.sum() * px ** 2 / 1e4, 3),
            umbral_ndvi=round(th, 3), cobertura_veg_pct=round(veg.sum() / ok.sum() * 100, 1),
            ndvi_medio=r2(nv.mean()) if nv.size else None, ndvi_std=r2(nv.std()) if nv.size else None,
            ndre_medio=r2(ne.mean()) if ne.size else None, ndre_std=r2(ne.std()) if ne.size else None,
            ndre_p10=r2(np.percentile(ne, 10)) if ne.size else None,
            ndre_p90=r2(np.percentile(ne, 90)) if ne.size else None,
            n_fallas=n_f, area_fallas_m2=round(area_f, 1),
            fallas_pct=round(area_f / (ok.sum() * px ** 2) * 100, 1),
            n_celdas=len(celdas), cv_ndre_entre_celdas_pct=r2(cv_cel, 1),
            ndre_q33=r2(q1), ndre_q67=r2(q2), geometry=g))
        log(f"  {nombre}: {g.area / 1e4:.2f} ha | NDVI {nv.mean():.2f} | NDRE {ne.mean():.2f} | "
            f"cobertura {veg.sum() / ok.sum() * 100:.0f}% | {n_f} fallas ({area_f:.0f} m2) | {len(celdas)} celdas")
    sn.close()
    se.close()
    if not filas:
        raise ValueError("Ningun poligono pudo analizarse.")

    gpkg = os.path.join(out_dir, "lote_analizado.gpkg")
    if os.path.exists(gpkg):
        os.remove(gpkg)
    crs = pol.crs
    gdf = gpd.GeoDataFrame(filas, crs=crs)
    gdf.to_file(gpkg, layer="lotes", driver="GPKG")
    if fallas_l:
        gpd.GeoDataFrame(fallas_l, crs=crs).to_file(gpkg, layer="fallas", driver="GPKG")
    if grilla_l:
        gpd.GeoDataFrame(grilla_l, crs=crs).to_file(gpkg, layer="grilla", driver="GPKG")
    pd.DataFrame(gdf.drop(columns="geometry")).to_csv(os.path.join(out_dir, "resumen.csv"), index=False)
    png = _vista(gdf, grilla_l, fallas_l, rn["ndre"], os.path.join(out_dir, "vista.png"))
    log(f"  Resultados en {out_dir}")
    return dict(resumen=gdf.drop(columns="geometry"), gpkg=gpkg, png=png, indices=idx)


def _vista(gdf, grilla, fallas, ndre_path, png):
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rasterio.windows import from_bounds
        import geopandas as gpd
    except Exception:
        return None
    col = {"baja": "#d73027", "media": "#fee08b", "alta": "#1a9850", "n/d": "#bbbbbb"}
    b = gdf.total_bounds
    m = 10
    with rasterio.open(ndre_path) as s:
        win = from_bounds(b[0] - m, b[1] - m, b[2] + m, b[3] + m, s.transform).round_offsets().round_lengths()
        v = s.read(1, window=win, boundless=True, fill_value=np.nan)
        tr = s.window_transform(win)
    ext = [tr.c, tr.c + tr.a * v.shape[1], tr.f + tr.e * v.shape[0], tr.f]
    f = v[np.isfinite(v)]
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a in ax:
        a.imshow(v, cmap="RdYlGn", vmin=np.percentile(f, 2), vmax=np.percentile(f, 98), extent=ext)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title("NDRE (relativo) y limite del lote")
    gdf.boundary.plot(ax=ax[0], color="black", linewidth=1.5)
    ax[1].set_title("Vigor relativo por celda (terciles de NDRE) y fallas")
    if grilla:
        gg = gpd.GeoDataFrame(grilla, crs=gdf.crs)
        gg.plot(ax=ax[1], color=[col[x] for x in gg["vigor"]], edgecolor="white", linewidth=0.3, alpha=0.85)
    if fallas:
        gpd.GeoDataFrame(fallas, crs=gdf.crs).plot(ax=ax[1], color="none", edgecolor="black", linewidth=0.8)
    gdf.boundary.plot(ax=ax[1], color="black", linewidth=1.5)
    ax[1].set_xlim(ext[0], ext[1])
    ax[1].set_ylim(ext[2], ext[3])
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("poligonos")
    ap.add_argument("salida")
    ap.add_argument("--capa", default=None)
    ap.add_argument("--grid", type=float, default=10.0, help="tamano de celda en metros")
    ap.add_argument("--umbral-ndvi", type=float, default=None)
    ap.add_argument("--min-falla", type=float, default=2.0, help="area minima de una falla en m2")
    a = ap.parse_args()
    analizar_lotes(a.tif, a.poligonos, a.salida, capa=a.capa, grid_m=a.grid,
                   umbral_ndvi=a.umbral_ndvi, min_falla_m2=a.min_falla)
