"""Estadisticas, fallas y zonificacion de vigor dentro de poligonos de lote que dibuja el usuario.

Uso:
    python lote.py ORTOMOSAICO.tif POLIGONOS.gpkg|shp CARPETA_SALIDA [--capa NOMBRE]
                   [--grid 10] [--umbral X] [--min-falla 2]

Los indices se eligen segun las bandas del ortomosaico:
  - multiespectral (NIR + Red): vegetacion = NDVI, vigor = NDRE (o NDVI si no hay RedEdge).
  - RGB comun: vegetacion = ExG, vigor = GLI.

Por cada poligono: estadisticas de los indices (media, desvio, p10/p90), cobertura vegetal, fallas
(manchas de suelo o vegetacion muy baja >= --min-falla m2) y una grilla (--grid m) con el vigor relativo
de cada celda: el valor medio del indice de vigor y una clase (baja / normal / alta) segun cuanto se
aparta de la mediana del lote. Un lote parejo queda casi todo "normal" (no se fuerzan clases).

Los valores son RELATIVOS: sirven para comparar zonas dentro del mismo vuelo, no entre fechas. El umbral
de vegetacion por defecto es automatico (Otsu sobre todo el mosaico); fijalo con --umbral si el
resultado no te convence.
"""
import argparse
import os
import numpy as np

from indices import INDICES, MPL_OSCURO, TEXTO_TENUE, aviso_para, calcular_indices, indices_para_lotes


def _leeme(veg, vig, aviso):
    nv, ng = INDICES[veg][0], INDICES[vig][0]
    return aviso + f"""

COMO LEER LOS RESULTADOS
- Indices usados: {nv} para separar vegetacion de suelo y {ng} para el vigor relativo.
- resumen.csv: una fila por lote (estadisticas de los indices, cobertura, fallas, celdas).
- lote_analizado.gpkg: capas lotes, fallas y grilla (abrir en QGIS).
- Vigor por celda: se muestra el valor medio de {ng} y una clase. Una celda es "baja" o "alta" solo si se
  aparta de la mediana del lote al menos {int(DESVIO_MIN * 100)} % del rango del indice en todo el ortomosaico
  Y al menos {Z_MIN} desvios robustos (MAD) respecto de las demas celdas; el resto es "normal". Un lote
  parejo queda casi todo "normal". Los umbrales (desvio_min, z) son arbitrarios y no estan validados con
  datos de campo. Mira tambien cv_{vig}_entre_celdas_pct (variacion entre celdas) y pct_area_baja/alta.
- Fallas: manchas de suelo o vegetacion muy baja segun el umbral de {nv} (columna umbral_{veg}). El
  umbral automatico (Otsu) puede no ser adecuado en cultivos muy densos o muy ralos: revisa vista.png y,
  si hace falta, fija --umbral.
- Una falla no implica falta de plantas: puede ser camino, suelo, maleza o un sector senescente.
"""


def _otsu(v, bins=256):
    """Umbral de Otsu y su separabilidad eta (varianza entre clases / varianza total; ~1 = muy bimodal)."""
    h, e = np.histogram(v, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    sb = w0 * w1 * (m0 - m1) ** 2 / max(1.0, float(w0[-1])) ** 2
    i = int(np.argmax(sb))
    var = float(np.var(v))
    return float(c[i]), (float(sb[i]) / var if var > 0 else 0.0)


def _umbral_global(path, default=0.3):
    """(umbral, eta) automaticos sobre todo el mosaico."""
    import rasterio
    with rasterio.open(path) as s:
        k = max(1, max(s.width, s.height) // 2000)
        v = s.read(1, out_shape=(max(1, s.height // k), max(1, s.width // k)))
    v = v[np.isfinite(v)]
    return _otsu(v) if v.size else (default, 0.0)


DESVIO_MIN = 0.10   # fraccion del rango del indice (p2-p98 del mosaico) que una celda debe apartarse de la mediana
Z_MIN = 1.5         # y desvios robustos (MAD) respecto de las demas celdas (si hay >= 5)
ETA_MIN = 0.7   # por debajo, la distribucion no es claramente bimodal y el umbral automatico es dudoso
FALLAS_MAX_PCT = 40.0   # si las fallas ocupan mas de esto del lote, casi seguro el poligono o el umbral no sirven


def _celda_auto(area_m2, px):
    """Tamano de celda (m) para un lote segun su superficie: el mayor de 10, 5, 3, 2, 1, 0.5 o 0.25 m que deje al
    menos ~16 celdas (lado del lote / 4) y no baje de ~6 pixeles. Un lote grande usa 10 m; uno chico (p. ej. de
    lechuga) usa celdas mas finas en lugar de quedarse sin celdas."""
    lado = max(area_m2, 0.0) ** 0.5
    for c in (10, 5, 3, 2, 1, 0.5, 0.25):
        if c <= lado / 4:
            break
    else:
        c = 0.25
    return round(max(c, 6 * px), 3)


def analizar_lotes(ortho_path, poligonos_path, out_dir, capa=None, grid_m=None,
                   umbral=None, min_falla_m2=2.0, log=print, umbral_ndvi=None, modo=None, desvio_min=DESVIO_MIN):
    """Analiza cada poligono del archivo. Devuelve dict con resumen (DataFrame), gpkg, png, aviso e indices.
    `modo`: 'multi' (NDVI/NDRE), 'rgb' (ExG/GLI) o None (automatico segun las bandas).
    `grid_m`: tamano de celda en metros; None = automatico segun la superficie de cada lote (ver _celda_auto).
    `desvio_min`: fraccion del rango del indice de vigor que una celda debe apartarse de la mediana del
    lote para clasificarse baja/alta (el resto es normal). `umbral_ndvi` es un alias antiguo de `umbral`."""
    import cv2
    import pandas as pd
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask, shapes
    from rasterio.windows import from_bounds, Window
    from shapely.geometry import shape, box

    if umbral is None:
        umbral = umbral_ndvi
    os.makedirs(out_dir, exist_ok=True)
    vn, gn = indices_para_lotes(ortho_path, modo)  # indice de vegetacion e indice de vigor
    idx = calcular_indices(ortho_path, os.path.join(out_dir, "indices"), log=log,
                           indices=list(dict.fromkeys([vn, gn])), modo=modo)
    rn = idx["rutas"]
    advertencias = []
    est = idx["stats"].get(gn)
    rango = float(est["p98"] - est["p2"]) if est else 0.0
    if umbral is not None:
        th, eta = float(umbral), None
    else:
        th, eta = _umbral_global(rn[vn])
    log(f"  Umbral {INDICES[vn][0]} de vegetacion: {th:.3f} ({'manual' if umbral is not None else 'automatico, Otsu'})")
    if eta is not None and eta < ETA_MIN:
        msg = (f"El umbral automatico de {INDICES[vn][0]} ({th:.3f}) separa mal vegetacion y suelo en este "
               f"ortomosaico (distribucion poco bimodal, separabilidad {eta:.2f} < {ETA_MIN}). La cobertura "
               "vegetal y las fallas pueden estar distorsionadas: revisa vista.png y fija un umbral manual.")
        advertencias.append(msg)
        log("  ADVERTENCIA: " + msg)

    sv, sg = rasterio.open(rn[vn]), rasterio.open(rn[gn])
    if abs(sv.transform.b) > 1e-9 or abs(sv.transform.d) > 1e-9:
        raise ValueError("Ortomosaico rotado: se espera una grilla norte-arriba (como la de ODM).")
    if sv.crs is None:
        raise ValueError("El ortomosaico no tiene sistema de coordenadas (CRS): hace falta un GeoTIFF georreferenciado "
                         "para ubicar los lotes y medir areas.")
    if sv.crs.is_geographic:                      # no deberia pasar: calcular_indices reproyecta a UTM
        raise ValueError("Hace falta un CRS proyectado (metros) para medir areas.")
    px = abs(sv.res[0])

    pol = gpd.read_file(poligonos_path, layer=capa) if capa else gpd.read_file(poligonos_path)
    pol = pol[pol.geometry.notna() & ~pol.geometry.is_empty].to_crs(sv.crs).reset_index(drop=True)
    namecol = next((c for c in ("nombre", "name", "Nombre", "NOMBRE") if c in pol.columns), None)
    kv, kg = f"{vn}_medio", f"{gn}_medio"          # nombres de columna (pueden coincidir si vn == gn)

    filas, fallas_l, grilla_l = [], [], []
    for i, r in pol.iterrows():
        nombre = str(r[namecol]) if namecol else f"lote_{i + 1}"
        g = r.geometry
        minx, miny, maxx, maxy = g.bounds
        win = from_bounds(minx, miny, maxx, maxy, sv.transform).round_offsets().round_lengths()
        try:
            win = win.intersection(Window(0, 0, sv.width, sv.height))
        except Exception:          # el poligono no toca el ortomosaico
            win = Window(0, 0, 0, 0)
        if win.width < 2 or win.height < 2:
            log(f"  {nombre}: fuera del ortomosaico, se omite")
            continue
        tr = sv.window_transform(win)
        iv, ig = sv.read(1, window=win), sg.read(1, window=win)
        dentro = geometry_mask([g], out_shape=iv.shape, transform=tr, invert=True)
        ok = dentro & np.isfinite(iv) & np.isfinite(ig)
        if ok.sum() * px ** 2 < 20:
            log(f"  {nombre}: sin datos suficientes dentro del poligono, se omite")
            continue
        veg = ok & (iv >= th)
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
            m = geometry_mask([pg], out_shape=iv.shape, transform=tr, invert=True)
            fallas_l.append({"lote": nombre, "area_m2": round(pg.area, 1),
                             kv: round(float(np.nanmean(iv[m & ok])), 3) if (m & ok).any() else None,
                             "geometry": pg})
        n_f = len(keep_ids)
        area_f = float(sum(st[j, cv2.CC_STAT_AREA] for j in keep_ids) * px ** 2)

        # grilla de vigor: estadisticas por celda con bincount
        gm = float(grid_m) if grid_m else _celda_auto(g.area, px)
        rows, cols = np.indices(iv.shape)
        xs = tr.c + (cols + 0.5) * tr.a
        ys = tr.f + (rows + 0.5) * tr.e
        gx = np.floor((xs - minx) / gm).astype(int)
        gy = np.floor((ys - miny) / gm).astype(int)
        nx = int((maxx - minx) // gm) + 1
        ncell = (int((maxy - miny) // gm) + 1) * nx
        cid = gy * nx + gx
        cnt = np.bincount(cid[ok], minlength=ncell)
        cvg = np.bincount(cid[veg], minlength=ncell)
        s_g = np.bincount(cid[veg], weights=ig[veg], minlength=ncell)
        s_v = np.bincount(cid[veg], weights=iv[veg], minlength=ncell)
        s_bf = np.bincount(cid[keep & ok], minlength=ncell)
        celdas = []
        for c in np.nonzero(cnt)[0]:
            cy_, cx_ = divmod(int(c), nx)
            cel = box(minx + cx_ * gm, miny + cy_ * gm,
                      minx + (cx_ + 1) * gm, miny + (cy_ + 1) * gm)
            inter = cel.intersection(g)
            if inter.area < 0.5 * gm ** 2 or cvg[c] == 0:
                continue
            d = {"lote": nombre, kv: round(float(s_v[c] / cvg[c]), 3)}
            d[kg] = round(float(s_g[c] / cvg[c]), 3)
            d.update(cobertura_pct=round(float(cvg[c] / cnt[c] * 100), 1),
                     falla_m2=round(float(s_bf[c] * px ** 2), 1), geometry=inter)
            celdas.append(d)
        med = None
        gate = desvio_min * rango
        if len(celdas) >= 3:
            v = np.array([c[kg] for c in celdas])
            med = float(np.median(v))
            sig = 1.4826 * float(np.median(np.abs(v - med)))
            for c in celdas:
                d = c[kg] - med
                z_ok = len(celdas) < 5 or sig <= 1e-9 or abs(d) / sig >= Z_MIN
                c["vigor"] = ("alta" if d > 0 else "baja") if (abs(d) > max(gate, 1e-9) and z_ok) else "normal"
                c["desvio"] = round(d, 3)
        else:
            for c in celdas:
                c["vigor"] = "n/d"
        grilla_l += celdas
        cv_cel = None
        if len(celdas) >= 2:
            mu = float(np.mean([c[kg] for c in celdas]))
            if mu != 0:
                cv_cel = float(np.std([c[kg] for c in celdas]) / abs(mu) * 100)
        nv, ne = iv[veg], ig[veg]
        r2 = lambda x, d=3: round(float(x), d) if x is not None else None
        fila = {"lote": nombre, "area_ha": round(g.area / 1e4, 3),
                "area_con_datos_ha": round(ok.sum() * px ** 2 / 1e4, 3),
                "indice_vegetacion": INDICES[vn][0], "indice_vigor": INDICES[gn][0],
                f"umbral_{vn}": round(th, 3), "umbral_separacion": (round(eta, 2) if eta is not None else None),
                "cobertura_veg_pct": round(veg.sum() / ok.sum() * 100, 1),
                kv: r2(nv.mean()) if nv.size else None, f"{vn}_std": r2(nv.std()) if nv.size else None}
        if gn != vn:
            fila[kg] = r2(ne.mean()) if ne.size else None
            fila[f"{gn}_std"] = r2(ne.std()) if ne.size else None
        fila[f"{gn}_p10"] = r2(np.percentile(ne, 10)) if ne.size else None
        fila[f"{gn}_p90"] = r2(np.percentile(ne, 90)) if ne.size else None
        fila.update(n_fallas=n_f, area_fallas_m2=round(area_f, 1),
                    fallas_pct=round(area_f / (ok.sum() * px ** 2) * 100, 1),
                    n_celdas=len(celdas), celda_m=gm)
        fila[f"cv_{gn}_entre_celdas_pct"] = r2(cv_cel, 1)
        fila[f"{gn}_mediana_celdas"] = r2(med)
        fila["desvio_min_abs"] = r2(gate) if med is not None else None
        tot = sum(c["geometry"].area for c in celdas) or 1.0
        for cl in ("baja", "normal", "alta"):
            fila[f"pct_area_{cl}"] = (round(sum(c["geometry"].area for c in celdas if c["vigor"] == cl) / tot * 100, 1)
                                      if med is not None else None)
        fila["geometry"] = g
        filas.append(fila)
        if fila["fallas_pct"] > FALLAS_MAX_PCT:
            msg = (f"En {nombre} las fallas ocupan el {fila['fallas_pct']:.0f} % del lote ({area_f:.0f} m2). Es mucho: "
                   "revisa que el poligono abarque un solo cultivo (sin parcelas sin sembrar, caminos u otros lotes), "
                   "que el umbral de vegetacion sea adecuado (mira vista.png y, si hace falta, fija un umbral manual) "
                   "y ten en cuenta que en un cultivo en surcos con poca cobertura el suelo entre surcos cuenta como falla.")
            advertencias.append(msg)
            log("  ADVERTENCIA: " + msg)
        log(f"  {nombre}: {g.area / 1e4:.2f} ha | {INDICES[vn][0]} {nv.mean():.2f} | {INDICES[gn][0]} {ne.mean():.2f} | "
            f"cobertura {veg.sum() / ok.sum() * 100:.0f}% | {n_f} fallas ({area_f:.0f} m2) | {len(celdas)} celdas")
    sv.close()
    sg.close()
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
    aviso = aviso_para(idx["indices"])
    png = _vista(gdf, grilla_l, fallas_l, rn[gn], INDICES[gn][0], os.path.join(out_dir, "vista.png"))
    with open(os.path.join(out_dir, "LEEME.txt"), "w", encoding="utf-8") as f:
        f.write(_leeme(vn, gn, aviso) + "".join(chr(10) + "ADVERTENCIA: " + a + chr(10) for a in advertencias))
    log(f"  Resultados en {out_dir}")
    log("  " + aviso)
    return dict(resumen=gdf.drop(columns="geometry"), gpkg=gpkg, png=png, indices=idx, aviso=aviso,
                indices_usados=(vn, gn), advertencias=advertencias)


def _vista(gdf, grilla, fallas, vigor_path, vigor_nombre, png):
    try:
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rasterio.windows import from_bounds
        import geopandas as gpd
    except Exception:
        return None
    col = {"baja": "#d73027", "normal": "#e0e0e0", "alta": "#1a9850", "n/d": "#bbbbbb"}
    b = gdf.total_bounds
    m = 10
    with rasterio.open(vigor_path) as s:
        win = from_bounds(b[0] - m, b[1] - m, b[2] + m, b[3] + m, s.transform).round_offsets().round_lengths()
        v = s.read(1, window=win, boundless=True, fill_value=np.nan)
        tr = s.window_transform(win)
    ext = [tr.c, tr.c + tr.a * v.shape[1], tr.f + tr.e * v.shape[0], tr.f]
    f = v[np.isfinite(v)]
    with plt.rc_context(MPL_OSCURO):
        return _dibujar_vista(plt, gdf, grilla, fallas, vigor_nombre, v, ext, f, col, png)


def _dibujar_vista(plt, gdf, grilla, fallas, vigor_nombre, v, ext, f, col, png):
    import geopandas as gpd
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    for a in ax:
        a.imshow(v, cmap="RdYlGn", vmin=np.percentile(f, 2), vmax=np.percentile(f, 98), extent=ext)
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title(f"{vigor_nombre} (relativo) y limite del lote")
    gdf.boundary.plot(ax=ax[0], color="#ffd400", linewidth=1.5)
    ax[1].set_title(f"Vigor por celda segun desvio de la mediana ({vigor_nombre}) y fallas")
    if grilla:
        gg = gpd.GeoDataFrame(grilla, crs=gdf.crs)
        gg.plot(ax=ax[1], color=[col[x] for x in gg["vigor"]], edgecolor="white", linewidth=0.3, alpha=0.85)
    if fallas:
        gpd.GeoDataFrame(fallas, crs=gdf.crs).plot(ax=ax[1], color="none", edgecolor="white", linewidth=0.8)
    gdf.boundary.plot(ax=ax[1], color="#ffd400", linewidth=1.5)
    ax[1].set_xlim(ext[0], ext[1])
    ax[1].set_ylim(ext[2], ext[3])
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(0.5, 0.012, "Valores relativos: comparables solo dentro de este vuelo. Celda baja/alta = se aparta claramente de la mediana del lote.",
             ha="center", fontsize=9, color=TEXTO_TENUE)
    fig.savefig(png, dpi=100)
    plt.close(fig)
    return png


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("poligonos")
    ap.add_argument("salida")
    ap.add_argument("--capa", default=None)
    ap.add_argument("--grid", type=float, default=None,
                    help="tamano de celda en metros (por defecto: automatico segun la superficie del lote)")
    ap.add_argument("--umbral", "--umbral-ndvi", dest="umbral", type=float, default=None,
                    help="umbral del indice de vegetacion (vacio = automatico)")
    ap.add_argument("--min-falla", type=float, default=2.0, help="area minima de una falla en m2")
    ap.add_argument("--desvio-min", type=float, default=DESVIO_MIN * 100,
                    help="desvio minimo respecto de la mediana, en %% del rango del indice (default 10)")
    ap.add_argument("--modo", choices=("multi", "rgb"), default=None,
                    help="multi = NDVI/NDRE, rgb = ExG/GLI (por defecto: automatico segun las bandas)")
    a = ap.parse_args()
    analizar_lotes(a.tif, a.poligonos, a.salida, capa=a.capa, grid_m=a.grid,
                   umbral=a.umbral, min_falla_m2=a.min_falla, modo=a.modo, desvio_min=a.desvio_min / 100)
