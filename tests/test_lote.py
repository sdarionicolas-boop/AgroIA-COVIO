import os, sys
import numpy as np
import pytest
rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lote


@pytest.fixture
def mosaico(tmp_path):
    """100x100 m a 0.5 m/px: vegetacion homogenea con un parche de suelo de 10x10 m."""
    n = 200
    d = np.zeros((6, n, n), "uint16")
    d[0], d[1], d[2] = 100, 100, 100                 # R, G, B
    d[3], d[4], d[5] = 300, 200, 255                 # NIR, RedEdge, alpha
    d[3, 20:40, 20:40] = 105                         # suelo: NIR ~ Red -> NDVI ~ 0.02
    p = str(tmp_path / "m.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=6, dtype="uint16",
                       crs="EPSG:32723", transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
        s.descriptions = ["Red", "Green", "Blue", "NIR", "Rededge", "alpha"]
    return p


def test_fallas_cobertura_y_grilla(mosaico, tmp_path):
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500100, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(mosaico, poly, str(tmp_path / "o"), grid_m=20, umbral_ndvi=0.3,
                            min_falla_m2=2, log=lambda *_: None)
    f = r["resumen"].iloc[0]
    assert f["area_ha"] == pytest.approx(1.0)
    assert f["n_fallas"] == 1
    assert f["area_fallas_m2"] == pytest.approx(100, rel=0.05)
    assert f["cobertura_veg_pct"] == pytest.approx(99, abs=0.5)
    assert f["ndvi_medio"] == pytest.approx(0.5, abs=0.01)     # (300-100)/(300+100)
    assert os.path.exists(os.path.join(str(tmp_path / "o"), "LEEME.txt"))
    g = gpd.read_file(r["gpkg"], layer="grilla")
    assert len(g) == 25 and set(g["vigor"]) <= {"baja", "normal", "alta", "n/d"}


def test_poligono_fuera_del_mosaico_no_rompe(mosaico, tmp_path):
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["fuera", "dentro"]},
                     geometry=[box(0, 0, 50, 50), box(500010, 7000010, 500060, 7000060)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(mosaico, poly, str(tmp_path / "o"), umbral_ndvi=0.3, log=lambda *_: None)
    assert list(r["resumen"]["lote"]) == ["dentro"]


@pytest.fixture
def mosaico_rgb(tmp_path):
    """100x100 m a 0.5 m/px, RGB uint8: vegetacion verde con un parche de suelo marron de 10x10 m."""
    n = 200
    d = np.zeros((3, n, n), "uint8")
    d[0], d[1], d[2] = 50, 150, 50
    d[0, 20:40, 20:40], d[1, 20:40, 20:40], d[2, 20:40, 20:40] = 150, 100, 70
    p = str(tmp_path / "rgb.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8",
                       crs="EPSG:32723", transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
    return p


def test_lote_en_rgb_usa_exg_y_gli(mosaico_rgb, tmp_path):
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500100, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(mosaico_rgb, poly, str(tmp_path / "o"), grid_m=20, umbral=0.3, log=lambda *_: None)
    f = r["resumen"].iloc[0]
    assert r["indices_usados"] == ("exg", "gli") and "RGB" in r["aviso"]
    assert f["indice_vegetacion"] == "ExG" and f["indice_vigor"] == "GLI"
    assert f["n_fallas"] == 1 and f["area_fallas_m2"] == pytest.approx(100, rel=0.05)
    assert f["exg_medio"] == pytest.approx(0.8, abs=0.01) and f["gli_medio"] == pytest.approx(0.5, abs=0.01)
    assert "umbral_exg" in r["resumen"].columns and "ndvi_medio" not in r["resumen"].columns
    txt = open(os.path.join(str(tmp_path / "o"), "LEEME.txt"), encoding="utf-8").read()
    assert "ExG" in txt and "GLI" in txt


def test_vista_previa_indices_rgb(mosaico_rgb):
    import lotes_io
    for modo in ("rgb", "exg", "gli", "vari"):
        v = lotes_io.vista_previa(mosaico_rgb, modo, max_px=100)
        assert v["img"].shape == (100, 100, 3)
    with pytest.raises(ValueError):
        lotes_io.vista_previa(mosaico_rgb, "ndvi", max_px=100)    # no hay NIR en un RGB


def test_advierte_si_el_umbral_automatico_separa_mal(tmp_path):
    """Distribucion unimodal (sin suelo): Otsu parte en dos una sola poblacion -> advertencia."""
    rng = np.random.default_rng(0)
    n = 200
    d = np.zeros((3, n, n), "uint8")
    g = np.clip(rng.normal(130, 25, (n, n)), 60, 220)
    d[0], d[1], d[2] = 60, g, 60
    p = str(tmp_path / "u.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8",
                       crs="EPSG:32723", transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500100, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(p, poly, str(tmp_path / "o"), log=lambda *_: None)
    assert r["advertencias"] and "separa mal" in r["advertencias"][0]
    assert r["resumen"].iloc[0]["umbral_separacion"] < lote.ETA_MIN
    assert "ADVERTENCIA" in open(os.path.join(str(tmp_path / "o"), "LEEME.txt"), encoding="utf-8").read()
    # con umbral manual no se advierte
    r2 = lote.analizar_lotes(p, poly, str(tmp_path / "o2"), umbral=0.3, log=lambda *_: None)
    assert r2["advertencias"] == []


def test_lote_modo_rgb_sobre_multiespectral(mosaico, tmp_path):
    """Un multiespectral analizado en modo RGB usa ExG/GLI y no NDVI/NDRE."""
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500100, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(mosaico, poly, str(tmp_path / "o"), umbral=0.0, modo="rgb", log=lambda *_: None)
    assert r["indices_usados"] == ("exg", "gli") and "exg_medio" in r["resumen"].columns
    assert "ndvi_medio" not in r["resumen"].columns


def _mosaico_vigor(tmp_path, bajo=False, ruido=2):
    """RGB 100x100 m (0.5 m/px). Poligono = mitad izquierda (50x100 m); la derecha es suelo desnudo (da rango
    al indice). Dentro del poligono: vegetacion pareja con ruido; opcionalmente un bloque de vigor bajo."""
    rng = np.random.default_rng(1)
    n = 200
    d = np.zeros((3, n, n), "uint8")
    d[0], d[1], d[2] = 50, 150, 50
    d[:, :, 100:] = np.array([150, 100, 70], "uint8").reshape(3, 1, 1)          # suelo (fuera del poligono)
    d[1, :, :100] = np.clip(d[1, :, :100] + rng.normal(0, ruido, (n, 100)), 0, 255).astype("uint8")
    if bajo:
        d[0, 100:160, :60], d[1, 100:160, :60], d[2, 100:160, :60] = 60, 115, 60    # vigor bajo, aun vegetacion (ExG 0.47 > umbral 0.3)
    p = str(tmp_path / "v.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8",
                       crs="EPSG:32723", transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
    poly = str(tmp_path / "pv.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500050, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    return p, poly


def test_lote_parejo_queda_normal(tmp_path):
    p, poly = _mosaico_vigor(tmp_path, bajo=False)
    r = lote.analizar_lotes(p, poly, str(tmp_path / "o"), grid_m=10, umbral=0.3, log=lambda *_: None)
    f = r["resumen"].iloc[0]
    g = gpd.read_file(r["gpkg"], layer="grilla")
    assert set(g["vigor"]) == {"normal"} and f["pct_area_normal"] == pytest.approx(100.0)
    assert f["pct_area_baja"] == 0 and f["pct_area_alta"] == 0


def test_lote_con_bloque_bajo_marca_solo_esas_celdas(tmp_path):
    p, poly = _mosaico_vigor(tmp_path, bajo=True)
    r = lote.analizar_lotes(p, poly, str(tmp_path / "o"), grid_m=10, umbral=0.3, log=lambda *_: None)
    g = gpd.read_file(r["gpkg"], layer="grilla")
    bajas = g[g["vigor"] == "baja"]
    assert len(bajas) >= 3 and not (g["vigor"] == "alta").any()
    # el bloque bajo ocupa x 0-30 m, y 20-50 m desde el borde superior (y: 7000100 - 50 .. -80)
    assert bajas.geometry.centroid.x.max() < 500035 and bajas.geometry.centroid.y.between(7000020, 7000050).all()
    f = r["resumen"].iloc[0]
    assert 0 < f["pct_area_baja"] < 50 and f["pct_area_normal"] > 50


def test_celda_automatica_segun_la_superficie():
    assert lote._celda_auto(10000, 0.5) == 10          # 1 ha -> 10 m (como antes)
    assert lote._celda_auto(2000, 0.05) == 10          # 0.2 ha
    assert lote._celda_auto(1000, 0.05) == 5           # 0.1 ha
    assert lote._celda_auto(130, 0.05) == 2            # lote chico de lechuga (~130 m2): lado 11.4 m -> 2 m
    assert lote._celda_auto(4, 0.01) == 0.5            # 2 x 2 m
    assert lote._celda_auto(130, 1.0) == 6             # nunca menos de ~6 pixeles


def test_lote_chico_forma_celdas_con_grilla_automatica(mosaico_rgb, tmp_path):
    poly = str(tmp_path / "p.gpkg")
    # 12 x 12 m dentro del mosaico: con celdas fijas de 10 m queda una sola celda (o ninguna si el lote es irregular)
    gpd.GeoDataFrame({"nombre": ["chico"]}, geometry=[box(500020, 7000020, 500032, 7000032)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(mosaico_rgb, poly, str(tmp_path / "o"), umbral=0.3, log=lambda *_: None)
    f = r["resumen"].iloc[0]
    assert f["celda_m"] == 3 and f["n_celdas"] >= 9
    r2 = lote.analizar_lotes(mosaico_rgb, poly, str(tmp_path / "o2"), grid_m=10, umbral=0.3, log=lambda *_: None)
    assert r2["resumen"].iloc[0]["celda_m"] == 10                       # una celda fija de 10 m sigue respetandose
    assert r2["resumen"].iloc[0]["n_celdas"] == 1                       # y deja una sola celda: la automatica da mucho mas detalle


def test_crs_utm_para_puntos_conocidos():
    import indices
    assert indices.crs_utm_para(-91.54, 14.88) == 32615        # Guatemala
    assert indices.crs_utm_para(-60.0, -33.0) == 32721         # Argentina (hemisferio sur)
    assert indices.crs_utm_para(3.0, 40.0) == 32631
    assert indices.crs_utm_para(179.9, 0.0) == 32660


@pytest.fixture
def mosaico_geografico(tmp_path):
    """RGB en EPSG:4326 (grados), ~1 m/px, cerca de Santa Fe (Argentina): vegetacion con un parche de suelo."""
    n = 200
    d = np.zeros((3, n, n), "uint8")
    d[0], d[1], d[2] = 50, 150, 50
    d[0, 40:70, 40:70], d[1, 40:70, 40:70], d[2, 40:70, 40:70] = 150, 100, 70
    p = str(tmp_path / "geo.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-60.00, -33.00, 1e-5, 1e-5)) as s:
        s.write(d)
    return p


def test_ortomosaico_en_grados_se_analiza_en_metros(mosaico_geografico, tmp_path):
    import indices
    poly = str(tmp_path / "pg.gpkg")
    g = box(-59.9992, -33.0018, -59.9984, -33.0010)                          # en grados
    gpd.GeoDataFrame({"nombre": ["G"]}, geometry=[g], crs="EPSG:4326").to_file(poly, driver="GPKG")
    esperada_ha = gpd.GeoSeries([g], crs="EPSG:4326").to_crs("EPSG:32721").area.iloc[0] / 1e4
    r = lote.analizar_lotes(mosaico_geografico, poly, str(tmp_path / "o"), umbral=0.3, log=lambda *_: None)
    f = r["resumen"].iloc[0]
    assert f["area_ha"] == pytest.approx(esperada_ha, rel=0.01)               # area en metros, no en grados
    assert f["n_celdas"] > 0 and f["celda_m"] >= 1
    for p in r["indices"]["rutas"].values():                                  # los indices quedan en UTM
        with rasterio.open(p) as s:
            assert s.crs.to_epsg() == 32721 and not s.crs.is_geographic and 0.5 < s.res[0] < 2
    assert gpd.read_file(r["gpkg"]).crs.to_epsg() == 32721


def test_sin_crs_da_un_mensaje_claro(tmp_path):
    n = 20
    p = str(tmp_path / "sincrs.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8") as s:
        s.write(np.full((3, n, n), 100, "uint8"))
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:32721").to_file(poly, driver="GPKG")
    with pytest.raises(ValueError, match="no tiene sistema de coordenadas"):
        lote.analizar_lotes(p, poly, str(tmp_path / "o"), log=lambda *_: None)


def test_advierte_si_las_fallas_ocupan_demasiado_del_lote(tmp_path):
    """Un poligono que abarca cultivo y suelo sin sembrar: la mitad del lote queda como 'falla' -> advertencia."""
    n = 200
    d = np.zeros((3, n, n), "uint8")
    d[0], d[1], d[2] = 50, 150, 50
    d[0, :, :100], d[1, :, :100], d[2, :, :100] = 150, 100, 70           # mitad izquierda: suelo desnudo
    p = str(tmp_path / "mix.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs="EPSG:32723",
                       transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
    poly = str(tmp_path / "pm.gpkg")
    gpd.GeoDataFrame({"nombre": ["mixto", "sano"]},
                     geometry=[box(500000, 7000000, 500100, 7000100), box(500060, 7000010, 500090, 7000090)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(p, poly, str(tmp_path / "o"), umbral=0.3, log=lambda *_: None)
    f = r["resumen"].set_index("lote")
    assert f.loc["mixto", "fallas_pct"] > lote.FALLAS_MAX_PCT and f.loc["sano", "fallas_pct"] == 0
    adv = [a for a in r["advertencias"] if "fallas ocupan" in a]
    assert len(adv) == 1 and "mixto" in adv[0] and "sano" not in adv[0]
    assert "surcos" in adv[0]                                              # menciona el caso de cultivos en surcos
    txt = open(os.path.join(str(tmp_path / "o"), "LEEME.txt"), encoding="utf-8").read()
    assert "ADVERTENCIA: En mixto las fallas ocupan" in txt


def test_la_advertencia_del_umbral_muestra_tres_decimales(tmp_path):
    """Un valor de 0.697 se mostraba como '0.70 < 0.7' (contradictorio); ahora se ve con tres decimales."""
    rng = np.random.default_rng(0)
    n = 200
    d = np.zeros((3, n, n), "uint8")
    g = np.clip(rng.normal(130, 25, (n, n)), 60, 220)
    d[0], d[1], d[2] = 60, g, 60
    p = str(tmp_path / "u.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs="EPSG:32723",
                       transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500100, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(p, poly, str(tmp_path / "o"), log=lambda *_: None)
    import re
    m = re.search(r"separabilidad (\d\.\d+) <", r["advertencias"][0])
    assert m and len(m.group(1).split(".")[1]) == 3
    assert float(m.group(1)) < lote.ETA_MIN
    assert round(float(r["resumen"].iloc[0]["umbral_separacion"]), 3) == float(m.group(1))
