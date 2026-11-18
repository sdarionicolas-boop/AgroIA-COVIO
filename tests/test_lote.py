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
