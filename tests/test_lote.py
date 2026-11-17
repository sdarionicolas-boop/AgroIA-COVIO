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
    assert len(g) == 25 and set(g["vigor"]) <= {"baja", "media", "alta", "n/d"}


def test_poligono_fuera_del_mosaico_no_rompe(mosaico, tmp_path):
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["fuera", "dentro"]},
                     geometry=[box(0, 0, 50, 50), box(500010, 7000010, 500060, 7000060)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    r = lote.analizar_lotes(mosaico, poly, str(tmp_path / "o"), umbral_ndvi=0.3, log=lambda *_: None)
    assert list(r["resumen"]["lote"]) == ["dentro"]
