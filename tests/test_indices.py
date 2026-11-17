import os, sys
import numpy as np
import pytest
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import indices


def _crear(path, nombres, datos, alpha_ci=False):
    with rasterio.open(path, "w", driver="GTiff", width=datos.shape[2], height=datos.shape[1],
                       count=len(nombres), dtype=str(datos.dtype), crs="EPSG:32723",
                       transform=from_origin(0, 10, 1, 1)) as d:
        d.write(datos)
        d.descriptions = nombres
        if alpha_ci:
            ci = [rasterio.enums.ColorInterp.gray] * len(nombres)
            ci[-1] = rasterio.enums.ColorInterp.alpha
            d.colorinterp = ci


def test_detecta_bandas_por_nombre_sin_importar_orden(tmp_path):
    p = str(tmp_path / "a.tif")
    _crear(p, ["Blue", "Green", "Red", "NIR", "Rededge"], np.ones((5, 4, 4), "uint16"))
    with rasterio.open(p) as s:
        assert indices.detectar_bandas(s) == dict(blue=1, green=2, red=3, nir=4, rededge=5)
        assert indices.bandas_rgb(s) == (3, 2, 1)


def test_rgb_simple_cae_a_123(tmp_path):
    p = str(tmp_path / "rgb.tif")
    with rasterio.open(p, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8") as d:
        d.write(np.ones((3, 4, 4), "uint8"))
    with rasterio.open(p) as s:
        assert indices.bandas_rgb(s) == (1, 2, 3)
    with pytest.raises(ValueError):
        indices.calcular_indices(p, str(tmp_path / "o"))


def test_ndvi_ndre_valores_y_nodata(tmp_path):
    p = str(tmp_path / "m.tif")
    d = np.zeros((6, 4, 4), "uint16")
    d[0], d[1], d[2], d[3], d[4], d[5] = 100, 100, 100, 300, 200, 255   # R,G,B,NIR,RE,alpha
    d[5, 0, 0] = 0                                                       # un pixel sin dato
    _crear(p, ["Red", "Green", "Blue", "NIR", "Rededge", "alpha"], d)
    r = indices.calcular_indices(p, str(tmp_path / "o"), log=lambda *_: None)
    with rasterio.open(r["rutas"]["ndvi"]) as s:
        v = s.read(1)
    with rasterio.open(r["rutas"]["ndre"]) as s:
        e = s.read(1)
    assert v[1, 1] == pytest.approx(0.5)        # (300-100)/(300+100)
    assert e[1, 1] == pytest.approx(0.2)        # (300-200)/(300+200)
    assert np.isnan(v[0, 0]) and np.isnan(e[0, 0])
    assert "RELATIVOS" in r["aviso"]


def test_alpha_sin_nombre_se_detecta_por_colorinterp():
    # ODM no nombra el alpha, solo lo marca en colorinterp (GDAL no deja crearlo facil en un test).
    from types import SimpleNamespace as NS
    ci = lambda n: NS(name=n)
    src = NS(count=6, descriptions=("Red", "Green", "Blue", "NIR", "Rededge", None),
             colorinterp=[ci("red"), ci("green"), ci("blue"), ci("gray"), ci("gray"), ci("alpha")])
    assert indices.detectar_bandas(src)["alpha"] == 6
