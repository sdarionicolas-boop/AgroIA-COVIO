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
        assert indices.indices_disponibles(indices.detectar_bandas(s)) == ["exg", "gli", "vari"]


def test_multiespectral_y_rgb_no_se_mezclan():
    assert indices.indices_disponibles(dict(red=1, green=2, blue=3, nir=4, rededge=5)) == ["ndvi", "ndre"]
    assert indices.indices_disponibles(dict(red=1, nir=2)) == ["ndvi"]
    assert indices.indices_disponibles(dict(red=1, green=2, blue=3)) == ["exg", "gli", "vari"]
    assert indices.indices_disponibles(dict(red=1)) == []
    assert indices.roles(["ndvi", "ndre"]) == ("ndvi", "ndre")
    assert indices.roles(["ndvi"]) == ("ndvi", "ndvi")
    assert indices.roles(["exg", "gli", "vari"]) == ("exg", "gli")
    assert "RGB" in indices.aviso_para(["exg"]) and "calibracion" in indices.aviso_para(["ndvi"])


def test_formulas_rgb_valores_conocidos():
    z = lambda v: np.array([[v]], "float32")
    ok = np.array([[True]])
    verde = dict(red=z(50), green=z(150), blue=z(50))      # vegetacion
    suelo = dict(red=z(150), green=z(100), blue=z(70))     # suelo marron
    assert indices.formula("exg", verde, ok)[0, 0] == pytest.approx((300 - 100) / 250)     # 0.8
    assert indices.formula("exg", suelo, ok)[0, 0] == pytest.approx((200 - 220) / 320)     # < 0
    assert indices.formula("gli", verde, ok)[0, 0] == pytest.approx(200 / 400)
    assert indices.formula("vari", verde, ok)[0, 0] == pytest.approx(100 / 150)
    # sin dato -> NaN
    assert np.isnan(indices.formula("exg", verde, np.array([[False]]))[0, 0])
    # el ExG normalizado no depende del brillo global
    doble = {k: v * 2 for k, v in verde.items()}
    assert indices.formula("exg", doble, ok)[0, 0] == pytest.approx(indices.formula("exg", verde, ok)[0, 0])


def test_calcular_indices_rgb(tmp_path):
    p = str(tmp_path / "rgb.tif")
    d = np.zeros((3, 20, 20), "uint8")
    d[0], d[1], d[2] = 50, 150, 50
    d[:, :10, :10] = 0                                       # esquina sin dato (negro)
    with rasterio.open(p, "w", driver="GTiff", width=20, height=20, count=3, dtype="uint8") as s:
        s.write(d)
    r = indices.calcular_indices(p, str(tmp_path / "o"), log=lambda *_: None)
    assert r["indices"] == ["exg", "gli", "vari"] and "RGB" in r["aviso"]
    with rasterio.open(r["rutas"]["exg"]) as s:
        v = s.read(1)
    assert v[15, 15] == pytest.approx(0.8) and np.isnan(v[0, 0])
    assert os.path.exists(os.path.join(str(tmp_path / "o"), "indices_stats.csv"))
    with pytest.raises(ValueError):
        indices.calcular_indices(p, str(tmp_path / "o2"), indices=["ndvi"], log=lambda *_: None)


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


def test_modos_multi_y_rgb():
    multi = dict(red=1, green=2, blue=3, nir=4, rededge=5)
    assert indices.indices_disponibles(multi, "multi") == ["ndvi", "ndre"]
    assert indices.indices_disponibles(multi, "rgb") == ["exg", "gli", "vari"]   # un multiespectral tambien tiene RGB
    assert indices.indices_disponibles(multi) == ["ndvi", "ndre"]                  # automatico: multiespectral
    rgb = dict(red=1, green=2, blue=3)
    assert indices.indices_disponibles(rgb, "multi") == []
    assert indices.indices_disponibles(rgb, "rgb") == ["exg", "gli", "vari"]


def test_calcular_indices_con_modo(tmp_path):
    p = str(tmp_path / "m.tif")
    d = np.zeros((6, 8, 8), "uint16")
    d[0], d[1], d[2], d[3], d[4], d[5] = 100, 150, 80, 300, 200, 255
    _crear(p, ["Red", "Green", "Blue", "NIR", "Rededge", "alpha"], d)
    r = indices.calcular_indices(p, str(tmp_path / "a"), modo="rgb", log=lambda *_: None)     # RGB sobre un multiespectral
    assert r["indices"] == ["exg", "gli", "vari"] and "RGB" in r["aviso"]
    r = indices.calcular_indices(p, str(tmp_path / "b"), modo="multi", log=lambda *_: None)
    assert r["indices"] == ["ndvi", "ndre"]
    prgb = str(tmp_path / "rgb.tif")
    with rasterio.open(prgb, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8") as s:
        s.write(np.ones((3, 4, 4), "uint8"))
    with pytest.raises(ValueError, match="NIR"):
        indices.calcular_indices(prgb, str(tmp_path / "c"), modo="multi", log=lambda *_: None)
