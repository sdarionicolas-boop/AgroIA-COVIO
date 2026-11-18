import os, sys
import numpy as np
import pytest
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import indices


@pytest.fixture(autouse=True)
def config_aislada(tmp_path, monkeypatch):
    """La asignacion de bandas se guarda en una carpeta de configuracion: cada test usa la suya."""
    monkeypatch.setenv("AGROIA_CONFIG_DIR", str(tmp_path / "cfg"))
    indices._ASIGNADAS.clear()
    yield
    indices._ASIGNADAS.clear()


def _cinco_bandas(path, nir_mayor_que_re=True):
    """5 bandas SIN nombre, como un export de Pix4D: valores de vegetacion en el orden B, G, R, RedEdge, NIR."""
    d = np.zeros((5, 20, 20), "int32")
    d[0], d[1], d[2] = 280, 570, 490            # azul, verde, rojo
    d[3], d[4] = (1460, 3490) if nir_mayor_que_re else (3490, 1460)
    with rasterio.open(path, "w", driver="GTiff", width=20, height=20, count=5, dtype="int32",
                       crs="EPSG:32718", transform=from_origin(0, 20, 1, 1), nodata=-2147483648) as s:
        s.write(d)
    return path


def test_necesita_asignacion(tmp_path):
    p = _cinco_bandas(str(tmp_path / "a.tif"))
    with rasterio.open(p) as s:
        assert indices.detectar_bandas(s) == {} and indices.necesita_asignacion(s) is True
    # con nombres de banda no hace falta
    q = str(tmp_path / "n.tif")
    with rasterio.open(q, "w", driver="GTiff", width=4, height=4, count=5, dtype="uint16") as s:
        s.write(np.ones((5, 4, 4), "uint16"))
        s.descriptions = ["Blue", "Green", "Red", "Rededge", "NIR"]
    with rasterio.open(q) as s:
        assert indices.necesita_asignacion(s) is False
    # RGB de 3 bandas tampoco
    r = str(tmp_path / "rgb.tif")
    with rasterio.open(r, "w", driver="GTiff", width=4, height=4, count=3, dtype="uint8") as s:
        s.write(np.ones((3, 4, 4), "uint8"))
    with rasterio.open(r) as s:
        assert indices.necesita_asignacion(s) is False
    # 4 bandas con alpha marcado en colorinterp: no; 4 sin alpha (p. ej. RGB + NIR): si
    ci = lambda n: NS(name=n)
    con_alpha = NS(name=None, count=4, descriptions=(None,) * 4, colorinterp=[ci("red"), ci("green"), ci("blue"), ci("alpha")])
    sin_alpha = NS(name=None, count=4, descriptions=(None,) * 4, colorinterp=[ci("gray")] * 4)
    assert indices.necesita_asignacion(con_alpha) is False and indices.necesita_asignacion(sin_alpha) is True


def test_sugerencia_segun_los_valores_medios(tmp_path):
    nombre, mapeo, medias = indices.sugerir_asignacion(_cinco_bandas(str(tmp_path / "a.tif"), nir_mayor_que_re=True))
    assert nombre.startswith("Azul, Verde, Rojo, RedEdge, NIR") and mapeo == dict(blue=1, green=2, red=3, rededge=4, nir=5)
    assert medias == pytest.approx([280, 570, 490, 1460, 3490])
    nombre, mapeo, _ = indices.sugerir_asignacion(_cinco_bandas(str(tmp_path / "b.tif"), nir_mayor_que_re=False))
    assert "MicaSense" in nombre and mapeo["nir"] == 4 and mapeo["rededge"] == 5


def test_registrar_usa_la_asignacion_y_se_recuerda(tmp_path):
    p = _cinco_bandas(str(tmp_path / "a.tif"))
    mapeo = indices.registrar_bandas(p, dict(blue=1, green=2, red=3, rededge=4, nir=5))
    with rasterio.open(p) as s:
        assert indices.detectar_bandas(s) == mapeo and indices.necesita_asignacion(s) is False
        assert indices.bandas_rgb(s) == (3, 2, 1)                # el conteo lee R, G, B en el orden correcto
    # "otra sesion": se vacia la memoria y se vuelve a leer del archivo de configuracion del usuario
    indices._ASIGNADAS.clear()
    assert indices.bandas_registradas(p) == mapeo
    # si el archivo cambia (otro tamano), la asignacion guardada se descarta
    indices._ASIGNADAS.clear()
    with open(p, "ab") as f:
        f.write(b"x" * 10)
    assert indices.bandas_registradas(p) is None
    with pytest.raises(ValueError):
        indices.registrar_bandas(p, dict(red=1, nir=1))        # una banda con dos roles


def test_indices_sobre_5_bandas_asignadas(tmp_path):
    p = _cinco_bandas(str(tmp_path / "a.tif"))
    with pytest.raises(ValueError):                              # sin asignar no se puede
        indices.calcular_indices(p, str(tmp_path / "o0"), modo="multi", log=lambda *_: None)
    indices.registrar_bandas(p, dict(blue=1, green=2, red=3, rededge=4, nir=5))
    r = indices.calcular_indices(p, str(tmp_path / "o"), modo="multi", log=lambda *_: None)
    assert r["indices"] == ["ndvi", "ndre"]
    with rasterio.open(r["rutas"]["ndvi"]) as s:
        assert s.read(1)[5, 5] == pytest.approx((3490 - 490) / (3490 + 490), abs=1e-4)
    with rasterio.open(r["rutas"]["ndre"]) as s:
        assert s.read(1)[5, 5] == pytest.approx((3490 - 1460) / (3490 + 1460), abs=1e-4)
    r = indices.calcular_indices(p, str(tmp_path / "o2"), modo="rgb", log=lambda *_: None)   # RGB con las bandas visibles
    assert r["indices"] == ["exg", "gli", "vari"]


def _tk_o_skip():
    import time
    tk = pytest.importorskip("tkinter")
    err = None
    for _ in range(4):
        try:
            return tk.Tk()
        except tk.TclError as e:
            err = e
            time.sleep(0.4)
    pytest.skip(f"sin entorno grafico: {err}")


def test_ventana_asignar_bandas(tmp_path, monkeypatch):
    from tkinter import messagebox
    mensajes = []
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: mensajes.append(a[0]))
    import asignar_bandas
    p = _cinco_bandas(str(tmp_path / "a.tif"))
    root = _tk_o_skip()
    root.withdraw()
    dlg = asignar_bandas.AsignarBandas(root, p)
    dlg.update()
    assert dlg.roles() == dict(blue=1, green=2, red=3, rededge=4, nir=5)        # sugerencia precargada
    # roles repetidos -> error y no se registra
    dlg.vars[4].set(dlg.vars[2].get())
    assert dlg.aceptar() is None and mensajes == ["Roles repetidos"] and indices.bandas_registradas(p) is None
    # sin rojo+nir ni rgb -> error
    dlg.set_roles(dict(blue=1, green=2))
    assert dlg.aceptar() is None and mensajes[-1] == "Faltan bandas"
    # cambio de sugerencia y aceptar
    dlg.aplicar_preset("Azul, Verde, Rojo, NIR, RedEdge (orden MicaSense)")
    assert dlg.roles()["nir"] == 4
    assert dlg.aceptar() == dict(blue=1, green=2, red=3, nir=4, rededge=5)
    assert indices.bandas_registradas(p) == dict(blue=1, green=2, red=3, nir=4, rededge=5) and dlg.result
    root.destroy()
