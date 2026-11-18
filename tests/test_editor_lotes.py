import os, sys
import numpy as np
import pytest
rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lotes_io


@pytest.fixture
def mosaico(tmp_path):
    """100x100 m a 0.5 m/px, 6 bandas (R,G,B,NIR,RE,alpha), esquina superior izquierda en (500000, 7000100)."""
    n = 200
    d = np.zeros((6, n, n), "uint16")
    d[0], d[1], d[2], d[3], d[4], d[5] = 100, 120, 90, 300, 200, 255
    p = str(tmp_path / "m.tif")
    with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=6, dtype="uint16",
                       crs="EPSG:32723", transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
        s.descriptions = ["Red", "Green", "Blue", "NIR", "Rededge", "alpha"]
    return p


@pytest.mark.parametrize("modo", ["rgb", "ndvi", "ndre"])
def test_vista_previa_modos_y_transformada(mosaico, modo):
    v = lotes_io.vista_previa(mosaico, modo, max_px=100)           # reduce 2x
    assert v["img"].shape == (100, 100, 3) and v["img"].dtype == np.uint8
    x, y = v["transform"] * (0, 0)                                   # esquina superior izquierda
    assert (x, y) == pytest.approx((500000, 7000100))
    x, y = v["transform"] * (100, 100)                               # esquina inferior derecha
    assert (x, y) == pytest.approx((500100, 7000000))


def test_guardar_y_cargar_poligonos_ida_y_vuelta(mosaico, tmp_path):
    v = lotes_io.vista_previa(mosaico, "rgb", max_px=100)
    pts = [(10, 10), (60, 10), (60, 50), (10, 50)]                   # 50 x 40 px de vista = 50 x 40 m
    out = str(tmp_path / "l.gpkg")
    lotes_io.guardar_poligonos([pts], ["A"], v["transform"], v["crs"], out)
    g = gpd.read_file(out)
    assert g.geometry.iloc[0].area == pytest.approx(50 * 40)          # 1 px de vista = 1 m
    nombres, polys = lotes_io.cargar_poligonos_px(out, v["transform"], v["crs"])
    assert nombres == ["A"]
    assert np.allclose(sorted(polys[0]), sorted(pts), atol=1e-6)


def test_poligono_con_menos_de_3_vertices_falla(mosaico, tmp_path):
    v = lotes_io.vista_previa(mosaico, "rgb", max_px=100)
    with pytest.raises(ValueError):
        lotes_io.guardar_poligonos([[(0, 0), (1, 1)]], ["x"], v["transform"], v["crs"], str(tmp_path / "l.gpkg"))


def _tk():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("sin entorno grafico")
    root.withdraw()
    return root


def test_editor_dibuja_cierra_y_guarda(mosaico, tmp_path):
    root = _tk()
    import editor_lotes
    guardados = []
    out = str(tmp_path / "o" / "lotes.gpkg")
    ed = editor_lotes.EditorLotes(root, mosaico, out, on_saved=guardados.append)
    ed._pedir_nombre = lambda defecto: "Lote Norte"
    ed.update()
    # cerrar con menos de 3 vertices no crea nada
    ed.agregar_vertice(10, 10); ed.agregar_vertice(40, 10)
    assert ed.cerrar_poligono() is False and not ed.polys
    ed.agregar_vertice(40, 40); ed.agregar_vertice(10, 40)
    ed.deshacer_vertice(); ed.agregar_vertice(10, 40)                 # deshacer y rehacer
    assert ed.cerrar_poligono() is True and ed.polys[0]["nombre"] == "Lote Norte"
    # un segundo poligono en curso se cierra solo al guardar
    for p in [(60, 60), (90, 60), (90, 90)]:
        ed.agregar_vertice(*p)
    ed._pedir_nombre = lambda defecto: "Lote Sur"
    ed.guardar()
    assert guardados == [out] and os.path.exists(out)
    g = gpd.read_file(out)
    assert list(g["nombre"]) == ["Lote Norte", "Lote Sur"] and g.crs.to_epsg() == 32723
    root.destroy()


def test_editor_recarga_poligonos_existentes(mosaico, tmp_path):
    root = _tk()
    import editor_lotes
    v = lotes_io.vista_previa(mosaico, "rgb")
    ex = str(tmp_path / "ex.gpkg")
    lotes_io.guardar_poligonos([[(5, 5), (50, 5), (50, 50)]], ["Previo"], v["transform"], v["crs"], ex)
    ed = editor_lotes.EditorLotes(root, mosaico, str(tmp_path / "nuevo.gpkg"), existing_path=ex)
    assert [p["nombre"] for p in ed.polys] == ["Previo"]
    root.destroy()
