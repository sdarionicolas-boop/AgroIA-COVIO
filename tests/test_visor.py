import os, sys
import numpy as np
import pytest
rasterio = pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import indices, lote, visor_resultados


def _rgb(tmp_path):
    d = np.zeros((3, 200, 200), "uint8")
    d[0], d[1], d[2] = 50, 150, 50
    p = str(tmp_path / "rgb.tif")
    with rasterio.open(p, "w", driver="GTiff", width=200, height=200, count=3, dtype="uint8",
                       crs="EPSG:32723", transform=from_origin(500000, 7000100, 0.5, 0.5)) as s:
        s.write(d)
    return p


def test_cargar_carpeta_sin_lotes(tmp_path):
    out = str(tmp_path / "o")
    indices.calcular_indices(_rgb(tmp_path), out, log=lambda *_: None)
    r = visor_resultados.cargar_carpeta(out)
    cols, datos = r["tabla"]
    assert cols == ["Índice", "Media", "P2", "P98"] and [x[0] for x in datos] == ["ExG", "GLI", "VARI"]
    assert r["indices_png"] and r["mapa"] is None and "RGB" in r["notas"]


def test_cargar_carpeta_con_lotes(tmp_path):
    out = str(tmp_path / "o")
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A", "B"]}, geometry=[box(500000, 7000050, 500050, 7000100), box(500050, 7000000, 500100, 7000050)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    lote.analizar_lotes(_rgb(tmp_path), poly, out, grid_m=20, umbral=0.3, log=lambda *_: None)
    r = visor_resultados.cargar_carpeta(out)
    cols, datos = r["tabla"]
    assert cols == ["Métrica", "A", "B"] and r["mapa"] and r["gpkg"] and r["indices_png"]
    etiquetas = [f[0] for f in datos]
    assert "Cobertura vegetal (%)" in etiquetas and "ExG media" in etiquetas and "Nº de fallas" in etiquetas
    assert "ExG" in r["notas"]


def test_visor_se_abre(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("sin entorno grafico")
    root.withdraw()
    out = str(tmp_path / "o")
    indices.calcular_indices(_rgb(tmp_path), out, log=lambda *_: None)
    v = visor_resultados.VisorResultados(root, out)
    v.update()
    assert len(v.winfo_children()) > 0
    v.destroy()
    root.destroy()


def _analizar(tmp_path):
    out = str(tmp_path / "o")
    poly = str(tmp_path / "p.gpkg")
    gpd.GeoDataFrame({"nombre": ["A"]}, geometry=[box(500000, 7000000, 500100, 7000100)],
                     crs="EPSG:32723").to_file(poly, driver="GPKG")
    lote.analizar_lotes(_rgb(tmp_path), poly, out, grid_m=20, umbral=0.3, log=lambda *_: None)
    return out


def test_mapa_lote_datos_y_valores(tmp_path):
    import mapa_lote
    out = _analizar(tmp_path)
    d = mapa_lote.cargar_datos(out)
    assert d["vigor"] == "gli" and set(d["tifs"]) == {"exg", "gli"} and len(d["grilla"]) == 25
    assert mapa_lote.cargar_datos(str(tmp_path / "vacio")) is None
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("sin entorno grafico")
    root.geometry("900x600")
    f = tk.Frame(root); f.pack(fill="both", expand=True)
    m = mapa_lote.MapaLote(f, out)
    m.pack(fill="both", expand=True)
    root.update()
    m._ajustar()
    z0 = m.z
    m._zoom_en(2.0, 200, 200)
    assert m.z == pytest.approx(z0 * 2)
    # hover en el centro: una celda con sus valores
    col, row = m.inv * (500050, 7000050)
    ev = type("E", (), {"x": int((col - m.ox) * m.z), "y": int((row - m.oy) * m.z)})()
    m._on_move(ev)
    assert "GLI" in m.info.get() and "EXG" not in m.info.get()
    for k in ("exg", "gli"):
        m.fondo.set(k)
        m._dibujar()
    m.v_clases.set(True)
    m._dibujar()
    root.destroy()
