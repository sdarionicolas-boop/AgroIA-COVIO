import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_base_dir_junto_al_exe_cuando_esta_congelado(tmp_path, monkeypatch):
    """En el .exe (PyInstaller), models/ y output/ quedan junto al ejecutable y no dentro de _internal."""
    pytest.importorskip("tkinter")
    exe = tmp_path / "AgroIA-COVIO" / "AgroIA-COVIO.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    import app
    try:
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))
        importlib.reload(app)
        assert app.BASE_DIR == str(exe.parent)
        assert app.MODELS_DIR == str(exe.parent / "models") and os.path.isdir(app.MODELS_DIR)
        assert "_internal" not in app.BASE_DIR
    finally:
        monkeypatch.undo()
        importlib.reload(app)
    assert app.BASE_DIR == os.path.dirname(os.path.abspath(app.__file__))      # sin congelar: igual que antes
