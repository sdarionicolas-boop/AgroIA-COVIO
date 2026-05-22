"""
build_exe.py — Genera el ejecutable de AgroIA-COVIO v2.1 con PyInstaller
Autor: Darío Sánchez Leguizamón

Copyright (c) 2026 Darío Sánchez Leguizamón

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional Terms (Section 7b of GNU GPL v3.0):
Any interactive user interface (including GUIs, CLIs, and web apps)
derived from or using this Program must prominently display the
attribution: "AgroIA-COVIO — desarrollado por Darío Sánchez Leguizamón"
"""
import subprocess
import sys
import os

try:
    import PyInstaller
except ImportError:
    print("Instalando PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",
    "--windowed",
    "--name", "AgroIA-COVIO",
    "--add-data", f"pipeline.py{os.pathsep}.",
    # rasterio
    "--hidden-import", "rasterio._shim",
    "--hidden-import", "rasterio.control",
    "--hidden-import", "rasterio.crs",
    "--hidden-import", "rasterio.transform",
    "--collect-all", "rasterio",
    # geopandas + motores de escritura
    "--hidden-import", "geopandas",
    "--hidden-import", "geopandas.io.file",
    "--collect-all", "geopandas",
    "--hidden-import", "pyogrio",
    "--hidden-import", "pyogrio._geometry",
    "--hidden-import", "pyogrio._lib",
    "--collect-all", "pyogrio",
    "--hidden-import", "fiona",
    "--hidden-import", "fiona.ogrext",
    "--collect-all", "fiona",
    # pyproj / shapely
    "--hidden-import", "pyproj",
    "--collect-all", "pyproj",
    "--hidden-import", "shapely",
    "--collect-all", "shapely",
    # ultralytics + SAM + MobileSAM
    "--hidden-import", "ultralytics",
    "--collect-all", "ultralytics",
    "--collect-all", "segment_anything",
    "--collect-all", "mobile_sam",
    "--hidden-import", "timm",
    "--collect-all", "timm",
    # otros
    "--hidden-import", "cv2",
    "--hidden-import", "sklearn",
    "--hidden-import", "scipy",
    "--hidden-import", "PIL",
    "--hidden-import", "PIL.ImageTk",
    "app.py",
]

print("Construyendo ejecutable AgroIA-COVIO v2.1...")
subprocess.check_call(cmd)
print("\n✓ Ejecutable en: dist/AgroIA-COVIO/AgroIA-COVIO.exe")
