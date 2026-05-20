"""
build_exe.py — Genera el ejecutable de COVIO v2.0 con PyInstaller
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
    "--name", "COVIO_2.0",
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

print("Construyendo ejecutable COVIO 2.0...")
subprocess.check_call(cmd)
print("\n✓ Ejecutable en: dist/COVIO_2.0/COVIO_2.0.exe")
