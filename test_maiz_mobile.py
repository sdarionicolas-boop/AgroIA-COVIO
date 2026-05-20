
import os
import threading
from pipeline import run_full_pipeline

# Configuración de la prueba
params = {
    "gsd":          0.015,
    "gsd_cm":       1.5,
    "tile":         1024,
    "conf":         0.15,      # Bajamos un poco para MobileSAM
    "nms_dist":     37.5 / 1.5,
    "nms_cm":       37.5,
    "min_size":     3.0 / 1.5,
    "min_cm":       3.0,
    "max_size":     120.0 / 1.5,
    "max_cm":       120.0,
    "epochs":       5,         # 5 épocas para rapidez en el test
    "grid_size":    10,
    "sam_fraction": 1.0,       # 100% para compensar
    "sam_type":     "mobile",
    "lote":         "MAIZ_TEST_MOBILE",
}

ortho_path = r"E:\Desktop_Movidos\PlantDetector\MAIZ\mosaico.tif"
models_dir = r"E:\Desktop_Movidos\PlantDetector\models"
output_dir = r"E:\Desktop_Movidos\PlantDetector\output\MAIZ_TEST_MOBILE"

os.makedirs(output_dir, exist_ok=True)

def dummy_log(msg, tag="info"):
    print(f"[{tag.upper()}] {msg}")

def dummy_status(msg):
    print(f"STATUS: {msg}")

def dummy_progress(val):
    pass

print("--- INICIANDO TEST MAIZ CON MOBILESAM ---")
try:
    result = run_full_pipeline(
        ortho_path=ortho_path,
        models_dir=models_dir,
        output_dir=output_dir,
        params=params,
        log_fn=dummy_log,
        progress_fn=dummy_progress,
        status_fn=dummy_status,
        export_shp=False,
        stop_event=threading.Event()
    )
    print("\n--- TEST FINALIZADO ---")
    print(f"Plantas detectadas: {result['n_plants']}")
except Exception as e:
    print(f"ERROR EN EL TEST: {e}")
