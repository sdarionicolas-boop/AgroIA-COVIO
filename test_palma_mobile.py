
import os
import threading
from pipeline import run_full_pipeline

# Configuración de la prueba para Palma Aceitera (GSD alto, objetos grandes)
params = {
    "gsd":          0.13,
    "gsd_cm":       13.0,
    "tile":         1024,
    "conf":         0.15,      # Umbral bajo para compensar MobileSAM
    "nms_dist":     650.0 / 13.0, # 6.5 metros de marco
    "nms_cm":       650.0,
    "min_size":     260.0 / 13.0, # 2.6 metros mín
    "min_cm":       260.0,
    "max_size":     1040.0 / 13.0,# 10.4 metros máx
    "max_cm":       1040.0,
    "epochs":       5,
    "grid_size":    10,
    "sam_fraction": 1.0,
    "sam_type":     "mobile",
    "lote":         "PALMA_TEST_MOBILE",
}

ortho_path = r"E:\Desktop_Movidos\PlantDetector\PALMA\RGB_13cm.tif"
models_dir = r"E:\Desktop_Movidos\PlantDetector\models"
output_dir = r"E:\Desktop_Movidos\PlantDetector\output\PALMA_TEST_MOBILE"

os.makedirs(output_dir, exist_ok=True)

def dummy_log(msg, tag="info"):
    print(f"[{tag.upper()}] {msg}")

def dummy_status(msg):
    print(f"STATUS: {msg}")

def dummy_progress(val):
    pass

print("--- INICIANDO TEST PALMA CON MOBILESAM ---")
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
    print(f"Palmas detectadas: {result['n_plants']}")
except Exception as e:
    print(f"ERROR EN EL TEST: {e}")
