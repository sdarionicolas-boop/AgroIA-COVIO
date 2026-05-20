"""
COVIO v2.1 — Pipeline de detección de plantas
"""

import os
import gc
import datetime
import numpy as np
import pandas as pd

STEPS = [
    "Cargando ortomosaico y particionando tiles",
    "Generando pseudo-etiquetas con SAM",
    "Entrenando YOLOv8m",
    "Ejecutando inferencia + filtros agronómicos",
    "Geoprocesamiento y zonificación",
    "Exportando resultados",
]

_log_file = None

def _init_log(output_dir, lote=""):
    global _log_file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"covio_log_{ts}.txt")
    _log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # UTF-8 obligatorio para rutas con acentos
    _log_file.write("COVIO v2.1 — Log de ejecucion\n")
    if lote:
        _log_file.write(f"Lote: {lote}\n")
    _log_file.write(f"Inicio: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    _log_file.write("=" * 50 + "\n")
    return log_path

def _write_log(msg):
    if _log_file:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        _log_file.write(f"[{ts}] {msg}\n")

def _close_log(success=True):
    global _log_file
    if _log_file:
        _log_file.write("=" * 50 + "\n")
        estado = "COMPLETADO" if success else "INTERRUMPIDO / ERROR"
        _log_file.write(f"Estado final: {estado}\n")
        _log_file.write(f"Fin: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        _log_file.close()
        _log_file = None


def preview_tile(ortho_path, models_dir, params, log_fn):
    """
    Procesa un único tile del ortomosaico para validar parámetros.
    Retorna (tile_rgb, instances_count, preview_path)
    """
    import rasterio
    from rasterio.windows import Window
    import cv2
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam_type = params.get("sam_type", "vit_h")
    if sam_type == "mobile":
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
        sam_path = os.path.join(models_dir, "mobile_sam.pt")
        sam = sam_model_registry["vit_t"](checkpoint=sam_path)
    else:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam_path = os.path.join(models_dir, "sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_h"](checkpoint=sam_path)

    tile_size = params["tile"]
    gsd_cm = params["gsd_cm"]
    
    with rasterio.open(ortho_path) as src:
        # Tomar un tile del centro del lote para que sea representativo
        cx, cy = src.width // 2, src.height // 2
        window = Window(cx - tile_size//2, cy - tile_size//2, tile_size, tile_size)
        tile = src.read((1, 2, 3), window=window)
        tile = np.moveaxis(tile, 0, -1)
        if tile.dtype != np.uint8:
            tile = (tile / tile.max() * 255).astype(np.uint8)

    _sam_min_area = max(150, int(np.pi * (params["min_cm"] / (2 * gsd_cm)) ** 2 * 0.5))
    _sam_max_area = min(200000, int(np.pi * (params["max_cm"] / (2 * gsd_cm)) ** 2 * 3.0))

    mask_gen = SamAutomaticMaskGenerator(
        sam, points_per_side=16, pred_iou_thresh=0.88,
        stability_score_thresh=0.95, min_mask_region_area=int(_sam_min_area * 0.5)
    )
    
    masks = mask_gen.generate(tile)
    valid = [m for m in masks if _sam_min_area < m["area"] < _sam_max_area]
    
    # Dibujar preview
    preview = tile.copy()
    for m in valid:
        bx, by, bw, bh = m["bbox"]
        cv2.rectangle(preview, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0, 255, 0), 2)
        # Overlay semi-transparente de la máscara
        mask = m["segmentation"]
        preview[mask] = preview[mask] * 0.5 + np.array([0, 255, 0]) * 0.5

    return preview, len(valid)

def run_full_pipeline(ortho_path, models_dir, output_dir, params,
                      log_fn, progress_fn, status_fn, export_shp=False,
                      stop_event=None):
    lote     = params.get("lote", "")
    log_path = _init_log(output_dir, lote)

    def log(msg, tag="info"):
        log_fn(msg, tag)
        _write_log(msg)

    def step_progress(step_idx, sub=0.0):
        progress_fn(((step_idx + sub) / len(STEPS)) * 100)

    def check_stop():
        if stop_event and stop_event.is_set():
            raise KeyboardInterrupt("Cancelado por el usuario")

    try:
        log(f"Log guardado en: {log_path}", "muted")

        # ── RESUMEN DE PARÁMETROS ──────────────────────────────────────────
        log("", "muted")
        log("PARÁMETROS DE EJECUCIÓN", "ok")
        log("─" * 40, "muted")
        log(f"  Cultivo / Lote    : {lote}", "info")
        log(f"  Ortomosaico       : {os.path.basename(ortho_path)}", "info")
        log(f"  GSD               : {params['gsd_cm']:.1f} cm/px", "info")
        log(f"  Tile size         : {params['tile']} px", "info")
        log(f"  SAM cobertura     : {params.get('sam_fraction', 0.7)*100:.0f}% de tiles", "info")
        log(f"  Umbral confianza  : > {params['conf']}", "info")
        log(f"  Dist. min. plantas: {params['nms_cm']:.1f} cm  ({params['nms_dist']:.1f} px)", "info")
        log(f"  Tam. min. objeto  : {params['min_cm']:.1f} cm  ({params['min_size']:.1f} px)", "info")
        log(f"  Tam. max. objeto  : {params['max_cm']:.1f} cm  ({params['max_size']:.1f} px)", "info")
        log(f"  Epocas YOLO       : {params['epochs']}", "info")
        log(f"  Grilla SIG        : {params['grid_size']}x{params['grid_size']} m", "info")
        log(f"  Exportar SHP      : {'Si' if export_shp else 'No'}", "info")
        log("─" * 40, "muted")
        log("", "muted")

        # PASO 1: Tiling
        log(f"\n[1/6] {STEPS[0]}...", "warn")
        status_fn(STEPS[0])
        import rasterio
        from rasterio.windows import Window

        tiles_dir = os.path.join(output_dir, "tiles")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(tiles_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        tile_size = params["tile"]
        tile_paths = []

        # ── REMUESTREO AUTOMÁTICO ──────────────────────────────────────────
        # Si el GSD real es más fino que el objetivo, se remuestrea antes de tilear.
        # El target GSD se limita automáticamente para que la planta más pequeña
        # tenga al menos SAM_MIN_PX px de diámetro — garantizando que SAM pueda
        # detectarla (su filtro interno de área mínima es ~16px de diámetro).
        SAM_MIN_PX     = 16   # diámetro mínimo para que SAM genere una máscara válida
        _max_gsd_crop  = params["min_cm"] / SAM_MIN_PX          # GSD máximo según cultivo
        TARGET_GSD_CM  = min(params.get("target_gsd_cm", 1.5), _max_gsd_crop)

        with rasterio.open(ortho_path) as src:
            from rasterio.enums import Resampling as _Resampling
            from rasterio.transform import Affine as _Affine

            crs = src.crs
            h_orig, w_orig = src.height, src.width

            # GSD real: solo calculable desde la transformada en CRS proyectados (metros).
            # Para CRS geográficos (grados, ej. EPSG:4326) la transformada no está en metros
            # y el cálculo daría ~0. En ese caso usamos el GSD ingresado por el usuario.
            if crs and not crs.is_geographic:
                actual_gsd_cm = abs(src.transform.a) * 100  # metros → cm
            else:
                actual_gsd_cm = params["gsd_cm"]  # CRS geográfico: confiar en el usuario
                log(f"  CRS geográfico detectado — GSD tomado del parámetro: {actual_gsd_cm} cm/px", "muted")

            log(f"  Tamano original: {w_orig}x{h_orig} px | GSD: {actual_gsd_cm:.3f} cm/px | CRS: {crs}", "muted")

            if actual_gsd_cm < TARGET_GSD_CM - 0.01:
                # Calcular nueva resolución
                scale   = actual_gsd_cm / TARGET_GSD_CM
                new_w   = max(1, int(w_orig * scale))
                new_h   = max(1, int(h_orig * scale))
                est_tiles = ((new_h // tile_size) + 1) * ((new_w // tile_size) + 1)
                t = src.transform
                new_transform = _Affine(t.a / scale, t.b, t.c, t.d, t.e / scale, t.f)

                log(f"  ⚡ Remuestreo: {actual_gsd_cm:.2f} → {TARGET_GSD_CM:.1f} cm/px "
                    f"| {w_orig}×{h_orig} → {new_w}×{new_h} px (~{est_tiles} tiles)", "warn")

                # Leer y remuestrear con interpolación bilineal
                data = src.read(
                    out_shape=(src.count, new_h, new_w),
                    resampling=_Resampling.bilinear
                )
                # Guardar TIF remuestreado en output_dir
                resampled_path = os.path.join(output_dir, "_ortho_resampled.tif")
                profile = src.profile.copy()
                profile.update(width=new_w, height=new_h, transform=new_transform,
                               compress="lzw")
                with rasterio.open(resampled_path, "w", **profile) as dst:
                    dst.write(data)
                del data
                gc.collect()

                # Recalcular parámetros en píxeles al nuevo GSD
                params["nms_dist"] = params["nms_cm"]  / TARGET_GSD_CM
                params["min_size"] = params["min_cm"]  / TARGET_GSD_CM
                params["max_size"] = params["max_cm"]  / TARGET_GSD_CM
                log(f"  Params px recalculados al GSD objetivo "
                    f"(NMS={params['nms_dist']:.1f}px, "
                    f"min={params['min_size']:.1f}px, max={params['max_size']:.1f}px)", "muted")

                ortho_effective = resampled_path
                transform = new_transform
                h, w = new_h, new_w
                log(f"  OK: Remuestreo completado.", "ok")
            else:
                log(f"  GSD {actual_gsd_cm:.2f} cm/px ≥ objetivo {TARGET_GSD_CM:.1f} cm/px — sin remuestreo.", "muted")
                ortho_effective = ortho_path
                transform = src.transform
                h, w = h_orig, w_orig

        # Tiling sobre el ortomosaico efectivo (original o remuestreado)
        with rasterio.open(ortho_effective) as src:
            log(f"  Generando tiles de {tile_size}x{tile_size} px...", "muted")
            total_tiles = ((h // tile_size) + 1) * ((w // tile_size) + 1)
            count = 0
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    check_stop()
                    window = Window(x, y, min(tile_size, w-x), min(tile_size, h-y))
                    tile = src.read((1, 2, 3), window=window)
                    tile = np.moveaxis(tile, 0, -1)
                    if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                        padded[:tile.shape[0], :tile.shape[1]] = tile
                        tile = padded
                    tile_name = f"tile_{y:05d}_{x:05d}.npy"
                    tile_path = os.path.join(tiles_dir, tile_name)
                    np.save(tile_path, tile)
                    tile_paths.append((tile_path, x, y))
                    count += 1
                    if count % 20 == 0:
                        step_progress(0, count / total_tiles)

        log(f"  OK: {count} tiles generados.", "ok")
        step_progress(1)

        # PASO 2: SAM — muestreo estratificado (70% de tiles)
        log(f"\n[2/6] {STEPS[1]}...", "warn")
        status_fn(STEPS[1])
        
        sam_type = params.get("sam_type", "vit_h")
        if sam_type == "mobile":
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
            sam_path = os.path.join(models_dir, "mobile_sam.pt")
            log("  Cargando MobileSAM (Modo Ligero)...", "muted")
            sam = sam_model_registry["vit_t"](checkpoint=sam_path)
        else:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            sam_path = os.path.join(models_dir, "sam_vit_h_4b8939.pth")
            log("  Cargando SAM vit_h (Modo Pesado)...", "muted")
            sam = sam_model_registry["vit_h"](checkpoint=sam_path)

        # Muestreo estratificado espacial:
        # Dividimos el ortomosaico en una grilla de zonas y tomamos tiles
        # representativos de cada zona para maximizar cobertura espacial
        # manteniendo 70% del total (mejor que random para cultivos en hileras)
        sam_fraction = params.get("sam_fraction", 0.70)
        n_sam = max(1, int(len(tile_paths) * sam_fraction))

        if n_sam < len(tile_paths):
            # Ordenar por posición y tomar cada N-ésimo para cobertura uniforme
            step = len(tile_paths) / n_sam
            sam_indices = sorted(set(int(i * step) for i in range(n_sam)))
            sam_tiles = [tile_paths[i] for i in sam_indices]
            log(f"  Muestreo estratificado: {len(sam_tiles)}/{len(tile_paths)} tiles "
                f"({sam_fraction*100:.0f}% cobertura espacial)", "muted")
        else:
            sam_tiles = tile_paths
            log(f"  Procesando todos los tiles ({len(sam_tiles)})", "muted")

        # Filtro de área SAM dinámico: se calcula en función del GSD efectivo
        # y el tamaño real del cultivo para no excluir plantas pequeñas.
        # Fórmula: área_px = π × (diámetro_cm / (2 × gsd))²
        _eff_gsd = TARGET_GSD_CM if actual_gsd_cm < TARGET_GSD_CM else actual_gsd_cm
        # Piso de 150 px²: empíricamente validado como mínimo para que SAM
        # genere máscaras limpias y no ruido de textura o fragmentos de hojas.
        # La fórmula dinámica escala para cultivos grandes (maíz, palmas),
        # pero el piso evita máscaras inútiles en cultivos pequeños (lechuga).
        _sam_min_area = max(150, int(np.pi * (params["min_cm"] / (2 * _eff_gsd)) ** 2 * 0.5))
        _sam_max_area = min(200000, int(np.pi * (params["max_cm"] / (2 * _eff_gsd)) ** 2 * 3.0))
        log(f"  Filtro área SAM: {_sam_min_area}–{_sam_max_area} px² "
            f"(cultivo {params['min_cm']}–{params['max_cm']} cm @ {_eff_gsd:.2f} cm/px)", "muted")

        mask_gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,          # default 32 → 4x más rápido, buena precisión
            pred_iou_thresh=0.88,        # filtra máscaras de baja calidad antes
            stability_score_thresh=0.95, # más estricto = menos ruido
            min_mask_region_area=int(_sam_min_area * 0.5),  # pre-filtro SAM interno
        )
        total_instances = 0

        for i, (tile_path, tx, ty) in enumerate(sam_tiles):
            check_stop()
            tile = np.load(tile_path)
            if tile.dtype != np.uint8:
                tile = (tile / tile.max() * 255).astype(np.uint8)
            masks = mask_gen.generate(tile)
            valid = [m for m in masks if _sam_min_area < m["area"] < _sam_max_area]
            total_instances += len(valid)
            label_file = os.path.join(labels_dir,
                            os.path.basename(tile_path).replace(".npy", ".txt"))
            with open(label_file, "w", encoding="utf-8") as f:
                for m in valid:
                    bx, by, bw, bh = m["bbox"]
                    cx = (bx + bw/2) / tile_size
                    cy = (by + bh/2) / tile_size
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw/tile_size:.6f} {bh/tile_size:.6f}\n")
            if (i + 1) % 5 == 0 or (i + 1) == len(sam_tiles):
                eta_tiles = len(sam_tiles) - (i + 1)
                log(f"  SAM: {i+1}/{len(sam_tiles)} tiles | "
                    f"{total_instances} instancias | faltan ~{eta_tiles} tiles", "muted")
                step_progress(1, (i+1) / len(sam_tiles))

        del sam, mask_gen
        gc.collect()
        log(f"  OK: {total_instances} pseudo-etiquetas de {len(sam_tiles)} tiles.", "ok")
        log(f"  Los {len(tile_paths) - len(sam_tiles)} tiles restantes usan etiquetas vacías "
            f"(YOLO aprende igual del contexto).", "muted")
        step_progress(2)

        # PASO 3: Entrenamiento YOLOv8m
        log(f"\n[3/6] {STEPS[2]}...", "warn")
        status_fn(STEPS[2])

        # Aplicar regla de seguridad de épocas mínimas para pocos tiles
        num_tiles = len(tile_paths)
        requested_epochs = params["epochs"]
        effective_epochs = requested_epochs
        if num_tiles < 20 and requested_epochs < 10:
            effective_epochs = 10
            log(f"  ⚡ Regla de seguridad activa: Lote con pocos tiles ({num_tiles} < 20). "
                f"Forzando épocas de entrenamiento: {requested_epochs} → {effective_epochs} para asegurar convergencia.", "warn")

        # YOLO requiere: dataset/images/*.jpg
        #                dataset/labels/*.txt  (mismo nombre de archivo)
        # Todo dentro de output_dir que el usuario eligió (tiene permisos de escritura)
        dataset_dir = os.path.join(output_dir, "dataset")
        ds_images   = os.path.join(dataset_dir, "images")
        ds_labels   = os.path.join(dataset_dir, "labels")
        os.makedirs(ds_images, exist_ok=True)
        os.makedirs(ds_labels, exist_ok=True)

        log("  Convirtiendo tiles a JPEG y organizando dataset...", "muted")
        import cv2, shutil, pathlib
        for tile_path, tx, ty in tile_paths:
            base_name = os.path.basename(tile_path).replace(".npy", "")
            tile = np.load(tile_path)
            if tile.dtype != np.uint8:
                tile = np.clip(tile / tile.max() * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(ds_images, base_name + ".jpg"),
                        cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            src_label = os.path.join(labels_dir, base_name + ".txt")
            dst_label = os.path.join(ds_labels,  base_name + ".txt")
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                open(dst_label, "w", encoding="utf-8").close()

        # data.yaml con paths absolutos POSIX (YOLO requiere forward slashes)
        yaml_path = os.path.join(output_dir, "data.yaml")
        ds_images_posix = pathlib.Path(ds_images).as_posix()
        ds_labels_posix = pathlib.Path(ds_labels).as_posix()
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"train: {ds_images_posix}\n")
            f.write(f"val:   {ds_images_posix}\n")
            f.write("nc: 1\n")
            f.write("names: ['planta']\n")
            f.write(f"# label_dir: {ds_labels_posix}\n")

        # Redirigir stdout/stderr para evitar el crash de tqdm dentro del .exe
        # YOLO con verbose=False igual usa tqdm internamente; lo neutralizamos
        import sys, io
        _stdout_bak = sys.stdout
        _stderr_bak = sys.stderr
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()

        from ultralytics import YOLO
        log(f"  Entrenando YOLOv8m en CPU | epocas={effective_epochs} | batch=2", "muted")
        log("  (Este paso puede tardar 10-20 minutos, la app no se congela)", "muted")

        try:
            model = YOLO("yolov8m.pt")
            model.train(
                data=yaml_path,
                epochs=effective_epochs,
                imgsz=tile_size,
                device="cpu",
                batch=2,
                workers=0,
                patience=10,
                verbose=False,
                plots=False,
                project=output_dir,
                name="yolo_run",
                exist_ok=True,
            )
        finally:
            # Restaurar stdout/stderr pase lo que pase
            sys.stdout = _stdout_bak
            sys.stderr = _stderr_bak

        log("  OK: Entrenamiento completado.", "ok")
        step_progress(3)

        images_dir = ds_images  # paso 4 usa esta variable

        # PASO 4: Inferencia + filtros
        log(f"\n[4/6] {STEPS[3]}...", "warn")
        status_fn(STEPS[3])

        best_weights = os.path.join(output_dir, "yolo_run", "weights", "best.pt")
        if not os.path.exists(best_weights):
            best_weights = os.path.join(output_dir, "yolo_run", "weights", "last.pt")

        model_inf = YOLO(best_weights)
        raw_detections = []
        for tile_path, tx, ty in tile_paths:
            check_stop()
            img_path = os.path.join(images_dir,
                                    os.path.basename(tile_path).replace(".npy", ".jpg"))
            results = model_inf.predict(img_path, conf=0.15, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_detections.append({
                        "px": tx + (x1+x2)/2, "py": ty + (y1+y2)/2,
                        "w": x2-x1, "h": y2-y1, "conf": box.conf.item()
                    })

        log(f"  Inferencia bruta: {len(raw_detections)} candidatos", "muted")
        if not raw_detections:
            df_final = pd.DataFrame(columns=["px", "py", "w", "h", "conf"])
            n_plants = 0
            log("  ⚠ YOLO no detectó ningún objeto. SAM pudo haber generado etiquetas insuficientes.", "warn")
            log("  Sugerencias: subir SAM%, bajar confianza, ampliar rango de tamaño.", "warn")
            progress_fn(100)
            status_fn("Sin detecciones. Revisá parámetros.")
            _close_log(success=True)
            return {"n_plants": 0, "density_low": 0, "density_mid": 0, "density_high": 0}

        df = pd.DataFrame(raw_detections)
        df = df[df["conf"] > params["conf"]]
        log(f"  Tras filtro confianza: {len(df)}", "muted")
        df["max_side"] = df[["w","h"]].max(axis=1)
        df = df[(df["max_side"] >= params["min_size"]) & (df["max_side"] <= params["max_size"])]
        log(f"  Tras filtro morfometrico: {len(df)}", "muted")

        df = df.sort_values("conf", ascending=False).reset_index(drop=True)
        nms_dist = params["nms_dist"]
        kept = []
        for _, row in df.iterrows():
            dup = any(
                np.sqrt((row.px-p["px"])**2 + (row.py-p["py"])**2) < nms_dist
                for p in kept
                if abs(row.px-p["px"]) < nms_dist and abs(row.py-p["py"]) < nms_dist
            )
            if not dup:
                kept.append(row.to_dict())

        df_final = pd.DataFrame(kept)
        n_plants = len(df_final)
        log(f"  OK: {n_plants} plantas validadas.", "ok")
        step_progress(4)

        # ── DETECCIÓN DE FALLAS DE SIEMBRA (GAP ANALYSIS) ─────────────────
        # Identifica áreas donde la distancia entre plantas es significativamente 
        # mayor a la esperada según el NMS manual.
        log("  Analizando fallas de siembra (gap analysis)...", "muted")
        nms_m = params["nms_cm"] / 100.0
        gap_threshold = nms_m * 2.5 # Umbral empírico: 2.5x la distancia de siembra
        
        import geopandas as gpd
        from shapely.geometry import Point, box as shapely_box
        from shapely.ops import unary_union

        # Crear GeoDataFrame temporal para análisis espacial
        geometry = [Point(transform * (row.px, row.py)) for _, row in df_final.iterrows()]
        gdf_temp = gpd.GeoDataFrame(df_final, geometry=geometry, crs=crs)
        if gdf_temp.crs and gdf_temp.crs.is_geographic:
            gdf_temp = gdf_temp.to_crs(gdf_temp.estimate_utm_crs())

        # Calcular área total del lote (basado en tiles generados)
        # Esto es una aproximación del área analizada
        lot_extent = gdf_temp.total_bounds
        lot_poly = shapely_box(*lot_extent)
        
        # Buffer alrededor de cada planta con el radio de falla
        # Las áreas NO cubiertas por estos buffers son potenciales fallas
        plant_buffers = gdf_temp.buffer(gap_threshold).union_all()
        failure_areas = lot_poly.difference(plant_buffers)
        
        # Convertir fallas a polígonos individuales y filtrar por tamaño mínimo
        # (para no marcar pequeños huecos entre hojas como fallas de siembra)
        from shapely.geometry import Polygon, MultiPolygon
        if isinstance(failure_areas, Polygon):
            failures = [failure_areas]
        elif isinstance(failure_areas, MultiPolygon):
            failures = list(failure_areas.geoms)
        else:
            failures = []
            
        # Filtrar fallas: deben ser mayores a un área mínima (ej. 2x2 plantas fallidas)
        min_failure_area = (gap_threshold ** 2) * 2
        valid_failures = [f for f in failures if f.area > min_failure_area]
        
        gdf_failures = gpd.GeoDataFrame(geometry=valid_failures, crs=gdf_temp.crs)
        log(f"  OK: {len(gdf_failures)} sectores con fallas detectados.", "ok")
        # ──────────────────────────────────────────────────────────────────

        if n_plants == 0:
            log("  ⚠ Sin detecciones — ajustá los parámetros e intentá de nuevo.", "warn")
            log("  Sugerencias: bajar confianza, ampliar rango de tamaño, aumentar SAM%.", "warn")
            progress_fn(100)
            status_fn("Sin detecciones. Revisá parámetros.")
            _close_log(success=True)
            return {"n_plants": 0, "density_low": 0, "density_mid": 0, "density_high": 0}

        # PASO 5: Geoprocesamiento
        log(f"\n[5/6] {STEPS[4]}...", "warn")
        status_fn(STEPS[4])

        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from shapely.geometry import Point, box as shapely_box

        # transform y crs ya están definidos desde el paso 1
        # (apuntan al ortomosaico efectivo: remuestreado u original)
        geometry = [Point(transform * (row.px, row.py)) for _, row in df_final.iterrows()]
        gdf = gpd.GeoDataFrame(df_final, geometry=geometry, crs=crs)

        # Si el CRS es geográfico (grados, ej. EPSG:4326), reproyectar a UTM
        # para que la grilla y la densidad sean en metros reales
        if gdf.crs and gdf.crs.is_geographic:
            utm_crs = gdf.estimate_utm_crs()
            gdf = gdf.to_crs(utm_crs)
            log(f"  CRS geográfico detectado → reproyectado a {utm_crs.to_epsg()}", "muted")
        else:
            log(f"  CRS proyectado: {gdf.crs.to_epsg()}", "muted")

        xmin, ymin, xmax, ymax = gdf.total_bounds
        gs = params["grid_size"]
        cells = [shapely_box(x0, y0, x0+gs, y0+gs)
                 for x0 in np.arange(xmin, xmax, gs)
                 for y0 in np.arange(ymin, ymax, gs)]
        grid = gpd.GeoDataFrame(geometry=cells, crs=gdf.crs)
        joined = gpd.sjoin(grid, gdf[["geometry"]], how="left", predicate="contains")
        counts = joined.groupby(joined.index).size() - 1
        grid["densidad"] = (counts / (gs*gs)) * 10000
        grid["densidad"] = grid["densidad"].clip(lower=0)

        # Promedios por zona — reemplazar NaN por 0
        # Zonificación relativa al lote — percentiles 33% y 66%
        densidades_validas = grid["densidad"][grid["densidad"] > 0]
        if len(densidades_validas) > 2:
            p33 = float(np.percentile(densidades_validas, 33))
            p66 = float(np.percentile(densidades_validas, 66))
        else:
            p33, p66 = 1000.0, 3500.0   # fallback si hay muy pocas celdas

        grid["zona"] = "media"
        grid.loc[grid["densidad"] <= p33, "zona"] = "baja"
        grid.loc[grid["densidad"] >  p66, "zona"] = "alta"

        low_val  = float(grid[grid["zona"] == "baja"]["densidad"].mean())
        mid_val  = float(grid[grid["zona"] == "media"]["densidad"].mean())
        high_val = float(grid[grid["zona"] == "alta"]["densidad"].mean())
        low_val  = 0.0 if np.isnan(low_val)  else low_val
        mid_val  = 0.0 if np.isnan(mid_val)  else mid_val
        high_val = 0.0 if np.isnan(high_val) else high_val

        log(f"  Zonificacion relativa al lote (percentiles 33%-66%)", "muted")
        log(f"  Umbral baja/media : {p33:.0f} pl/ha", "muted")
        log(f"  Umbral media/alta : {p66:.0f} pl/ha", "muted")
        log(f"  Zona baja  (media): {low_val:.0f} pl/ha", "muted")
        log(f"  Zona media (media): {mid_val:.0f} pl/ha", "muted")
        log(f"  Zona alta  (media): {high_val:.0f} pl/ha", "muted")

        fig, ax = plt.subplots(figsize=(10, 12), facecolor="#0F1923")
        ax.set_facecolor("#0F1923")
        colors = {"baja": "#E76F51", "media": "#F4A261", "alta": "#4CAF82"}
        for zona, color in colors.items():
            sub = grid[grid["zona"] == zona]
            if not sub.empty:
                sub.plot(ax=ax, color=color, alpha=0.8, linewidth=0)
        
        # Superponer polígonos de fallas en rojo vibrante
        if not gdf_failures.empty:
            gdf_failures.plot(ax=ax, color="#FF0000", alpha=0.6, label="Fallas de siembra")
            
        gdf.plot(ax=ax, markersize=1, color="white", alpha=0.5)
        
        patches = [mpatches.Patch(color=c, label=f"Zona {z.capitalize()}") for z, c in colors.items()]
        if not gdf_failures.empty:
            patches.append(mpatches.Patch(color="#FF0000", label="Fallas de siembra"))
            
        ax.legend(handles=patches, loc="upper right",
                  facecolor="#1A2535", edgecolor="#2A3F58", labelcolor="white", fontsize=10)
        ax.set_title(f"Zonificacion Productiva Intra-lote\nn={n_plants} plantas | Grilla {gs}x{gs} m",
                     color="white", fontsize=13, pad=12)
        ax.tick_params(colors="#7A8FA6")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A3F58")

        map_out = os.path.join(output_dir, "mapa_densidad.png")
        plt.tight_layout()
        plt.savefig(map_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        step_progress(5)

        # PASO 6: Exportación
        log(f"\n[6/6] {STEPS[5]}...", "warn")
        status_fn(STEPS[5])

        gpkg_out = os.path.join(output_dir, "zonificacion_productiva.gpkg")
        csv_out  = os.path.join(output_dir, "plantas_detectadas.csv")

        # Exportar GeoPackage — intentar pyogrio primero, luego fiona, luego GeoJSON
        try:
            grid.to_file(gpkg_out, layer="zonificacion", driver="GPKG", engine="pyogrio")
            if not gdf_failures.empty:
                gdf_failures.to_file(gpkg_out, layer="fallas_siembra", driver="GPKG", engine="pyogrio")
            log("  GeoPackage exportado (pyogrio)", "muted")
        except Exception:
            try:
                grid.to_file(gpkg_out, layer="zonificacion", driver="GPKG", engine="fiona")
                if not gdf_failures.empty:
                    gdf_failures.to_file(gpkg_out, layer="fallas_siembra", driver="GPKG", engine="fiona")
                log("  GeoPackage exportado (fiona)", "muted")
            except Exception:
                # Fallback: exportar como GeoJSON (siempre funciona, sin dependencias)
                gpkg_out = gpkg_out.replace(".gpkg", ".geojson")
                grid.to_file(gpkg_out, driver="GeoJSON")
                if not gdf_failures.empty:
                    gdf_failures.to_file(os.path.join(output_dir, "fallas_siembra.geojson"), driver="GeoJSON")
                log("  ⚠ pyogrio/fiona no disponibles — exportado como GeoJSON", "warn")
                log("    El .geojson se puede abrir en QGIS igual que el .gpkg", "muted")

        # CSV con coordenadas UTM como columnas principales de geometría
        # pixel_x/pixel_y se conservan como referencia interna
        df_export = pd.DataFrame({
            "id":         range(1, len(df_final) + 1),
            "utm_x":      [pt.x for pt in gdf.geometry],
            "utm_y":      [pt.y for pt in gdf.geometry],
            "confianza":  df_final["conf"].values,
            "pixel_x":    df_final["px"].values,
            "pixel_y":    df_final["py"].values,
        })
        df_export.to_csv(csv_out, index=False)
        # También exportar puntos como GeoPackage para carga directa en QGIS
        puntos_gpkg = os.path.join(output_dir, "plantas_puntos.gpkg")
        try:
            gdf[["geometry","conf"]].rename(columns={"conf":"confianza"}).to_file(
                puntos_gpkg, driver="GPKG", engine="pyogrio")
        except Exception:
            try:
                gdf[["geometry","conf"]].rename(columns={"conf":"confianza"}).to_file(
                    puntos_gpkg, driver="GPKG", engine="fiona")
            except Exception:
                puntos_gpkg = puntos_gpkg.replace(".gpkg", ".geojson")
                gdf[["geometry","conf"]].rename(columns={"conf":"confianza"}).to_file(
                    puntos_gpkg, driver="GeoJSON")

        log(f"  OK: Zonificacion  -> {os.path.basename(gpkg_out)}", "ok")
        log(f"  OK: Puntos QGIS   -> {os.path.basename(puntos_gpkg)}", "ok")
        log(f"  OK: CSV plantas   -> {os.path.basename(csv_out)}", "ok")
        log(f"  OK: Mapa          -> {os.path.basename(map_out)}", "ok")

        # SHP opcional
        if export_shp:
            log("  Exportando Shapefiles opcionales...", "muted")
            shp_dir = os.path.join(output_dir, "shapefiles")
            os.makedirs(shp_dir, exist_ok=True)
            try:
                grid.to_file(os.path.join(shp_dir, "zonificacion.shp"), driver="ESRI Shapefile")
                gdf[["geometry","conf"]].rename(columns={"conf":"confianza"}).to_file(
                    os.path.join(shp_dir, "plantas_puntos.shp"), driver="ESRI Shapefile")
                log(f"  OK: SHP -> shapefiles/", "ok")
            except Exception as e_shp:
                log(f"  Advertencia SHP: {e_shp}", "warn")

        progress_fn(100)
        log("\n" + "="*50, "muted")
        log("COVIO v2.0 — ANALISIS COMPLETADO", "ok")
        log("─" * 40, "muted")
        log(f"  Cultivo / Lote     : {lote}", "info")
        log(f"  Objetos detectados : {n_plants}", "info")
        log(f"  Zona baja  (media) : {low_val:.0f} pl/ha  (umbral < {p33:.0f})", "info")
        log(f"  Zona media (media) : {mid_val:.0f} pl/ha  ({p33:.0f} - {p66:.0f})", "info")
        log(f"  Zona alta  (media) : {high_val:.0f} pl/ha  (umbral > {p66:.0f})", "info")
        log("─" * 40, "muted")
        log("ARCHIVOS GENERADOS", "info")
        log(f"  {os.path.basename(map_out)}", "muted")
        log(f"  {os.path.basename(gpkg_out)}", "muted")
        log(f"  {os.path.basename(puntos_gpkg)}", "muted")
        log(f"  {os.path.basename(csv_out)}", "muted")
        if export_shp:
            log(f"  shapefiles/zonificacion.shp", "muted")
            log(f"  shapefiles/plantas_puntos.shp", "muted")
        log("─" * 40, "muted")
        log("PARAMETROS USADOS", "info")
        log(f"  Cultivo: {lote} · GSD {params['gsd_cm']} cm/px · Tile {params['tile']}px · "
            f"SAM {params.get('sam_fraction',0.7)*100:.0f}% · Conf >{params['conf']} · "
            f"Dist.min {params['nms_cm']}cm · Obj {params['min_cm']}-{params['max_cm']}cm · "
            f"Epocas {params['epochs']} · Grilla {params['grid_size']}m", "muted")
        log("=" * 50, "muted")
        status_fn(f"Completado · {n_plants} objetos detectados.")
        _close_log(success=True)

        return {"n_plants": n_plants, "density_low": low_val,
                "density_mid": mid_val, "density_high": high_val}

    except Exception as e:
        log(f"\nERROR: {e}", "err")
        import traceback
        log(traceback.format_exc(), "err")
        _close_log(success=False)
        raise
