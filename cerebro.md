# CEREBRO - AgroIA-COVIO v2.1 (PlantDetector)

Este documento define la arquitectura, filosofía y reglas inviolables del sistema AgroIA-COVIO v2.1. Funciona como la fuente de la verdad para cualquier decisión técnica o de diseño en el proyecto.

---

## 1. Identidad y Filosofía (Basado en YO.md)
*   **Propósito:** Democratizar herramientas de agricultura de precisión (conteo y zonificación de densidad de plantas) para productores pequeños y medianos en LATAM.
*   **Enfoque:** No gadgets. No cajas negras. Herramientas reproducibles, de bajo costo, que respeten el juicio del agrónomo.
*   **Operación:** Procesamiento local offline, orientado a CPU para máxima accesibilidad, con una interfaz amigable.
*   **Métricas Reales:** Validado empíricamente (ej. Maíz en AMBA Sur, Palma Aceitera en Perú, Girasol en INTA).

---

## 2. Arquitectura del Sistema
El proyecto se basa en una arquitectura modular sin fragmentación excesiva:

### Frontend (`app.py`)
*   Interfaz gráfica construida con Tkinter (tema oscuro "Valorant-style").
*   Ejecución de procesos pesados en hilos secundarios (`threading`) para evitar bloqueos de la UI.
*   **Preajustes de Cultivo (Crop Presets - v2.1):** Un selector integrado que precarga las mejores configuraciones empíricas conocidas (GSD, confianza, NMS, tamaño de objeto y épocas) según el cultivo.
*   **Auto-detección del GSD:** Carga automática del tamaño de píxel en centímetros desde el GeoTIFF al examinar el archivo mediante la librería `rasterio` (solo en CRS proyectados).
*   **Modo de Previsualización:** Botón "👁" que procesa un único tile central para validar el GSD y los filtros de SAM visualmente antes de iniciar el proceso completo.
*   **Mecanismo de Parada:** Botón "✕" (Cancelar) que permite abortar el análisis de forma segura mediante un `threading.Event`, cerrando el log y restaurando la UI.

### Pipeline Core (`pipeline.py`)
El flujo de procesamiento consta de 6 pasos secuenciales:
1.  **Tiling y Remuestreo:** Ajuste automático del GSD (remuestreo) para asegurar que la planta mínima tenga al menos 16px (garantizando su detección por el modelo).
2.  **Pseudo-etiquetado (Multimodelo):** 
    *   Soporte para **SAM ViT-H** (Modo Pesado, ~2.4GB) para máxima precisión.
    *   Soporte para **MobileSAM** (Modo Ligero, ~40MB) para procesamiento rápido en CPU.
    *   Muestreo estratificado (ej. 70% de cobertura espacial) y filtros dinámicos de área.
3.  **Entrenamiento (YOLOv8m):** Entrenamiento al vuelo (*on-the-fly*) sobre las pseudo-etiquetas para aprender la textura local del cultivo.
4.  **Inferencia y Filtros Agronómicos:**
    *   **NMS Agronómico:** Filtrado de duplicados derivado de la geometría/marco de siembra real (`nms_cm`), un diferenciador clave frente al NMS puramente visual.
    *   Filtros morfométricos estrictos (`min_cm`, `max_cm`).
5.  **Geoprocesamiento y Análisis Espacial:**
    *   Conversión de píxeles a coordenadas proyectadas (UTM). 
    *   **Detección de Fallas (Gap Analysis):** Identificación automática de huecos en la siembra mayores a 2.5x el NMS. Exportación de polígonos de falla para navegación a campo.
    *   **Zonificación:** Clasificación relativa al lote usando percentiles de densidad (33% baja, 66% alta).
6.  **Exportación:** Mapas PNG (con fallas superpuestas), bases de datos espaciales (GeoPackage multicapa/Shapefile) y tablas (CSV).

---

## 3. Reglas de Desarrollo e Integración
*   **Sentido Agronómico:** Todo cálculo espacial debe tener sentido agronómico trazable (cm, metros).
*   **Regla de Épocas Mínimas (Evitar Sub-entrenamiento):** Si el conjunto de datos tiene pocos tiles (ej. < 20 tiles), **nunca se debe entrenar a menos de 10 épocas** (e idealmente de 15 a 20). Entrenamientos de 5 épocas en lotes pequeños (como en Palma) no logran la convergencia de YOLOv8 y desploman las detecciones.
*   **Selección de SAM según el Cultivo:**
    *   *MobileSAM:* Adecuado para cultivos de cobertura continua (como Girasol) donde el objetivo es la zonificación de heterogeneidad y se prioriza el tiempo de CPU (ahorro >50%).
    *   *SAM ViT-H:* Obligatorio para plantas discretas pequeñas (como Maíz temprano) o lotes con muy pocos tiles, donde el encoder de MobileSAM no logra capturar suficientes pseudo-etiquetas.
*   **Preservación del Criterio:** Los presets son guías; cualquier modificación manual de parámetros por parte del usuario debe cambiar automáticamente el selector de la GUI a "Personalizado".

---

## 4. Validaciones Empíricas (Benchmarking Real)

A continuación se detallan las corridas exitosas registradas en el sistema:

### A. Maíz
*   **Baseline (ViT-H - `pipeline_log_20260320_173255.txt`):**
    *   *Configuración:* GSD 1.5 cm/px, Confianza > 0.25, NMS 37.5 cm, Épocas=10.
    *   *Salida:* **5,958 plantas** | Tiempo Total: 49m 49s (SAM: 22m 52s).
*   **Prueba MobileSAM (`covio_log_20260520_122616.txt`):**
    *   *Configuración:* GSD 1.5 cm/px, Confianza > 0.15, NMS 37.5 cm, Épocas=5.
    *   *Salida:* **3,794 plantas** | Tiempo Total: 22m 33s (SAM: 9m 16s).
    *   *Análisis:* Reducción de tiempo a la mitad, pero pérdida del **36.3%** en detecciones. El encoder ligero de MobileSAM redujo las pseudo-etiquetas iniciales de 1,538 a 732, mermando el dataset de YOLO.

### B. Palma Aceitera
*   **Baseline (ViT-H - `pipeline_log_20260318_234341.txt`):**
    *   *Configuración:* GSD 13.0 cm/px, Confianza > 0.25, NMS 6.5 m (50 px), Épocas=10.
    *   *Salida:* **379 palmas** | Tiempo Total: 5m 55s (SAM: 2m 35s).
*   **Prueba MobileSAM (`covio_log_20260520_130131.txt`):**
    *   *Configuración:* GSD 13.0 cm/px, Confianza > 0.15, NMS 6.5 m (50 px), Épocas=5.
    *   *Salida:* **42 palmas**.
    *   *Análisis:* Pérdida catastrófica del **88.9%**. Debido a que el dataset consta de solo 4 tiles, configurar 5 épocas significó entrenar YOLO en solo 10 pasos de gradiente totales. La red no convergió.

### C. Girasol (INTA Parcela 2)
*   **Baseline (ViT-H - `pipeline_log_20260320_141544.txt`):**
    *   *Configuración:* GSD 1.5 cm/px, Confianza > 0.12 (Modo Canopea), NMS 37.5 cm, Épocas=10.
    *   *Salida:* **2,805 plantas** | Tiempo Total: 2h 43m 08s (SAM: 1h 16m).
*   **Prueba MobileSAM (`covio_log_20260518_171323.txt`):**
    *   *Configuración:* GSD 1.5 cm/px, Confianza > 0.25, NMS 37.5 cm, Épocas=10.
    *   *Salida:* **75 plantas** | Tiempo Total: 2h 16m 26s (SAM: 37m 15s).
    *   *Análisis:* MobileSAM segmentó muy bien (7,177 pseudo-etiquetas frente a 8,626 de ViT-H), reduciendo el tiempo de SAM en un 51%. Sin embargo, al configurar por error una confianza restrictiva de >0.25 en un cultivo continuo (donde el promedio de score es 0.176), se descartó el 97% de las plantas.

### D. Lechuga (INTA Parcela 2)
*   **Baseline (ViT-H - `covio_log_20260424_132605.txt`):**
    *   *Configuración:* GSD 0.6 cm/px, Confianza > 0.20, NMS 15.0 cm, Épocas=10.
    *   *Salida:* **966 plantas** | Tiempo Total: 15m 39s.
    *   *Análisis:* Logró una **precisión de campo del 96%** debido a dos factores: remuestreo a GSD sub-centimétrico y el recorte previo del área productiva (máscara de lote) que eliminó falsos positivos externos.

---

## Instrucción clave
Para cualquier tarea de este proyecto, priorizá el contexto de este archivo sobre conocimiento general del modelo.
Ante duda sobre arquitectura o flujo, consultá este CEREBRO.md antes de asumir.