# 🌱 AgroIA-COVIO v2.1

**Detección automática y análisis espacial de densidad de plantas**  
Desarrollado por Darío Sánchez Leguizamón — CLAP 2026

---

## ¿Qué hace?

COVIO toma un ortomosaico RGB de dron (GeoTIFF) y produce:

- **Conteo de plantas** individuales detectadas
- **Mapa de zonificación productiva** (alta / media / baja densidad)
- **GeoPackage (.gpkg)** con la grilla de densidades (integrable en QGIS / monitores agrícolas)
- **CSV georreferenciado** con coordenadas UTM de cada planta

### Novedades en v2.1:
*   **Preajustes de Cultivo (Presets):** Selector que precarga de forma inmediata parámetros optimizados para Maíz, Palma, Lechuga y Girasol.
*   **Auto-detección de GSD:** Extrae automáticamente la resolución del ortomosaico en centímetros (`cm/px`) al cargar el GeoTIFF (requiere CRS proyectado).
*   **Regla de Seguridad de Épocas:** Alertas automáticas y ajuste a un piso de 10 épocas para lotes pequeños (<20 tiles) para garantizar la convergencia del entrenamiento de YOLOv8m.

Pipeline interno: SAM (pseudo-etiquetado) → YOLOv8m (detección) → Filtros agronómicos → SIG

---

## Instalación (primera vez)

### Opción A: Ejecutable (recomendada para usuarios no técnicos)

1. Descomprimí la carpeta `PlantDetector/`
2. Hacé doble clic en `PlantDetector.exe`
3. La primera vez, la app descargará automáticamente el modelo SAM (~2.4 GB)

### Opción B: Desde código fuente (para desarrolladores)

```bash
# Clonar / descomprimir el proyecto
cd PlantDetector

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Lanzar la app
python app.py
```

---

## Estructura del proyecto

```
PlantDetector/
├── app.py              # Interfaz gráfica (Tkinter)
├── pipeline.py         # Lógica del pipeline
├── requirements.txt    # Dependencias Python
├── build_exe.py        # Script para generar el .exe
├── models/             # Aquí se guarda el modelo SAM (se crea automáticamente)
│   └── sam_vit_h_4b8939.pth   (~2.4 GB, descarga automática)
└── output/             # Resultados generados
    ├── mapa_densidad.png
    ├── zonificacion_productiva.gpkg
    └── plantas_detectadas.csv
```

---

## Uso

1. **Abrí la app**
2. **Cargá el ortomosaico** con el botón "Examinar" (archivo `.tif` georreferenciado)
3. **Verificá el modelo SAM** — si no está, hacé clic en "Descargar modelo"
4. **Ajustá los parámetros** si es necesario (los valores por defecto son los del paper)
5. **Hacé clic en "CORRER PIPELINE"**
6. Al finalizar, usá los botones para ver el mapa o descargar los archivos

---

## Parámetros

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| GSD (cm/px) | 1.5 | Resolución espacial del ortomosaico |
| Tile size (px) | 512 | Tamaño de ventana de procesamiento |
| Umbral confianza | 0.25 | Mínimo score para aceptar una detección |
| Tam. mín caja (px) | 20 | Descarta objetos muy pequeños (artefactos) |
| Tam. máx caja (px) | 80 | Descarta objetos muy grandes (malezas) |
| Radio NMS manual (px) | 25 | Distancia mínima entre plantas (~37.5 cm) |
| Épocas entrenamiento | 10 | Inferencia temprana (óptima en CPU) |
| Tamaño grilla SIG (m) | 10 | Resolución de la zonificación productiva |

---

## Requisitos de hardware

- CPU moderna (AMD Ryzen / Intel Core, 4+ núcleos)
- **16 GB RAM mínimo** (recomendado para ortomosaicos > 500 MB)
- ~5 GB de espacio libre en disco (modelo SAM + archivos temporales)
- GPU *no* requerida — todo corre en CPU

---

## Generar el ejecutable (.exe)

```bash
pip install pyinstaller
python build_exe.py
```

El ejecutable queda en `dist/PlantDetector/PlantDetector.exe`.  
Distribuir la carpeta completa `dist/PlantDetector/` (no solo el .exe).

---

## Cita

Si usás este trabajo, por favor citar:

> Sánchez Leguizamón, D. N. (2026). Detección automática y análisis espacial de densidad de plantas mediante imágenes RGB de dron y aprendizaje profundo. *IV Congreso Latinoamericano de Agricultura de Precisión (CLAP 2026)*. Santiago, Chile.

---

## Licencia y Atribución

Este proyecto está distribuido bajo la Licencia **GNU GPL v3.0** con **Términos Adicionales** (Sección 7b) para garantizar la atribución de autoría en la interfaz de usuario.

Cualquier versión modificada, copia o trabajo derivado de este software (incluyendo interfaces gráficas, herramientas de línea de comandos o aplicaciones web) **debe mostrar de forma visible y destacada** el siguiente crédito en su interfaz de usuario final:
> **AgroIA-COVIO — desarrollado por Darío Sánchez Leguizamón**

Para más detalles, consulta los archivos [LICENSE](file:///e:/Desktop_Movidos/PlantDetector/LICENSE) y [NOTICE](file:///e:/Desktop_Movidos/PlantDetector/NOTICE).

---

## Licenciamiento Comercial / Integración Propietaria

Como autor único y titular exclusivo de los derechos de autor de AgroIA-COVIO, ofrezco este software bajo un esquema de **licenciamiento dual**:
*   **Comunitario y Abierto:** Gratuito bajo los términos de la licencia GNU GPL v3.0 (con atribución 7b).
*   **Comercial y Cerrado:** Licencias comerciales exclusivas diseñadas para empresas y AgTechs que deseen integrar los algoritmos y capacidades de COVIO dentro de sus productos cerrados o propietarios, eximiéndose de las obligaciones de copyleft de la GPL.

Para cotizaciones de licencias comerciales, integraciones personalizadas o consultoría, contactar a:
📩 **sdarionicolas@gmail.com**

