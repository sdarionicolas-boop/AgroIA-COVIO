# 🌱 AgroIA-COVIO v2.1

**Detección automática y análisis espacial de densidad de plantas**  
Desarrollado por Darío Sánchez Leguizamón — CLAP 2026

---

## ¿Qué hace?

AgroIA-COVIO toma un ortomosaico RGB de dron (GeoTIFF) y produce:

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

### Requisitos
- Windows, Linux o macOS con **Python 3.10 o superior** ([python.org](https://www.python.org/downloads/); en Windows marcá "Add Python to PATH").
- **Git** instalado ([git-scm.com](https://git-scm.com/downloads)): hace falta porque `segment-anything` y `mobile-sam` se instalan desde GitHub.
- **16 GB de RAM** como mínimo (ideal 32 GB para ortomosaicos grandes) y ~10 GB libres en disco.
- Internet la primera vez: se descargan el modelo SAM (~2,4 GB, o ~40 MB en modo MobileSAM) y los pesos de YOLOv8m.

### Desde el código fuente (recomendado)

```bash
git clone https://github.com/sdarionicolas-boop/AgroIA-COVIO.git
cd AgroIA-COVIO

# Entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt   # tarda varios minutos (instala PyTorch)

python app.py
```

Al abrir la app: cargá el GeoTIFF, descargá el modelo SAM con el botón del panel "Modelo SAM" y usá primero **"Previsualizar Parámetros (1 Tile)"** para validar GSD y filtros antes de correr el análisis completo. Si algo falla en la instalación, abrí un [issue](https://github.com/sdarionicolas-boop/AgroIA-COVIO/issues) con el mensaje de error.

### Ejecutable para Windows (opcional)
No hay ejecutable publicado en los releases. Podés generarlo vos con `python build_exe.py`; queda en `dist/AgroIA-COVIO/AgroIA-COVIO.exe` (distribuí la carpeta completa, no solo el `.exe`).

---

## Estructura del proyecto

```
PlantDetector/
├── app.py              # Interfaz gráfica (Tkinter)
├── pipeline.py         # Lógica del pipeline
├── indices.py          # Índices de vegetación (NDVI/NDRE multiespectral; ExG/GLI/VARI RGB)
├── lote.py             # Estadísticas, fallas y vigor por lote (experimental)
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

## Índices de vegetación (experimental)

Además del conteo de plantas, el repositorio incluye herramientas para calcular **índices de vegetación**,
tanto en ortomosaicos **multiespectrales** (p. ej. MicaSense RedEdge-M procesado con WebODM/ODM) como en
ortomosaicos **RGB** comunes. **Es un módulo experimental: no está validado con datos de campo.** Se usa
desde la interfaz, con el ortomosaico ya seleccionado, usando uno de los dos botones del encabezado —
**🛰 Índices multiespectrales (NDVI / NDRE)** o **🎨 Índices RGB (ExG / GLI / VARI)**— o por línea de comandos
(`--modo multi|rgb`). El botón RGB también sirve para un ortomosaico multiespectral (usa sus bandas visibles).

| Ortomosaico | Índices | Vegetación / vigor |
|---|---|---|
| Multiespectral (bandas NIR y Red por nombre) | NDVI y NDRE | NDVI separa vegetación de suelo; NDRE zonifica el vigor |
| RGB común (3 o 4 bandas) | ExG, GLI y VARI | ExG separa vegetación de suelo; GLI zonifica el vigor |

Los dos grupos se mantienen separados: no se mezclan índices RGB con multiespectrales.

**En la interfaz** se puede, sin salir de la aplicación: dibujar los lotes con el mouse sobre el mapa (fondo RGB
o cualquiera de los índices), calcular, y ver los resultados en un visor con el mapa del lote, los índices del
ortomosaico, la tabla y las notas. El mapa del lote es interactivo: zoom con la rueda, desplazar arrastrando,
capas que se prenden y apagan (índice de fondo, celdas con su valor, clases, fallas, límite) y al pasar el mouse,
las coordenadas, el **valor exacto de cada índice en ese píxel** (leído del archivo del índice, no de la
imagen reducida) y los datos de la celda (cobertura, falla, clase). Al terminar también quedan los archivos en la carpeta de resultados
(`indices_multiespectral` o `indices_rgb`, según el botón).

| Herramienta | Qué hace |
|---|---|
| `pipeline.py` | Si el ortomosaico no es de 8 bits (p. ej. uint16 de ODM), aplica un estiramiento único (percentiles 2–98) para que todos los tiles tengan el mismo brillo. Lee R, G y B por nombre de banda. |
| `indices.py` | Calcula los índices que corresponden a las bandas del ortomosaico: `python indices.py ORTO.tif SALIDA` |
| `editor_lotes.py` / `lotes_io.py` | Editor para **dibujar los lotes con el mouse** (clic izquierdo = vértice, clic derecho o Enter = cerrar polígono); guarda un GeoPackage. |
| `lote.py` | Dentro de los polígonos de lote: estadísticas, fallas y zonificación de vigor en grilla: `python lote.py ORTO.tif LOTES.gpkg SALIDA --grid 10` |
| `visor_resultados.py` | Visor de resultados dentro de la interfaz (mapa, índices, tabla y notas). |
| `mapa_lote.py` | Mapa interactivo de los lotes analizados (zoom, capas, valores por celda) que usa el visor. |

**Cómo leer los resultados**
- Los índices son **relativos**. Sirven para comparar zonas dentro de un mismo vuelo; no para comparar
  fechas o vuelos distintos, ni contra valores de la bibliografía, salvo que el ortomosaico tenga
  calibración radiométrica verificada (el archivo no permite comprobarlo).
- Los índices **RGB** son además más sensibles a la cámara, la hora del día, las sombras y el brillo del
  ortomosaico, y no distinguen bien vegetación verde de otras superficies verdes ni vegetación senescente.
- La zonificación por celdas muestra el valor medio del índice de vigor y una clase **baja / normal / alta**.
  Una celda es baja o alta solo si se aparta de la mediana del lote al menos un 10 % del rango del índice en
  todo el ortomosaico (`--desvio-min`) y además al menos 1,5 desvíos robustos respecto de las demás celdas;
  si no, es normal. Un lote parejo queda casi todo "normal". Esos umbrales son arbitrarios y no están
  validados con datos de campo. Mirá también `cv_<índice>_entre_celdas_pct` y `pct_area_baja/normal/alta`
  en `resumen.csv`.
- Una "falla" es una mancha de suelo o vegetación muy baja según un umbral del índice de vegetación
  (automático, Otsu, o fijado con `--umbral`). No implica falta de plantas: puede ser camino, maleza o un
  sector senescente. Revisá `vista.png`.
- **El umbral automático puede equivocarse.** Si la distribución del índice no es claramente bimodal (no hay
  una población de suelo y otra de vegetación, por ejemplo un cultivo senescente o muy parejo), el programa
  lo advierte (`umbral_separacion` < 0,7 en `resumen.csv`) y la cobertura y las fallas quedan en duda: fijá un
  umbral manual.
- En cultivo cerrado (plantas que se tocan) no tiene sentido contar plantas individuales; para ese caso
  usá los índices y no el conteo.

**Opciones de WebODM sugeridas** (sin validar con un reprocesado completo): `orthophoto-resolution` 1.5 cm/px,
`radiometric-calibration: camera`, `auto-boundary: false`, `crop: 0`, `skip-3dmodel: true`. Si el
ortomosaico sale con parte del lote sin cubrir, revisá primero `auto-boundary` y `crop`.

---|---|
| `pipeline.py` | Si el ortomosaico no es de 8 bits (p. ej. uint16 de ODM), aplica un estiramiento único (percentiles 2–98) para que todos los tiles tengan el mismo brillo. Lee R, G y B por nombre de banda. |
| `indices.py` | Calcula NDVI (NIR, Red) y NDRE (NIR, RedEdge): `python indices.py ORTO.tif SALIDA` |
| `editor_lotes.py` / `lotes_io.py` | Editor para **dibujar los lotes con el mouse** sobre el mapa (fondo RGB, NDVI o NDRE) desde el botón 🛰 de la interfaz; guarda un GeoPackage. Clic izquierdo = vértice, clic derecho o Enter = cerrar polígono. |
| `lote.py` | Dentro de polígonos de lote que dibujás vos: estadísticas, fallas y zonificación de vigor en grilla: `python lote.py ORTO.tif LOTES.gpkg SALIDA --grid 10` |

**Cómo leer los resultados**
- Los índices son **relativos**. Sirven para comparar zonas dentro de un mismo vuelo; no para comparar
  fechas o vuelos distintos, ni contra valores de la bibliografía, salvo que el ortomosaico tenga
  calibración radiométrica verificada (el archivo no permite comprobarlo).
- La zonificación (baja/media/alta) usa terciles de NDRE **dentro de cada lote**: siempre hay un tercio
  por clase. Mirá `cv_ndre_entre_celdas_pct` en `resumen.csv`: si es bajo, las diferencias son pequeñas.
- Una "falla" es una mancha de suelo o vegetación muy baja según un umbral de NDVI (automático, Otsu,
  o fijado con `--umbral-ndvi`). No implica falta de plantas: puede ser camino, maleza o un sector
  senescente. Revisá `vista.png`.
- En cultivo cerrado (plantas que se tocan) no tiene sentido contar plantas individuales; para ese caso
  usá los índices y no el conteo.

**Opciones de WebODM sugeridas** (sin validar con un reprocesado completo): `orthophoto-resolution` 1.5 cm/px,
`radiometric-calibration: camera`, `auto-boundary: false`, `crop: 0`, `skip-3dmodel: true`. Si el
ortomosaico sale con parte del lote sin cubrir, revisá primero `auto-boundary` y `crop`.

---

## Parámetros

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| GSD (cm/px) | 1.5 | Resolución espacial del ortomosaico |
| Tile size (px) | 1024 | Tamaño de ventana de procesamiento. **Usar 1024**: en pruebas sobre el ortomosaico de maíz (GSD ~1,5 cm), tile 1024 detectó ~5.500 plantas y tile 512 solo 2 (el modelo no converge con 10 épocas). Bajar a 512 solo si la RAM no alcanza, y verificar el conteo. |
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
*   **Comunitario y Abierto:** Gratuito bajo los términos de la licencia GNU GPL v3.0 (con atribución de preservación 7b).
*   **Comercial y Cerrado:** Licencias comerciales exclusivas diseñadas para empresas y AgTechs que deseen integrar los algoritmos y capacidades de AgroIA-COVIO dentro de sus productos cerrados o propietarios, eximiéndose de las obligaciones de copyleft de la GPL.

Para cotizaciones de licencias comerciales, integraciones personalizadas o consultoría, contactar a:
📩 **sdarionicolas@gmail.com**

---

## Contribuciones y Desarrollo

¡Las contribuciones son bienvenidas! Si deseas aportar código, corregir errores o mejorar la documentación, por favor lee las instrucciones en [CONTRIBUTING.md](file:///e:/Desktop_Movidos/PlantDetector/CONTRIBUTING.md).

Para poder aceptar contribuciones de terceros en el código principal sin comprometer el modelo comercial dual, es obligatorio aceptar los términos descritos en el [Acuerdo de Licencia de Contribuidor (CLA)](file:///e:/Desktop_Movidos/PlantDetector/CLA.md).


