# AGENT.md | Darío Nicolás

## IDENTIDAD

**Rol:** Ingeniero Agrónomo + Desarrollador Agtech | Licenciatura en Ciencias Agrarias (UNAJ, Buenos Aires, diciembre 2026)

**Propósito:** Democratizar herramientas de agricultura de precisión para productores pequeños y medianos en LATAM. No gadgets. No cajas negras. Herramientas reproducibles, de bajo costo, que respeten el juicio del agrónomo.

**Ubicación:** Buenos Aires, Argentina  
**Red Primaria:** LATAM (CLAP 2026, contactos post-Santiago, WhatsApp agronomía)

---

## STACK TÉCNICO

### Core Competencies
- **Computer Vision:** YOLO (v8m, v8s, v8n), SAM (vit_b, vit_l), pseudo-labeling pipelines
- **Remote Sensing:** Sentinel-2, GEE, NASA POWER, super-resolución (Gamma Earth)
- **Ecophysiological Simulation:** Dinámica de cultivos, balance de agua, soluciones nutrientes
- **Species Distribution Modeling:** R/biomod2, AUC/TSS, leaflet mapas interactivos
- **Development Stack:** Python (Streamlit, FastAPI, Ollama), Tkinter+PyInstaller, R, JavaScript
- **Infrastructure:** Colab-native (reproducibilidad), GitHub, bases de datos (pgvector, Postgres)

### Arquitectura Característica
- **Modularidad sin fragmentación:** Scripts unificados para portabilidad Colab
- **NMS derivado de geometría de siembra:** Diferenciador técnico core
- **RAG + LLM local:** Ollama + Gemma 3 4B para asistentes agronomía
- **Geoprocessing:** Zonificación K-means, detección anomalías IsolationForest

---

## PROYECTOS ACTIVOS (PRIORIDAD DESCENDENTE)

### 1. **POLIGONIZACIÓN AUTOMÁTICA DE LOTES (Delineación SAM + Sentinel-2)**

**Estado:** MVP validado. En pitch comercial.

**Pipeline Técnico:**
Punto GPS + Fecha → GEE (Sentinel-2) → SAM ViT-B → Vectorización → GeoJSON georreferenciado

**Validaciones Realizadas:**

- **TAYPE Siniestros (313 puntos granizo):** 
  - 85.6% hit rate (14.4% sin imagen = nubes, ESPERADO)
  - SAM score promedio: 0.923 (rango 0.683-0.999)
  - Error área: 43.6% promedio (ESPERADO para geometría, no magnitud daño)
  - Error <20%: 246/313 polígonos (72%)
  - Tiempo: 3.1 segundos/punto

- **INTA Balcarce Círculos (454 puntos riego):**
  - 74.9% hit rate + 25.1% QC automático (nubes, suelo desnudo) = feature
  - SAM score promedio: 0.962 (excelente)
  - Error vs manual INTA: 8.5% (MUY PRECISO)
  - Error <10%: 246/340 polígonos (72%)
  - Tiempo: 5 segundos/punto

**Diferencial Técnico:**
- Bounding box dinámico (adapta visión al tamaño esperado del lote)
- Anti-fuga (negative point injection evita sobre-segmentación de 4 lotes como 1)
- QC automático (rechaza polígonos con SAM score <0.85, nubes, suelo desnudo)

**Economía:**
- **Costo:** USD 0.50-2/lote (vs USD 10-25 manual, USD 5-15 MapMyCrop, USD 1000-5000 ortofoto)
- **Velocidad:** 5-9 segundos/punto (escalable)
- **Infraestructura:** GEE + SAM ViT-B (gratuita), GPU T4 Colab

**Mercado Objetivo:**

- **Demandantes:**
  - Aseguradoras: Allianz, Zurich, Victoria, Sura, Mapfre (~2.8M ha, 500k-1M lotes/año Argentina)
  - Agrotech: LAYERS, MapMyCrop, TraceX, Kalibre, Agrofy
  - Gobiernos/INTA: catastros, LPIS público (500k-1M lotes nuevos/año Argentina)
  - Seguros paramétricos: indemnización automática por NDVI

- **Pain Point:** No existe LPIS público en LATAM. Cada empresa digitaliza manualmente (15-30 min/lote). Soluciones cloud internacionales (USD 5-15/ha) son caras para LATAM.

- **Tamaño:** USD 10M-50M oportunidad anual LATAM
  - Argentina: 35M ha
  - Brasil: 80M ha
  - LATAM: 300M+ ha sin LPIS público

**Contactos Calientes:**

- **Esteban Videla (FAUNO):** Fundador experto agritech. Validó "pre loteo es problema histórico no resuelto". Intros a AuraBand, CIMA, Optimus, Bold Agro. **HIGH VALUE.**

- **Néstor Barrionuevo (INTA SIG):** Digitalizó 454 círculos manualmente. Feedback: "Excelente, mil gracias". Va a presentarte con Guillermo (TAYPE) próxima semana. Presentación charla institucional = credibilidad pública. También solicita polygon-based clustering para AgroIA.

- **Guillermo (TAYPE):** Dueño datos 313 siniestros granizo. Potencial cliente GRANDE (500+ lotes/año). Intro via Néstor. **CALIENTE, PRÓXIMA SEMANA.**

- **AuraBand:** Plataforma SaaS agrícola. Solicitud deck poligonización. Evaluando white-label integration. Warm intro.

**Modelos de Negocio (Fases Secuenciales):**

- **FASE 1 (Ahora) - Opción A + D:**
  - A (API/SaaS por consumo): USD 2-5/lote O plan mensual fijo (100-1000 lotes)
  - D (White-label): Embedded en plataformas agrotech. Revenue share 70-30.
  - Objetivo: Validación rápida product-market fit con 2-3 clientes pagantes
  - Timeline: 1-2 meses
  - Revenue esperado: USD 2k-10k/mes

- **FASE 2 (Meses 2-3) - Opción B:**
  - Datos masivos: Poligonizar 1-2 provincias. Vender dataset a gobiernos/INTA/aseguradoras
  - Timeline: 2-4 semanas procesamiento
  - Revenue esperado: USD 10k-50k por dataset

- **FASE 3 (Meses 4+) - Opción C:**
  - SaaS integrado: Plataforma web/mobile completa, auditoría, export, QGIS plugins
  - Timeline: 8-12 semanas desarrollo
  - Revenue esperado: USD 500-2000/mes por cliente

**Timeline Inmediato:**
- **Semana 1:** Esperar respuesta AuraBand (reunión 20 min si dicen sí). Prepararse para charla Guillermo.
- **Semana 2-3:** Charla Guillermo (TAYPE) + AuraBand + Bold Agro. Procesar feedback.
- **Semana 4-6:** Procesamiento pilotos, documentación, cierre cliente.
- **META: Cerrar 1 cliente pagante en próximas 4 semanas (Guillermo O AuraBand)**

---

### 2. **Gemelo Digital Tomate (Irrigación Ecofisiológica)**

**Estado:** Código final congelado. Modelado completado. Campo listo.

**Socios Nuevos:** Braian Pezet (IoT/TICAPPS) + Prof. Jorge Osio (Lab IoT UNAJ)

**Rol Darío:** "Cerebro" ecofisiológico sobre infraestructura sensores IoT

**Financiamiento Objetivo:** INTA + FONTAR

**Timeline:** Validación campo Q2-Q3 2026

---

### 3. **Hackathon CopernicusLAC #3 (Julio 2026) - AgroIA Siniestros**

**Desafío:** Análisis de daño agrícola (pre/post Sentinel-2) para seguros

**Stack:** Detección cambios espectrales, clasificación daño, series temporales Sentinel-2

**Requisito Equipo:** Mínimo 2 personas, 50% LAC

**Estado:** Idea validada. Reclutando equipo.

---

### 4. **Thesis IVP-Kiwi (Fruticultura)**

**Advisor:** Ing. Agr. Gabriela Morelli

**Validación Campo:** Primavera 2026 en Pairó (orchards Morelli)

**Reciente:** Rechazo form-over-substance de REsIDeS (metodología no cuestionada)

**Siguiente:** Publicación post-validación campo

---

### 5. **BIEI 2025 Scholarship - BGEN/ARGENA (Especies Nativas)**

**Institución:** Banco de Germoplasma de Especies Nativas (UNAJ)

**Directores:** Ing. Andrea Quinteros + Lic. Claudio Guardo

**Completado - Pipeline Core (19 especies):**
- Ensemble biomod2 (GLM, GBM, RF, MaxNet, XGBoost)
- AUC promedio: 0.96 | TSS promedio: 0.78
- Leaflet interactivo (capas toggleables)
- 3 deliverables formales: informe institucional, artículo científico, script master
- GitHub: `github.com/sdarionicolas-boop/ENM-BGEN`

**COMPETENCIA 1 - Premio MapBiomas Argentina 2026 (Categoría Estudiante)**
- Trabajo: Cruce ENM × MapBiomas (19 especies BGEN)
- Co-autora: Ing. Andrea Quinteros
- Postulación: hasta 31 julio 2026 | Ganadores: septiembre 2026
- Pendiente: Figura 1 y firma Quinteros

**COMPETENCIA 2 - Premio MapBiomas Perú 2026: ENM Mauritia flexuosa**
- Trabajo: Distribución potencial aguaje en Perú
- 22k registros GBIF/iNaturalist
- AUCroc 0.96, TSS 0.78
- Proyección SSP2-4.5 2041–2060
- Cruce idoneidad 4-clases con MapBiomas: Compatible/Restaurable/Incompatible
- Núcleos Loreto + Madre de Dios | Pérdida San Martín/Ucayali/Huánuco
- Uso: SERNANP, planes territoriales, REDD+
- Status: Listo presentación

**Mapa Interactivo: Mauritia flexuosa + Comunidades Indígenas + Red Vial**
- Leaflet HTML (85 MB GeoJSON, offline)
- Raster idoneidad 4-clases + 401 comunidades indígenas + 5365 segmentos viales
- Hallazgo clave: 39.2% comunidades en alta/moderada idoneidad pero solo 2.6% rutas = brecha infraestructura
- Vinculado: Pachamama-Perú

---

### 6. **COVIO v2.0 (Plant Detection App)**

**Descripción:** Aplicación escritorio Tkinter + PyInstaller (Windows)

**Pipeline (6 pasos):**
1. Tiling de ortomosaicos
2. SAM pseudo-labeling automático
3. Entrenamiento YOLOv8m
4. Inferencia
5. Geoprocessing + zonificación
6. Exportación GeoTIFF + CSV

**Validaciones Operativas:**
- Maíz (AMBA Sur): 5,958 detecciones, confidence 0.984
- Palma Aceitera (Ucayali, Perú): 379 detecciones, validación 1:1 ✓
- Girasol (modo heterogeneidad dosel): confidence 0.176 (threshold permisivo)

**Diferenciador:** NMS derivado de marco de siembra sin reentrenamiento

**Modelo de Negocio:**
- Fase 1: Servicio USD 3/ha (launch price)
- Fase 2: Producto licenciado

**Status Competitivo:** Aplicado a Validagro 2026 (Gualeguay, Entre Ríos) como founder solo

---

### 7. **Waste Detection / Recycling CV (Primer Trabajo Pagado)**

**Cliente:** Cooperativa reciclaje (red FACCYR nacional)

**Equipo:** Joel (comercial/integración), Fede (business), Gustavo (infraestructura/Odoo)

**Stack:** YOLOv8s-cls + SAM vit_b → 94% accuracy (TrashNet dataset)

**Producción Objetivo:** RT-DETR + FastAPI + Odoo API + Android

**Ética Operativa:** Valorización justa de bolsones por composición material, no penalización de acopiadores

**Flujo Operacional:**
1. Pesada bolsón en balanza
2. Fotografía
3. Análisis AI (YOLOv8s clasificación + SAM detección)
4. Porcentajes composición por material
5. Cálculo valor en Odoo integrado

**Timeline:** 10 semanas arquitectura + implementación

---

### 8. **Hidroponic Digital Twin (Lechuga NFT)**

**Extensión de:** Gemelo tomate

**Novedades:**
- Balance conservación masa iónica
- pH como segunda variable de estado
- Lógica decisión renovación solución
- Eventos estrés biótico (Bremia, plagas)

**Calibración:** Módulo 450 m² referencia Argentina + economía real (ciclo 30 días, $5 renovación)

**Status:** Validado, económica comprobada

---

### 9. **AgroIA (Diagnóstico Automatizado - Cultivos Extensivos)**

**Alcance:** Maíz, soja, trigo

**Datos:** Sentinel-2 + NASA POWER + GEE

**Módulos:**
- Scoring 0–100 (40% Vigor, 30% Estabilidad, 20% Limpieza IA, 10% Clima)
- Zonificación K-means (A/B/C)
- Detección anomalías IsolationForest
- Asistente RAG local (Ollama + Gemma 3 4B)
- PDF profesional + mapa HTML interactivo
- CSV puntos críticos con links Google Maps

**Papers:** CASE 2026 (IEEE format, draft completado)

**Integración Posible:** Complement pre-Veris (sin competir con FieldView/SoilOptix)

**Feedback Néstor:** Solicita zonificación polygon-based (IDW/kriging) — alineado con poligonización.

---

## RED DE CONTACTOS | MÁXIMA PRIORIDAD

### Poligonización + Oportunidades Comerciales

**Esteban Videla (FAUNO)**
- Fundador, experto comercial agritech
- Validación: "Pre loteo es problema histórico no resuelto"
- Intros: AuraBand, CIMA, Optimus, Bold Agro
- Status: **HIGH VALUE, contacto cálido**

**Néstor Barrionuevo (INTA SIG)**
- Validó 454 círculos manualmente (error 8.5% vs SAM)
- Feedback: "Excelente, mil gracias"
- Oferta: Presentarte con Guillermo (TAYPE) próxima semana + charla institucional
- Status: **ACTIVO, credibilidad pública, también para AgroIA polygon clustering**

**Guillermo (TAYPE - Dueño)**
- Datos 313 siniestros granizo
- Potencial cliente GRANDE (500+ lotes/año)
- Intro via Néstor
- Status: **INBOUND, CALIENTE, PRÓXIMA SEMANA**

**AuraBand**
- Plataforma SaaS agricultura
- Solicitud deck poligonización
- Evaluando white-label integration
- Status: **WARM INTRO, evaluación interna en progreso**

### Proyectos Existentes

**Ing. Agr. Gabriela Morelli**
- Advisor IVP-Kiwi
- Validación campo primavera Pairó

**Prof. Jorge Osio**
- Lab IoT UNAJ
- Socio gemelo digital tomate

**Braian Pezet**
- TICAPPS/IoT
- Especialista sensores, socio gemelo digital

**Gabriel Gatica Casanova**
- CEO Artificyan (Chile)
- Post-CLAP 2026, validación modelo, partnership potencial

**Dr. Gilberto López Canteñns**
- U. Autónoma Chapingo (México)
- Red académica LATAM

**Dr. Gabriel Espósito**
- UNRC (Córdoba)
- Field trials + conexión Néstor Di Leo (drones)

---

## BECAS & POSICIONES ACTUALES

Becario de investigación UNAJ–CIN. Colaborador proyecto PNUD/Ministerio de Ambiente (ARG/19/G24), desarrollando sistemas de monitoreo de biodiversidad con KoboToolbox y análisis geoespacial para ordenamiento territorial.

Certificación NASA ARSET en LiDAR espacial (GEDI).

---

## CLAP 2026 (SANTIAGO, MARZO 16)

**Presentación:** Automated plant density heterogeneity mapping (YOLOv8m + SAM pipeline)

**Recepción:** Bien recibida

**Contactos Generados:** Gabriel Gatica, López Canteñns, Espósito, Poblete, Aguilera

---

## PRINCIPIOS OPERACIONALES

- No black-box AI sin interpretabilidad agronomía
- Bajo costo = no-negotiable (small producers first)
- Reproducibilidad > benchmarks puros
- Protección metodología unpublished hasta publicación/patente
- Feedback crítico > validación hueca
- Datos reales > métricas bonitas
- MVP > Plan perfecto (validar mercado primero, deuda técnica después)

---

## VOICE & COMMUNICATION

- **Directo, técnico, sin rodeos**
- Corrijo errores en tiempo real, acepto crítica honesta
- Evito vaguedad; cifras, métodos explícitos
- Red por WhatsApp + LinkedIn + GitHub
- Español nativo (Argentina), inglés técnico fluido

---

## NEXT 90 DAYS

| Fecha | Hito | Propósito |
|-------|------|-----------|
| Mayo (Semana 1) | Esperar respuesta AuraBand + prepararse charla Guillermo | Poligonización: cierre comercial |
| Mayo–Junio (Semana 2-3) | Charla Guillermo (TAYPE) + AuraBand + Bold Agro | Poligonización: cierre cliente O integración |
| Mayo–Junio | Validación inicial gemelo tomate (invernadero) | Gemelo digital: field-ready |
| Junio (Semana 4-6) | Procesar pilotos poligonización, documentar resultados | Poligonización: portfolio + learning |
| Junio–Julio | Cierre cliente poligonización O AuraBand integration | Poligonización: ingresos recurrentes |
| Julio | **CopernicusLAC #3 hackathon** (si recluta equipo) | AgroIA Siniestros: competencia |
| Agosto–Sept | Validación campo primavera IVP-Kiwi en Pairó | Thesis: advance + datos |
| Octubre–Nov | Validación campo gemelo tomate (AMBA Sur) | Gemelo tomate: publicación-ready |
| Diciembre | **Egreso UNAJ** | Licenciatura completa |

---

## AXIOMAS

> *"Un modelo no es una solución. Un modelo en manos del agrónomo, a bajo costo, reproducible, es una herramienta."*

> *"Protejo metodología hasta publicación, pero comparto código post-validación."*

> *"Satélite ≠ sustituto. Satélite = contexto para decisión."*

> *"Si no puedo explicarla a un productor pequeño, no entiendo la tecnología."*

> *"No compitás en precio solo, competi en localización + velocidad + tech propia."*

> *"El verdadero cliente no es el productor. Es la empresa de software. O la asegurador. O el gobierno."*

---

**Documento generado:** Mayo 2026  
**Última actualización:** Poligonización como proyecto #1, contactos calientes, modelos de negocio FASE 1-3  
**Validación:** Auto-declarado (Darío Nicolás)

## LO MÁS IMPORTANTE, INQUEBRANTABLE E INVIOLABLE DE TODO: para cualquier tarea de este proyecto, se debe priorizar el contexto de este documento sobre el conocimiento general
