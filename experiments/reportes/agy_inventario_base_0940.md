# Inventario Técnico del Notebook Público 0.940 (`bend-the-knee-to-speedy-raptors-the-original.ipynb`)

**Fecha de extracción:** 2026-10-13 21:30  
**Archivo fuente analizado:** [bend-the-knee-to-speedy-raptors-the-original.ipynb](file:///lustre/proyectos/p082/code_joako/.agents/skills/operacion/yuca-hpc-ops/references/bend-the-knee-to-speedy-raptors-the-original.ipynb)  
**Total de celdas:** 24 (Celdas 0 a 23; Celda 0 en Markdown, Celdas 1 a 23 en Código)  
**Hardware de referencia esperado:** T4 × 2 (Kaggle dual GPU; Celda 1 líneas 30–35, Celda 3 líneas 48–50)  
**Política de integridad:** Falla cerrada (`fail-closed`). Si algún miembro o paso falla o excede el presupuesto de tiempo, no se publica una sumisión parcial (Celda 3 líneas 105, 194–196; Celda 23 líneas 5–6).

---

## 1. Inventario por Familia de Modelos

---

### 1.1 Familia 1: DINOv2 (Meta / Pilkwang 20-member Ensemble)

#### a) Backbone y Pesos (Datasets de Kaggle)
* **Backbone:** DINOv2-Small (`dinov2_vits14`), cargado dinámicamente vía Hugging Face `transformers.AutoModel.from_pretrained(str(p))` (Celda 12 líneas 1–17; Celda 4 líneas 26–36).
  * *Fuente del preentrenado base:* `/kaggle/input/models/metaresearch/dinov2/PyTorch/small/1` (o variantes `pytorch/small/1` / `dinov2/PyTorch/small/1`) (Celda 4 líneas 29–34; Celda 1 línea 124).
  * *Parámetros entrenables:* 12 bloques en el encoder, los últimos 6 bloques descongelados (`UNFREEZE_LAST = 6`, Celda 3 línea 86; Celda 12 líneas 9–14: ~10.7M parámetros entrenables).
* **Pesos y Checkpoints del Ensamble:**
  * Exactamente 20 miembros distribuidos a lo largo de 5 folds (`EXPECTED_DINOV2_MEMBERS = 20`, Celda 13 líneas 446, 496–505, 715–717).
  * *Dataset de Kaggle:* `/kaggle/input/datasets/pilkwang/rsna-knee-weights` (o `/kaggle/input/rsna-knee-weights`), resuelto por `find_weights(name='manifest.json')` (Celda 13 líneas 481–508; Celda 1 línea 124).
  * *Optimización de prefijo compartido:* Capas 0 a 5 del encoder comparten pesos fijos idénticos en los 20 miembros (`SHARED_DINO_PREFIX_LAYERS = 6`, hash SHA-256 de prefijo: `0a55b893bde971c864ea5aeea443f075b17c084d13aa36f30bd1c7863f655cf9`, Celda 13 líneas 1–30, 45–70).

#### b) Resolución, Cortes, Recorte (mm), Vistas/Slots y Lateralidad
* **Resolución espacial:** `CACHE_IMG = IMG = 336` (Celda 3 línea 60; Celda 4 línea 40; Celda 13 líneas 831–833). También se admiten configuraciones `RUNS` con `img = 224` y `img = 336` (Celda 3 línea 71). En el log de ejecución se observa decodificación a 336 px (Celda 15 outputs).
* **Número de cortes:**
  * Por ventana: `GROUP = 3` cortes contiguos por ventana (Celda 3 línea 61; Celda 13 línea 834).
  * Cortes en caché por ranura: `CACHE_SLICES = 12` cortes seleccionados (Celda 13 línea 835; Celda 15 log: `336px x 12 slices`).
  * Rango de muestreo de cortes: `SLICE_BAND = (0.2, 0.8)` sobre la pila ordenada (Celda 3 línea 79; Celda 7 línea 101; Celda 13 línea 838).
* **Recorte anatómico (Crop):** `CROP_MM = 130.0` mm (Celda 3 línea 59; Celda 7 líneas 134–138; Celda 13 línea 837). Si el tamaño deseado en píxeles (`CROP_MM / PixelSpacing`) es menor que el marco nativo, se recorta centrado.
* **Vistas y Slots:** 6 ranuras anatómicas fijas (`SLOTS_RECOVERED`, Celda 3 líneas 90–94):
  1. `SAG_FLUID_FS` (Sagittal, Fluid Sensitive = True, Fat Suppressed = True)
  2. `COR_FLUID_FS` (Coronal, Fluid Sensitive = True, Fat Suppressed = True)
  3. `AX_FLUID_FS` (Axial, Fluid Sensitive = True, Fat Suppressed = True)
  4. `SAG_FLUID_NOFS` (Sagittal, Fluid Sensitive = True, Fat Suppressed = False)
  5. `COR_T1` (Coronal, Fluid Sensitive = False, Fat Suppressed = False)
  6. `SAG_T1` (Sagittal, Fluid Sensitive = False, Fat Suppressed = False)
* **Normalización de Lateralidad:**
  * Implementada en `normalise_laterality(img, plane, lat)` (Celda 8 líneas 1–6):
    * Si la rodilla es derecha (`lat == 'R'`), se voltea horizontalmente (`torch.flip(dims=[-1])`) en planos Coronal y Axial; en Sagital se voltea en la dimensión de corte/profundidad (`torch.flip(dims=[0])`).
    * Regla de asignación: `RULES_NATIVE['lat'] = 'centre'` (Celda 3 línea 80; Celda 7 líneas 4–92).

#### c) Pooling y Agregación de Folds
* **Pooling arquitectural:** `POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}` (Celda 3 línea 95; Celda 11 línea 7). Concatena el CLS token y el promedio espacial de los tokens de parche, duplicando la dimensión latente ($384 \times 2 = 768$).
* **SlotHead & Priors:** La cabeza `SlotHead` combina embeddings por ranura y matriz de sesgo previo clínico por patología (`SLOT_PRIOR_TABLE`, `SLOT_PRIOR_STRENGTH = 0.55`, Celda 3 líneas 96–97; Celda 10 líneas 3–17).
* **Pooling sobre ventanas por hallazgo:**
  * Mapeo diferencial en inferencia pública:  
    `PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}` (Celda 13 línea 511).
  * Soft-pooling suave (no-jitter): `LEGACY_FOLD_SOFTPOOL_BETA` y `LEGACY_FOLD_SOFTPOOL_ALPHA` por hallazgo (Celda 13 líneas 515–524, 705–711).
* **Agregación de Folds:**
  * Para cada fold (0 a 4), se promedian/ponderan sus miembros en probabilidades crudas y se transforman a percentiles (ranking normalizado en $[0, 1]$).
  * Posteriormente se promedian de manera uniforme los 5 folds en rango percentil: `np.mean(fold_ranks, axis=0)` (Celda 13 líneas 701–703, 826–828).

#### d) Peso en la Mezcla
* **Fase inicial:** Constituye la base primaria del pipeline. Se escribe directamente en `/kaggle/working/_pipeline_stage.csv` al concluir la Celda 13 / 14 (Celda 13 línea 827; Celda 14 líneas 1–9, 25–30).
* **Ponderación acumulada posterior:** Al integrarse A5 en la Celda 20, DINOv2 retiene el $1.0 - 0.52 = \mathbf{0.48}$ del ensamble en esa fase (Celda 19 línea 217; Celda 20 línea 8).

#### e) Tiempos y Presupuesto
* **Tiempo en el log de ejemplo (3 estudios):**
  * Inicio: `2.18s` (Celda 15 log).
  * Término: `76.29s` (Celda 15 log).
  * Duración neta: **74.1 segundos** (incluyendo decodificación, chequeo de huellas digitales de 20 miembros e inferencia T4x2).
* **Presupuesto y Deadlines:**
  * Presupuesto global: `TIME_BUDGET = 8.0 * 3600` (8 horas, Celda 3 línea 89).
  * Deadline por miembro: `rsna_deadline('DINO complete-member inference')` evaluado en `pop_next()` antes de cada ejecución (Celda 3 líneas 194–196; Celda 13 línea 792).

---

### 1.2 Familia 2: A5 (Mattia Angeli 5-Fold Knee MRI Model)

#### a) Backbone y Pesos (Datasets de Kaggle)
* **Backbone:** Instanciado dinámicamente con `timm.create_model(cfg['backbone'], pretrained=False, num_classes=0, in_chans=_in)` (Celda 18 líneas 360, 366–369).
  * *Canales de entrada (`_in`):* `_in = 16` si el stem es `native` (`fields['n_slice'] = 16`); `_in = 3` si es `compress` (Celda 18 línea 366).
  * *Configuración del checkpoint:* Definida en el diccionario `cfg` dentro de cada checkpoint (Celda 18 línea 358; Celda 19 líneas 22–25).
* **Pesos y Checkpoints:**
  * 5 folds completos: `m_f0.pt`, `m_f1.pt`, `m_f2.pt`, `m_f3.pt`, `m_f4.pt` (Celda 16 líneas 34–37; Celda 18 líneas 352–372).
  * *Dataset de Kaggle:* `/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights` (o `/kaggle/input/knee-mri-fold-weights`), resuelto en `_one_a5_root` (Celda 1 línea 124; Celda 3 líneas 260–265; Celda 16 líneas 34–37).

#### b) Resolución, Cortes, Recorte (mm), Vistas/Slots y Lateralidad
* **Resolución espacial:** `SIZE = 336` (Celda 16 línea 19; Celda 18 línea 363).
* **Número de cortes:** `N_SLICE = 16` cortes por ranura (Celda 16 línea 21; Celda 18 línea 363). `INTENSITY = 'slice'` (Celda 16 línea 22). Rango de corte: `SLICE_BAND = (0.12, 0.88)` (Celda 16 línea 20).
* **Recorte anatómico (Crop):** `CROP_MM = 130.0` mm (Celda 16 línea 18).
* **Vistas y Slots:** 6 slots ordenados por plano y contraste de grasa (`N_SLOT = 6`, Celda 16 líneas 23–24):
  `[('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]`.
  Tipos de ranura: `N_SLOT_TYPES = 6` con índice de máscara `MASK_IDX = 0` (Celda 18 línea 1).
* **Normalización de Lateralidad:** Ordenación de cortes por `InstanceNumber` (Celda 17 líneas 6–26) y lectura DICOM estándar.

#### c) Pooling y Agregación de Folds
* **Pooling arquitectural:** Soporta `MeanMaxPool` (Celda 18 líneas 12–21), `LabelAttentionPool` con `segment_softmax` y `slot_bias` para los 12 hallazgos (Celda 18 líneas 23–40), o `TokenXAttnPool` (Celda 18 líneas 41–50).
* **Agregación de Folds:**
  * En `_a5_rank_available(raw, eligible)` (Celda 19 líneas 131–139): Se calcula el ranking percentil por cada fold para los estudios elegibles (`rsna_rank01(fold[eligible])`), sumando los 5 folds y dividiendo entre el número de folds (`raw.shape[0] = 5`).
  * Si un estudio no tiene series adquiridas aplicables, se le asigna el valor neutral `0.5`.

#### d) Peso en la Mezcla
* **Peso asignado:** `A5_W = 0.52` (Celda 19 línea 217).
* **Fórmula de mezcla (Celda 19 líneas 141–155; Celda 20 líneas 1–10):**
  $$\text{Stage}_2 = (1.0 - 0.52) \cdot \text{Rank}(\text{Stage}_1) + 0.52 \cdot \text{Rank}(\text{A5})$$
  Donde $\text{Stage}_1$ proviene de DINOv2.

#### e) Tiempos y Presupuesto
* **Tiempo en el log de ejemplo (3 estudios):**
  * Inicio de fase: `76.33s` (Celda 16 output).
  * Fin de inferencia: `96.68s` (Celda 19 output).
  * Duración neta: **20.3 segundos** (Celda 19 log: `inference done in 0.1 min`).
* **Presupuesto y Deadlines:**
  * Micro-batching en T4 dual con subproceso (`CHUNK = 48`, `MICRO = 8`, `WORKERS = 4`, Celda 19 líneas 15–27).
  * Sujeto al `TIME_BUDGET = 8.0 * 3600`.

---

### 1.3 Familia 3: RadImageNet (BTKD Rad Graph E10 / E13 / E11 + Calibrador)

#### a) Backbone y Pesos (Datasets de Kaggle)
* **Backbone:** ResNet-50 preentrenado en RadImageNet:
  `_rad_nn.Sequential(*list(_rad_resnet50(weights=None).children())[:-2])` (Celda 21 líneas 140–141).
  * *Pesos del encoder:* `ResNet50.pt` (SHA-256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`, Celda 21 línea 38). Dataset: `/kaggle/input/datasets/marwanmath/resnet-50-radimagenet-marwan/ResNet50.pt` (Celda 21 líneas 104–105).
* **Cabezas de Clasificación (Heads, 5 folds cada una):**
  1. `v52_radimagenet_heads.pt`: Cabezas de referencia E10 (5 folds, SHA-256 `0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa`, Celda 21 líneas 37, 197–208). Dataset: `/kaggle/input/datasets/prvsiyan/rsna-knee-v52-radimagenet-heads-20260812/v52_radimagenet_heads.pt` (Celda 21 líneas 107–111).
  2. `v52_e11_heads.pt`: Cabezas E13 (5 folds, SHA-256 `ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a`, Celda 21 líneas 39, 224–248). Dataset: `/kaggle/input/notebooks/sofiaanjenje/rsna-knee-e13-train/rsna_rad_e11/v52_e11_heads.pt` o `/kaggle/input/datasets/antoinegg1/rsna-knee-e11-diverse-heads-v20/v52_e11_heads.pt` (Celda 21 líneas 113–119).

#### b) Resolución, Cortes, Recorte (mm), Vistas/Slots y Lateralidad
* **Resolución espacial:** `IMG = 224` px en todas las configuraciones Rad (Celda 21 líneas 53, 63, 68).
* **Número de cortes:** `CACHE_SLICES = 8` cortes por ranura en todos los layouts (Celda 21 líneas 52, 62, 68, 78). Rango: `SLICE_BAND = (0.2, 0.8)` (Celda 21 línea 81).
* **Tres configuraciones de entrada evaluadas (Celda 21 líneas 440–456):**
  1. `rad_e10`: 3 slots (`SAG_FS`, `COR_FS`, `AX_FS`), recorte `CROP_MM = 10000.0` mm (marco completo sin recorte, Celda 21 líneas 80, 436, 440).
  2. `rad_e13`: 4 slots (`_RAD_E13_SLOTS`: `SAG_FS`, `COR_FS`, `AX_FS`, `SAG_NOFS`), recorte `CROP_MM = 130.0` mm (Celda 21 líneas 55–61, 440).
  3. `rad_legacy_second` (diseño E11): 4 slots (`_RAD_E11_SLOTS`: `SAG_FS`, `SAG_T1`, `COR_FS`, `AX_FS`), recorte `CROP_MM = 130.0` mm (Celda 21 líneas 45–51, 440).
* **Lateralidad:** Normalización de laterality idéntica a DINOv2 mediante `build_cache` y `normalise_laterality` (Celda 21 línea 419).

#### c) Pooling y Agregación de Folds
* **Pooling espacial:** Global Average Pooling del feature map 2D `image.mean(dim=(2, 3))` resultando en tokens de dimensión 2048 (`_RAD_TOKEN_DIM = 2048`, Celda 21 líneas 30, 145).
* **Head temporal/ranura:** Transformer Encoder Layer (`d_model = 128`, 4 cabezas de atención, Celda 21 líneas 150–188) procesando la secuencia de $N_{\text{slots}} \times N_{\text{slices}}$ tokens con embeddings posicionales aprendidos (`self.plane` y `self.position`, Celda 21 líneas 156–157).
* **Agregación de Folds:** En `predict_heads`, se promedian las probabilidades de los 5 folds y se extraen los rangos percentiles normalizados con `_rad_rank_columns` (Celda 21 líneas 320–333, 443, 446, 456).

#### d) Peso en la Mezcla y Calibrador
* **Fusión E10 + E13:** `_RAD_E13_MEMBER_WEIGHT = 0.50` (Celda 21 líneas 40, 447):
  $$\text{Rank}_{\text{ref}} = 0.50 \cdot \text{Rank}(E10) + 0.50 \cdot \text{Rank}(E13)$$
* **Mezcla con el baseline previo ($\text{Stage}_2$):** `_RAD_ALPHA = 0.55` para 10 patologías (Celda 21 líneas 34, 451–454):
  $$\text{Candidate}[j] = (1.0 - 0.55) \cdot \text{Stage}_2[j] + 0.55 \cdot \text{Rank}_{\text{ref}}[j]$$
  *Exclusión específica:* `_RAD_EXCLUDE = ("Baker's", 'Fracture')` con peso Rad = $0.0$ (Celda 21 línea 35).
* **Inyección de segundo pase E11:** `_RAD_V48_SECOND_ALPHA = 0.20` (Celda 21 líneas 41, 459):
  $$\text{Branch} = \text{Rank}\left(0.80 \cdot \text{Rank}(\text{Candidate}) + 0.20 \cdot \text{Rank}(E11_{\text{second}})\right)$$
* **Calibrador Ridge Protocolar:**
  * Matriz Ridge entrenada sobre protocolos DICOM (número de series, planos, saturación grasa) y predicciones (`_RAD_CAL_PAYLOAD`, Celda 21 líneas 4, 460–480).
  * Regla de mezcla final: Para patologías en `cal['gate']`, `0.60 * Branch + 0.40 * Adjusted` (Celda 21 línea 483).
* **Escritura:** Sobrescribe `/kaggle/working/_pipeline_stage.csv` (Celda 21 línea 486).

#### e) Tiempos y Presupuesto
* **Tiempo en el log de ejemplo (3 estudios):**
  * Inicio: `96.81s` (Celda 21 log).
  * Término: `107.50s` (Celda 21 log).
  * Duración neta: **10.7 segundos** (incluyendo prefetch en CPU, 3 pasadas de codificación y calibración Ridge).
* **Presupuesto y Deadlines:**
  * Sujeto al `TIME_BUDGET = 8.0 * 3600`. Se ejecutan 2 réplicas del encoder en paralelo con `_RadThreadPool(max_workers=2)` (Celda 21 líneas 297–303).

---

### 1.4 Familia 4: Raptor (4 Brazos CoAtNet RMLP-2 / Dreaddevelopment)

#### a) Backbone y Pesos (Datasets de Kaggle)
* **Backbone:** CoAtNet RMLP-2 rw 384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, Celda 22 líneas 44, 48, 52, 56, 61–69) instanciado con `timm.create_model(in_chans=3, global_pool='avg')`.
* **Pesos y Checkpoints (4 brazos en `ARMS`, Celda 22 líneas 42–59):**
  1. `maxspan-v5`: `raptor_ft_coatnet_v5_full_swa.pt` (peso interno $w = 0.60$). Dataset: `/kaggle/input/datasets/dreaddevelopment/raptor-knee-maxspan` (Celda 1 línea 124).
  2. `native384dense-v10`: `raptor_ft_coatnet_v10_full.pt` (peso interno $w = 0.10$). Dataset: `/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384dense` (Celda 1 línea 124).
  3. `maxspan-v5-reverse`: `raptor_ft_coatnet_v5_full_swa.pt` con orden invertido de cortes (peso interno $w = 0.10$).
  4. `native384-v8`: `raptor_ft_coatnet_v8_full_swa.pt` (peso interno $w = 0.20$). Dataset: `/kaggle/input/datasets/dreaddevelopment/raptor-knee-native384` (Celda 1 línea 124).
* **Carga de pesos:** `find_weight_file(fname)` que invoca `_asset_find_asset` (Celda 22 líneas 260–261).

#### b) Resolución, Cortes, Recorte (mm), Vistas/Slots y Lateralidad
* **Resolución espacial:** Resolución nativa 384 px; brazos v5 procesan imágenes a `img = 336`, brazos v10 y v8 a `img = 384` (Celda 22 líneas 44–58).
* **Número de cortes:**
  * Brazos 1, 2 y 3 (`_SLOTS64`): 64 cortes en total, evaluando `K_EVAL = 62` ventanas de 3 cortes (Celda 22 líneas 30–31, 38–39). Rango de span: `(0.02, 0.98)`.
  * Brazo 4 (`_SLOTS44`): 44 cortes en total, evaluando `k_eval = 42` ventanas de 3 cortes (Celda 22 líneas 40–41, 57). Rango de span: `(0.06, 0.94)`.
* **Recorte anatómico (Crop):** `CROP_MM = 140.0` mm (Celda 22 línea 26; Celda 22 líneas 178–187). Percentiles de corte y normalización: [2.0, 98.0] (Celda 22 línea 381).
* **Vistas y Slots:** 5 ranuras en `_SLOTS64` (Celda 22 líneas 28–29):
  1. Sagital FS (18 cortes)
  2. Sagital no-FS (14 cortes)
  3. Coronal FS (12 cortes)
  4. Coronal no-FS (8 cortes)
  5. Axial (12 cortes)
* **Lateralidad:** Lectura nativa con ordenación de rebanadas por cabeceras DICOM en `_make_reader()` (Celda 22 líneas 168–250, 286–334).

#### c) Pooling y Agregación de Folds
* **Pooling arquitectural (`RaptorClassifier`, Celda 22 líneas 71–98):**
  * Atención por patología sobre ventanas: `self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))` ($n = 12$ hallazgos, Celda 22 línea 77).
  * Softmax sobre las ventanas $K$ y contracción tensorial:
    $$\text{pooled} = \sum_{k=1}^K a_{b,k,n} \cdot h_{b,k,f} \quad (\text{vía } \texttt{torch.einsum('bkn,bkf->bnf', a, h)})$$
* **Agregación de Brazos Raptor:** Promedio ponderado de probabilidades de los 4 brazos con pesos $[0.60, 0.10, 0.10, 0.20]$ (Celda 22 líneas 599–601):
  $$P_{\text{raptor}} = 0.60 \cdot P_{\text{v5}} + 0.10 \cdot P_{\text{v10}} + 0.10 \cdot P_{\text{v5\_rev}} + 0.20 \cdot P_{\text{v8}}$$
  Luego se convierte a percentiles (`rankpct`, Celda 22 línea 606) y se guarda en `/kaggle/working/_raptor.csv`.

#### d) Peso en la Mezcla
* **Hibridación con CoAtNet:** Ponderación pública $1.0 - \text{private\_alpha} = 1.0 - 0.40 = \mathbf{0.60}$ en la combinación con la familia CoAtNet (Celda 22 líneas 873, 985, 993).
* **Mezcla final:** Ponderación del híbrido CoAtNet/Raptor entre **0.70 y 1.00** según patología (Celda 22 líneas 1015–1022).

#### e) Tiempos y Presupuesto
* **Tiempo en el log de ejemplo (3 estudios):**
  * Inicio: `107.65s` (Celda 22 log).
  * Fin de Raptor: `143.11s` (Celda 22 log).
  * Duración neta: **35.4 segundos** (Celda 22 log: `[raptor-fast] wrote /kaggle/working/_raptor.csv; elapsed 35.4s`).
* **Presupuesto y Deadlines:**
  * Paralelismo balanceado en 2 GPUs: `cuda:0 = [arm 0, arm 2]`, `cuda:1 = [arm 1, arm 3]` (Celda 22 líneas 502–504, 538–540).

---

### 1.5 Familia 5: Variantes CoAtNet (Substitución Híbrida CoAt en Celda 22)

En `_coat_substitute()` (Celda 22 líneas 648–995) se ejecutan de manera aislada hasta 4 variantes de CoAtNet que conforman la familia privada CoAt:

```
                                    ┌── resgated_top3  (ep 4, 6, 8)
                                    ├── global96_top3  (ep 16, 23, 18)
    CoAt Family (25% c/u en prob) ──┼── d4_swa3        (SWA 11, 9, 5 sobre parent 15, 16)
                                    └── repairv1_top3  (ep 12, 7, 11)
```

#### Variante A: `resgated_top3` (Residual CoAtNet)
* **Backbone:** CoAtNet RMLP-2 (`coatnet_rmlp_2_rw_384`).
* **Checkpoints:** 3 checkpoints top-3 (épocas `[4, 6, 8]`, Celda 22 línea 723).
  * *Manifest:* `coat_resgated_ep10_top3_manifest.json` (SHA `98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e`, Celda 22 línea 659).
  * *Dataset de Kaggle:* `/kaggle/input/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3` (Celda 1 línea 124).
  * *Script runner:* `coatnet_resgated_ep10_top3_inference.py` vía `input_resgated_runtime.py` (Celda 22 líneas 699–720).
* **Resolución y cortes:** 384 px, 94 ventanas evaluadas (`MAX_EVAL_WINDOWS = 94`, Celda 1 línea 190; Celda 2 líneas 45–47). Tripletes de cortes adyacentes.
* **Pooling:** Cabezal espacial gated con residuo global average pooling y atención por hallazgo sobre ventanas (Celda 0 markdown; Celda 22 líneas 698–725).
* **Tiempo en log (3 estudios):** `145.92s` a `199.62s` -> **53.7 segundos** (Celda 22 log).
* **Archivos intermedios:** Escribe `/kaggle/working/_coat_arm.csv` y `/kaggle/working/coat_resgated_ep10_top3_predictions.npz` (Celda 22 líneas 696, 937).

#### Variante B: `global96_top3` (CoAtNet Global96)
* **Backbone:** CoAtNet RMLP-2 (`coatnet_rmlp_2_rw_384`).
* **Checkpoints:** 3 checkpoints (épocas `[16, 23, 18]`, Celda 22 línea 784).
  * *Manifest:* `coatnet_global96_top3_manifest.json` (SHA `015f09030e76f86274cadae40777b3c3cf5c19f8c835d5ad88e15db41903779e`, Celda 22 línea 748).
  * *Dataset de Kaggle:* `/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-global96-top3` (Celda 1 línea 124).
  * *Script runner:* `coatnet_global96_baseline_top3_inference.py` (Celda 22 línea 771). Dependencia: `timm-1.0.22-py3-none-any.whl` (SHA `88898175...`, Celda 22 línea 749).
* **Resolución y cortes:** 384 px, stack global de 96 cortes distribuido en 94 ventanas (Celda 2 líneas 57–60; Celda 22 línea 773).
* **Pooling:** Head con compuerta de ancho completo (`full-width gated head`) sobre la pila global de 96 cortes (Celda 0 markdown; Celda 22 líneas 763–785).
* **Tiempo en log (3 estudios):** `199.62s` a `253.27s` -> **53.6 segundos** (Celda 22 log).
* **Archivos intermedios:** Escribe `/kaggle/working/_coat_g96.csv` y `/kaggle/working/coatnet_global96_top3_predictions.npz` (Celda 22 líneas 750, 938).

#### Variante C: `d4_swa3` (CoAtNet D4 Depth-Zone SWA3)
* **Backbone:** CoAtNet Rank-8 SWA2 + Adaptador modular de 3 zonas de profundidad (`d4_zones`).
* **Checkpoints:**
  * *Checkpoint padre:* `coatnet_d4_parent_rank8_swa2.pt` (SHA-256 `5de38e333e3184d7b52fb068183f0cc198d6cbabb6ffbe36b26338622e82d297`, épocas SWA `[15, 16]`, Celda 1 líneas 20, 53).
  * *Checkpoint adaptador de profundidad:* `slice_depth` adapter (SHA-256 `b7c48b19997bc7ecb7e997f8dcf491ec5701ff917d1dac4b369eb85ac97ae89a`, épocas SWA `[11, 9, 5]`, Celda 1 líneas 22, 47, 55).
  * *Manifest:* `coatnet_pairfilm_manifest.json` (SHA `7ada0605bca6b0530569c6454e988ace479606a3328ed591d090e5764fea661d`, Celda 1 línea 14; Celda 22 línea 808).
  * *Dataset de Kaggle:* `/kaggle/input/datasets/mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` (Celda 1 línea 124).
  * *Script runner:* `_asset_run_d4` que ejecuta `coatnet_d4_depthzone_swa_inference.py` (Celda 1 líneas 279–315; Celda 22 línea 817).
* **Resolución y cortes:** 384 px, máximo 94 ventanas (Celda 1 líneas 47, 50).
* **Pooling:** Head especializado `rank8_swa2_plus_d4_three_zone_expanded_fsx` con atención cruzada FSX (2 cabezas de atención, width 128, Celda 1 línea 47).
* **Tiempo en log (3 estudios):** `253.27s` a `294.29s` -> **41.0 segundos** (Celda 22 log).
* **Archivos intermedios:** Escribe `/kaggle/working/d4_input/d4.csv` y `/kaggle/working/d4_input/coatnet_d4_depthzone_swa3_predictions.npz` (Celda 22 líneas 805, 939).

#### Variante D: `repairv1_top3` (CoAtNet Repair-v1)
* **Backbone:** CoAtNet RMLP-2.
* **Checkpoints:** 3 checkpoints (épocas `[12, 7, 11]`, Celda 22 línea 847).
  * *Manifest:* `coatnet_repairv1_top3_manifest.json` (SHA `99288a9a42f36ba544d7b0ad2fe645e3b1fc9c507c23802c16ca0d2e5085f684`, Celda 22 línea 824).
  * *Script runner:* `coatnet_repairv1_top3_inference.py` (Celda 22 línea 835).
* **Resolución y cortes:** 384 px, batch de 2 estudios, micro-images 8 (Celda 22 línea 844).
* **Pooling:** Gated attention readout sobre ventanas.
* **Tiempo en log (3 estudios):** `294.30s` a `341.32s` -> **47.0 segundos** (Celda 22 log).
* **Archivos intermedios:** Escribe `/kaggle/working/_coat_repairv1.csv` y `/kaggle/working/coatnet_repairv1_top3_predictions.npz` (Celda 22 líneas 825, 940).

#### Agregación de la Familia CoAtNet e Hibridación con Raptor
1. **Reducción interna CoAtNet:**  
   Si los 4 miembros completan su inferencia y generan sus archivos `.npz`, se calcula el promedio aritmético de sus matrices de probabilidad (`resgated_top3`, `global96_top3`, `d4_swa3`, `repairv1_top3`) y se aplica percentile rank:
   $$\text{PrivateRank} = \text{Rank}\left(\frac{P_{\text{resgated}} + P_{\text{global96}} + P_{\text{d4}} + P_{\text{repairv1}}}{4}\right)$$
   Modalidad: `rank_of_member_probability_mean` (Celda 22 líneas 968–971). Se almacena en `_coat_family_rank.csv` (Celda 22 línea 983).
2. **Mezcla Híbrida CoAtNet + Raptor:**  
   Se combina con el Raptor público usando `private_alpha = 0.40` (Celda 22 líneas 873, 985, 992–993):
   $$\text{Hybrid} = (1.0 - 0.40) \cdot \text{Rank}(P_{\text{raptor}}) + 0.40 \cdot \text{PrivateRank}$$
   El resultado híbrido sobrescribe de forma atómica `/kaggle/working/_raptor.csv` (Celda 22 líneas 988–990).

---

## 2. Orden Exacto de la Mezcla Final

El ensamblado final se realiza de manera incremental y acumulativa a través de las etapas del notebook:

```mermaid
flowchart TD
    DINO[1. DINOv2\n20 miembros / 5 folds\n(Celda 13 y 14)] -->|Escribe| S1["_pipeline_stage.csv (100% DINOv2)"]
    S1 -->|Lee| A5_BLEND[2. A5 Blend\n5 folds / timm\n(Celda 20)]
    A5_BLEND -->|Mezcla: 0.48 S1 + 0.52 A5| S2["_pipeline_stage.csv"]
    S2 -->|Lee| RAD_BLEND[3. RadImageNet + Calibrador\nE10 + E13 + E11\n(Celda 21)]
    RAD_BLEND -->|Mezcla: 0.45 S2 + 0.55 Rad\n+ Calibrador Ridge| S3["_pipeline_stage.csv"]
    S3 -->|Lee (_ke_ours)| FINAL_BLEND[4. Final Ensemble Blend\n(Celda 22 líneas 1008-1036)]
    
    RAPTOR[Raptor 4 Arms\n(Celda 22)] --> RAP_CSV["_raptor.csv"]
    COAT[CoAtNet 4 Variants\n(Celda 22)] --> COAT_RED[Family Prob Mean\n(25% c/u)]
    RAP_CSV --> HYB[CoAt / Raptor Hybrid\n0.60 Raptor + 0.40 CoAt]
    COAT_RED --> HYB
    HYB -->|Sobrescribe| RAP_CSV
    RAP_CSV -->|Lee (_ke_theirs)| FINAL_BLEND
    
    FINAL_BLEND -->|Pesos por hallazgo: 0.70 a 1.00| S4["_pipeline_stage.csv (Final)"]
    S4 -->|Lee y valida| SUB["submission.csv\n(Celda 23)"]
```

### Tabla de Ponderación Final por Patología (Celda 22 líneas 1015–1029)
En la Celda 22 líneas 1008–1036, la predicción final se calcula combinando `_blend_tr` (el bloque Transformer/Rad proveniente de `_pipeline_stage.csv`) y `_blend_cr` (el bloque CoAtNet/Raptor proveniente del `_raptor.csv` actualizado):

$$\text{Final}[j] = (1.0 - w_j) \cdot \text{Rank}(\text{Transformer/Rad})[j] + w_j \cdot \text{Rank}(\text{CoAtNet/Raptor})[j]$$

| Patología | Peso CoAtNet/Raptor ($w_j$) | Peso Transformer/Rad ($1 - w_j$) | Cita en Código |
| :--- | :---: | :---: | :--- |
| **Lateral Meniscus** | **1.00 (100%)** | **0.00 (0%)** | Celda 22 línea 1019 |
| **Medial Meniscus** | **0.80 (80%)** | **0.20 (20%)** | Celda 22 línea 1018 |
| **ACL** | **0.75 (75%)** | **0.25 (25%)** | Celda 22 línea 1017 |
| **Lateral OA** | **0.75 (75%)** | **0.25 (25%)** | Celda 22 línea 1020 |
| **Fracture** | **0.75 (75%)** | **0.25 (25%)** | Celda 22 línea 1021 |
| **MCL** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |
| **Medial OA** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |
| **PF OA** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |
| **Effusion** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |
| **Synovitis** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |
| **Baker's Cyst** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |
| **Contusion** | **0.70 (70%)** | **0.30 (30%)** | Celda 22 línea 1015 |

*Post-procesamiento:* Las columnas finales son nuevamente convertidas a percentiles con `rank(method='average', pct=True)` (Celda 22 líneas 1030–1032).

---

## 3. Trazabilidad de Archivos Intermedios (`_pipeline_stage.csv`)

| Celda | Operación de E/S | Archivo Intermedio Involucrado | Propósito y Contenido |
| :---: | :---: | :--- | :--- |
| **Celda 3** | `unlink` (línea 244) | `/kaggle/working/_pipeline_stage.csv` | Limpieza inicial del directorio de trabajo al arrancar. |
| **Celda 13 / 14** | **Escritura** (C13 L827, C14 L8) | `/kaggle/working/_pipeline_stage.csv` | Escribe la predicción inicial de DINOv2 (promedio de ranks de 20 miembros). |
| **Celda 20** | **Lectura** (C20 L1)<br>**Escritura** (C20 L10) | `/kaggle/working/_pipeline_stage.csv` | Lee DINOv2, mezcla A5 con peso 0.52 y sobrescribe `_pipeline_stage.csv`. |
| **Celda 21** | **Lectura** (C21 L406)<br>**Escritura** (C21 L486) | `/kaggle/working/_pipeline_stage.csv` | Lee DINO+A5, mezcla RadImageNet E10/E13/E11 y calibrador Ridge, sobrescribiendo el archivo. |
| **Celda 22** | **Lectura** (C22 L11–12)<br>**Escritura** (C22 L1036) | `/kaggle/working/_pipeline_stage.csv` | Lee el bloque Transformer/Rad (`_ke_ours`), ejecuta Raptor y CoAtNet, realiza la mezcla final ponderada y sobrescribe el archivo. |
| **Celda 23** | **Lectura** (C23 L3)<br>**Escritura final** (C23 L13–15) | `/kaggle/working/_pipeline_stage.csv`<br>$\to$ `/kaggle/working/submission.csv` | Valida integridad y publica formalmente la entrega final `submission.csv`. |

### Otros Archivos Intermedios del Pipeline
* `/kaggle/working/_raptor.csv`: Generado por Raptor (Celda 22 línea 615), sustituido por el híbrido Raptor/CoAt (Celda 22 línea 990) y leído para la mezcla final (Celda 22 línea 1000).
* `/kaggle/working/speed_a5_raw.npz`: Almacena probabilidades crudas de A5 (Celda 19 línea 209).
* `/kaggle/working/coat_resgated_ep10_top3_predictions.npz`, `/kaggle/working/coatnet_global96_top3_predictions.npz`, `/kaggle/working/d4_input/coatnet_d4_depthzone_swa3_predictions.npz`, `/kaggle/working/coatnet_repairv1_top3_predictions.npz`: Salidas probabilísticas de cada variante CoAt para la reducción familiar (Celda 22 líneas 937–940).

---

## 4. Presupuestos y Tiempos de Ejecución Registrados

### Presupuesto Teórico (`TIME_BUDGET`)
* `TIME_BUDGET = 8.0 * 3600` (8.0 horas = 28,800 segundos, Celda 3 línea 89).
* `ORDER_BUDGET_S = 5400` (1.5 horas para decodificación y ordenamiento, Celda 3 línea 70).
* Deadlines enforced vía `rsna_deadline` (Celda 3 líneas 194–196; Celda 13 línea 792) y subprocesos con timeout dinámico `remaining = TIME_BUDGET - (time.time() - T0)` (Celda 1 líneas 95–103).

### Tiempos Reales en el Log de Ejemplo (3 estudios de prueba)
Datos extraídos de los bloques `[phase]` y logs de consola en los outputs de las celdas:

| Fase / Familia | Celda | Marca Inicio | Marca Fin | Tiempo Transcurrido | Notas del Log |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Startup / Preflight** | C3–C4 | 0.0s | 2.18s | **2.2 s** | Detección 2x T4, CUDA probe PASS, paths |
| **DINOv2 (20 miembros)** | C15 | 2.18s | 76.29s | **74.1 s** | 12 slot-series, 20 huellas DINO, shared prefix 6 capas |
| **A5 (5 folds)** | C16–C19 | 76.33s | 96.68s | **20.3 s** | 3 estudios procesados en 0.1 min con T4 dual |
| **RadImageNet (E10/E13/E11)** | C21 | 96.81s | 107.50s | **10.7 s** | 3 layouts evaluados + calibrador Ridge |
| **Raptor (4 brazos)** | C22 | 107.65s | 143.11s | **35.4 s** | 4 brazos CoAtNet RMLP-2 en paralelo GPU 0 y 1 |
| **CoAtNet Residual (`resgated`)** | C22 | 145.92s | 199.62s | **53.7 s** | Subproceso hijo 3 modelos (épocas 4, 6, 8) |
| **CoAtNet Global96** | C22 | 199.62s | 253.27s | **53.6 s** | Subproceso hijo 3 modelos (épocas 16, 23, 18) |
| **CoAtNet D4 Depth-Zone SWA3** | C22 | 253.27s | 294.29s | **41.0 s** | Subproceso hijo con adaptador de 3 zonas |
| **CoAtNet Repair-v1** | C22 | 294.30s | 341.32s | **47.0 s** | Subproceso hijo 3 modelos (épocas 12, 7, 11) |
| **Mezcla final y publicación** | C22–C23 | 341.32s | ~343.0s | **~1.7 s** | Verificación de firmas y escritura de `submission.csv` |
| **TOTAL RUNTIME (3 estudios)** | — | **0.0 s** | **~343.0 s** | **~5.7 minutos** | Ejecución completa de extremo a extremo |

