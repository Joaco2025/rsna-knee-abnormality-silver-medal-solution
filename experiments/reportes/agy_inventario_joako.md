# Inventario Técnico del Lado Joako (`code_joako` y `runs_joako`)

**Fecha de corte:** 2026-10-13 19:55  
**Usuario:** `jcastro`  
**Entorno:** Clúster Yuca (ACARUS, UniSon), nodo `login-01`  
**Repositorio de trabajo:** `/lustre/proyectos/p082/code_joako` (rama `joako`)  
**Directorio de artefactos y corridas:** `/lustre/proyectos/p082/runs_joako`  
**Política de acceso:** Solo lectura de rutas propias. Aislamiento total de rutas ajenas (`/lustre/proyectos/p082/{code,runs,envios}`).

---

## 1. Paquetes y Ensembles Armados

Resumen consolidado de los ensambles principales diseñados, evaluados y/o empaquetados en el lado de Joako. Para cada ensamble se detalla su composición, pesos de agregación, checkpoints y las métricas reportadas con su fuente documental o el rótulo **"sin fuente"** cuando corresponda.

### 1.1 Ensamble Cuádruple ABCD (`cuadruple_abcd_promedio`)
* **Estado:** Empaquetado y verificado en `envios_joako/candidato_cuadruple_abcd/` (archivo autónomo `candidato_cuadruple_abcd_paquete.zip`, 385 MB).
* **Fórmula de combinación:** Promedio simple equitativo de probabilidades:
  $$\hat{P} = 0.25 \cdot P_{\text{resnet\_base}} + 0.25 \cdot P_{\text{convnext\_base}} + 0.25 \cdot P_{\text{resnet\_hib}} + 0.25 \cdot P_{\text{convnext\_hib}}$$
* **Composición y Checkpoints:**
  1. **A — ResNet34 Base:** Job `2080214`, época 7 (`pesos/resnet34_job2080214_ep7.pt`, SHA-256: `9452d38ae4d884f6222dcbd63402d6625f0fef56b7b6f566e60ac27b0337ea01`). Supervisión: `etiquetas_v1_2079203.parquet`. Peso: $0.25$.
  2. **B — ConvNeXt-Tiny Base:** Job `2080215`, época 5 (`pesos/convnext_tiny_job2080215_ep5.pt`, SHA-256: `0d927cdb84f090333dfec9a4d88947f1d6ada4d3f5b772415a2ccb8a95b11d9d`). Supervisión: `etiquetas_v1_2079203.parquet`. Peso: $0.25$.
  3. **C — ResNet34 Híbrido:** Job `2084568`, época 7 (`pesos/resnet34_job2084568_ep7_hibrido.pt`, SHA-256: `02c0fceefd10932a08c219a4bb3114dc373e948e58f61bb742e7a8dee310b30a`). Supervisión: `etiquetas_candidatas_v2_hibridas.parquet`. Peso: $0.25$.
  4. **D — ConvNeXt-Tiny Híbrido:** Job `2084609`, época 7 (`pesos/convnext_tiny_job2084609_ep7_hibrido.pt`, SHA-256: `25481e98caca2c5e2f72ecf256defc6fc47a143f14a9dc73123d40f0c014bea3`). Supervisión: `etiquetas_candidatas_v2_hibridas.parquet`. Peso: $0.25$.
* **Métricas reportadas:**
  * **`val_int` macro AUC (415 estudios):** **0.8653** ($\Delta = \mathbf{+0.0079}$ vs baseline AB $0.8574$, IC 95%: `[+0.0032, +0.0124]`, $P = 100\%$).  
    *Fuente:* `envios_joako/candidato_cuadruple_abcd/config_candidato.json`, `experiments/journal.md` (línea 177).
  * **OOF macro AUC (4,349 estudios):** **sin fuente** (no evaluado en matriz OOF completa; los modelos C y D solo contaron con inferencia en los 415 estudios de `val_int`).
  * **Kaggle Público:** **0.905** (enviado mediante notebook `notebook8425e5d90f - cuadruple_abcd_corregido`).  
    *Fuente:* `runs_joako/2084846/comparacion_paquetes_oof.json` (tag `envio-joako-cuadruple-abcd`).

---

### 1.2 Ensamble Quíntuple ABCDE
* **Estado:** Evaluado rigurosamente offline sobre `val_int`.
* **Fórmula de combinación:** Promedio simple de 5 arquitecturas:
  $$\hat{P} = 0.20 \cdot P_A + 0.20 \cdot P_B + 0.20 \cdot P_C + 0.20 \cdot P_D + 0.20 \cdot P_E$$
* **Composición y Checkpoints:**
  1. **A — ResNet34 Base:** Job `2080214`, época 7 (`resnet34_job2080214_ep7.pt`). Peso: $0.20$.
  2. **B — ConvNeXt-Tiny Base:** Job `2080215`, época 5 (`convnext_tiny_job2080215_ep5.pt`). Peso: $0.20$.
  3. **C — ResNet34 Híbrido:** Job `2084568`, época 7 (`resnet34_job2084568_ep7_hibrido.pt`). Peso: $0.20$.
  4. **D — ConvNeXt-Tiny Híbrido:** Job `2084609`, época 7 (`convnext_tiny_job2084609_ep7_hibrido.pt`). Peso: $0.20$.
  5. **E — EfficientNetV2-S:** Job `2084365`, época 8 (`last.pt`, SHA-256: `124514845f595e81d200eda7038e8f6bfe69afed796e37dc0e6e7b1941ac4dd6`). Peso: $0.20$.
* **Métricas reportadas:**
  * **`val_int` macro AUC (415 estudios):** **0.8670** ($\Delta = \mathbf{+0.0018}$ vs ABCD $0.8653$, IC 95%: `[-0.0003, +0.0039]`, $P = 95.0\%$). Récords en meniscos: Lateral 0.8626, Medial 0.9281.  
    *Fuente:* `docs/evaluacion_efficientnetv2_s.md` (sección 3 y 4), `runs_joako/eval_efficientnet_2084365/resumen_evaluacion_efficientnet.json`.
  * **OOF macro AUC (4,349 estudios):** **sin fuente** (no evaluado a nivel OOF global de 4,349 estudios).
  * **Kaggle Público:** **sin fuente** (no enviado a Kaggle; proyectado a ~0.905–0.908 en informe).

---

### 1.3 Paquete v4 (4 familias × 5 folds = 20 modelos)
* **Estado:** Comparado y verificado en OOF mediante Job `2084846`.
* **Fórmula de combinación:** Promedio simple de 4 familias (en OOF, 1 modelo por familia por estudio; en inferencia completa, 20 modelos):
  $$\text{Familias:} \quad \text{r34} + \text{eff} + \text{cnxhib} + \text{effhib}$$
* **Composición y Checkpoints (20 modelos verificados):**
  * `r34` (5 folds): Jobs `2084050` (f4), `2084051` (f0), `2084052` (f1), `2084350` (f2), `2084351` (f3).
  * `eff` (5 folds): Jobs `2084584` (f4), `2084365` (f0), `2084585` (f1), `2084586` (f2), `2084626` (f3).
  * `cnxhib` (5 folds): Jobs `2084633` (f4), `2084671` (f0), `2084673` (f1), `2084676` (f2), `2084684` (f3).
  * `effhib` (5 folds): Jobs `2084634` (f4), `2084674` (f0), `2084683` (f1), `2084689` (f2), `2084694` (f3).
  * *Verificación:* Cobertura estricta de 4,349 estudios sin solapamiento ($0.0$ diferencia parquet vs checkpoints).
* **Métricas reportadas:**
  * **`val_int` macro AUC:** **sin fuente** (no evaluado como ensamble completo de 20 modelos en el subconjunto de 415 estudios).
  * **OOF macro AUC (4,349 estudios, 42,637 celdas válidas):** **0.875597** (~**0.8756**).  
    *Delta vs v3 (0.8726):* $\mathbf{+0.0030}$ puntual (IC 95%: `[+0.0021, +0.0039]`, $P = 100\%$).  
    *Fuente:* `runs_joako/2084846/comparacion_paquetes_oof.json`, `experiments/journal.md` (líneas 508–516).
  * **Kaggle Público:** **sin fuente** (paquete preparado pero sin envío registrado en MLflow al 2026-10-13; solo prueba técnica de inferencia en Job `2084790` con 160 estudios).

---

### 1.4 Paquete v5 (4 familias × 5 folds = 20 modelos)
* **Estado:** Comparado y verificado en OOF mediante Job `2084846`.
* **Fórmula de combinación:** Promedio simple de 4 familias (sustituye `r34` estándar por `r34att` con atención):
  $$\text{Familias:} \quad \text{eff} + \text{cnxhib} + \text{effhib} + \text{r34att}$$
* **Composición y Checkpoints (20 modelos verificados):**
  * `eff`, `cnxhib`, `effhib`: Idénticos a v4 (15 modelos).
  * `r34att` (5 folds con atención MIL): Jobs `2084722` (f4), `2084756` (f0), `2084769` (f1), `2084771` (f2), `2084783` (f3).
* **Métricas reportadas:**
  * **`val_int` macro AUC:** **sin fuente** (no evaluado como ensamble completo en los 415 estudios de `val_int`).
  * **OOF macro AUC (4,349 estudios):** **0.875863** (~**0.8759**).  
    *Delta vs v3 (0.8726):* $\mathbf{+0.0032}$ (IC 95%: `[+0.0019, +0.0045]`, $P = 100\%$).  
    *Delta vs v4 (0.8756):* $\mathbf{+0.0003}$ (IC 95%: `[-0.0005, +0.0011]`, $P = 74.1\%$, estadísticamente **inconcluso**).  
    *Fuente:* `runs_joako/2084846/comparacion_paquetes_oof.json`.
  * **Kaggle Público:** **sin fuente** (sin envío en MLflow; solo prueba de viabilidad de 6 estudios en Job `2084831`).

---

### 1.5 Otros Ensambles Propios Armados por Joako

1. **Ensamble Base v1 (50/50 ResNet34 + ConvNeXt-Tiny):**
   * *Composición:* $0.50 \cdot \text{ResNet34 ep7 (`2080214`)} + 0.50 \cdot \text{ConvNeXt ep5 (`2080215`)}$.
   * *Checkpoints:* `envios_joako/v1_ensamble/pesos/` (`resnet34_job2080214_ep7.pt` y `convnext_tiny_job2080215_ep5.pt`).
   * *`val_int` macro AUC:* **0.8574** (IC 95%: `[0.8401, 0.8772]`). Fuente: `runs_joako/2081432/resumen_evaluacion.json`.
   * *OOF macro AUC:* **sin fuente**.
   * *Kaggle Público:* **0.897** (Notebook `RSNA Joako Ensemble v1`). Fuente: MLflow tag en `runs_joako/2084846/comparacion_paquetes_oof.json`.

2. **Trío Base+D v2 (`trio_base_d_convnext`):**
   * *Composición:* $0.25 \cdot \text{ResNet34 Base} + 0.25 \cdot \text{ResNet34-D (slice\_patho `2084614` ep6)} + 0.50 \cdot \text{ConvNeXt-Tiny Base}$.
   * *Checkpoints:* `envios_joako/candidato_trio_d/pesos/` (`resnet34_d_job2084614_ep6.pt`, SHA-256: `638c9624...`).
   * *`val_int` macro AUC:* **0.8628** ($\Delta = +0.0054$ vs 50/50, IC 95%: `[+0.0017, +0.0092]`, $P = 99.8\%$). Fuente: `envios_joako/candidato_trio_d/config_candidato.json`.
   * *OOF macro AUC:* **sin fuente**.
   * *Kaggle Público:* **0.904** (Notebook `RSNA Joako | Trio Base+D v2`). Fuente: MLflow tag en `runs_joako/2084846/comparacion_paquetes_oof.json`.

3. **Candidato Fusión C5 (Fusión Offline 17 Modelos):**
   * *Composición:* $0.60 \cdot \text{Owen 15 (CV15)} + 0.20 \cdot \text{ResNet34-D Slice-Att (`2084614`)} + 0.20 \cdot \text{ConvNeXt Híbrido (`2084609`)}$.
   * *`val_int` macro AUC:* **0.8679** ($\Delta = \mathbf{+0.0049}$ vs Owen 15 $0.8630$, IC 95%: `[+0.0014, +0.0084]`, $P = 99.7\%$). Fuente: `runs_joako/eval_fusion_offline/resumen_fusion_offline.json`.
   * *OOF / Público:* **sin fuente** (diseñado como ensayo analítico offline de coste cero GPU).

---

## 2. Inventario Exhaustivo de Jobs Propios de Joako

Tabla cronológica de los trabajos ejecutados desde `code_joako` con salida o registro en `runs_joako` y `logs/`:

| Job ID | Fecha y Hora | Estado SLURM | Tipo / Nombre del Trabajo | Recursos | Métrica Principal / Resultado | Fuentes en Disco |
| :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| **`2081432`** | 2026-10-12 02:41 | `COMPLETED` | Evaluación Ensamble Base 50/50 | GPU MI210 | `val_int` macro AUC: **0.8574**; `val_oficial`: **0.8793** | `runs_joako/2081432/`, `logs/rsna-eval-ensamble_2081432.out` |
| **`2084348`** | 2026-10-12 14:43 | `COMPLETED` | Evaluación TTA Ensamble 50/50 | GPU MI210 (3m) | `val_int` TTA: **0.8585** ($\Delta = +0.0011$, $P=91.6\%$) | `runs_joako/2084348/`, `logs/rsna-eval-tta_2084348.out` |
| **`2084404`** | 2026-10-12 15:14 | `COMPLETED` | Preprocesamiento v2 (5 vistas) | 48 CPU (3.4m) | 4,407/4,407 estudios OK, 0 errores, 43 GB generados | `logs/rsna-preproc-v2_2084404.out`, `/lustre/proyectos/p082/preproc_joako/v2_train` |
| **`2084537`** | 2026-10-12 15:50 | `COMPLETED` | Auditoría de integridad preproc v2 | 16 CPU (114s) | 4,407 estudios verificados, 0 corruptos, 73.7% cor_nofs | `logs/auditar-v2_2084537.out` |
| **`2084557`** | 2026-10-12 17:49 | `COMPLETED` | Exp. Vistas — Brazo A (4 vistas, s0) | GPU MI210 (31m) | `val_int` macro AUC: **0.8398** (ep 7), `val_oficial`: 0.8590 | `runs_joako/cornofs_5view/brazoA_4v_seed0/`, `logs/r34-4v-s0_2084557.out` |
| **`2084558`** | 2026-10-12 18:09 | `COMPLETED` | Exp. Vistas — Brazo B (5 vistas, s0) | GPU MI210 (41m) | `val_int` macro AUC: **0.8419** ($\Delta = +0.0021$, no significativo) | `runs_joako/cornofs_5view/brazoB_5v_seed0/`, `logs/r34-5v-s0_2084558.out` |
| **`2084611`** | 2026-10-12 21:46 | `COMPLETED` | Validación infra arquitecturas CPU | 8 CPU (1.2m) | **18/18 tests PASS** en `test_arquitecturas.py`; smoke OK | `runs_joako/2084611/`, `logs/rsna-arq-valid_2084611.out` |
| **`2084617`** | 2026-10-12 21:50 | `COMPLETED` | Smoke física v3 (24 estudios) | 8 CPU (7s) | 24/24 estudios OK (0.079 GB, 0 errores) | `runs_joako/2084617/`, `runs_joako/prueba_geometria_v3/`, `logs/rsna-p3-smoke_2084617.out` |
| **`2084621`** | 2026-10-12 21:53 | `COMPLETED` | Smoke test 1 KneeNet3D en CPU | 8 CPU (<1m) | Verificación inicial de forward pass | `logs/rsna-val-3d_2084621.out` |
| **`2084622`** | 2026-10-12 21:54 | `COMPLETED` | Validación sintética KneeNet3D | 8 CPU (<1m) | **11/11 tests PASS**, inflado I3D exacto, VRAM ~322 MB | `logs/rsna-val-3d_2084622.out` |
| **`2084614`** | 2026-10-12 22:27 | `COMPLETED` | Entrenamiento ResNet34-D (`slice_patho`) | GPU MI210 (30m) | `val_int` macro AUC: **0.8633** (ep 6, `best_interna.pt`) | `runs_joako/2084614/`, `logs/rsna-arq-D_2084614.out` |
| **`2084615`** | 2026-10-12 22:57 | `COMPLETED` | Entrenamiento ResNet34-E (`view_patho`) | GPU MI210 (30m) | `val_int` macro AUC: **0.8590** (ep 8, `best_interna.pt`) | `runs_joako/2084615/`, `logs/rsna-arq-E_2084615.out` |
| **`2084609`** | 2026-10-12 22:21 | `COMPLETED` | Entrenamiento ConvNeXt-Tiny Híbrido | GPU MI210 (3h 02m) | `val_int` macro AUC: **0.8582** (ep 7, `best_interna.pt`); ABCD: 0.8653 | `runs_joako/2084609/`, `logs/rsna-cnx-hib_2084609.out` |
| **`2084663`** | 2026-10-13 01:41 | `FAILED` | Intento eval ConvNeXt CPU | 16 CPU (3s) | FAILED: `TypeError` por firma incompatible en script | `logs/eval-cnx-cpu_2084663.err` |
| **`2084700`** | 2026-10-13 11:05 | `COMPLETED` | Entrenamiento DenseNet121 RadImageNet | GPU MI210 (18m) | 5 épocas completadas con backbone congelado | `runs_joako/2084700/`, `logs/rsna-dense-rad_2084700.out` |
| **`2084747`** | 2026-10-13 12:38 | `COMPLETED` | Evaluación CPU DenseNet121 Rad | 16 CPU (8.8m) | `val_int` macro AUC: **0.6902** ($\Delta = -0.1534$ vs base) | `runs_joako/2084700/evaluacion/`, `logs/eval-dense-cpu_2084747.out` |
| **`2084750`** | 2026-10-13 12:56 | `FAILED` | Intento eval EfficientNetV2-S CPU | 16 CPU (3s) | FAILED: ajuste de rutas de entrada | `logs/eval-eff-cpu_2084750.err` |
| **`2084751`** | 2026-10-13 13:00 | `COMPLETED` | Evaluación CPU EfficientNetV2-S | 16 CPU (7.8m) | `val_int` individual: **0.8517**; Quíntuple ABCDE: **0.8670** | `runs_joako/eval_efficientnet_2084365/`, `logs/eval-eff-cpu_2084751.out` |
| **`2084773`** | 2026-10-13 13:39 | `COMPLETED` | Fusión Multi-Modelo Offline (0h GPU) | 8 CPU (4.7m) | Ensamble C5: **0.8679** ($\Delta = +0.0049$ vs Owen 15, $P=99.7\%$) | `runs_joako/eval_fusion_offline/`, `logs/rsna-eval-fusion_2084773.out` |
| **`2084782`** | 2026-10-13 14:00 | `CANCELLED` | Piloto Fold 0 `effhib` (Joako) | GPU MI210 | CANCELLED tras 18m28s (evitar duplicidad con Owen) | `runs_joako/2084782/`, `logs/rsna-eff-hib0_2084782.out` |
| **`2084837`** | 2026-10-13 16:14 | `FAILED` | Destilación objetivos v3 (intento 1) | 8 CPU (1s) | FAILED: `AssertionError` directorio vacío (corregido) | `logs/rsna-destilar-v3_2084837.err` |
| **`2084838`** | 2026-10-13 16:14 | `FAILED` | Stacking OOF (intento 1) | 8 CPU (1s) | FAILED: `AssertionError` directorio vacío (corregido) | `logs/rsna-stacking-oof_2084838.err` |
| **`2084839`** | 2026-10-13 16:15 | `COMPLETED` | Destilación objetivos v3 (Synovitis) | 8 CPU (10s) | Generación `etiquetas_v3_distilled_gold.parquet` | `runs_joako/2084839/`, `logs/rsna-destilar-v3_2084839.out` |
| **`2084840`** | 2026-10-13 16:15 | `COMPLETED` | Meta-stacking 12x12 sobre OOF | 8 CPU (7.2m) | Stacking 12x12: **0.8740** ($\Delta = -0.0016$ vs v4; refutado) | `runs_joako/2084840/`, `logs/rsna-stacking-oof_2084840.out` |
| **`2084841`** | 2026-10-13 16:16 | `COMPLETED` | Diagnóstico destilación `--rellenar all` | 8 CPU (12s) | 551 celdas rellenadas (263 en Baker's) | `runs_joako/2084841/`, `logs/rsna-destilar-v3_2084841.out` |
| **`2084846`** | 2026-10-13 16:33 | `COMPLETED` | Comparación OOF paquetes v3/v4/v5 | 8 CPU (1m) | v4 vs v3: $\Delta = \mathbf{+0.0030}$ ($P=100\%$); v5 vs v4 inconcluso | `runs_joako/2084846/`, `logs/rsna-cmp-paquetes_2084846.out` |

---

## 3. Scripts y Archivos `.slurm` Sin Versionar (`git status`)

Estado obtenido mediante `git --no-optional-locks status` sobre `/lustre/proyectos/p082/code_joako`:

### 3.1 Scripts Python sin versionar en `rsna/` (26 archivos)

1. `rsna/analizar_series_metadatos.py`: Análisis exploratorio de metadatos de series DICOM y diagnóstico de vistas descartadas.
2. `rsna/auditar_datos_arq.py`: Auditoría estricta de solo lectura de tensores de entrada para la campaña de arquitecturas.
3. `rsna/auditar_integridad_v2.py`: Auditoría de integridad técnica, dimensional y bit a bit de tensores v2 frente a v1.
4. `rsna/auditar_predicciones.py`: Análisis de concordancia y desacuerdos entre predicciones para optimización de pesos $\alpha$.
5. `rsna/comparar_paquetes_oof.py`: Comparador OOF pareado con bootstrap (1,000 réplicas) para paquetes v3, v4 y v5.
6. `rsna/destilar_etiquetas_v3.py`: Generador de tabla destilada v3 mezclando supervisiones con predicciones OOF y ponderación Gold.
7. `rsna/entrenar_arq.py`: Entrenador de arquitecturas alternativas con variantes de atención (`slice_patho`, `view_patho`).
8. `rsna/entrenar_densenet.py`: Script de entrenamiento para extracción de características con DenseNet121 RadImageNet.
9. `rsna/entrenar_vistas.py`: Entrenador controlado para la prueba de hipótesis de 4 vistas vs 5 vistas (`cor_nofs`).
10. `rsna/evaluar_arquitecturas.py`: Evaluador pareado CPU con bootstrap para modelos de la campaña de arquitecturas.
11. `rsna/evaluar_convnext_hibrido.py`: Script de evaluación pareada de ConvNeXt-Tiny híbrido (documentado error de firma pendiente).
12. `rsna/evaluar_densenet.py`: Evaluador pareado sobre `val_int` para el piloto de DenseNet121 RadImageNet.
13. `rsna/evaluar_efficientnet.py`: Evaluador pareado CPU sobre `val_int` para el modelo EfficientNetV2-S (Job `2084365`).
14. `rsna/evaluar_experimento_vistas.py`: Evaluador pareado y análisis de bootstrap para el experimento de 5 vistas.
15. `rsna/evaluar_tta.py`: Evaluador controlado de Test-Time Augmentation por paridad de corte (offset 0 vs offset 1).
16. `rsna/modelo_3d.py`: Implementación de arquitectura `KneeNet3D` y función de inflado I3D de pesos 2D a 3D.
17. `rsna/modelo_att.py`: Definición de `KneeNetAtt` con atención MIL específica por patología sobre cortes y vistas.
18. `rsna/modelo_densenet.py`: Definición de arquitectura con backbone DenseNet121 RadImageNet y cabeza multietiqueta.
19. `rsna/preproc_v2.py`: Lógica de extracción y empaquetado de tensores en 5 ranuras anatómicas fijas.
20. `rsna/preprocesar_v2.py`: Pipeline CLI para preprocesamiento paralelo v2 sobre un split completo.
21. `rsna/preprocesar_v3.py`: Pipeline CLI para normalización física volumétrica tridimensional v3 (FOV 200 mm, 4 mm en $z$).
22. `rsna/probar_geometria_v3.py`: Diagnóstico preliminar e inspección visual de hojas de contacto para geometría v3.
23. `rsna/stacking_biomecanico_oof.py`: Algoritmo de meta-stacking 12x12 regularizado sobre probabilidades OOF.
24. `rsna/test_arquitecturas.py`: Suite sintética de pruebas unitarias (18 tests) para `modelo_att.py`.
25. `rsna/test_modelo_3d.py`: Suite sintética de pruebas unitarias (11 tests) para `modelo_3d.py`.
26. `rsna/evaluar_fusion_offline.py` (modificado): Evaluador pareado de fusiones multi-modelo offline (Owen 15 + Joako).

### 3.2 Scripts SLURM sin versionar en `jobs/` (18 archivos)

1. `jobs/auditar_v2.slurm`: Ejecuta `rsna/auditar_integridad_v2.py` en partición `cpu` (16 cores).
2. `jobs/comparar_paquetes_oof_cpu.slurm`: Ejecuta `rsna/comparar_paquetes_oof.py` en partición `cpu` (8 cores).
3. `jobs/destilar_etiquetas_v3_cpu.slurm`: Lanza `rsna/destilar_etiquetas_v3.py` en partición `cpu` (8 cores).
4. `jobs/entrenar_arq.slurm`: Lanza entrenamientos de variantes `slice_patho` / `view_patho` en GPU MI210.
5. `jobs/entrenar_convnext_hibrido.slurm`: Lanza entrenamiento de ConvNeXt-Tiny híbrido (Job `2084609`) en GPU MI210.
6. `jobs/entrenar_densenet_rad.slurm`: Lanza entrenamiento de DenseNet121 RadImageNet en GPU MI210.
7. `jobs/entrenar_vistas.slurm`: Lanza entrenamientos comparativos de 4 vs 5 vistas en GPU MI210.
8. `jobs/evaluar_arq.slurm`: Lanza evaluación CPU de variantes de atención de la campaña de arquitecturas.
9. `jobs/evaluar_convnext_hibrido.slurm`: Job GPU de evaluación de ConvNeXt híbrido.
10. `jobs/evaluar_convnext_hibrido_cpu.slurm`: Job CPU de evaluación de ConvNeXt híbrido.
11. `jobs/evaluar_densenet_cpu.slurm`: Lanza `rsna/evaluar_densenet.py` en partición `cpu` (16 cores).
12. `jobs/evaluar_efficientnet_cpu.slurm`: Lanza `rsna/evaluar_efficientnet.py` en partición `cpu` (16 cores).
13. `jobs/evaluar_tta.slurm`: Lanza evaluación de TTA en GPU MI210 (`rsna/evaluar_tta.py`).
14. `jobs/preprocesar_v2.slurm`: Lanza preprocesamiento paralelo v2 en partición `cpu` (48 cores).
15. `jobs/preprocesar_v3_smoke.slurm`: Lanza smoke test de preprocesamiento físico v3 en partición `cpu` (8 cores).
16. `jobs/stacking_oof_cpu.slurm`: Lanza `rsna/stacking_biomecanico_oof.py` en partición `cpu` (8 cores).
17. `jobs/validar_arquitecturas_cpu.slurm`: Ejecuta suite `test_arquitecturas.py` y smoke test en CPU.
18. `jobs/validar_modelo_3d_cpu.slurm`: Ejecuta suite `test_modelo_3d.py` y cálculo de VRAM en CPU.

*Nota:* Archivos `.slurm` versionados pero con cambios locales no commiteados:
* `jobs/entrenar_effhib_fold0.slurm` (modificado)
* `jobs/evaluar_fusion_offline_cpu.slurm` (modificado)

### 3.3 Otros archivos y artefactos sin versionar

* **Documentación técnica (`docs/`):**
  * `docs/auditoria_compatibilidad_dinov2.md`: Dictamen técnico sobre incompatibilidad geométrica y ausencia de pesos locales de DINOv2.
  * `docs/auditoria_dinov2_radimagenet.md`: Auditoría de autenticidad y compatibilidad de checkpoints de RadImageNet.
  * `docs/auditoria_owen_0912_vs_joako.md`: Análisis forense cuantitativo de la ventaja competitiva de Owen (0.912) vs Joako (0.904/0.905).
  * `docs/evaluacion_densenet_radimagenet.md`: Informe formal de la evaluación de DenseNet121 RadImageNet (0.6902).
  * `docs/evaluacion_efficientnetv2_s.md`: Informe formal de la evaluación de EfficientNetV2-S (0.8517) y ensambles Quíntuple ABCDE (0.8670).
* **Experimentos:**
  * `experiments/campana_arquitecturas_progreso.md`: Checkpoint de estado de la campaña de arquitecturas.
* **Directorios de pesos y modelos:**
  * `models/RadImageNet_pytorch/`: Contiene pesos locales descargados `DenseNet121.pt` (28.4 MB) e `InceptionV3.pt` (87.5 MB).
  * `models/dinov2/`: Código fuente clonado del repositorio oficial de DINOv2 (sin pesos `.pth`).
* **Skills del agente:**
  * `.agents/`: Nueva ubicación centralizada de las skills reorganizadas del proyecto.
  * `.claude/skills/*`: Symlinks apuntando a `.agents/skills/*`.

---

## 4. Trazabilidad y Verificación de Integridad

1. **Aislamiento de Entornos:** Todas las métricas y rutas aquí tabuladas provienen exclusivamente de archivos residentes en `/lustre/proyectos/p082/code_joako` y `/lustre/proyectos/p082/runs_joako`.
2. **Ausencia de Procesos Invasivos:** Durante la recopilación de este inventario no se realizaron llamadas de red, comandos de escritura fuera del archivo de reporte, envíos de jobs SLURM (`sbatch`) ni ejecuciones pesadas de Python en el nodo de login.
3. **Métricas Faltantes:** Cualquier combinación o métrica teórica no respaldada por un archivo `.json`, `.csv` o log formal en disco ha sido catalogada explícitamente como **"sin fuente"**.

