# Receta CoAtNet: piloto del fold 0 (hilo 746792, mensaje #53, sin OAI)

Fecha: 2026-10-14. Código: [rsna/entrenar_coatnet.py](../../rsna/entrenar_coatnet.py), [rsna/raptor_corpus.py](../../rsna/raptor_corpus.py), [rsna/etiquetas_coatnet.py](../../rsna/etiquetas_coatnet.py), [jobs/entrenar_coatnet_piloto.slurm](../../jobs/entrenar_coatnet_piloto.slurm).

## Fuentes, en orden de autoridad

| Fuente | Ruta | Qué fija |
|---|---|---|
| #53 de Danial Zakaria | `referencias/rsna_research/foro/verbatim/746792.md`, líneas 422-465 | Etiquetas, pérdida, optimizador, ventanas, selección |
| Instrucciones de jcastro (2026-10-14) | esta sesión | Sin OAI, fp16, geometría de 140 mm, fold 0, blend con pesos, Baker's, criterio |
| Notebook de entrenamiento de dreaddevelopment | `referencias/rsna_research/notebooks/dreaddevelopment/knee-mri-training-the-twelve-finding-model/` | `RaptorClassifier`, muestreo de ventanas, jitter, `pos_weight`, promedio de pesos, defaults de OneCycle |
| Notebook 0.940, celda 22 | `referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb` | `RaptorClassifier` (idéntico), inferencia por trozos, preprocesado native384 |
| Benchmark propio | `experiments/reportes/bench_coatnet_v2.md` | bf16 cae con SIGSEGV; fp16 + GradScaler y atención sin fusionar funcionan |

En la columna "Origen", **decisión nuestra** marca lo que ninguna fuente fija; está todo listado al final.

## La receta

| Elemento | Qué se hace | Origen |
|---|---|---|
| Estudios de entrenamiento | 4,349 de la competición, sin OAI. Piloto: partición 0 de 5 (`md5(uid) % 5`, la de Owen). Entrena con 3,458 y deja 891 fuera; los 58 Gold nunca entrenan | jcastro; #53 excluye los Gold |
| Corpus | Brazo `native384dense-v10` de `rsna/raptor_preproc.py`: 5 ranuras (SLOTS64), banda 0.02-0.98, hasta 96 cortes por capacidad, percentiles 2-98, **recorte de 140 mm**, 384 px sin reescalado posterior. Paridad bit a bit con el notebook (job 2085083) | jcastro (el #53 usa campo completo, pero lo combina con OAI) |
| Caché | `/scratch` local del nodo dentro del job (~62 GB), borrado con `trap` al salir. Nada en Lustre | jcastro |
| Ventana | Triplete (i−1, i, i+1) del volumen concatenado; puede cruzar el borde entre ranuras, igual que en el notebook y el 0.940 | notebook |
| Ventanas de entrenamiento | 12 centros al azar por estudio y paso, sin reemplazo (se repite la lista si hay menos de 12) | #53, notebook |
| Ventanas de evaluación | 24 centros equiespaciados (`linspace` redondeado), idénticas a `raptor_preproc.eval_windows` (comprobado, dif. máxima 0.0) | #53, notebook |
| Aumento | Ganancia multiplicativa por estudio, uniforme en [0.9, 1.1], recortada a [0, 1] antes de normalizar. Sin flips ni transformaciones afines | #53, notebook |
| Normalización | ImageNet (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225), no el 0.5 del `config.json` de timm | #53 |
| Backbone | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` de timm 1.0.30, pesos locales `model.safetensors` (sha256 `91570c7af10f…`; 478 de 478 tensores cargados, `head.fc` descartada), `num_classes=0`, `global_pool="avg"`, F = 1,024 | #53 |
| Atención del backbone | Sin fusionar (`TIMM_FUSED_ATTN=0`; 0 de 16 capas fusionadas) | bench v2 |
| Cabeza | `RaptorClassifier` copiado tal cual: LayerNorm → `att` = Linear(F,256)–Tanh–Dropout(0.2)–Linear(256,12) → softmax sobre ventanas → pooling por hallazgo → `logits = (pooled · clsW).sum(−1) + clsb`; `clsW` con trunc_normal(0.02) | jcastro, notebook, 0.940 |
| Etiquetas | `blendjev` (sección siguiente) | #53, jcastro |
| Pérdida | `BCEWithLogits(pos_weight)` por celda × peso de celda `w`; Σ / (12 × estudios del lote), celdas enmascaradas incluidas en el denominador | #53 (el `w` continuo lo acordó jcastro) |
| `pos_weight` | `(1 − p)/p`, p recortado a [0.03, 0.7] y el peso a [1, 10]. p = media de `t` en las celdas con `w > 0` de los 3,458 estudios de entrenamiento | #53, notebook; **decisión nuestra**: excluir el fold 0 de p y usar solo celdas con objetivo |
| Lote | 1 estudio, acumulación 16; 216 pasos por época (3,458 // 16; el resto se descarta cada época) | #53 (4,349 // 16 = 271 → sus 4,336 pasos) |
| Optimizador | AdamW, wd 0.0588 en todos los parámetros (también cabeza, LayerNorm y sesgos). lr backbone 4.01e-5; lr cabeza 1.95e-3 (`norm`, `att`, `clsW`, `clsb`) | #53, jcastro, notebook |
| Planificador | OneCycleLR por grupo, `pct_start` 0.15, 16 épocas = 3,456 pasos; defaults de torch (coseno, `div_factor` 25, `final_div_factor` 1e4, `cycle_momentum` True, que mueve beta1 entre 0.85 y 0.95) | #53, notebook |
| Recorte de gradiente | Norma 3.0, tras `unscale_` | #53 |
| Precisión | fp16 autocast + GradScaler. Inferencia en fp16, backbone por trozos de 16 ventanas y una sola llamada a la cabeza | jcastro (el #53 usa bf16), bench v2, #53 |
| EMA | No | #53 |
| Semilla | 42 (python, numpy, torch). No es bit a bit reproducible: MIOpen no es determinista | #53 |
| Selección | Top-3 épocas por macro AUC Gold (24 ventanas); promedio uniforme de los `state_dict` en doble precisión, buffers de BatchNorm incluidos; `num_batches_tracked` del mejor | #53, notebook |
| Evaluación por época | Gold (y real, 58) y OOF del fold 0 (891) con tres jueces. Se guardan predicciones de cada época y del SWA | jcastro |

## Etiquetas

Construidas con `rsna/etiquetas_coatnet.py` y guardadas en `runs_joako/etiquetas_coatnet_20261014/` (`etiquetas_blendjev.parquet` sha256 `5347cc5bbbe8…`, `etiquetas_blend.parquet` `c5b6050da2ee…`).

**Nuestra.** Tabla híbrida (`runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet`). El #53 usa aquí sus etiquetas de Gemini, que son densas.

**blend** = nuestra + stevenleehans v4 (`llm_labels_v4_blend.csv`), con `w_v4 = 2·|p − 0.5|`. Regla acordada con jcastro:

| Caso | `t` | `w` |
|---|---|---|
| Las dos con valor | media | media |
| Solo la nuestra (v4 = 0.25 exacto) | nuestra | nuestra |
| Solo v4 (la nuestra es NaN) | v4 | `w_v4` |
| Ninguna | NaN | 0 |

**blendjev** = media de blend y JEV HYBRID (`referencias/rsna_research/jev/HYBRID_labels_scores.csv`) donde JEV tiene valor; `w` = el de blend.

- JEV no tiene valor en 1 estudio (ACL, MCL y Fracture vacíos); ahí queda blend, como dice el #53.
- **Decisión nuestra:** donde blend es NaN, la celda sigue enmascarada aunque JEV tenga valor. En esas celdas JEV es el 0.25 de v4 reescalado (~0.24): es silencio, no lectura.

**Baker's.** Cada celda que sigue enmascarada pasa a `t = 0.05, w = 0.5` (1,758 estudios no oficiales). Lo pidió jcastro; en Gold, P(positivo | silencio) = 1/29 (`silencio_etiquetas.md`).

Cobertura de blendjev en los 4,349: de 87.2 % (Medial/Lateral OA) a 95.6 % (Medial Meniscus); Baker's 100 % con el relleno.

## Criterio de aceptación (fijado antes de lanzar)

- **Pasa si** el macro AUC OOF del fold 0 con el juez v1 supera al de effhib en esos mismos 891 estudios por ≥ 0.005, y no hay NaN ni divergencia en fp16.
- **Referencia.** effhib, fold 0, juez v1: **0.8628** (`runs/2084813/oof_effhib.parquet`, calculado con el mismo `macro_auc` de `entrenar.py`). El 0.8644 citado es sobre las 5 particiones juntas. **Umbral: 0.8678.**
- **Checkpoint que decide (decisión nuestra).** El SWA top-3 por Gold, que es el que la receta entregaría. Gold y fold 0 son disjuntos, así que elegir las épocas con Gold no sesga el OOF. También se reporta la última época; si el SWA y la última discrepan sobre el criterio, se dice.
- **Gold.** Se reporta en cada época y para el SWA, sin decidir. El AUC Gold del SWA está sesgado al alza: es el máximo de una selección sobre esos mismos 58 estudios.
- **IC.** Δ con bootstrap pareado de 1,000 réplicas (semilla 20261014), como información. El criterio es el punto, como lo fijó jcastro.
- **NaN (decisión nuestra).** Cualquier logit no finito en el forward de entrenamiento, pérdida media de época no finita o predicción de evaluación no finita.
- **Divergencia (decisión nuestra).** GradScaler salta más del 25 % de los pasos de una época tras la primera. El job aborta y lo registra.
- **fp16 sano (jcastro).** Cada época registra el % de pasos que salta GradScaler y la escala final (`historial.jsonl`, `resumen.json`, MLflow). Si **ninguna** época tiene menos del 5 % de pasos saltados, el veredicto es **NO PASA** aunque el AUC supere 0.8678.

## Jueces del OOF

Todos con `y = t ≥ 0.5` y AUC solo en celdas `w > 0`, la regla de `entrenar.py` y `evaluar_fusion_offline.py`.

- **v1** (`preproc/etiquetas_v1_2079203.parquet`): decide. Synovitis solo tiene 269 celdas evaluables en el fold 0 y Baker's 497 (`silencio_etiquetas.md`, sección 5).
- **blend**: sin el relleno de Baker's.
- **blendjev**: el objetivo de entrenamiento. Incluye los negativos débiles de Baker's, así que su AUC de Baker's no es comparable con el de los otros jueces. Medir contra el propio objetivo favorece al modelo.

## Lista de decisiones nuestras

1. p de `pos_weight` sobre las celdas con objetivo de los estudios de entrenamiento del fold, no sobre los 4,349.
2. Celdas sin blend se mantienen enmascaradas aunque JEV tenga valor (salvo el relleno de Baker's).
3. El resto de estudios que no completa un paso de acumulación se descarta cada época (como el recuento del #53).
4. El checkpoint que decide el criterio es el SWA; la última época se reporta.
5. Definiciones operativas de NaN y divergencia (arriba).
6. Estudios con error en el corpus o menos de 3 cortes se excluyen y se cuentan en `config.json`.
7. Inferencia por trozos de 16 ventanas (el 0.940 usa 8; el #53, 16). Solo cambia memoria y velocidad.

## Diferencias deliberadas con el #53, y su efecto esperado

| #53 | Piloto | Por qué |
|---|---|---|
| +2,399 rodillas OAI (6,748 filas) | Sin OAI | jcastro; su fila "sin OAI" (0.940-0.942) es la referencia |
| Campo completo | Recorte de 140 mm | jcastro; el campo completo venía con OAI |
| bf16 | fp16 + GradScaler | bf16 cae en la MI210 |
| Etiquetas propias de Gemini, densas | Tabla híbrida con `w` y máscaras | No tenemos las suyas |
| All-data | Fold 0 (3,458 de entrenamiento) | OOF limpio para decidir; 20 % menos de datos y de pasos |
