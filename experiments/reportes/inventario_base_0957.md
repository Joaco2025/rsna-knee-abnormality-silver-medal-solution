# Inventario del notebook 0.957 (`referencias/0-957-latest-dataset-score-in-one-click.ipynb`)

**Fecha:** 2026-10-14. **Solo lectura:** el notebook no se ha tocado; se volcó a texto en el scratchpad.
**Cobertura:** las 7 celdas completas (0–1 Markdown, 2–6 código). De la celda 4 se extrajeron los 7 fuentes incrustados (`SOURCES`: `px_cache.py`, `compact_train.py`, `compact_infer.py`, `probe_infer.py`, `probe_dual.py`, `knee_anchor_bank_v2.py`, `side_common.py`) y se leyeron por búsqueda dirigida (backbones, argumentos, plazos), no línea a línea.
**Sin salidas guardadas:** ninguna celda tiene outputs. No hay tiempos medidos ni log de ejemplo, a diferencia del 0.940 (que trae 3 estudios de prueba).
**Comparado con:** `bend-the-knee-to-speedy-raptors-the-original.ipynb` (0.940) y su inventario previo, [agy_inventario_base_0940.md](agy_inventario_base_0940.md). Contrasté con el notebook mismo la mezcla final, los pesos por hallazgo, `TIME_BUDGET`, la publicación en la celda 23, y la ausencia de ConvNeXt/EfficientNet.

---

## 1. Estructura: dos modelos, mezcla por rangos

El notebook se titula «R7 + Raptor» y se describe como «Private candidate». Bundle `8c7b666e27dc0e8f`. Tiene solo **dos familias**; no hay DINOv2, A5, RadImageNet ni variantes CoAt.

| Celda | Qué hace |
|---|---|
| 0–1 (md) | Descripción, créditos y reconocimiento OAI/NDA. |
| 2 | `%writefile /tmp/raptor_run.py`: la inferencia pública de Raptor, con comprobación de SHA-256 del checkpoint. |
| 3 | **Etapa 1.** Fija `NB_T0 = time.time()`, ejecuta `raptor_run.py` como subproceso (timeout 2 h) y exige `raptor_sub.csv` y `raptor_raw.csv`. |
| 4 | Prepara la **etapa 2**: monta datasets, verifica hashes, instala ruedas `--no-index --no-deps`, construye `WORKER_ARGS`. Fija `T0 = NB_T0`. |
| 5 | **Etapa 2.** Lanza `probe_dual.py` con `--deadline T0 + DEADLINE_HOURS*3600`; escribe `ours_sub.csv` y `submission_receipt.json`. |
| 6 | **Etapa 3.** Mezcla por rangos y **publica `/kaggle/working/submission.csv`**. |

Falla cerrada: si la etapa 1 falla, el notebook se detiene («no silent single-model submission»).

## 2. Familias, encoders y pesos

### 2.1 Raptor (público, dreaddevelopment; checkpoints reentrenados por ellos)

- **Encoder:** `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm), cabeza MIL con atención (`RaptorClassifier`, `F_dim=768`). Un solo brazo (`ARMS`, `w = 1.0`).
- **Checkpoint:** `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, SHA-256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` (verificado antes de inferir).
- **Datasets de Kaggle** (en este orden de búsqueda): `rsna-knee-publication-swa-weights-20261007` y `nartaa/rsna-knee-publication-swa-weights-20261007`; alternativa `rsna-knee-oai-original384-s42-pair` (también bajo `nartaa/`). Los nombres `raptor-knee-arms` y `raptor-cnn336` de `find_weight_file` quedan como heredados: `ARMS` solo pide el archivo anterior.
- **Entrada:** ventanas de 3 cortes, `K_EVAL = 94` (los 94 desplazamientos de un volumen de 96 cortes), `CROP_MM = 140`, normalización ImageNet, lector rápido `study_lut`.
- **TTA:** hace un pase normal y otro con espejo izquierda/derecha y promedia las probabilidades (`infer_pair_probs`). El docstring de la cabecera dice «no TTA», pero el código sí lo hace (el docstring está desfasado).
- **Dudas sin resolver:** `INPUT_RES = 320` y el `assert` de `infer_pair_probs` exige `(94,3,320,320)`, mientras `ARMS` declara `res: 384`. No pude comprobar cuál manda sin ejecutar `load_model`.
- **Salida:** `raptor_raw.csv` (probabilidades crudas; un estudio fallido queda en 0.5) y `raptor_sub.csv` (rangos).

### 2.2 «Ours» (R7 ConvNeXt-small, compacto)

- **Encoder:** ConvNeXt-small, 4 particiones, `crop 352`, entrenamiento auxiliar fastMRI, apiladores de estado reajustados para Effusion y Baker (cabecera de la celda 0). El nombre exacto del backbone, `views`, `slices` y `crop` salen de `manifest.json`, que está en un dataset y **no en el notebook**; solo se confirma por el texto de la celda 0 y por el nombre `r7cnxfsss`.
- **Código:** el entrenador exige ConvNeXt para la ruta auxiliar fastMRI (`compact_train.py`, línea 1953). Con CoAtNet o ResNet la rechaza porque movería las estadísticas de BatchNorm.
- **Datasets de Kaggle:**
  - `rsna-knee-compact-r7cnxfsss`: `manifest.json` y pesos de los 4 pliegues.
  - `rsna-orthofoundation-fold0-epoch7-20261004`: `receipt.json`, `cloud_code.zip`, `dependencies.zip` (ruedas) y, si `views` es `v3k` o `v4`, `anatomy.pt`.
  - `rsna-knee-ofl-ssl-weights`: solo se usa si `backbone == 'ofl_l'`, lo que no parece el caso.
- **Brazo:** `ARM = 'E_lrprior'`. El ensamble de 4 pliegues se promedia dentro del worker.
- **Etiquetas:** pesos débiles de reportes: `stevenleehans/rsna-knee-llm-report-labels`, `pilkwang/rsna-knee-llm-labels` y `riadmohamed42/jev-knee-labels-verification`, más etiquetas propias derivadas de reportes. Los reportes y las tablas de etiquetas no están incluidos.
- **Control de integridad:** SHA-256 de 8 módulos del código principal y de los 7 fuentes incrustados; `bundle_id` contra el manifiesto; `trainer_md5` si el manifiesto lo declara.
- **Salida:** `ours_sub.csv`. Un estudio fallido recibe la media de la columna; la celda 6 los detecta releyendo `worker*/predictions.csv`.

## 3. Mezcla final (celda 6)

Rango percentil por hallazgo, **sin ajuste con etiquetas**:

```
final[c] = w_c · rango(Raptor)[c] + (1 − w_c) · rango(ours)[c]
```

| Hallazgo | Peso Raptor | Peso ours |
|---|---|---|
| Los 12 (por defecto `W_RAPTOR`) | 0.5 | 0.5 |
| Medial OA, Lateral OA, PF OA (`W_RAPTOR_C`, «decisión del usuario 10-10 17:15») | 0.5 | 0.5 |

El diccionario por hallazgo coincide con el valor por defecto, así que en la práctica **todos los hallazgos pesan 0.5 / 0.5**. Los pesos admiten 0–1 (hay una aserción).

**Reglas de fallo:**

- Cada modelo se rankea solo sobre sus estudios no fallidos.
- Si un modelo falla en un estudio, vale el rango del otro; si fallan los dos, 0.5.
- Más del 3 % de fallos de Raptor detiene el notebook.
- Más del 50 % de estudios sin puntuar en «ours» detiene el notebook.
- Cualquier sustitución imprime `BLEND_FALLBACK`.
- Correlación de Spearman entre `raptor_raw` y `raptor_sub`: debe ser ≥ 0.99.

## 4. Presupuesto de tiempo

- **No existe `TIME_BUDGET`.** El reloj es `NB_T0` (celda 3), copiado a `T0` en la celda 4.
- `DEADLINE_HOURS = 8.3`, contado desde `NB_T0`. Se pasa a `probe_dual.py` como `--deadline T0 + 8.3·3600`. Deja 0.7 h hasta el límite de 9 h de Kaggle.
- La etapa 1 lleva `timeout = 2 h` por subproceso.
- La celda 6 escribe `RAPTOR_SECONDS` y `total_seconds` en `blend_receipt.json`. `submission_receipt.json` trae `seconds` y `seconds_including_notebook_setup`. Esos archivos son la forma de medir cuánto tarda de verdad.

## 5. Dónde se publica `submission.csv`

En la **celda 6**, con `sub.to_csv(WORK / 'submission.csv', index=False)`, sin escritura atómica ni archivo de etapa intermedio. La celda 6 es la última. Las aserciones previas comprueban el orden de UIDs, que todo sea finito y que esté en [0, 1].

---

## 6. Comparación con el 0.940

| | 0.940 | 0.957 |
|---|---|---|
| Celdas | 24 | 7 |
| Familias | DINOv2-S (20 miembros, pilkwang), A5 (Mattia Angeli, 5 pliegues), RadImageNet ResNet-50 (E10/E13/E11 + calibrador Ridge), Raptor (4 brazos CoAtNet), 4 variantes CoAt (`resgated`, `global96`, `d4_swa3`, `repairv1`) | Raptor (1 brazo CoAtNet, checkpoint reentrenado con OAI) + R7 ConvNeXt-small (4 pliegues) |
| Mezcla final | Pesos Raptor/CoAt por hallazgo de 0.70 a 1.00 (Lateral Meniscus 1.00; Medial Meniscus 0.80; ACL, Lateral OA y Fracture 0.75; el resto 0.70) | 0.5 / 0.5 en todos |
| Reloj | `T0` y `TIME_BUDGET = 8 h` (celda 3) | `NB_T0`/`T0` y `DEADLINE_HOURS = 8.3` |
| Etapa intermedia | `/kaggle/working/_pipeline_stage.csv` | Ninguna; `ours_sub.csv`, `raptor_raw.csv`, `raptor_sub.csv` |
| Publica | Celda 23 (relee y valida la etapa, escribe `.tmp`, `fsync`, `os.replace`) | Celda 6 (escritura directa) |
| Funciones de apoyo | `rsna_frame`, `rsna_phase`, `rsna_event`, `rsna_save_predictions` | Ninguna |
| GPU de referencia | 2 × T4 | T4 (el aviso de la celda 2 dice «a T4 GPU»; `ngpu` se detecta en ejecución) |
| Aporta | — | Pesos de Raptor entrenados con OAI y R7 con fastMRI auxiliar y apiladores de estado |
| Pierde | — | DINOv2, A5, RadImageNet, las 4 variantes CoAt y los 3 brazos extra de Raptor |

### ¿Hay CNN de ImageNet tipo ConvNeXt o EfficientNet?

- **0.940: no.** Las familias son transformador (DINOv2-S), CoAtNet (híbrido conv-atención, preentrenado en ImageNet-12k) y ResNet-50 con pesos RadImageNet (no ImageNet). A5 construye su backbone con `timm.create_model(cfg['backbone'])` y el nombre viene de `cfg`, que el notebook lee de la configuración de los pesos de Mattia Angeli (no la inspeccioné). Desde el notebook no se puede saber cuál es; el código admite ViT (`'vit_' in backbone`). «convnext» aparece solo dentro de la tupla `hybrid = arch.startswith(...)`, que decide el pooling de timm. No hay EfficientNet.
- **0.957: sí, ConvNeXt-small.** La rama de entrenamiento carga pesos públicos de ImageNet de timm desde la caché local (`HF_HUB_OFFLINE=1`). No hay EfficientNet ni ResNet de ImageNet.

**Consecuencia para el brazo v5** (`eff`, `cnxhib`, `effhib`, `r34att`; 20 CNN ajustadas desde ImageNet):

- **En el 0.940** las cuatro familias son nuevas.
- **En el 0.957** `cnxhib` compite con el R7 ConvNeXt-small: misma familia de backbone y probablemente predicciones más correlacionadas con la mitad «ours» de la mezcla. `eff`, `effhib` y `r34att` siguen siendo familias distintas.
- **El peso `V5_W = 0.20` plano** se pensó contra la mezcla del 0.940. Aquí la mezcla tiene solo dos fuentes con 0.5/0.5, así que el 0.20 mueve una proporción distinta del resultado. No lo he medido: habría que correlacionar las predicciones OOF de v5 con las de un R7 reentrenado, o aceptar el peso por analogía.

---

## 7. Qué habría que cambiar en `kaggle/brazo_v5/celda_brazo_v5.py` para insertarla

Dónde insertar: una celda nueva **después de la celda 6**. La v5 es fail-soft y deja `submission.csv` intacto si algo falla, así que ya publicado el archivo no se pierde nada si se omite. El 0.940 la insertaba entre la 22 y la 23, pero este notebook no tiene celda de publicación aparte.

### 7.1 Variables de tiempo

| Cambio | Motivo |
|---|---|
| Definir `TIME_BUDGET` antes de la celda. Sugerencia: `TIME_BUDGET = 8.0 * 3600`; la alternativa es `DEADLINE_HOURS * 3600` (8.3 h). | `_v5_restante()` devuelve `None` si falta `TIME_BUDGET`, y `brazo_v5()` omite el brazo con «no se pueden leer T0/TIME_BUDGET». Sin este cambio v5 **no se ejecuta nunca y no da error visible**. |
| `T0` ya existe (celda 4: `T0 = NB_T0`). No hay que tocarlo. | Debe seguir siendo float de `time.time()`. |
| Decidir 8.0 h o 8.3 h. | Con 8.3 h, el margen hasta las 9 h de Kaggle baja de 1 h a 0.7 h (menos los 600 s de `V5_MARGEN_S`). Con 8.0 h se conserva la reserva del 0.940. Si la ejecución real ya consume casi todo el tiempo, v5 se omite en cualquier caso. |
| Medir cuánto queda de verdad al llegar a la celda. | Las 8.3 h no son un objetivo: no sabemos cuánto tarda el notebook. Los datos están en `blend_receipt.json` (`total_seconds`) y `submission_receipt.json`. Con el test oculto de tamaño desconocido, la regla `restante < FIJOS + N·s_estudio + MARGEN` decide sola. |

### 7.2 Función de validación

- `rsna_frame` no existe en este notebook. `brazo_v5()` ya lo trata como opcional (`callable(globals().get("rsna_frame"))`), así que `_v5_validar` es el único validador. No hay que cambiar nada, pero conviene saber que la validación equivalente a la de la celda 6 queda solo en `_v5_validar`: UIDs únicos y completos, columnas en orden, valores finitos y en [0, 1].
- `_RSNA_TEST_IDS` y `_RSNA_LABELS` tampoco existen. La celda usa `test.csv` y `V5_LABELS`, que coincide con las 12 columnas de `sample_submission.csv`. Opcional: definir `_RSNA_TEST_IDS = test_ids` y `_RSNA_LABELS = labels` para que la comparación con el notebook sea explícita.
- Raíz de datos: `_v5_datos()` busca `ROOT`. Aquí la variable se llama **`DATA`** (celda 4, `Path`). Poner `ROOT = DATA` antes de la celda, o `_v5_datos()` recurre a una búsqueda por `test_series.csv` hasta 6 niveles (más lenta con el test completo).
- `rsna_phase`, `rsna_event` y `rsna_save_predictions` faltan; `_v5_nb` los ignora. La traza queda solo en `print` y en `brazo_v5/estado.json`. Opcional: volcar `BRAZO_V5_ESTADO` en `blend_receipt.json`.

### 7.3 Archivo de etapa

- `V5_STAGE = "/kaggle/working/_pipeline_stage.csv"` no existe aquí. Cambiarlo a **`/kaggle/working/submission.csv`**, que es lo que escribe la celda 6.
- La celda sobrescribe ese archivo con `os.replace` y guarda una copia previa en `/kaggle/working/brazo_v5/_pipeline_stage_antes_v5.csv`. Esa copia tendría que renombrarse a algo como `submission_antes_v5.csv`.
- La mezcla opera sobre los rangos de `submission.csv`, que ya son la mezcla Raptor + R7: `(1−W)·rango(submission) + W·rango(v5)`, vuelta a rankear.
- `blend_receipt.json` seguirá describiendo solo los pesos 0.5/0.5. Si v5 se aplica, la receta final deja de coincidir con ese recibo, salvo que se amplíe.

### 7.4 Dataset y entorno

- Añadir como entrada el dataset de Owen con el paquete v5 (`rsna/`, `pesos/`, `wheels/`); este notebook no lo trae.
- La celda 4 ejecuta `pip install --no-index --no-deps` con ruedas fijadas del dataset de código (`dependencies.zip`). Se desconoce su contenido; si incluye `torch`, `torchvision` u `opencv`, el paquete v5 (`torch`, `torchvision` con `weights=None`, `cv2`, `pydicom`, `pandas`) usaría esas versiones. Hay que imprimir `pip list` en una ejecución y comprobar que `torchvision.models.get_model` y los decodificadores siguen disponibles.
- `HF_HUB_OFFLINE=1` ya está puesto: no afecta a v5 (pesos propios, sin descarga).
- Los nombres globales que define la celda (`V5_*`, `brazo_v5`, `BRAZO_V5_ESTADO`) no chocan con los del 0.957.
- Memoria de GPU: las etapas 1 y 2 corren en subprocesos y liberan la GPU, así que `_v5_gpus()` debería ver VRAM libre. Hay que verificarlo en la primera ejecución.

### 7.5 Resumen de cambios

1. `TIME_BUDGET = 8.0 * 3600` (o `DEADLINE_HOURS * 3600`) antes de la celda. **Obligatorio.**
2. `ROOT = DATA`. Recomendado.
3. `V5_STAGE = "/kaggle/working/submission.csv"` y renombrar el archivo de copia previa. **Obligatorio.**
4. Adjuntar el dataset con el paquete v5.
5. Revisar `V5_W` (0.20 plano) a la luz del solape ConvNeXt, y decidir si `cnxhib` debe entrar o pesar menos.
6. Decidir si se amplía `blend_receipt.json` con el estado de v5.
7. Antes de enviar: ensayar con el test falso de `ensayo_tiempo_kaggle.py`, adaptado a las variables de este notebook (`NB_T0`, `DATA`, sin `rsna_frame`), y medir los segundos reales de las etapas 1 y 2.

---

**Lo que no pude confirmar:** el backbone exacto y `views` de «ours» (están en `manifest.json`, fuera del notebook), el backbone de A5 en el 0.940, el contenido de `dependencies.zip`, cualquier tiempo de ejecución del 0.957 y la correlación real entre `cnxhib` y R7.
