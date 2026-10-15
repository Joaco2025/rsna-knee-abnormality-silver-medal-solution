# Avisos de código de terceros

Este repositorio contiene código copiado o adaptado de notebooks públicos de la competición de Kaggle [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). Aquí se lista, para cada archivo, de qué notebook sale, quién lo escribió, bajo qué licencia y qué partes se copiaron literalmente.

## Licencia

Ninguno de los cuatro notebooks de origen declara licencia en su `kernel-metadata.json` (se comprobó el 2026-10-14 sobre las copias descargadas en `referencias/rsna_research/notebooks/`). Por eso se aplica:

- **Reglas de la competición, regla 3.6.b:** el código compartido públicamente en la competición debe publicarse bajo una licencia de código abierto aprobada por la OSI. La cita la aportó jcastro; no se verificó el texto de las reglas desde el clúster, que no tiene acceso de red.
- **Kaggle:** los notebooks públicos se publican por defecto bajo **Apache License 2.0**.

Por tanto, el código de estos notebooks se usa bajo Apache 2.0, conservando la atribución de abajo. Los derechos de autor del código copiado siguen siendo de sus autores. Los pesos, conjuntos de datos y etiquetas que esos notebooks usan **no** están incluidos en este repositorio.

## Notebooks de origen

| Id. | Notebook | Autor (Kaggle) | URL | Copia local consultada (sha256, 16 primeros) |
|---|---|---|---|---|
| **N957** | ≈0.957 \| Latest dataset \| score in one click! | heliosli | https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click | `referencias/0-957-latest-dataset-score-in-one-click.ipynb`, `cb2a0f57962ca9b6` |
| **N940** | Bend the Knee to Speedy Raptors – The Original | mattiaangeli | https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original | `referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb`, `32065fc04a644fe5` |
| **NDD** | Knee MRI: training the twelve-finding model | dreaddevelopment | https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model | `referencias/rsna_research/notebooks/dreaddevelopment/…/knee-mri-training-the-twelve-finding-model.py` |
| **NDD2** | Knee MRI: twelve findings from a single model | dreaddevelopment | https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model | `referencias/rsna_research/notebooks/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model/knee-mri-twelve-findings-from-a-single-model.py` |

Estos notebooks son a su vez trabajo derivado, y sus autores acreditan a otros:

- **N957** se declara "built on dreaddevelopment's Raptor CoAtNet/MIL inference recipe" y cita las etiquetas de stevenleehans, pilkwang y riadmohamed42.
- **N940** acredita a Jiwei Liu, Pilkwang, Sofia Anjenje, Antoine G., prvsiyan, Marwan Mahmoud, dreaddevelopment / Roman Tamrazov, renta.k y Anvith Pothula.

Esa atribución se extiende a este repositorio.

N940 tiene varias versiones. La que se copió es la de sha256 `32065fc04a644fe5…`, que cita `kaggle/brazo_dinov2/script_dinov2.py`. La copia en `referencias/rsna_research/` es otra versión (`2fc5a4206d76c13a…`).

## Archivos con código de terceros

| Archivo | Origen | Qué se copió literalmente | Qué es nuestro |
|---|---|---|---|
| `kaggle/notebooks/0957_mas_v5_diag.ipynb` | N957 | Las 7 celdas del notebook (2 de markdown con su descripción y créditos, y 5 de código), idénticas al original | La celda 8, el brazo v5, que es el contenido de `kaggle/brazo_v5/celda_brazo_v5_diagnostico.py` |
| `kaggle/brazo_dinov2/script_dinov2.py` | N940 | Celdas 3 a 14 completas, por nodos de primer nivel del AST y sin reescribir la lógica. Única sustitución de texto: la ruta `'/kaggle/working/diagnostics'` pasa a `DIAG_DIR` (2 apariciones) | El "pegamento": argumentos, plazo y comprobación de entorno. Lo genera `generar_brazo_dinov2.py` |
| `kaggle/brazo_dinov2/celda_para_pegar_0957.py` | N940 (vía `script_dinov2.py`) | Incrusta `script_dinov2.py` completo, es decir, las celdas 3 a 14 de N940 | La celda que lo escribe, verifica su sha256, lo lanza y mezcla los rangos |
| `kaggle/brazo_dinov2/celda_para_pegar_0957_diag.py` | N940 (vía `script_dinov2.py`) | Igual que la anterior | Igual, más la línea `BRAZO_DINOV2_EXIGIR = True` |
| `rsna/raptor_preproc.py` | N940, celdas 2 y 22 (`_KE_SRC`) | Medido línea a línea contra N940. **Literal completo:** `_eval_centers` (10/10 líneas). **Copia casi literal**, con renombres (`_dense_allocate` → `dense_allocate`, etc.) y cambios de formato: `dense_allocate` (18/21 líneas idénticas), `dense_unique_linspace` (5/8), `pick_series_for_slot` (19/25), `eval_windows` (22/30), `make_reader` (35/53, de `_make_reader`) y `prepare_volume` (33/57, de `_ke_prepare_windows`). Las constantes (`IMG`, `CROP_MM`, percentiles, medias, `SLOTS64`/`SLOTS44`, `ARMS`) tienen los mismos valores con otro formato | `reader_con_cache`, `ventanas`, la partición en volumen uint8 y ventanas, la eliminación de la auditoría JSONL y de `rsna_event`, y la importación diferida de `torch`. Paridad bit a bit con el original comprobada (job 2085083, 18/18) |
| `rsna/entrenar_coatnet.py` | NDD, igual a la de la celda 22 de N940 | La clase `RaptorClassifier` completa. La lógica de `centros()` sale de `StudyWindows._centers` y la del promedio de pesos del top-k, de `main()`, ambas de NDD | Todo lo demás: datos, pérdida con máscara, fp16 con GradScaler, evaluación, jueces y criterio |
| `rsna/r7_sources/*.py` (7 archivos) | N957, celda 4 (fuentes incrustadas `SOURCES`, de R7 y Raptor del equipo de heliosli) | **Literal completo**: `px_cache.py`, `compact_train.py`, `compact_infer.py`, `probe_infer.py`, `probe_dual.py`, `knee_anchor_bank_v2.py` y `side_common.py`, extraídos con `ast.literal_eval` sin ejecutar nada. Cada sha256 coincide con `SOURCE_SHA` del propio notebook (`MANIFEST.json`, `coincide: true` en los 7) | Solo `rsna/r7_extraer_sources.py` (el extractor) y `MANIFEST.json`. Son material para leer y auditar; no se ejecutan desde el repositorio |
| `rsna/entrenar_coatnet_dd.py` | NDD | Port casi literal: `StudyWindows` sin la rama ROI, `collate`, `build_backbone`, grupos AdamW, OneCycle y bucle de entrenamiento. Desviaciones listadas en el docstring del archivo | Importa `RaptorClassifier`, métricas y jueces de `entrenar_coatnet.py`; fp16 con GradScaler, peso `w` en la BCE, evaluación por época |
| `rsna/dd44_corpus.py` | NDD2: «Knee MRI: twelve findings from a single model» (dreaddevelopment; `knee-mri-twelve-findings-from-a-single-model.py`) | `_make_reader`, `_pick_series_for_slot` y `build_study`, copiados literalmente. Única diferencia: el import de `apply_modality_lut` cae a `pydicom.pixels` en pydicom 3 | Las ranuras de 44 cortes y la banda 0.06-0.94 (constantes), el multiproceso y el índice |
| `rsna/seeds_entrenar_coatnet.py` | Copia de `rsna/entrenar_coatnet.py` | Hereda lo de la fila de `entrenar_coatnet.py` (`RaptorClassifier`, `centros()` y el promedio top-k de NDD) | Opciones `--all-data`, pesos en fp16 y criterio sin OOF |

## Archivos que leen código de terceros pero no lo contienen

- **`rsna/medir_raptor_preproc.py`:** carga y ejecuta en tiempo de prueba el código de N940 desde `referencias/` para comparar bit a bit. No copia ese código.
- **`kaggle/brazo_dinov2/generar_brazo_dinov2.py`:** lee N940 para generar `script_dinov2.py`. Contiene solo nombres de celdas y de variables de N940, no su código.
- **`kaggle/brazo_dinov2/celda_brazo_dinov2.py`:** plantilla sin el script incrustado.

La receta de entrenamiento descrita en `experiments/reportes/receta_coatnet.md` (hiperparámetros del mensaje #53 del hilo 746792 de Danial Zakaria) se documenta, no se copia como código.

Las copias completas de los notebooks y del foro en `referencias/` están excluidas del repositorio por `.gitignore`.
