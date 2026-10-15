# C. R7 limpio (prefijo `r7_`): auditoría de `compact_train.py`

Fecha: 2026-10-14. **Veredicto: NO VIABLE en limpio en esta sesión.** No se encoló nada. Se pasa a D.

## Extracción

[rsna/r7_extraer_sources.py](../../rsna/r7_extraer_sources.py) lee el notebook como JSON, analiza la celda 4 con `ast` y evalúa solo los literales `SOURCES` y `SOURCE_SHA` (`ast.literal_eval`); no ejecuta nada. Los 7 fuentes están en [rsna/r7_sources/](../../rsna/r7_sources/), con `MANIFEST.json`. **Los 7 sha256 coinciden** con `SOURCE_SHA` del notebook.

| Fuente | Bytes | Líneas |
|---|---|---|
| `compact_train.py` | 230,308 | 3,073 |
| `compact_infer.py` | 84,290 | 1,300 |
| `px_cache.py` | 20,263 | 360 |
| `probe_infer.py` | 16,584 | 333 |
| `knee_anchor_bank_v2.py` | 14,529 | 280 |
| `probe_dual.py` | 8,490 | 165 |
| `side_common.py` | 1,571 | 36 |

Material de terceros: se audita como datos, no se ejecuta.

## Respuestas a las preguntas

Las líneas citadas son de `rsna/r7_sources/<archivo>`.

### ¿Entrena sin `--fm-aux`? **Sí.**

- `--fm-aux` tiene `default=None` (`compact_train.py:1897`).
- El docstring dice: «Off (no --fm-aux, the default) = v9.7 bit for bit in every output and printed line» (`compact_train.py:32-33`).
- Pero entonces **ya no es el R7**: la celda 0 del 0.957 describe el R7 como «fastMRI auxiliary training».

### ¿Con views sin SKM-TEA? **El entrenador sí; la caché no se puede construir.**

- Solo los conjuntos `v3k` y `v4` usan anclas SKM-TEA (`px_cache.py:84-100`; `compact_infer.py:329-330` exige `--anchor-weights` para ellos). `v1`, `v2` y `v3` no las usan (`px_cache.py:101-118`).
- El R7 probablemente usó `v4` a 384 px: el estabilizador de fluidos «se ajustó sobre la caché de 384 px (px384_v4)» (`compact_infer.py:78`), y el notebook verifica `anatomy.pt` si `views` es `v3k`/`v4` (celda 4, líneas 131-134 del volcado). Sin SKM-TEA habría que bajar a `v1`/`v2`/`v3`, otra configuración.

### ¿Qué datos y formatos necesita?

| Entrada | Formato | Línea | ¿La tenemos? |
|---|---|---|---|
| `--px` (caché de píxeles por corte) | `px.u8.npy` uint8 [N, S, S] + `meta.f32.npy` [N, 12] + `index.json` (bloques por serie y vista) | `compact_train.py:191, 234, 1996`; `px_cache.py:5-7` | **No, y no se puede construir** (abajo) |
| `--targets` | CSV con `StudyInstanceUID`, `is_gold`, `review_holdout`, `fold` y por hallazgo `<Hallazgo>\|prior`, `\|strict`, `\|lr_prior` | `compact_train.py:1998-2012, 2034-2047` | Adaptable: un brazo `nombre=bce:col:wcol` admite blendjev con pesos por celda (`compact_train.py:2040-2047`) |
| `--state-targets`, `--state-spec` (apiladores de estado de Effusion y Baker's del R7) | CSV y JSON de `make_state_targets.py` | `compact_train.py:1818-1820, 2057-2060` | **No**: `make_state_targets.py` y `analysis/fit_state_stacker.py` no están en los fuentes |
| Pesos ConvNeXt-small ImageNet | timm o safetensors | `compact_train.py:293-300, 367, 481-497` | Solo en formato torchvision (`modelos/torch/convnext_small-0c510722.pth`): habría que convertir claves |
| Configuración del R7 | `manifest.json`: `views`, `size`, `slices`, `phys_mm`, `eval_stride`, `state_spec`… | celda 4, líneas 76-131 | **No**: está en el dataset `rsna-knee-compact-r7cnxfsss`, no en el notebook |

## Bloqueos

1. **La caché de píxeles depende de código que no tenemos.** `px_cache.py` importa el paquete `training` del «mainline» (`CODE_ROOT`, `px_cache.py:14-15`): `training.coronal_views` (l. 104), `training.input_provenance.view_geometry`, `training.intensity_mapping.normalize_intensity`, `training.native_data.select_display_window` y `training.exact_duplicates.ExactDuplicateResolver` (l. 141-144). Además lee una caché nativa previa, `F:/rsnaknee/cache_native` (l. 3), que construye `training.native_cache.prepare` (`compact_infer.py:4`; `probe_infer.py:43-67`). Ese paquete viaja en `cloud_code.zip` del dataset de Kaggle `rsna-orthofoundation-fold0-epoch7-20261004`. No está entre los 7 fuentes, no está en `referencias/` (solo se descargaron notebooks) y la sesión no tiene red.
2. **Reimplementarlo no sería el R7.** Habría que adivinar la intensidad `series_companded`, la geometría de vistas, las vistas bilaterales coronales, la promediación en losas de 3.5 mm (`px_cache.py:20`) y la resolución de duplicados. Además, en Kaggle `compact_infer.py` infiere con el código original del mainline, así que una reimplementación rompería la paridad entre entrenamiento e inferencia.
3. **bf16 fijo en el código.** Las regiones de entrenamiento usan `torch.autocast("cuda", dtype=torch.bfloat16)` (`compact_train.py:1705, 1748, 2486, 2525, 2823`). En la MI210 con ROCm 6.4, el backward en bf16 cae con SIGSEGV en CoAtNet, y lo más probable es que la causa sean las convoluciones de MIOpen (`bench_coatnet_v2.md`). ConvNeXt es casi todo convoluciones. Habría que parchear a fp16 + GradScaler: otra desviación, sin probar.
4. **Particiones distintas.** El R7 parte con `sha1(uid)[:8] % K` (`compact_train.py:2012`), no con nuestro `md5 % 5`. El criterio «OOF del fold 0 ≥ 0.8628 (effhib)» habría que recalcularlo con effhib sobre los estudios del fold 0 del R7. Es posible (effhib tiene OOF de los 4,349), pero no es el mismo número.
5. **Coordinación con Owen.** El R7 es un **ConvNeXt-small, una familia CNN**. La instrucción de coordinación dice «NO entrenes familias CNN ni dupliques ese brazo» (Owen entrena ConvNeXt-Tiny). Aunque se desbloqueara lo técnico, C chocaría con esa regla: decisión de jcastro y Owen.

## Disco y GPU-h (si se desbloqueara; nada medido)

- **Caché de píxeles:** filas ≈ series × vistas × cortes ≈ 5 × 2-3 × ~30 ≈ 300-450 por estudio. A 384 px son ~45-65 MB por estudio, es decir **~200-290 GB** para 4,407 estudios (~110-160 GB a 288 px). Más la caché nativa uint16, de tamaño desconocido. Solo cabe en `/scratch` (~720 GB), no en Lustre.
- **GPU:** sin dato publicado. Los defaults son 10 épocas, 48 ventanas por estudio y lote 8 (`compact_train.py:1780-1782`). A ojo, 2-4 GPU-h por fold en una MI210 si bf16 → fp16 funciona; 4 folds ≈ 8-16 GPU-h.

## Qué lo desbloquearía

1. Obtener `cloud_code.zip` (paquete `training`) y `manifest.json` del R7. Eso exige red o una descarga hecha por jcastro fuera de esta sesión, revisando la licencia de esos datasets.
2. Decidir con Owen si un ConvNeXt-small entra o no en conflicto con su brazo CNN.
3. Un smoke en GPU de ConvNeXt-small con fp16 en lugar de bf16.

Sin 1 no hay R7 limpio. Con 1 y 2, el siguiente paso sería un fold 0 con `views v2`, sin `--fm-aux`, sin apiladores y en fp16. Esa configuración ya no sería la del R7 original.
