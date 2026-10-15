# Brazo DINOv2 ×20 para el notebook 0.957: diseño, datasets, tiempo y pruebas

**Fecha:** 2026-10-14
**Alcance:** celda para pegar en el notebook 0.957, `script_dinov2.py` y su prueba en CPU. No hubo GPU, ni red, ni envío. El código de DINOv2 **no se ejecutó nunca de verdad**: en Yuca falta `transformers` y los pesos no están en el clúster (§5).
**Etiquetas:** [hecho] = verificado en archivo o log citado; [estimación] = número con supuestos explícitos; [no verificado] = no se pudo comprobar con lo que hay en el clúster.

## Resumen

- **Qué es.** Una celda que se pega **al final** del 0.957 (después de la celda 6, la que escribe `submission.csv`). Lanza como subproceso la familia DINOv2-S ×20 del notebook 0.940 y mezcla su salida por rangos sobre `/kaggle/working/submission.csv` con **W = 0.15** en los 12 hallazgos: `rango final = rank((1 − W)·rango(submission) + W·rango(DINOv2))`.
- **Qué se entrega** (`kaggle/brazo_dinov2/`):

| Archivo | Qué es |
|---|---|
| `celda_para_pegar_0957.py` | La celda. Lleva el script incrustado (1,944 líneas, 94 KB) |
| `celda_para_pegar_0957_diag.py` | Lo mismo con `BRAZO_DINOV2_EXIGIR = True` delante: si el brazo se omite o falla, lanza excepción |
| `script_dinov2.py` | El script independiente (1,626 líneas). Generado, no se edita a mano |
| `generar_brazo_dinov2.py` | Extrae el script de las celdas 3–14 del 0.940 con `ast` y arma las celdas. `--comprobar` verifica que todo está sincronizado |
| `celda_brazo_dinov2.py` | La lógica de la celda sin el script (plantilla; el generador le inserta el script) |
| `test_celda_brazo_dinov2.py`, `inventario_nodos.json` | La prueba de CPU y la lista de nodos extraídos por celda |

- **Tiempo para ~1,300 estudios con 2 × T4:** **22 / 34 / 62 min** (optimista / central / pesimista), más 10 min de margen que la celda se reserva (§3). La celda usa el escenario pesimista como guarda: pide **~72 min** restantes con 2 GPU y **~106 min** con una [estimación].
- **Probado:** 343/343 comprobaciones PASS en CPU (job **2085259**), con la inferencia simulada. **No probado:** DINOv2 real, GPU, transformers, los pesos y el entorno del 0.957 (§5, §6).
- **Dos cosas que debes decidir o saber antes de enviar:** el W = 0.15 es varias veces el peso efectivo que DINOv2 tiene en el 0.940 (§6.1); y los datasets de §2 hay que adjuntarlos a mano.

## 1. Cómo se construyó el script (sin reescribir la lógica)

`generar_brazo_dinov2.py` lee `referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb` y copia **por líneas** el texto de nodos de primer nivel de las celdas 3 a 14 (sin `ast.unparse`, así que comentarios y formato interiores se conservan). Lo que toma y lo que no, por celda:

| Celdas | Se toma | No se toma, y por qué |
|---|---|---|
| 3 | imports, `_cuda_execution_probe`, `rsna_sha/json/event/finite/rank01/rankpct/frame/save_predictions/strict_load/deadline/array/release_pixels/phase`, todas las constantes (`T0`, `TARGETS`, `SLOTS_RECOVERED`, `RULES_*`, `TIME_BUDGET`…) | `DEVS` y la exigencia de **exactamente 2 × T4** (la decide la celda, que acepta 1 o 2 GPU); **`rsna_initialize`** (borra `submission.csv` y `_pipeline_stage.csv` y hace `chdir`); **`rsna_asset_preflight`** (exige los datasets de Raptor, CoAt y A5); el gancho de IPython |
| 4 | `log`, `_one_direct`, `find_dinov2`, `IMG = CACHE_IMG` | `ROOT = find_root()`, `plan_cache`, `rsna_initialize(...)`, el preflight y la lectura de `train.csv` |
| 5–12 | todo: cabeceras, lateralidad, ranuras, ordenación, lectura de cortes, `build_cache`, `SlotHead`, `Model`, `build_model` (con su caché de plantillas) | — |
| 13 | todo: prefijo compartido de 6 bloques, `infer_from_package`, `find_weights`, `check_fingerprint`, `_combine`, `adopt_config_globals`, pooling por ventana… | — |
| 14 | `write_submission`, `_v37_validate_submission`, `main` | — |
| 15–16 | no se copian; el pegamento hace lo mismo en 6 líneas: `rsna_phase`, `main()`, limpieza, y la compuerta 20/20 | — |

**Lo que se añade (pegamento, claramente marcado en el archivo):** argumentos `--data`, `--out`, `--deadline`, `--comprobar`; construir `DEVS` con la sonda de la celda 3 sobre las GPU visibles; fijar `ROOT`, `_RSNA_TEST_IDS`, `DINOV2_SOURCE`; reajustar `TIME_BUDGET` al plazo recibido; renombrar `_pipeline_stage.csv` a `dinov2_sub.csv`; y escribir `informe.json` (miembros, estudios sin series, eventos). `DINOV2_SOURCE = find_dinov2('small')` se mueve del nivel superior al pegamento para que el script se pueda importar sin los datasets.

**Una sola sustitución de texto, declarada:** la ruta `'/kaggle/working/diagnostics'` (2 apariciones, en `rsna_save_predictions` y `rsna_phase`) pasa a la variable `DIAG_DIR`, para que los diagnósticos se escriban bajo `--out`. El test comprueba que sigue habiendo exactamente 2.

**Dos efectos secundarios del notebook que se pierden, a propósito:** `np.random.seed` / `torch.manual_seed` (eran llamadas sueltas; la inferencia no usa aleatoriedad: el *jitter* está desactivado) y los `print` de arranque.

**Verificación independiente de que el texto es el del notebook:** el test vuelve a parsear el notebook y compara con `ast.dump` las **61 funciones y clases** extraídas; las 61 son idénticas (59 sin cambios más `rsna_save_predictions` y `rsna_phase` con la sustitución anterior) [hecho, job 2085259]. Un análisis de nombres sin resolver sobre el script completo deja un único nombre, `augment`, que ya falta en el notebook y solo se llama con `jitter=True`; el notebook nunca lo activa [hecho].

## 2. Datasets de Kaggle que hay que adjuntar

Además de los datasets que ya trae el 0.957. Sacado del código del 0.940 (`find_weights`, celda 13; `find_dinov2`, celda 4); no verifiqué que sigan existiendo ni que tu cuenta los vea [no verificado].

| # | Qué | Ruta que busca el código | Archivos que usa |
|---|---|---|---|
| 1 | Dataset **`pilkwang/rsna-knee-weights`** | `/kaggle/input/datasets/pilkwang/rsna-knee-weights` (o `/kaggle/input/rsna-knee-weights`) | `manifest.json` con **exactamente 20 miembros** de ids distintos, y los 20 `.pt` que ese manifiesto nombra |
| 2 | **Modelo `metaresearch/dinov2`**, framework PyTorch, variación **small**, versión **1** (en el selector «Add Input → Models») | `/kaggle/input/models/metaresearch/dinov2/PyTorch/small/1` (también acepta `.../pytorch/small/1`, `/kaggle/input/dinov2/PyTorch/small/1` y `/kaggle/input/dinov2/pytorch/small/1`) | `config.json` y `pytorch_model.bin`, los dos obligatorios |
| — | Los datos de la competencia | los que ya tiene el 0.957 (`DATA`) | `test.csv`, `test_series.csv`, `test_series/` |

Reglas que el código impone y que conviene saber:
- Debe existir **exactamente un** directorio candidato para cada uno (`_one_direct` y `find_weights` fallan si hay cero o dos). No adjuntes dos versiones.
- Si faltan, la celda **no falla a ciegas**: `--comprobar` lo detecta y el brazo queda `omitido` con el motivo.
- **No hacen falta:** el *bundle* heredado `rsna_20260807_v1.pt` (`legacy_group_members()` devuelve `{}`), los datasets de A5, RadImageNet, Raptor ni CoAt, ni ruedas (`wheels`). Ninguna de las celdas 3–15 del 0.940 instala nada con `pip`.
- **Librerías que el entorno de Kaggle debe traer:** `torch`, `transformers` (`AutoModel.from_pretrained`), `pydicom`, `numpy`, `pandas`. La celda 4 del 0.957 instala ruedas de `dependencies.zip` (contenido desconocido) y podría cambiar `torch` o `numpy` [no verificado, §6.4].
- `pilkwang/rsna-knee-weights` y `metaresearch/dinov2` son los mismos que usa el notebook público 0.940. No son datasets de Owen ni privados de nadie del equipo [inferencia: el 0.940 es público y los monta].

## 3. Estimación de tiempo (~1,300 estudios, 2 × T4)

El 1,300 es tu cifra; el tamaño real del test no se conoce (el 1,322 del log del 0.940 es `0.3 × 4,407`, no el test). La celda usa el `test.csv` real.

**A diferencia del brazo v5, aquí el preproceso en CPU de *todos* los estudios ocurre antes de la inferencia y no se solapa con ella** (`build_cache` rellena la caché entera y después se cargan los 20 miembros). Los términos se suman:

`segundos = FIJOS + N · (CPU_estudio + GPU_estudio / n_gpus)`

### Supuestos explícitos

| Término | Optimista | Central | Pesimista (la celda) | De dónde sale |
|---|---|---|---|---|
| `FIJOS` (s) | 200 | 250 | **300** | Log del 0.940 (3 estudios), solo como **cota del coste fijo**: 74 s la fase DINOv2, de los que ~66 s son fijos (26 s desde el fin de la decodificación hasta construir el primer modelo, donde entra la importación de `transformers`; 40 s las 20 cargas con huella) + ~8 s de importar torch y sondear las GPU. Se suman, sin medir, lectura en frío de ~1.8 GB de pesos [estimación: 20 × 22 M parámetros × 4 B; el único ViT-S local pesa 88 MB], la reserva de la caché y la pasada de cabeceras. **4 × lo del log** en el pesimista |
| `CPU_estudio` (s) | 0.23 | 0.40 | **1.0** | **Medido**: 0.232 s/estudio en Yuca con 4 CPU y 64 estudios de train (§4). Central = ×1.75 por vCPU de Kaggle más lentos (el factor 1.5–2 que supuso el plan de v5, sin medir). Pesimista = ×4: añade el relleno de 10 GB en disco, E/S fría y contención del GIL |
| `GPU_estudio`, una T4 (s) | 1.29 | 1.93 | **3.2** | FLOPs contados abajo ÷ rendimiento efectivo 15 / 10 / 6 TFLOPS |

**FLOPs por estudio [hecho, cuenta aritmética]:** 336 px / parche 14 = 24 × 24 = 576 parches + CLS = 577 *tokens*. Un bloque de ViT-S (d = 384, MLP 1,536) cuesta 2.04 GFLOP en las proyecciones y 0.51 GFLOP en la atención = **2.55 GFLOP**. Cada estudio tiene 6 ranuras × 10 ventanas = 60 imágenes de 3 canales (`window_starts(12, 3)` = 10 ventanas, comprobado en el test). Los 6 primeros bloques se calculan **una vez** por imagen y se comparten entre los 20 miembros; los 6 últimos, por miembro: 60·6 + 20·60·6 = **7,560 pasadas de bloque ≈ 19.3 TFLOP por estudio**.

**El rendimiento efectivo de la T4 (15 / 10 / 6 TFLOPS) no está medido.** El pico de la T4 en fp16 es 65 TFLOPS; un ViT-S con d = 384, atención de PyTorch/Hugging Face y *autocast* se queda en una fracción del pico. Es el mayor factor de incertidumbre. *Contraste, no usado para fijar nada:* en el log del 0.940, la última GPU en cargar su miembro terminó los 10 miembros de 3 estudios en ≲ 3.7 s, lo que implica ≳ 8 TFLOPS con lotes pequeños (18 imágenes). Es coherente con el central y deja el pesimista como holgado.

### Resultado

| N = 1,300 | Por estudio | Total | Con el margen de 10 min |
|---|---|---|---|
| Optimista, 2 GPU | 0.87 s | **22 min** | 32 min |
| Central, 2 GPU | 1.36 s | **34 min** | 44 min |
| **Pesimista (guarda de la celda), 2 GPU** | 2.61 s | **62 min** (61.5) | **72 min** |
| Pesimista, 1 GPU | 4.22 s | 96 min | 106 min |
| Central, 1 GPU | 2.33 s | 55 min | 65 min |

Sensibilidad al tamaño (pesimista, 2 GPU): N = 1,000 → 49 min; 1,300 → 62; 1,600 → 75.

**Lo que la celda hace con esto:** si `restante < FIJOS + N·(CPU + GPU/n_gpus) + 600 s`, omite el brazo. Además fija un plazo duro (`restante − margen`) y mata el subproceso (y sus hijos) si lo alcanza. El script recibe ese plazo 30 s antes, para que sus propias comprobaciones (`rsna_deadline`, que en el camino que usa este script —prefijo compartido— solo se comprueba en la ordenación y la decodificación de la caché; durante la carga de los miembros y la inferencia solo manda el corte de la celda) tengan ocasión de parar solas. La estimación mala no puede llevar al notebook más allá de su `TIME_BUDGET`.

**Memoria y disco:** la caché de píxeles es `N × 6 × 12 × 336 × 336` bytes = **9.8 GiB** (10.6 GB) para 1,300 estudios. El código la pone en un `memmap` en `/kaggle/temp` (o `/tmp` si no hay sitio) porque pasa de 1 GiB, y la borra al terminar. La RAM con N = 1,300 no está medida (los pesos se cargan de uno en uno en CPU y los lotes son de 8 estudios); el *cgroup* del 0.940 es de 30 GiB.

## 4. Medición del preproceso real (lo único del 0.940 que se pudo ejecutar de verdad)

El test ejecuta el código extraído (`walk`, `annotate`, `pick_slots`, `lat_of`, `build_cache`) sobre **64 estudios de train** (semilla 2026), con una raíz falsa de enlaces simbólicos, bajo un trabajo de **4 CPU**:

| | Valor |
|---|---|
| Estudios / series | 64 / 337 (5.27 series por estudio) |
| Ranuras llenas | 285 de 384 (**74 %**; el ejemplo del 0.940 llenó 12 de 18 = 67 %) |
| Cabeceras + ranuras + lateralidad | 0.011 s por estudio |
| Caché (ordenación + decodificación + remuestreo) | 0.220 s por estudio |
| **Total** | **0.232 s por estudio** (14.8 s para 64) |
| Series con cortes ilegibles / estudios sin ninguna ranura | 0 / 0 |
| Forma y rango de la caché | (64, 6, 12, 336, 336) `uint8`, 0–255 |

Matices honestos: son estudios de **train**, no del test oculto; Yuca lee de Lustre, no del disco de Kaggle; la caché cabe en RAM (520 MB), mientras que con N = 1,300 irá a `memmap`; y el resultado de 0.232 s es una sola medición de un solo job (el primero, antes de corregir un umbral de la prueba, dio 0.248 s). Por eso la celda usa 1.0, no 0.23. Un umbral de la prueba (exigía llenar el 80 % de las ranuras) lo había puesto sin datos; al ver 74 % lo cambié por «ningún estudio vacío y al menos la mitad», con el valor medido en el mensaje.

## 5. Qué se probó y qué no

**Job 2085259** (CPU, 4 núcleos, 58 s), `venvs/rsna`, commit de trabajo `5d0ffb7` + cambios sin versionar (los hashes están en el log): **343/343 comprobaciones PASS, 0 FAIL**.

| Hash sha256 | Archivo |
|---|---|
| `1c4427e4702dfa07d0fdabab07f908527382510de1fcaccbf4f175b0e5f2d7c4` | `celda_para_pegar_0957.py` |
| `863e843920fa70e2e432e55b7545085a2e96228e21e53c6e4a48b642acf3ed82` | `celda_para_pegar_0957_diag.py` |
| `fc5375509d949b5de22d42af94a7b3066e5489e1456a6617316682c00e1b9b4f` | `script_dinov2.py` |
| `5a4c40d2280f568824a54800765ba77c6177f6de6dc5b0800ef60382dad41d22` | `generar_brazo_dinov2.py` |
| `9f2044c0b84727c4c0a2ad2f0859cbef1a46a67f4423ec2572127a3f46530276` | `test_celda_brazo_dinov2.py` |
| `0e74108509102e93c86d3bd6c78164da2dd9f8050efccf6d32a52a78b9a2785f` | `celda_brazo_dinov2.py` |

Logs: `logs/rsna-probar-dinov2_2085259.out`; resultados completos en `runs_joako/2085259/casos/resultados.json`.

**Qué cubre:**
1. **Procedencia (A).** El generador reproduce los tres archivos byte a byte; 61 definiciones idénticas por AST al notebook; el script no toca `submission.csv` ni borra nada fuera de su caché; solo escribe bajo `--out`; se importa limpio; `_combine`, `write_submission`, `window_starts` y los 12 hallazgos coinciden con una reimplementación aparte y con `LAB` del 0.957.
2. **La celda con inferencia simulada (B)**, para N = 7 y N = 1400, normal y diagnóstica: mezcla recalculada de forma independiente (diferencia < 1e-9), columnas y orden intactos, las aserciones de la celda 6 del 0.957, valores en (0, 1], copia previa idéntica, sin temporal; sin tiempo, sin `T0`, solo `NB_T0`, borde de la guarda (±60 s), una GPU, ninguna GPU, requisito que falta, etapa inválida (4 formas), más del 5 % de estudios sin series, 19 miembros en vez de 20, predicciones anticorrelacionadas (n ≥ 100), sha256 del script alterado, W = 0 en un hallazgo.
3. **El mecanismo real de subproceso (B)**, con un script falso: argumentos exactos (`--data`, `--out`, `--deadline = límite − 30 s`), `CUDA_VISIBLE_DEVICES`, una GPU con poca VRAM descartada (con un `nvidia-smi` falso), código de salida ≠ 0, comprobación previa que falla, y **un script colgado se corta en el plazo y su proceso nieto muere** (6 s de plazo, nieto muerto).
4. **La cola real de la celda**, sin simular nada salvo `nvidia-smi`: con el **script real** y sin `transformers`, la celda queda `omitido` por «falta algún requisito» (no por `T0`/`TIME_BUDGET`) y la diagnóstica lanza `DV2_EXIGIR`.
5. **Preproceso real (C)**, §4.

**Qué NO cubre, y por qué:**
- **DINOv2 real no se probó.** En el `venvs/rsna` de Yuca **no existe `transformers`** (`ModuleNotFoundError`), que `build_model` usa en `AutoModel.from_pretrained`, y está prohibido instalar paquetes. Además `pilkwang/rsna-knee-weights` y `metaresearch/dinov2` no están en el clúster, y el único peso de DINOv2 local es `modelos/torch/dinov2_vits14_pretrain.pth` (de Owen, 88 MB, sin `config.json`: no es el formato `from_pretrained` de Hugging Face [inferencia]); no lo toqué. Ni la carga de los 20 miembros, ni las huellas (`check_fingerprint`), ni el prefijo compartido (`_check_shared_dino_path`), ni la inferencia, ni la mezcla con tensores reales se ejecutaron.
- **GPU:** ninguna. No hay medida de rendimiento en T4 ni en MI210; probarlo exige un job de GPU, que necesitaría tu autorización.
- **El entorno del 0.957 en Kaggle** (versiones de `torch`, `transformers`, `numpy` tras instalar `dependencies.zip`), la VRAM libre al llegar a la celda y la RAM con N = 1,300.
- **Lo que haga el modelo con el *mismo* preproceso:** el log del 0.940 muestra que las huellas coinciden dentro de 4.6e-6, pero eso es en su entorno.

## 6. Riesgos y decisiones

1. **W = 0.15 es varias veces el peso efectivo de DINOv2 en el 0.940.** Según `arena_v2_2026-10-13.md` §1.2, DINOv2 pesa **0.00–0.12** en la mezcla final del 0.940 (0.03–0.05 en la mayoría de hallazgos; 0.10–0.12 solo en Fracture y Baker's), porque está dentro de un bloque con A5 y RadImageNet. En el 0.957 la base tiene solo dos fuentes (Raptor y R7) con 0.5/0.5, así que 0.15 es un peso mucho mayor para una familia que nadie ha validado aquí. Es tu decisión y no la cambié; no hay OOF de DINOv2 sobre los datos del equipo con los que medirlo [inferencia]. Si quieres un valor distinto: `DV2_W` en la celda, o el diccionario por hallazgo (`W = 0` deja una columna intacta).
2. **Coste del envío.** Con el escenario pesimista el brazo necesita ~72 min de holgura. El 0.957 no trae log de tiempos (`inventario_base_0957.md` §4), así que no se sabe cuánto queda al llegar a la celda. La variante diagnóstica responde a «¿cupo?» con un envío.
3. **Una sola ejecución con 3 estudios no sirve para el tiempo.** El «Save & Run All» visible usa el test público (3 estudios): confirma que el brazo corre (`applied`), no que cabe.
4. **Entorno:** `transformers` y la versión de `torch` que deje la celda 4 del 0.957 [no verificado]. `--comprobar` lo detecta y omite. Si `transformers` no está en la imagen de Kaggle sin internet, el brazo nunca correrá.
5. **Estudios sin series.** Un estudio sin ninguna ranura utilizable queda como fila de ceros enmascarada en el script (política del 0.940). La celda lo lee de `informe.json` y **no lo deja votar** (conserva el rango del notebook); si pasan del 5 %, no mezcla. Es el mismo criterio que el brazo v5.
6. **Guarda de correlación.** Con N ≥ 100, si la correlación media de rangos con el notebook es ≤ 0, no se mezcla. El umbral 0 lo puse yo: solo atrapa predicciones sin sentido, no un brazo malo.
7. **Convivencia con el brazo v5.** Los nombres no chocan (`DV2_`, `_dv2_`, `brazo_dinov2`, carpeta propia). Si pegas las dos celdas, la segunda ve menos tiempo restante.
8. **El peso de los `.pt` en frío** (1.8 GB) y el relleno de 10 GB de la caché no están medidos; entran en `FIJOS` y `CPU_estudio` pesimistas.

## 7. Cómo usarla

1. Adjunta los dos recursos de §2 al notebook 0.957.
2. Acelerador **GPU T4 ×2**, internet apagado.
3. Pega `celda_para_pegar_0957.py` como celda nueva **después de la última** (la 7.ª). Si pegas `celda_para_pegar_0957_diag.py` en su lugar, un brazo omitido o fallido rompe el rerun.
4. **Save Version → Save & Run All** (test público de 3 estudios). Busca en el log `[dinov2-brazo] applied: 20 miembros` o `skipped (...)` / `FAILED (...)`. En *Output* quedan `brazo_dinov2/estado.json`, `submission_antes_dinov2.csv`, `run/dinov2_sub.csv`, `run/informe.json` y los diagnósticos por miembro.
5. Si la celda se pega incompleta, `_dv2_script()` lo detecta (`el script DINOv2 incrustado no coincide con su sha256`).
6. Regenerar: `python kaggle/brazo_dinov2/generar_brazo_dinov2.py`; verificar: `... --comprobar`.
