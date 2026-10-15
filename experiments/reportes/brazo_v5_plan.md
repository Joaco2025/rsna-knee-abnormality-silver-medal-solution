# Brazo v5 en el notebook público 0.940: diseño, coste y requisitos

**Fecha:** 2026-10-13
**Alcance:** diseño y código sin ejecutar. No hubo GPU, jobs, Python, red ni commits. Owen, solo lectura. Escrituras: `code_joako/kaggle/brazo_v5/` y este archivo.
**Etiquetas:** [hecho] = verificado en archivo o log citado; [inferencia] = deducido de hechos citados; [estimación] = número con supuestos explícitos; [no verificado] = no se pudo comprobar con las fuentes autorizadas.

> **Corrección [2026-10-14]: 1,322 no es el tamaño del test oculto.** Es `int(0.3 × 4,407)`, la heurística con que la celda 4 del notebook dimensiona su caché (el log dice «sizing for 4407 train + 1322 test» con 3 estudios reales). La celda ya no exige ese número: usa los UIDs de `test.csv`. Las cifras en minutos de esta página que dicen «1,322 estudios» son una escala de referencia, no el tamaño real del test. Hay además una variante diagnóstica (`celda_brazo_v5_diagnostico.py`). Detalle y pruebas: `experiments/reportes/noche_2026-10-14.md`, sección A. El texto de abajo se conserva como estaba.

## Resumen

- **Coste marginal de v5 en el rerun:** unos **1.8 s por estudio con las dos T4** (rango 1.3–2.6), es decir, **unos 44 min para 1,322 estudios** contando la carga de pesos (rango 31–62 min) [estimación, §1]. Lo marginal incluye leer los DICOM y preprocesarlos: v5 no puede reutilizar los tensores del notebook público.
- La incertidumbre principal es el factor T4/MI210. Se resuelve con una sesión interactiva sobre estudios de train (§3), que cuesta unos 20 min de cuota de GPU y ningún envío.
- **Recortar folds ahorra poco:** por debajo de unos 16 modelos manda el piso de CPU (preproceso con 4 vCPU), y 12, 8 o 4 modelos tardan lo mismo que 16 (~36 min). En el escenario pesimista no se ahorra casi nada (§2).
- **Dataset:** el paquete privado de Owen (`v5_cv20_eff_cnxhib_effhib_r34att_paquete.zip`, 1.84 GB). Para adjuntarlo, Owen tiene que compartirlo con la cuenta que corre el notebook y autorizar que se use en un envío del equipo (§4).
- **Encaje en el notebook** (leído directamente: `referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb`): la celda va entre la 22 (mezcla final) y la 23 (publicación). Sigue el patrón fail-soft de las cuatro CoAt y valida la etapa con el mismo `rsna_frame` que usará la celda 23 (§0).
- **Riesgo principal:** no sabemos cuánto le queda al notebook de sus 8 h de `TIME_BUDGET` al llegar a la celda 22 en el rerun. Si le queda menos de ~70 min (pesimista) o de ~55 min (central), el brazo se omite solo. Como el log del rerun oculto no se ve, un envío con el brazo omitido sería indistinguible del 0.940 (§3).

## Archivos

| Archivo | Qué es |
|---|---|
| `kaggle/brazo_v5/rsna/{inferir,modelo,preproc}.py` | Copia del código del paquete de Owen (`envios/v5_cv20_eff_cnxhib_effhib_r34att/rsna/`, commit `ff92320`). `cmp`: idéntica byte a byte [hecho] |
| `kaggle/brazo_v5/celda_brazo_v5.py` | La celda que se inserta entre las celdas 22 y 23 del notebook (índices desde 0) |
| `kaggle/brazo_v5/ensayo_tiempo_kaggle.py` | Celda de ensayo para una sesión interactiva de Kaggle: mide el tiempo con estudios de train (§3) |

La celda no ejecuta la copia de `code_joako`: ejecuta el código del dataset adjunto, pero antes compara su sha256 con el de la copia revisada (`V5_SHA256`). Si Owen sube otra versión con código distinto, el brazo se omite en vez de correr código que nadie revisó. También comprueba nombre y tamaño de los 20 pesos (`V5_MIEMBROS`, `V5_BYTES`, sacados del MANIFIESTO).

### Qué hace la celda

1. **Etapa y datos.** Toma los UIDs de `_RSNA_TEST_IDS`, comprobados contra `test.csv`, y el orden de columnas de `_RSNA_LABELS`, que debe coincidir con `sample_submission.csv`. La etapa tiene que tener esas 13 columnas en ese orden, los mismos UIDs sin duplicados y valores finitos en [0, 1]. Además debe pasar `rsna_frame`. En el rerun (`KAGGLE_IS_COMPETITION_RERUN`) exige exactamente **1,322** estudios; en la versión guardada no puede, porque el test público tiene 3. Si la etapa no es válida, no la toca.
2. **Guarda de tiempo.** `restante = TIME_BUDGET − (time.time() − T0)`. Necesita `V5_FIJOS_S + n · max(V5_PISO_CPU_S, modelos · V5_GPU_MODELO_S / n_gpus)` más `V5_MARGEN_S`. Con los valores pesimistas son 62 + 10 = 72 min. Si no alcanza, registra el motivo y no hace nada. Además fija un plazo duro (`restante − margen`). Si la inferencia lo alcanza, mata los procesos y deja la etapa intacta. Así una estimación equivocada no puede llevar al notebook más allá de su `TIME_BUDGET`.
3. **Inferencia.** Instala los decodificadores de DICOM del paquete en una carpeta propia (`--target`, sin tocar el entorno del notebook). Elige las GPU con al menos 6 GB libres según `nvidia-smi`. Reparte los estudios intercalados entre ellas (carpeta por GPU con su parte de `test.csv`/`test_series.csv` y un enlace a `test_series/`). Lanza `inferir.py` sin modificar, un proceso por GPU, en su propia sesión para poder matar también a los workers. Se reparte por estudios y no por modelos para que cada estudio se preprocese una sola vez: la CPU es el recurso escaso.
4. **Mezcla por rangos.** `(1 − W[h]) · rango_notebook + W[h] · rango_v5`, columna a columna, y luego se vuelve a rankear en (0, 1], igual que la mezcla final de la celda 22. Por defecto `W = 0.2` en los 12 hallazgos; con `W = 0`, la columna del notebook queda intacta. Un estudio que v5 no pudo leer (0.5 en todo) no vota y conserva su rango del notebook. Si esos estudios pasan del 5 %, se considera un fallo sistemático (p. ej. faltan decodificadores) y no se mezcla.
5. **Escritura atómica.** Escribe a `_pipeline_stage.csv.tmp_v5`, lo relee y le pasa mis comprobaciones y `rsna_frame`, la misma validación con la que la celda 23 lee la etapa. Guarda una copia de la etapa original en `/kaggle/working/brazo_v5/_pipeline_stage_antes_v5.csv` y solo entonces hace `os.replace`.
6. **Registro**, con el patrón de las CoAt:
   - `rsna_phase('yuca_v5', 'START')` al entrar y `rsna_phase('yuca_v5', 'COMPLETE', ok=…)` al salir.
   - `rsna_event('yuca_v5_applied' | 'yuca_v5_skipped' | 'yuca_v5_failed', …)`. Estos eventos acaban en `btkd_v559_complete.json` cuando lo escribe la celda 23.
   - Una línea `[yuca-v5] … continuing without it (flagged)`.
   - `rsna_save_predictions('yuca_v5_prob', …)` y `brazo_v5/estado.json`, con el tiempo restante, el estimado y el real, las GPU, los estudios ilegibles y la correlación media de rangos con el notebook.

   Cualquier excepción queda capturada.
7. **`V5_EXIGIR`** (por defecto `False`). Solo para un envío de diagnóstico: si el brazo no se aplica, lanza una excepción y el rerun falla. Es la única manera de saber desde fuera que el brazo no corrió (§3).

## 0. Lo que dice el notebook (leído directamente)

Índices de celda desde 0. No usé `experiments/reportes/agy_inventario_base_0940.md`, que tiene errores conocidos.

| Hecho | Dónde | Consecuencia para la celda |
|---|---|---|
| `T0 = time.time()` tras la comprobación de las dos GPU; `TIME_BUDGET = 8.0 * 3600` | celda 3, líneas 54 y 89 | Guarda en segundos sobre `time.time()`. Las 8 h dejan 1 h por debajo del límite de 9 h que cita la introducción. `V5_MARGEN_S` baja a 10 min |
| `rsna_deadline` y `_run_required_child` usan el mismo `TIME_BUDGET − (time.time() − T0)` | celda 3:194; celda 1:89-123 | Mismo criterio; el corte de procesos es por grupo (`start_new_session`), como el suyo |
| `_RSNA_TEST_IDS`, `_RSNA_LABELS` y `rsna_frame` (columnas exactas, UIDs, finitos en [0, 1]) | celda 3:116-172 | Orden y validación canónicos del notebook |
| La celda 22 termina con la mezcla final transformer/CoAt-Raptor, la vuelve a rankear y escribe `_pipeline_stage.csv` | celda 22:1000-1036 | La etapa ya está en rangos pct; la celda vuelve a rankear tras mezclar |
| La celda 23 lee la etapa con `rsna_frame(…, 'FINAL')`, guarda `final` y publica de forma atómica | celda 23 | Insertar justo antes. La salida tiene que pasar `rsna_frame` |
| Las cuatro CoAt son fail-soft: `try` → hijo → `ok=True`; `except` → `rsna_event('…_child_failed')`, `print('… FAILED; continuing without it (flagged)')`, `ok=False`; luego `rsna_phase(…, 'COMPLETE', ok=…)` | celda 22:735-866 | Patrón replicado |
| En la versión guardada (3 estudios), la VRAM reservada por el kernel al final es ≤ 63 MB por GPU, y `memory.current` del cgroup es 15.2 de 30 GiB | salidas de `[phase]` | Las dos T4 deberían llegar libres a la celda; la RAM tiene margen para dos procesos de unos 3 GB cada uno [inferencia, con 3 estudios] |
| Las salidas guardadas son de 3 estudios: el notebook termina en 341 s, dominados por el coste fijo | salidas de `[phase]` | No sirven para estimar el tiempo del rerun con 1,322 |
| `find_root()` acepta `RSNA_COMP_ROOT` | celda 4 | Permite cronometrar el notebook completo con estudios de train (§3.4) |

## 1. Coste marginal por estudio en T4

### Lo que dicen los ensayos

| Job | Qué | Máquina | Estudios | s/estudio | Fuente |
|---|---|---|---|---|---|
| 2084790 | **v4** (r34, eff, cnxhib, effhib ×5 = 20) | 1 MI210, 8 CPU, 6 workers, lote 4 | 160 de train | **1.04** | `code/logs/rsna-inferir-v4_2084790.out` [hecho] |
| 2084831 | **v5** (20 modelos) | CPU, 32 núcleos, 30 workers | 6 de train | **12.08** | `code/logs/rsna-inferir-v5-cpu_2084831.out`; sacct [hecho] |
| 2084989 | v5, repetición | CPU, 32 núcleos | 6 | 11.07 | `rsna-inferir-cpu_2084989.out` [hecho] |
| 2084712 | v3 (r34, eff, cnxhib ×5 = 15) | 1 MI210, 6 workers | los mismos 160 | 0.77 | `rsna-inferir-v3_2084712.out` [hecho] |
| 2083826 | v2 (2 modelos) | 1 MI210, 6 workers | los mismos 160 | 0.33 | `rsna-inferir-v2_2083826.out` [hecho] |
| 2081215 | 1 modelo | 1 MI210, 6 workers | los mismos 160 | 0.41 | `rsna-inferir-tiempo_2081215.out` [hecho] |

El cronómetro de `inferir.py` arranca después de cargar los pesos (`inferir.py:87`). Mide lectura de DICOM, preproceso y red, solapados por el DataLoader [hecho].

### Descomposición

- **v4 sirve como medida de v5.** v5 cambia r34 por r34att: la misma ResNet-34, con 16 cortes y recorte 288 [hecho, cabeceras de los logs]. Lo único que añade es la atención por hallazgo (`modelo.py:62-68`), un `einsum` de tamaño (lote × 12 × 4 × 512), despreciable frente a 64 imágenes de 288² por estudio. El coste de GPU de v4 y de v5 es el mismo dentro del ruido [inferencia].
- **GPU por modelo en la MI210: ≈ 0.052 s por estudio.** De 15 a 20 modelos (+5 effhib), el tiempo pasa de 0.77 a 1.04 s: 0.054 s por modelo. La media de v4 es 1.04 / 20 = 0.052, y la ordenada en el origen sale ≈ 0. Con 15 y 20 modelos la inferencia está limitada por la GPU y el preproceso queda oculto [inferencia].
- **Piso de CPU: 0.33–0.41 s por estudio con 6 workers en Yuca**, es decir, 2.0–2.5 s de worker por estudio. Con 1 y 2 modelos la GPU necesitaría solo 0.05–0.10 s, pero el tiempo no baja de 0.33–0.41 s. Ese suelo es la lectura y el preproceso [inferencia].
- **Qué aporta el ensayo en CPU (2084831).** (a) El paquete v5, con r34att y `per_label`, corre de punta a punta y da un archivo válido [hecho]. (b) En CPU, en fp32 porque `autocast` solo se activa en CUDA (`inferir.py:64,93`), cuesta ~12 s por estudio con 32 núcleos, 11.6 veces la MI210. Es coherente con un coste dominado por la red. (c) Con los 4 vCPU de Kaggle (8 veces menos hilos, y más lentos) serían ≥ 90 s por estudio, más de 33 h para 1,322. **Sin GPU no hay brazo**, y la celda se omite si no la encuentra [inferencia]. Con 6 estudios en 2 lotes, el arranque de 30 workers pesa mucho, así que 12 s es una cota superior.

### Traslado a Kaggle (2 × T4, 4 vCPU)

- **Factor T4/MI210 por GPU: 3.5 (rango 2.5–5)** [estimación]. En fp16 denso, la MI210 tiene 181 TFLOPS de pico y la T4 65 (2.8×). En memoria, 1.6 frente a 0.32 TB/s (5.1×). Además, la T4 baja de frecuencia con carga sostenida (70 W). EfficientNetV2 (convoluciones depthwise, limitadas por memoria) queda más cerca de 5×; ResNet y ConvNeXt, más cerca de 3×. Este factor **no está medido** y es la mayor fuente de error.
  → GPU por modelo y estudio en una T4: **g ≈ 0.18 s** (0.13–0.26). Los 20 modelos en una T4 tardan 3.6 s por estudio; repartidos entre dos, **1.8 s por estudio**.
- **Piso de CPU en Kaggle: ≈ 1.45 s por estudio** (1.0–2.2) [estimación]: 2.0–2.5 s de worker por estudio × 1.5–2 de hilo más lento (Xeon de 2.0–2.2 GHz con hyperthreading) ÷ 2.3–3 hilos efectivos. Son 4 vCPU compartidos por 4 workers y los 2 procesos principales. El piso es el mismo con 1 o con 20 modelos.
- **Tiempo por estudio = max(piso, modelos · g / 2) ≈ max(1.45, 1.8) ≈ 1.8 s** (1.3–2.6). Cuando los dos términos se acercan, el solapamiento no es perfecto y el tiempo real queda algo por encima del máximo.
- **Fijos:** 2.5–5 min. pip de 8 ruedas pequeñas, lectura de los 1.92 GB de pesos en cada uno de los dos procesos, arranque de CUDA y de cuDNN, y arranque de los workers.

| Escenario | Piso CPU | g (T4) | s/estudio, 20 modelos, 2 GPU | **Total 1,322** |
|---|---|---|---|---|
| Optimista | 1.0 | 0.13 | 1.3 | **31 min** |
| **Central** | 1.45 | 0.18 | 1.8 | **44 min** |
| Pesimista | 2.2 | 0.26 | 2.6 | **62 min** |
| Central, con una sola GPU libre | 1.45 | 0.18 | 3.6 | 83 min |

La celda usa por defecto el escenario pesimista (`V5_PISO_CPU_S = 2.2`, `V5_GPU_MODELO_S = 0.26`, `V5_FIJOS_S = 300`). Así nunca se mete en un hueco donde no cabe, a costa de omitirse en algunos donde sí cabría. Tras el ensayo del §3 se sustituyen por lo medido.

## 2. Recortes: menos folds por familia

Regla de recorte fija: las particiones 0..k−1 de cada familia (`V5_FOLDS_POR_FAMILIA`). Elegir los folds por su AUC en los 58 oficiales sería seleccionar sobre ruido. Las cuatro familias conservan el mismo peso, porque `inferir.py` promedia por igual.

| Modelos | Configuración | Optimista | **Central** | Pesimista | Comentario |
|---|---|---|---|---|---|
| 20 | 5 por familia | 31 | **44** | 62 | Lo que se envió con 0.918 |
| 16 | 4 por familia | 25 | **36** | 54 | Punto donde GPU ≈ CPU en el escenario central |
| 15 | sin `eff` (la familia con peor OOF, 0.8592) | 25 | **36** | 54 | Cambia la composición: deja de ser el v5 medido |
| 12 | 3 por familia | 25 | **36** | 54 | Limitado por CPU: no ahorra frente a 16 |
| 8 / 4 | 2 / 1 por familia | 25 | **36** | 54 | Ídem. Solo pierde diversidad |

(Minutos para 1,322 estudios con 2 T4, fijos incluidos [estimación].)

- Con 20 modelos y en el escenario central, la parte de GPU es la que manda. Por eso, bajar a 16 ahorra unos 8 min, y bajar más no ahorra nada. En el pesimista, el ahorro es de 8–9 min.
- Para bajar de ~36 min habría que abaratar el **preproceso** (p. ej. leer solo los cortes que se usan), no quitar modelos. Eso es cambiar el código de Owen y queda fuera de este encargo.
- Con una sola GPU disponible, la que manda es la GPU: 3 por familia tardaría max(1.45, 12 · 0.18) ≈ 2.2 s → 52 min, frente a 83 con 20.
- Ningún recorte tiene medido su efecto en AUC. 4 por familia es la opción con menos riesgo si el tiempo aprieta.

## 3. Medir el tiempo real en Kaggle sin gastar un envío

1. **Dato gratuito, si existe.** Owen ya corrió v5 sobre el test completo en Kaggle ("v5_cv20_eff_cnxhib_effhib_r34att - Version 2", 0.918; `code/registros/2026-10-13_envio_v5.json`) [hecho que lo registró]. Ese notebook usa una sola GPU (`inferir.py` no reparte) y todos los vCPU como workers. Si Kaggle le muestra la duración de la ejecución puntuada y el acelerador, eso ya mide el caso "1 GPU, 20 modelos" [no verificado que Kaggle exponga esa duración]. Pregúntaselo.
2. **Sesión interactiva** (la crea quien vaya a enviar, con el dataset de Owen ya compartido, §4):
   - Notebook nuevo. Acelerador **GPU T4 x2**, internet **apagado** (igual que en el rerun). Entradas: los datos de la competencia y el dataset v5.
   - Celda 1: `BRAZO_V5_SOLO_DEFINIR = True`. Celda 2: el contenido de `celda_brazo_v5.py`, que solo define funciones. Celda 3: `ensayo_tiempo_kaggle.py`.
   - El ensayo toma 320 estudios de **train** al azar (semilla 2026) y corre dos pasadas sobre estudios distintos, para que la caché de disco de la primera no abarate la segunda. La **pasada A**, con un solo modelo (r34att, partición 0), da el piso de CPU. La **pasada B**, con los 20, da el tiempo completo. De ahí sale g, con su diagnóstico: si B no supera a A en más de un 15 %, está limitada por CPU y g es solo una cota superior.
   - Imprime la proyección para 1,322 estudios con 5, 4, 3, 2 y 1 particiones por familia, y los valores de `V5_PISO_CPU_S`, `V5_GPU_MODELO_S` y `V5_FIJOS_S`, con un 20 % de holgura, para pegarlos en la celda del envío.
   - Coste: unos 120 · 1.45 + 200 · 1.8 s más dos veces los fijos ≈ **15–25 min de cuota de GPU**. Ningún envío.
   - Los estudios de train están en el entrenamiento de 16 de los 20 modelos. Las predicciones del ensayo solo sirven para medir tiempo y formato, **no AUC**.
3. **Prueba de integración** (también sin envío): "Save Version" del notebook 0.940 con la celda insertada. Corre sobre los 3 estudios del test público y comprueba lo siguiente:
   - Que la celda encuentra `T0`, `TIME_BUDGET`, la etapa y el paquete.
   - Que el log dice `[yuca-v5] applied` y `diagnostics/phase_events.jsonl` tiene `yuca_v5` con `ok: true`.
   - Que la celda 23 publica sin error.

   No mide el tiempo del rerun.
4. **Lo que de verdad falta: cuánto le queda al notebook al llegar a la celda 22 con 1,322 estudios.** Las salidas guardadas son de 3 estudios (341 s) y el log del rerun oculto no se ve. Hay dos maneras de saberlo:
   - **Cronometrar el notebook con estudios de train**, en una sesión interactiva. `find_root()` acepta `RSNA_COMP_ROOT` (celda 4). Se le puede dar una raíz falsa en `/kaggle/working` con `test.csv`, `test_series.csv` y `sample_submission.csv` de N estudios de train, más un enlace `test_series → train_series`, igual que hace la celda con sus fragmentos. Con N = 200 y la marca `elapsed_seconds` de cada `rsna_phase`, se separa el fijo del coste por estudio y se extrapola a 1,322. Coste aproximado: 200/1,322 de lo que tarde el notebook entero, más su fijo; probablemente 1–1.5 h de cuota. No verifiqué que `train_series.csv` tenga las mismas columnas que `test_series.csv` ni que ninguna fase lea algo exclusivo de test [no verificado].
   - **Un envío de diagnóstico** con `V5_EXIGIR = True`. Si el rerun termina con error de puntuación, el brazo no corrió (omitido o fallido); si puntúa, corrió. Cuesta un envío y da un solo bit, pero es la única confirmación directa sobre el test oculto. Antes conviene hacer la vía anterior.
   - Si al 0.940 le quedan menos de ~70 min de su presupuesto, las opciones son recortar folds (§2), que sirve de poco, o usar parte de la hora que hay entre `TIME_BUDGET` (8 h) y el límite de 9 h de Kaggle. Lo segundo exige subir `TIME_BUDGET`, con riesgo para todo el envío: es una decisión del equipo, no un valor por defecto.

## 4. Dataset de Kaggle y permisos

- **Qué se adjunta:** el dataset que Owen subió a partir de `envios/v5_cv20_eff_cnxhib_effhib_r34att_paquete.zip` (1,842,252,378 B) [hecho que el zip existe y que su registro lo cita]. Contiene `rsna/` (los tres .py revisados), `pesos/` (los 20 `.pt`, 1.92 GB) y `wheels/` (pydicom 3.0.2, pylibjpeg y los decodificadores libjpeg/openjpeg para cp311–cp313) [hecho, en la copia de Yuca]. Ni el nombre ni el slug en Kaggle están registrados en el clúster [no verificado]: hay que pedírselos a Owen. La celda lo localiza por contenido, así que el nombre da igual. Lo que importa es la **versión**: tiene que ser la que coincide con los hashes de la celda.
- **Permisos:**
  - Es un dataset **privado de Owen**. Quien abra la sesión interactiva y quien haga el envío necesita acceso de lectura. Tener el mismo equipo en la competencia no da acceso automático a los datasets privados de los compañeros [no verificado; así lo entiendo de Kaggle].
  - Opción recomendada: Owen añade la cuenta de Kaggle de jcastro como colaborador del dataset. Basta el rol de lectura ("Viewer") para adjuntarlo a un notebook.
  - Alternativa: el envío se hace desde la cuenta de Owen, con el notebook copiado a su cuenta, y no hace falta compartir nada.
  - No recomendado: volver a subir el paquete desde otra cuenta. Duplica artefactos de Owen, obliga a sacar 1.8 GB del clúster (requiere autorización expresa) y sigue necesitando su permiso.
  - **Autorización de Owen** para usar sus modelos en un envío del equipo. Es requisito del proyecto (lo de Owen no se toca sin que él lo autorice), además del permiso técnico.
  - **Mantenerlo privado.** Compartirlo dentro del equipo está permitido. Hacerlo público equivale a publicar modelos de la competencia y es otra decisión [inferencia sobre las reglas habituales de Kaggle; no las verifiqué].
- **Ajustes del notebook de envío:** acelerador GPU T4 x2, internet apagado, el dataset v5 adjuntado (Kaggle fija la versión al adjuntarlo y no la actualiza sola).

## 5. Supuestos, riesgos y lo no verificado

- **Encaje: resuelto leyendo el notebook** (§0). `T0`, `TIME_BUDGET`, la ruta de la etapa, el orden de columnas, el validador de la celda 23 y el patrón fail-soft salen del código, no de supuestos.
- **Tiempo restante en el rerun: desconocido.** Es el riesgo principal (§3.4).
- **Código sin ejecutar.** No se corrió Python ni ningún job en esta sesión. La sintaxis y la lógica de la mezcla no tienen prueba automática. Propuesta: un job corto de CPU que ejecute la celda contra una etapa y un test sintéticos, con `inferir.py` simulado y `rsna_frame` copiado del notebook.
- **VRAM.** En la versión guardada, el kernel reserva ≤ 63 MB por GPU al final y las CoAt corren en subprocesos que terminan, así que se espera llegar con las dos T4 libres [inferencia, con 3 estudios]. Si no fuera así, la celda descarta las GPU con menos de 6 GB libres y, con una sola, recalcula el tiempo con 1 GPU.
- **RAM.** Cada proceso carga 1.92 GB de pesos en CPU antes de pasarlos a la GPU, y cada worker tiene un tensor de 20 MB. Con 3 estudios, el cgroup iba por 15.2 de 30 GiB al final. Con 1,322 no está medido, y las cachés de píxeles del notebook van a disco (`rsna_array`) y se liberan en la celda 23, no antes.
- **Entorno.** El notebook no fija torch: usa el de la imagen de Kaggle, como el envío v5 de Owen. Los dos `inferir.py` corren con el mismo intérprete. El notebook pone `OMP_NUM_THREADS=4` en el entorno; los hijos de v5 lo bajan a 1.
- **Recursos de CPU durante el brazo.** El brazo corre cuando el notebook ya terminó todas sus fases, así que no compite con ellas por los 4 vCPU [hecho por el orden de las celdas].
- **Efecto en AUC de W = 0.2:** desconocido. Este encargo no lo evalúa. Medirlo cuesta un envío.
