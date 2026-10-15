# Bug del CoAtNet: por qué el piloto da 0.869 en Gold y no ~0.917

Fecha: 2026-10-15. Piloto: job **2085360** (`rsna/entrenar_coatnet.py`, commit `ffa42d8`; el sha256 de `inputs.txt` coincide con el archivo commiteado, así que **las líneas "nuestro" se refieren a `ffa42d8`**). Receta: [receta_coatnet.md](receta_coatnet.md).

Fuentes comparadas:

- **dd-train**: `referencias/rsna_research/notebooks/dreaddevelopment/knee-mri-training-the-twelve-finding-model/knee-mri-training-the-twelve-finding-model.py`.
- **dd-inf**: `…/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model/knee-mri-twelve-findings-from-a-single-model.py`, que muestra lo que su modelo «espera».
- **0.940/0.957**: la inferencia de Raptor portada en `rsna/raptor_preproc.py` (celda 22 del 0.940) y `ft0949.md` (cómo el 0.957 infiere raptor_0949).

Que la evaluación es correcta se da por establecido: raptor_0949 saca ~0.918 en Gold con ella (2085552/2085553). Además, en este análisis la variante A del diagnóstico de CPU (más abajo) reproduce en fp32 y en CPU el Gold del SWA del piloto: **0.8690**.

## Resumen

1. **El bucle de entrenamiento no tiene errores de mecánica.** Están bien el OneCycle por paso del optimizador, el orden de GradScaler con la acumulación, los grupos de lr, la carga de pesos (478/478 tensores idénticos), la entrada, las ventanas, las máscaras, `train()`/`eval()` y el SWA. Ver la tabla «Revisado, sin discrepancia».
2. **La discrepancia estructural más probable es BatchNorm con micro-lotes de 1 estudio.** El backbone tiene **25 BatchNorm** (stem y etapas 0-1). El piloto pasa un estudio por forward. dreaddevelopment pasa 8, es decir, 96 ventanas de 8 rodillas. Un diagnóstico de CPU (job 2088310) descarta que sea un desajuste en la **evaluación**: recalibrar BN no mueve el Gold (0.8675-0.8725 frente a 0.8690). Si es el bug, actúa durante el aprendizaje.
3. Las otras dos discrepancias fuertes con dreaddevelopment son:
   - **2.5× menos pasos de optimizador**: 3,456 de lote 16, frente a ~8,700 de lote 8.
   - **El corpus**: hasta 96 cortes densos a 384 px, frente a 44 cortes a 336 px con máscaras. Con 12 ventanas por paso se ve el 12.8 % de los centros, frente al 28.6 %.
4. Las etiquetas y la pérdida están sanas: no hay celdas NaN que entren como 0 y `w` medio ~0.94. No se pueden comparar con las de dreaddevelopment, que no son públicas.
5. Firma del piloto: aprendizaje **lento y aún creciendo** al final (Gold 0.68 → 0.87 de forma casi monótona; OOF v1 subiendo en la época 15). La norma del gradiente supera el recorte de 3.0 en casi todos los pasos: mediana 15.8 en la ép. 0 y de 3 a 8 después. No es un colapso. ft0949, el mismo bucle partiendo del 0.949, solo baja Gold de 0.918 a 0.914 en 5 épocas: el bucle no destruye un modelo ya bueno. El problema está en **aprender** desde ImageNet.

## Discrepancias, ordenadas por impacto probable

| # | Discrepancia | Nuestro (`ffa42d8`) | Referencia | Evidencia y por qué importa | Impacto probable |
|---|---|---|---|---|---|
| 1 | **BatchNorm con lote de 1 estudio** | `DataLoader(batch_size=1)` en `entrenar_coatnet.py:278-280`, con `un_estudio` que lo exige (`:90-93`); acumulación de 16 micro-lotes (`:350-365`) | dd-train: `batch_size=a.bs` = 8 (`:349`, `:449`), sin acumulación (`:511-519`); 8 × 12 = 96 ventanas por BN | `timm` construye 25 `BatchNormAct2d` en `stem` y `stages.0-1` (comprobado al instanciar). En train, cada BN normaliza con la media y la varianza de **una rodilla**, lo que borra el contraste entre estudios en las primeras capas. Las `running_*` son una EMA (momentum 0.1) de los ~10 últimos estudios: ruidosas y distintas de lo visto en train. En eval se usan esas `running_*`. El #53 dice usar lote 1 × 16, pero no hay forma de comprobar cómo trató BN. El diagnóstico de CPU (abajo) descarta el desajuste en eval; queda el efecto sobre el aprendizaje. | **Medio-alto** |
| 2 | **Menos pasos de optimizador** | `steps_ep = len(train) // accum` = 3,456 // 16 = **216** por época, **3,456** en total (`:266-267`); lote efectivo 16 | dd-train: `steps = len(tl) * epochs` (`:488`), lote 8 → 432 por época con 3,456 estudios (543 con 4,349) → **~6,900-8,700** | Con Adam el tamaño del paso apenas depende de la escala del gradiente, así que la mitad de pasos es casi la mitad de «recorrido». El piloto sigue mejorando en la ép. 15 (pérdida 0.595 → 0.589; OOF v1 0.8399 → 0.8410): firma de entrenamiento corto. Integral de lr del backbone: 4.01e-5 × 3,456 ≈ 0.14, frente a 3e-5 × 6,912 ≈ 0.21. | **Medio-alto** |
| 3 | **Corpus**: 96 cortes densos frente a 44 | `raptor_corpus.py:31` usa el brazo `native384dense-v10`: `OBJETIVO_CORTES = 96` (`raptor_preproc.py:33, 48`), banda 0.02-0.98, 384 px nativos, sin máscara (n_cortes = `vol.shape[0]`) | dd-train: corpus fijo 44 × 336 × 336 con máscaras (`:15-16`, `:106`, `:229-230`), reescalado bilineal a 384 (`:216-222`). Banda 0.06-0.94 (dd-inf `:319-320`; hilo 737696: «44 over 6 to 94» → 0.926 frente a 0.917 con 2-98) | Con 12 ventanas al azar sobre ~94 centros, cada bolsa de entrenamiento ve el 12.8 % del volumen, frente al 28.6 % sobre 42 centros. Un hallazgo focal (LCA, menisco, fractura en 2-3 cortes) falta de la bolsa mucho más a menudo, y la etiqueta positiva no tiene evidencia (ruido MIL). En contra: raptor_0949 (`d96_r384`) se entrenó con 96 densos y funciona, aunque con OAI y all-data. | **Medio** |
| 4 | Hiperparámetros del #53 frente a los de dd | lr 4.01e-5 / 1.95e-3, wd 0.0588 (`:206-208`) | dd-train: 3e-5 / 1e-3, wd 0.02 (`:350-352`) | El lr de la cabeza es 2× más alto con el doble de wd; el del backbone, 1.3×. Por sí solo no explica 0.05. Entra en el control. | Bajo-medio |
| 5 | Etiquetas y pérdida | blendjev con máscara y `w` continuo (`:252-253`, `:348-349`); relleno de Baker's (t 0.05, w 0.5, 1,758 estudios); `pos_weight` con celdas `w > 0` (`:260-263`) | dd-train: etiquetas blandas densas propias (no públicas), `BCEWithLogitsLoss(pos_weight)` media (`:491`); prevalencia media simple (`:425-426`) | Revisado: 0 celdas con `t` NaN y `w > 0` (las 3,489 NaN tienen `w = 0`, así que `nan_to_num` no mete ceros con peso); `w` medio 0.90-0.97 salvo Synovitis (0.56) y Baker's (0.78); acuerdo `t ≥ 0.5` blendjev/v1 = 0.884. La fórmula Σ(w·BCE)/(12·B) equivale a la media de dd con `w = 1`. **No se puede descartar** que sus etiquetas sean mejores, pero con las públicas no se puede medir. | Desconocido; no testeable |
| 6 | Precisión | fp16 autocast + GradScaler (`:344-345`, `:296`) | dd-train: bf16 sin escalador (`:514`) | Secuencia correcta (abajo). 2.3 % de pasos saltados en la ép. 0 y 0 % después; escala 2,048-4,096; 0 logits no finitos. Forzado por la MI210. | Bajo |
| 7 | Normalización | ImageNet (`:50-51`, `:96-102`) | dd-train: default `--norm none` (`:184`, `:360`), pero dd-inf usa `NORM = "imagenet"` («must match training exactly», `:100`), igual que el #53 y el 0.940. timm declara mean/std 0.5 para estos pesos | Lo que el modelo público «espera» es ImageNet; el piloto hace eso. Mezclar 0.5 e ImageNet es un desplazamiento de entrada que el ajuste fino absorbe. | Bajo |

## Revisado, sin discrepancia

| Punto pedido | Qué se comprobó | Líneas |
|---|---|---|
| OneCycle por paso del optimizador | `sched.step()` solo dentro de `if (i+1) % accum == 0` | `:354-365` |
| `total_steps` frente a la acumulación de 16 | `total_steps = steps_ep × epochs` con `steps_ep = len(train) // 16`; el bucle corta en `steps_ep × accum` micro-lotes. El log confirma lr_bb 1.60e-6 → pico 4.01e-5 en la ép. 2.4 → final | `:266-267`, `:340-341` |
| Aviso «`lr_scheduler.step()` antes de `optimizer.step()`» | Inocuo: el paso 1 tuvo gradiente infinito (`|g| inf` en el log) y GradScaler lo saltó | log `.err` |
| Orden de GradScaler | `scale(loss/accum).backward()` en cada micro-lote → `unscale_` → `clip_grad_norm_(3.0)` → `step` → `update` → `zero_grad` → `sched.step`. Correcto. La pérdida se divide entre 16 | `:350`, `:355-365` |
| `zero_grad` | Al inicio de la época y tras cada paso | `:338`, `:364` |
| Grupos de parámetros | Cabeza = todo lo que no es `backbone.` (`norm`, `att.*`, `clsW`, `clsb`) con lr 1.95e-3; el log muestra lr_cab 7.8e-5 → 1.95e-3. wd 0.0588 en todo, también en normas y sesgos, **igual que dd-train** (`:484-487`) y el #53 | `:290-295` |
| Pesos de inicio | 478 de 478 tensores del backbone **idénticos** al safetensors tras `create_model` (solo se descarta `head.fc`). `pretrained_cfg`: mean/std 0.5, bicúbica, 384, crop 1.0; no se usa ninguna transformación de timm | `:136-140` |
| Entrada | uint8/255 → ganancia ±10 % → clamp [0, 1] → ImageNet, en el mismo orden que dd-train (`:239`, `:261-266`). Canales = cortes (c−1, c, c+1). Orden geométrico por IPP·normal (mismo lector que el 0.940). Train y eval usan el mismo corpus y la misma `preparar` | `:84-85`, `:96-102`, `:169` |
| Ventanas de train | 12 centros por estudio, sin reemplazo; la lista solo se repite si hay menos de 12 centros, como dd-train `_centers` (`:202-214`). No hay ranuras vacías: el corpus es denso, así que no hace falta máscara | `:56-65` |
| Ventanas de eval | 24 equiespaciadas (`linspace` redondeado), idénticas a `raptor_preproc.eval_windows` | `:66-67` |
| `train()` / `eval()` | `model.train()` al empezar cada época, `model.eval()` en `predecir` | `:335`, `:166` |
| Dropout y drop_path | Dropout 0.2 en la cabeza y 0.125 en el MLP de posición relativa de las etapas 2-3 (cfg de timm, igual en dd); drop_path 0 | — |
| SWA | Promedio en float64 de los `state_dict`, buffers de BN incluidos y `num_batches_tracked` del primero; igual que dd-train (`:610-622`) | `:419-423` |
| Semillas de los workers | PyTorch siembra `random` por worker; la ganancia y la elección de ventanas varían por época | — |

## Diagnóstico sin GPU de BatchNorm (job de CPU 2088310)

`rsna/diag_bn_eval.py` sobre el `swa.pt` del piloto, con los 58 Gold y 24 ventanas en fp32 y CPU, en cuatro variantes:

- **A**: eval normal.
- **B**: BN con la estadística del propio estudio, como en train.
- **C**: `running_*` recalibradas con media acumulada sobre 128 estudios de train, uno por lote.
- **D**: lo mismo con lotes de 8 estudios.

| Variante | Gold (58) |
|---|---|
| A: eval normal | **0.8690** (reproduce el piloto: la evaluación es correcta también en fp32 y en CPU) |
| B: BN por estudio (como en train) | 0.8725 |
| C: recalibrada, 1 estudio por lote | 0.8685 |
| D: recalibrada, 8 estudios por lote | 0.8675 |

Las cuatro variantes caen dentro de ±0.004, que es ruido con n = 58. **No hay desajuste de BN entre train y eval**: ni las estadísticas por estudio ni las recalibradas recuperan nada. Esto descarta un arreglo «gratis» en la inferencia. Si BN por estudio es el problema, actúa en la **dinámica de aprendizaje**: las primeras capas aprenden sin el contraste entre estudios. Eso solo lo mide D1. La sospecha 1 baja de «alto» a «medio-alto» y sigue primera porque es la única diferencia mecánica con dd que toca todas las capas convolucionales. Salida: `runs_joako/2088310/` (23 min de CPU).

## Fase 2: corridas de diagnóstico propuestas

Fold 0 (md5 % 5). 6 épocas con el OneCycle reescalado a 6, que sale solo porque `total_steps = steps_ep × 6`. Cada corrida cambia **una** cosa respecto al piloto. Métrica por época: Gold (58) y OOF fold 0 con el juez v1, en `historial.jsonl`. Todo pasó el humo en el login y `sbatch --test-only`.

| Job | Modo | Qué cambia (solo esto) | Sospecha | Comando |
|---|---|---|---|---|
| D1 | `bn8` | Micro-lote de **8 estudios** × acumulación 2: BN ve 96 ventanas de 8 rodillas. Mismo lote efectivo (16), mismos 216 pasos por época, misma pérdida. Necesita gradient checkpointing por memoria (matemáticamente igual, salvo que BN actualiza sus `running_*` dos veces por paso, como dd) | #1 | `sbatch -J diag-bn8 --export=ALL,MODO=bn8 jobs/diag_coatnet.slurm` |
| D2 | `accum8` | Acumulación **8** en vez de 16: lote efectivo 8, **432** pasos por época (el doble) | #2 | `… MODO=accum8 …` |
| D3 | `dd44` | Corpus **dd44** (`rsna/dd44_corpus.py`, port literal de `build_study` de dd-inf con 44 ranuras y banda 0.06-0.94), reescalado bilineal a 384 en la GPU; ventanas dentro del rango de la máscara. Todo lo demás, el piloto | #3 | `… MODO=dd44 …` |
| D4 | `control` | `rsna/entrenar_coatnet_dd.py`: el bucle de dd-train portado casi literal sobre dd44, con lote 8 sin acumulación, lr 3e-5 / 1e-3, wd 0.02 y BCE media con `pos_weight`. Desviaciones forzadas: fp16 + GradScaler, `w` como peso de la BCE, ImageNet como en dd-inf, pesos locales | control | `… MODO=control …` |
| (D0) | `base6` | **Nada**: el piloto con 6 épocas | referencia | `… MODO=base6 …` |

**Criterio fijado antes de lanzar (el de jcastro):** una corrida «encuentra el bug» si en su **6.ª época (índice 5)** supera al piloto en su 6.ª época por **≥ 0.02 en Gold y ≥ 0.01 en OOF v1**. Piloto, índice 5: Gold **0.8152**, OOF v1 **0.7942**. Umbrales: Gold **≥ 0.8352** y OOF v1 **≥ 0.8042**.

**Confusión del criterio:** con el OneCycle reescalado, la época 6 de cada diagnóstico termina con lr ≈ 0 (recocida). La época 6 del piloto está en lr casi máximo: es la 6.ª de 16. El recocido solo puede dar +0.02 de Gold sin arreglar nada, y todas las corridas «encontrarían el bug». D0 (`base6`) mide ese efecto; con él, el criterio se aplica contra D0 en vez de contra el historial del piloto. `qos082` limita p082 a **4 GPUs**, así que D0 sería el quinto job y empezaría cuando acabe uno de los cuatro (unas 3 h después). La alternativa sin quinto job es no reescalar: OneCycle de 16 épocas cortado tras la 6.ª, comparable 1:1 con el historial. Decide jcastro.

**Ruido:** con n = 58, cambiar de época mueve el Gold ±0.02-0.03 (`seeds_coatnet.md`); el OOF (891) es la medida fiable. La regla de `rsna-experiment-lifecycle` prohíbe elegir hiperparámetros con Gold: aquí el Gold solo acompaña al OOF, que es condición necesaria.

**Recursos por job:** partición `gpu`, 1 MI210, 16 CPUs, 64 GB, `--time` 6 h (presupuesto interno 5.5 h).

- Tiempo estimado: corpus 6-11 min; ~20 min por época de entrenamiento + 3 min de evaluación → ~2.5 h. D1 y D4 llevan checkpointing: ~3-3.5 h.
- `/scratch`: ~62 GB (native) o ~22 GB (dd44).
- Lustre: historial, predicciones por época y `last_fp16.pt` (~146 MB): **< 1 GB en total**. Cuota antes de lanzar: 548.7 GB de 700 GB.
- El código no está commiteado (regla de la sesión): cada job copia el código y su sha256 en `runs_joako/<job>/codigo/` e `inputs.txt`.

**Cambios de código para la fase 2.** Los valores por defecto de `entrenar_coatnet.py` siguen dando exactamente el piloto. Se añaden:

- `--micro-bs`, `--grad-ckpt`, `--corpus-tipo dd44` y `--pesos-fp16`.
- `centros` acepta un rango [lo, hi]. El recorte a [1, n−2] no cambia nada en native, porque los centros ya caían ahí.
- `preparar` admite (B, K, …) y reescala si el corpus no está a 384.

Archivos nuevos: `rsna/dd44_corpus.py`, `rsna/entrenar_coatnet_dd.py`, `rsna/diag_bn_eval.py`, `jobs/diag_coatnet.slurm` y `jobs/diag_bn_eval_cpu.slurm`.

## Cómo leer el resultado

- **D1 pasa y D2/D3 no:** BN por estudio es el bug. Hay que entrenar con micro-lotes de varios estudios, congelar BN o usar GroupNorm, y repetir el piloto de 16 épocas.
- **D4 pasa y ninguno de D1-D3:** el bug está en la combinación (lote 8 = BN + pasos) o en los hiperparámetros de dd. Siguiente ronda: D1 + D2 juntos.
- **D4 tampoco pasa:** con el bucle de dreaddevelopment y nuestros datos se queda igual, así que el hueco está en las **etiquetas**, en el tamaño de datos (fold de 3,456 frente a all-data de 4,349) o en que el ~0.917 de referencia no es reproducible. Siguiente paso: etiquetas, no código.
