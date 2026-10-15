# Auditoría de solo lectura: entrega de Owen con 0.918 público

**Fecha de corte:** 2026-10-13 20:01 MST (el estado de Owen cambió durante la auditoría: a las 19:56 registró la entrega; ver E).
**Alcance:** solo lectura sobre `/lustre/proyectos/p082/{code, runs, envios}`, `code_joako` y `runs_joako`. No se editó, copió, hasheó ni ejecutó nada de Owen. No hubo jobs, Python, red ni commits. La única escritura es este archivo.
**Etiquetas:** [hecho] = verificado en archivo o log citado; [inferencia] = deducido de hechos citados; [no verificado] = no se pudo comprobar con las fuentes autorizadas.
**Formato A–F:** las skills del proyecto no lo definen. Aquí A–F siguen los puntos 1–7 del encargo: A = 1, B = 2, C = 3, D = 4, E = 5 y 6, F = 7.

Abreviaturas de familias (las de Owen): `r34` ResNet-34 con tabla v1; `eff` EfficientNetV2-S con tabla v1; `cnxhib` ConvNeXt-Tiny con tabla híbrida; `effhib` EfficientNetV2-S con tabla híbrida; `r34att` ResNet-34 con tabla híbrida y atención por hallazgo.

---

## Resumen

- La entrega de 0.918 es **`v5_cv20_eff_cnxhib_effhib_r34att`**: 20 modelos (eff ×5, cnxhib ×5, effhib ×5, r34att ×5), `last.pt` de la época 8 y media simple de probabilidades, sin TTA ni calibración [hecho]. **No es nuestro v4**: v5 quita `r34` (v1) y pone `r34att` en su lugar. Comparten 15 de 20 modelos [hecho].
- La evidencia del 0.918 la escribió el propio Owen: un registro JSON, `EXPERIMENTOS.md`, el commit `1883220` y la corrida de MLflow que creó el job 2084935, todo a las 19:56 [hecho]. Kaggle no se consultó (sin red) [no verificado].
- En OOF (4,349 estudios, tabla v1, mismos UIDs), v5 frente a v4 da Δ +0.0003, IC95 [−0.0005, +0.0011], P = 74.1 %: **inconcluso**. v5 frente a v3: Δ +0.0032, IC95 [+0.0019, +0.0045] [hecho, `runs_joako/2084846`].
- Los 58 Gold no entran al entrenamiento de ningún miembro [hecho en el código actual, inferencia para commits anteriores]. **Nuestros 415 de `val_int` sí**: 16 de los 20 modelos de v5 entrenaron con ellos [hecho].

---

## A. Identificación de la entrega y evidencia del 0.918

| Elemento | Valor | Fuente | Etiqueta |
|---|---|---|---|
| Carpeta del paquete | `/lustre/proyectos/p082/envios/v5_cv20_eff_cnxhib_effhib_r34att/` | `ls envios/` | [hecho] |
| Zip subido como dataset | `envios/v5_cv20_eff_cnxhib_effhib_r34att_paquete.zip` (1,842,252,378 B, 15:32) | `ls`; `registros/2026-10-13_envio_v5.json:14` | [hecho] que existe y que el registro lo cita; [no verificado] que sea byte a byte lo que se subió a Kaggle |
| Armado | Job 2084817, 15:31:35–15:32:52, commit `ff92320` | `code/logs/rsna-armar-envio_2084817.out`; `envios/v5…/MANIFIESTO.txt:1`; `envios/v5…/commit.txt` | [hecho] |
| Notebook de Kaggle | "v5_cv20_eff_cnxhib_effhib_r34att - Version 2" | `registros/2026-10-13_envio_v5.json:24` | [hecho] que está registrado; [no verificado] su contenido en Kaggle |
| Notebook local | `envios/v5…/inferencia.ipynb`, idéntico (`cmp`) a los de v3, v4 y `code/kaggle/inferencia.ipynb` | `cmp` | [hecho] |
| Archivo de submission | Lo genera el notebook en `/kaggle/working/submission.csv`. En el clúster no hay submission de test, solo ensayos sobre estudios de **train** (`runs/2084790`, `runs/2084831`) | notebook, celda 4; logs de inferencia | [hecho] |
| Puntuación 0.918 | `"kaggle_publica": 0.918`, `"kaggle.fecha": "2026-10-13"` | `registros/2026-10-13_envio_v5.json:5,23` | [hecho] que Owen la registró; [no verificado] en Kaggle |
| | Fila de la tabla de envíos con 0.918, OOF 0.8759 y oficial 0.901 | `code/EXPERIMENTOS.md:27` (archivo de 19:56:27, cabecera "20:00") | [hecho] |
| | Commit `1883220`, "Envío v5 de 20 modelos: 0.918 en Kaggle; registro y bitácora", 19:56:27 | `code/.git/logs/HEAD` (última línea); `code/.git/COMMIT_EDITMSG` | [hecho] |
| | MLflow: corrida `envio-owen-v5-cv20` creada con 4 métricas y 6 tags | `code/logs/rsna-registrar-lote_2084935.out:2`; sacct: 2084935, 19:56:28–19:56:38, COMPLETED | [hecho] que se escribió; [no verificado] leída en `mlflow.db` (ruta no autorizada) |
| Hora del envío a Kaggle | Entre 15:59:54 (fin del ensayo 2084831, citado como ensayo en el registro) y 19:56:27 | registro:16; log 2084831 | [inferencia] |

Nota: a las 19:53, al empezar, nada de esto existía todavía (`EXPERIMENTOS.md` de 15:09 decía "mejor 0.912"). El registro apareció durante la auditoría.

---

## B. Composición

**Miembros** (`envios/v5…/MANIFIESTO.txt:3-22`, `code/logs/rsna-inferir-v5-cpu_2084831.out:3-22`) [hecho]:

| Familia | Jobs (partición 0→4) | Checkpoint | AUC oficial (58) por modelo, del log de inferencia |
|---|---|---|---|
| eff (v1) | 2084365, 2084585, 2084586, 2084626, 2084584 | `last.pt`, época 8 | 0.8600, 0.8751, 0.8671, 0.8604, 0.8675 |
| cnxhib | 2084671, 2084673, 2084676, 2084684, 2084633 | `last.pt`, época 8 | 0.8768, 0.8773, 0.8709, 0.8616, 0.8782 |
| effhib | 2084674, 2084683, 2084689, 2084694, 2084634 | `last.pt`, época 8 | 0.8844, 0.8666, 0.8769, 0.8772, 0.8781 |
| r34att | 2084756, 2084769, 2084771, 2084783, 2084722 | `last.pt`, época 8 | 0.8622, 0.8534, 0.8631, 0.8718, 0.8581 |

- **Ruta de origen:** `/lustre/proyectos/p082/runs/<job>/last.pt`. `armar_envio.slurm:15,40-42` copia `last.pt` "sin elegir con los 58 oficiales" [hecho]. Los 20 archivos de `pesos/` miden lo mismo que su `last.pt` de origen (92,755,652, 118,054,918 y 89,768,328 B), mientras que los `best_*.pt` miden más (92,803,932, 118,068,302 y 89,782,144 B) [hecho, `stat`]. El log de inferencia muestra `época=8` en los 20 [hecho].
- **Hashes:** el MANIFIESTO guarda los primeros 16 caracteres del sha256 de cada `last.pt` (`armar_envio.slurm:43`). No calculé hashes: la identificación ya es inequívoca. Hay un control cruzado: nuestro journal registró para `runs/2084365/last.pt` el sha256 `124514845f595e81d200…` (`experiments/journal.md:359`), que coincide con el MANIFIESTO [hecho].
- **Mezcla:** media simple de sigmoides de los 20 modelos, sin pesos (`envios/v5…/rsna/inferir.py:91-94`; MANIFIESTO:2) [hecho]. Los pesos por grupo y por hallazgo que ajustó Owen en el job 2084811, dentro de muestra, **no** se usan [hecho].
- **TTA:** ninguna. Un solo pase con recorte central de 288 sobre tensores de 320 (`inferir.py:44-45`) y `prepare(..., train=False)` [hecho].
- **Calibración:** ninguna [hecho].
- **Postproceso:** estudio ilegible o sin series → 0.5 en todo; luego `nan_to_num(0.5)` y `clip[0,1]` (`inferir.py:99-102`) [hecho].
- **Código de inferencia:** `rsna/{inferir,modelo,preproc}.py` del commit `ff92320`. Frente a v3, admite 6 ranuras, atención por hallazgo (`per_label`, que necesita r34att) y GRU/mediamax. Los valores por defecto reproducen v3 (`diff` v3→v4→v5) [hecho]. Ningún miembro usa GRU ni mediamax: r34att se entrenó con `6cbef75`/`2440227`, antes de que existiera esa opción, y su cabecera de log no tiene `contexto=` [inferencia].

---

## C. Entrenamiento de cada familia

Común a las 25 corridas: `rsna/entrenar.py`, lanzado con `jobs/entrenar.slurm` (array `--folds 5`); tensores `preproc/v1_train` (4 series a 320); 16 cortes, recorte 288, 8 épocas, AMP, `aug=suave` y sin mixup, consistencia ni quitar serie; AdamW con wd 1e-2, medio epoch de calentamiento y luego coseno (`entrenar.py:147-151`); partición `md5(UID) % 5` sobre los estudios no oficiales (`entrenar.py:126-129`). Cabeceras de log en `code/logs/rsna-cv-*_<job>.out:1-4` [hecho]. Seed 0: ninguna línea de `opciones=` pasa `--seed` y el valor por defecto actual es 0 (`entrenar.py:99`). Para commits anteriores es [inferencia].

| Familia | Etiquetas | Red / pesos de partida | Lote / lr | Commits | Duración por partición | OOF (v1) / oficial |
|---|---|---|---|---|---|---|
| eff | `preproc/etiquetas_v1_2079203.parquet` | efficientnet_v2_s ImageNet (`modelos/torch/efficientnet_v2_s-dd5fe13b.pth`) | 4 / 1e-4 | 4a45a34 (f0–2), cd95870 (f3), 2cc4dca (f4) | ~2 h 17 | 0.8592 / 0.8858 |
| cnxhib | `runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet` | convnext_tiny ImageNet | 8 / 2e-4 (por defecto) | 42fae0d | ~2 h 43 | 0.8610 / 0.8871 |
| effhib | híbrida (misma ruta) | efficientnet_v2_s ImageNet | 4 / 1e-4 | 42fae0d (f0–3), 3cc2d71 (f4) | ~2 h 15 | 0.8644 / 0.8928 |
| r34att | híbrida | resnet34 ImageNet, `--atencion hallazgo` | 8 / 2e-4 | 6cbef75 (f0–3), 2440227 (f4) | ~29 min | 0.8502 / 0.8765 |
| *(r34, sale en v5)* | v1 | resnet34 ImageNet | 8 / 2e-4 | 505cfc5, 4445ec9 | ~29 min | 0.8413 / 0.8600 |

- Las cifras OOF y oficial salen de `code/logs/rsna-evaluar-oof_2084813.out` (época 8, tabla v1) [hecho]. Contra la tabla híbrida (`…_2084812.out`): eff 0.8614, cnxhib 0.8711, effhib 0.8736, r34att 0.8578 [hecho].
- **Ni destiladas ni pseudoetiquetas:** solo v1 e híbrida de Cons. Owen deja sin entrenar la tabla corregida/rellenada (`EXPERIMENTOS.md:44`) y sin probar el pseudoetiquetado de test (`EXPERIMENTOS.md:122`) [hecho].
- **Varios commits en una misma familia** (eff, effhib, r34att): por los mensajes del reflog, los commits intermedios tocan evaluación, registro u opciones apagadas por defecto. Las cabeceras confirman que esas opciones estaban apagadas [hecho]. Que el código de entrenamiento fuera equivalente es [inferencia]: el diff no se pudo ver (F.1).
- r34att es la implementación de Owen de nuestra idea del modelo D. El comentario de `modelo.py` dice "idea de Joako" (`envios/v4…/rsna/modelo.py`, diff v3→v4) y `EXPERIMENTOS.md:74` lo atribuye igual [hecho].

---

## D. Datos: Gold, `val_int` y riesgo de leakage

1. **Los 58 Gold no entran al entrenamiento.** El código actual separa `official = table[table.oficial]` y entrena solo con `rest[fold != k]` (`entrenar.py:126-131`) [hecho]. En las 25 corridas, las particiones internas (891 + 897 + 847 + 835 + 879) suman exactamente 4,349 = 4,407 − 58, y el log informa "validación oficial=58" por separado [hecho]. Que los commits usados (505cfc5…2440227) tuvieran la misma exclusión es [inferencia], muy probable por la consistencia de los conteos. Matices: los Gold calibraron la tabla v1 (`references/contexto_actual_rsna_knee.md`, sección 2) y Owen usó la métrica oficial (0.9009 frente a 0.8985) como uno de los criterios para preferir v5 sobre v4. La cifra "oficial" de v5 es, por tanto, optimista [inferencia].
2. **Nuestro `val_int` (415) sí entró al entrenamiento de v5.** Los 415 UIDs (`runs_joako/2081432/predicciones_val_int.csv`) cumplen md5 % 10 = 0, calculado con `md5sum` en shell, y por tanto md5 % 5 = 0: todos están en la **partición 0** de Owen [hecho]. Los modelos de la partición 0 (2084365, 2084671, 2084674, 2084756) no los vieron. Los de las particiones 1–4, **16 de los 20 miembros de v5**, sí entrenaron con ellos [hecho, por construcción en `entrenar.py:129`]. Consecuencia: evaluar el paquete v5 completo en `val_int` está contaminado. Las únicas evaluaciones limpias son el OOF (un modelo por familia) o `val_int` con solo los miembros de la partición 0 [inferencia].
3. **Nuestros análisis previos no están contaminados, pero sí mal rotulados.** El "Owen 15 modelos en `val_int` = 0.8630" (`journal.md:412`) es en realidad la media OOF de 3 familias, una predicción por familia y estudio (`rsna/evaluar_fusion_offline.py:218-222`), no el ensamble de 15 [hecho]. Es limpio, porque viene de los modelos de la partición 0, pero no mide lo que se despliega.
4. **Stacking:** v5 no apila: es una media fija [hecho]. Los pesos de 2084811 (0.8861 por hallazgo) están ajustados dentro de muestra y no se usaron [hecho]. Nuestro stacking (2084840) ya declaró su sesgo y quedó refutado (`journal.md:480-491`).
5. **Test:** el pipeline no entrena ni pseudoetiqueta sobre test (`inferir.py` completo) [hecho]. No hay vía de leakage hacia test en el código revisado [inferencia].
6. **Métrica OOF híbrida** (0.8856): mide modelos entrenados con la tabla híbrida contra esa misma tabla, así que favorece a cnxhib, effhib y r34att. No es comparable con la cifra v1 [inferencia].

---

## E. Línea de tiempo v3 (0.912) → v5 (0.918) y comparación con nuestro v4

**Línea de tiempo** (reflog `code/.git/logs/HEAD`; sacct de una sola consulta; logs) [hecho salvo lo indicado]:

| Hora (13 oct) | Evento |
|---|---|
| 04:23–13:25 | effhib, particiones 0–4 (2084674…2084634) |
| 11:10:35 | commit `d16ebea` → **v3** armado por el job 2084711 (11:10:37) |
| 11:11–11:24 | commits `f1805f3`, `b2b91e2` (series según tensores, acumulación), **`a9043d6` (11:23, `--atencion hallazgo`)**, `3f41fce`, `6cbef75` (6 series) |
| 11:23–11:26 | 2084712: ensayo de inferencia de v3 |
| 11:26–14:00 | v3 enviado a Kaggle → 0.912 ("Version 2") [la hora es inferencia] |
| 13:07–14:34 | r34att, particiones 0–4 (array 2084722) |
| 13:59–14:00 | 2084786/2084787 evaluar-oof; commit `2440227` (registra 0.912/0.905/0.904 en MLflow, job 2084788); **v4** armado por 2084789 |
| 15:07 | 2084799/2084800 evaluar-oof (resumen r34att) |
| 15:09:34 | commit `f1da3b2` (bitácora: "v4 preparado y sin enviar") |
| 15:30 | commit `817dc57`; 2084811 (pesos), 2084812 (OOF híbrida), 2084813 (OOF v1): v5 queda en 0.8759 / 0.8856 / 0.9009 |
| 15:31:34 | commit `ff92320` (GRU/mediamax, opciones apagadas) → **v5** armado por 2084817 (15:31:35) |
| 15:51–15:55 | 2084790: ensayo de v4 en GPU, 160 estudios de train, OK |
| 15:58–15:59 | 2084831: ensayo de v5 en **CPU con 6 estudios** de train, OK |
| — | 2084818 (`rsna-inferir-v5`, GPU): **PENDING** al consultar sacct, nunca corrió [hecho] |
| 16:00–19:56 | v5 enviado a Kaggle → 0.918 ("Version 2") [inferencia] |
| 19:56:27 | commit `1883220` + `registros/2026-10-13_envio_v5.json` + `EXPERIMENTOS.md`; 2084935 lo registra en MLflow |

**Qué cambió exactamente de v3 a v5:**
- Miembros: −r34 (v1, ×5); +effhib (×5); +r34att (×5). Pasa de 15 a 20 modelos [hecho].
- Código de inferencia: solo compatibilidad hacia atrás. El notebook no cambió [hecho].
- No cambiaron ni el preprocesado (`preproc/v1_train`) ni la mezcla, la TTA o la calibración [hecho].
- Coste de GPU de los miembros nuevos: ≈ 11.3 GPU-h de effhib + ≈ 2.4 de r34att [inferencia, por las duraciones de sacct y los logs].
- Kaggle: +0.006. Owen lo había anticipado con OOF +0.003 y oficial +0.008 (`EXPERIMENTOS.md:33`) [hecho que lo escribió].

**Comparación con nuestro v4** (v3 + effhib = `envios/v4_cv20_r34_eff_cnxhib_effhib`; `runs_joako/2084846/comparacion_paquetes_oof.json`, OOF 4,349, tabla v1, 1,000 réplicas pareadas, semilla 20261013) [hecho]:

| | Composición | OOF v1 | OOF híbrida (Owen, solo el punto) | Oficial 58 |
|---|---|---|---|---|
| v3 (0.912) | r34 + eff + cnxhib | 0.8726 | 0.8802 | 0.8928 |
| **v4 (nuestro candidato)** | r34 + eff + cnxhib + effhib | 0.8756 | 0.8843 | 0.8985 |
| **v5 (0.918)** | eff + cnxhib + effhib + r34att | 0.8759 | 0.8856 | 0.9009 |

- v4 frente a v3: Δ +0.0030, IC95 [+0.0021, +0.0039], P = 100 %.
- v5 frente a v3: Δ +0.0032, IC95 [+0.0019, +0.0045], P = 100 %; el peor hallazgo es Synovitis, −0.0115.
- v5 frente a v4: Δ +0.0003, IC95 [−0.0005, +0.0011], P = 74.1 %: **inconcluso**. El peor hallazgo es Synovitis, −0.0062.
- **Veredicto:** v4 no es la misma composición que v5, pero comparte 15 de 20 modelos y no se distingue de v5 en OOF [hecho]. Como envío, v4 solo mediría el cambio r34 → r34att en el público, que según el propio Owen está por debajo del ruido de ±0.001 (`EXPERIMENTOS.md:10`). En la práctica es redundante [inferencia]. v4 nunca se envió, así que su puntuación en Kaggle es desconocida y "v5 supera a v4" en Kaggle es [no verificado]. Owen tomó esa decisión con métricas locales (`EXPERIMENTOS.md:31`).

---

## F. Inconsistencias, huecos y lo no verificado

1. **Git bloqueado:** `git --no-optional-locks -C /lustre/proyectos/p082/code log` falla con `detected dubious ownership`. No toqué `safe.directory` (reglas, sección 4). Lo reconstruí con el reflog en texto plano (`.git/logs/HEAD`), `COMMIT_EDITMSG`, `refs/heads/owen` y las copias de código de cada paquete (`diff`). **No pude ver** los diffs de `entrenar.py` entre los commits de entrenamiento [no verificado].
2. **MLflow no leído:** `/lustre/proyectos/p082/mlflow/` no está en las rutas autorizadas. Lo que se escribió se deduce del JSON de entrada y del log de 2084935. El mtime de las 19:50 es compatible con los entrenamientos en curso, que registran en MLflow (`jobs/entrenar.slurm:53`) [inferencia].
3. **El 0.918 está autorreportado:** sin red no se contrastó con Kaggle ni se comprobó que el dataset y el notebook "Version 2" correspondan a este zip [no verificado].
4. **v5 se envió sin ensayo en GPU:** 2084818 seguía PENDING. El registro cita un ensayo en CPU con 6 estudios, y el "Pendiente 1" de `EXPERIMENTOS.md:117` pedía enviar tras el ensayo [hecho]. Que el notebook terminara en Kaggle se deduce de que hay puntuación [inferencia].
5. **`EXPERIMENTOS.md` desactualizado en parte:** la sección "En cola o corriendo (13 de octubre, 15:15)" (`EXPERIMENTOS.md:102-113`) y "Pendiente 1" no se actualizaron, aunque la cabecera dice 20:00, y la línea 29 ("Paquetes:") no menciona v5 [hecho].
6. **Nuestro journal desactualizado o mal rotulado:** `journal.md:515` dice que no hay corrida de MLflow para v5 (ya la hay). `journal.md:409-428` (entrada 13:45) llama "Owen 15 Modelos" a una media OOF de 3 familias (D.3) [hecho]. *(Erratum 20:19: la versión inicial citaba `412,434`. La entrada 13:30 usa "15 modelos" correctamente para el paquete v3. Las notas de corrección ya están en el journal.)*
7. **Tabla híbrida:** `runs_cons/` no está autorizada. No verifiqué su construcción ("etiquetas públicas de otros participantes", `EXPERIMENTOS.md:43`), si los Gold influyeron en ella, ni si las reglas de la competencia permiten esos datos externos [no verificado].
8. **Falsos positivos del guardarraíl** (sin reformular para esquivarlos): (a) un `sed 's/<[^>]*>//'` sobre el reflog se leyó como redirección, y el archivo se leyó con la herramienta de lectura; (b) `cd /lustre/proyectos/p082; grep … 2>/dev/null` se bloqueó por "escritura tras cd", y se repitió como `grep` con rutas absolutas y sin redirección.
9. **Selección v5 frente a v4 dentro del ruido:** OOF v1 inconcluso (E), OOF híbrida sin IC y oficial con margen de ±0.03 (`EXPERIMENTOS.md:11`). El +0.006 en Kaggle es una sola observación del público.

**Operaciones realizadas:** `ls`, `stat`, `cat`, `head`, `tail`, `grep`, `find -maxdepth 2` en `runs/`, `cmp`, `diff`, `unzip -l` (solo lista), `md5sum` de nuestros 415 UIDs y **una** consulta `sacct -u owen -S 2026-10-13T11:00 -X`. Ningún hash de checkpoints, ningún git funcional, ningún sqlite, nada de Python, ningún job, ninguna escritura fuera de este archivo.

---

## Implicaciones para nuestra estrategia

- **La referencia ahora es v5 (0.918, OOF v1 0.8759).** v4 queda superado y es redundante como envío: no gastar en él uno de los 5 envíos diarios.
- **Nuestro diferenciador principal ya está en v5.** La atención por hallazgo (modelo D) entra como r34att en 5 particiones, así que la fusión C5 (Owen v3 + D + cnxhib de Joako, `journal.md:424`) y los ensambles Trío/ABCD pierden su ventaja. Hay que volver a medir cualquier aporte nuestro **contra v5**, no contra v3.
- **Protocolo de evaluación limpio obligatorio.** Para fusionar con v5, usar el OOF de 4,349 o `val_int` con solo los miembros de la partición 0 de cada familia. El paquete completo vio `val_int` en 16 de 20 modelos.
- **No duplicar la cola de Owen:** eff-att-hib (2084801) y cnx-att-hib (2084802) están corriendo; 6s288-hib (2084819), la prueba GRU (2084814) y el ensayo de v5 (2084818) están pendientes. Son su probable v6. Coordinar antes de entrenar familias equivalentes (misma lección que la cancelación de 2084782).

---

## Entrada propuesta para `experiments/journal.md` (no escrita)

```markdown
### [2026-10-13 20:05] — Auditoría de solo lectura: entrega 0.918 de Owen (v5)
- **Estado**: COMPLETED
- **Job SLURM**: N/A (auditoría de solo lectura; una consulta sacct; sin jobs)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `8b21ebb` + cambios sin versionar (informe en `experiments/reportes/auditoria_owen_0918.md`)
- **Hipótesis**: el 0.918 corresponde a un paquete existente en `envios/` y podría coincidir con nuestro v4 (v3 + effhib).
- **Resultados Observados**:
  - Entrega = `envios/v5_cv20_eff_cnxhib_effhib_r34att` (20 modelos: eff, cnxhib, effhib y r34att ×5; `last.pt` época 8; media simple; sin TTA ni calibración). Evidencia: `code/registros/2026-10-13_envio_v5.json`, `EXPERIMENTOS.md:27`, commit `1883220` y MLflow vía job 2084935 (19:56). Kaggle no verificado (sin red).
  - No es v4: v5 cambia r34 (v1) por r34att (híbrida, atención por hallazgo); comparten 15/20 modelos.
  - OOF 4,349 (tabla v1, job 2084846): v5 vs v4 Δ +0.0003, IC95 [−0.0005, +0.0011], P = 74.1 % (inconcluso); v5 vs v3 Δ +0.0032, IC95 [+0.0019, +0.0045].
  - Gold fuera del entrenamiento; los 415 de `val_int` caen en la partición 0 de Owen (md5 % 10 = 0): 16/20 miembros de v5 entrenaron con ellos.
- **Lección Aprendida**: v5 ya incorpora la atención por hallazgo en 5 particiones; v4 es redundante y cualquier aporte nuestro debe medirse contra v5 con OOF o con los miembros de la partición 0.
- **Siguiente Paso**: actualizar la referencia a v5 (0.918 / OOF 0.8759); corregir el rótulo "Owen 15 modelos" de la entrada 13:45 (en 13:30 basta una aclaración); no enviar v4.
```
