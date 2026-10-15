# B. Semillas ALL-DATA del CoAtNet propio (prefijo `seeds_`)

Fecha: 2026-10-14. Código: [rsna/seeds_entrenar_coatnet.py](../../rsna/seeds_entrenar_coatnet.py) (copia del piloto), [rsna/seeds_gate.py](../../rsna/seeds_gate.py), [rsna/seeds_corpus_lado.py](../../rsna/seeds_corpus_lado.py) (variante opcional), [jobs/seeds_coatnet.slurm](../../jobs/seeds_coatnet.slurm).

Sin OAI: estas semillas valen para las dos finales (como el piloto).

## Qué se corre

Dos corridas all-data con semillas **7** y **2026**, receta idéntica al piloto 2085360 ([receta_coatnet.md](receta_coatnet.md)).

`rsna/seeds_entrenar_coatnet.py` es una copia de `rsna/entrenar_coatnet.py` (sin editar el original). Cambia solo esto:

| Cambio | Por qué |
|---|---|
| `--all-data`: entrena con los 4,349 no Gold; no hay fold fuera ni OOF | Pedido |
| Guarda `last_fp16.pt` y `swa_fp16.pt` (flotantes en fp16, sin optimizador, ~146 MB c/u) en lugar de los `.pt` en fp32 | Regla de disco de la sesión |
| `--smoke-pasos` y una prevalencia que no da NaN si un hallazgo no tiene celdas | Solo afecta a los smokes de login (1-2 estudios) |
| `--ref-resumen`, `--corpus-tipo` y el bloque `criterio_all_data` | Criterio sin OOF (abajo) y trazabilidad |
| Sigue admitiendo `--folds 5 --fold 0` | Para la medición pareada del lado unificado (abajo) |

Con all-data hay 4,349 // 16 = **271 pasos por época** y 4,336 en 16 épocas, el mismo recuento del #53. El piloto tenía 216 pasos por época.

## Gate (dentro del job)

1. Al arrancar, antes de construir el corpus, `rsna/seeds_gate.py runs_joako/2085360/resumen.json`.
2. **Abre** solo si `criterio_swa.veredicto == "PASA"` (el SWA es el checkpoint que decide en el piloto).
3. Si el archivo no existe, el JSON está roto, falta la clave o dice «NO PASA», el job escribe `runs_joako/<job>/gate.txt` con el motivo y **termina con código 0 en segundos, sin entrenar**.
4. Se encola con `--dependency=afterany:2085360`, así que arranca cuando el piloto termina, falle o no.
5. Probado en el login con 5 casos: PASA → abre; NO PASA, JSON roto, archivo ausente, resumen sin `criterio_swa` → cierra con código 3.

## Selección y su sesgo

- **Selección:** top-3 épocas por macro AUC en los 58 Gold y promedio uniforme de pesos (SWA), igual que el #53 y el piloto.
- **Sesgo:** el AUC Gold del SWA es el máximo de una selección sobre esos mismos 58 estudios, así que está **inflado**. Los Gold calibraron además las etiquetas. Con all-data no hay OOF, así que **no queda ninguna medida limpia** de estas corridas. Su valor real solo lo dirá la tabla pública (o un ensamble con ellas).
- La selección por Gold también puede sobreajustar ruido con n = 58: cambiar de época puede mover el Gold ±0.03 sin cambiar nada real.

## Criterio (fijado antes de lanzar)

- **Sano:** sin aborto, al menos una época con < 5 % de pasos saltados por GradScaler, sin logits ni predicciones no finitas (las mismas definiciones del piloto).
- **Coherente con el piloto:** Gold del SWA ≥ Gold del SWA del piloto − 0.02. Los dos tienen el mismo sesgo de selección; la tolerancia cubre que el piloto entrenó con un 20 % menos de datos y el ruido de n = 58.
- **PASA** = sano y coherente. Va en `resumen.json` → `criterio_all_data.veredicto`. Si el Gold es NaN o falta la referencia, «NO PASA».
- No hay criterio de mejora: sin OOF no se puede medir. Las semillas sirven para promediar en el envío (reducir varianza), no para decidir nada.

## Recursos

| Recurso | Por job (×2) |
|---|---|
| Partición, GPU | gpu, 1 MI210 |
| CPU, RAM | 16, 64 GB |
| `--time` | 12 h, presupuesto interno 11.3 h (no empieza una época que no quepa); `--deadline=2026-10-19T23:59` (si no puede terminar antes, SLURM no lo arranca) |
| Estimación | corpus ~5-10 min; ~25 min por época (52k ventanas a ~35 img/s con 12 ventanas por paso, bench v2) → ~7 h para 16 épocas |
| `/scratch` | ~62 GB (exige ≥ 90 GB libres) |
| Lustre | ~300 MB (dos pesos fp16, preds, logs) |
| Dependencia | `afterany:2085360` + gate |

Si el gate cierra, cada job ocupa la GPU unos segundos.

## Smokes (login, ≤ 2 estudios, ≤ 2 pasos)

- **All-data, semilla 7:** completo. 2 épocas, SWA, `last_fp16.pt` y `swa_fp16.pt`, `criterio_all_data` = «NO PASA» por Gold NaN con 1 estudio y sin referencia, como debe.
- **Fold 0 con el corpus de lado unificado:** el cableado del corpus funciona. Con 2 estudios no puede haber a la vez entrenamiento, OOF y Gold, y falla en `predecir` con 0 Gold (con 58 no ocurre; el piloto se comporta igual). La ruta fold 0 es la del piloto, ya probada en su smoke de CPU con 14 estudios. Mis cambios en esa ruta son solo guardas `if oof`. **No se ha probado la ruta fold 0 con OOF en esta sesión.**

## Variante opcional: lado unificado (encolada como 2085558, fold 0)

**Propuesta (si el piloto pasa):** una corrida con lado unificado. Owen midió +0.010 a +0.020 en sus CNN. Propongo correrla primero **en el fold 0 con la semilla 42** (`LADO=1 FOLD0=1`). Así su OOF se compara de forma pareada con el piloto: mismos estudios, misma semilla, solo cambia la orientación. Una semilla all-data con lado unificado no se puede medir: no tiene OOF. Si el fold 0 mejora, se pasa a all-data.

### Cómo se obtiene el lado

1. **Laterality del DICOM** (R/L) de las series usadas, si es unánime. Está en ~50 % de los estudios (2,203 de 4,407 según la tabla de Owen; 25 son «mixto»).
2. **Si falta o es mixta: posición de la serie sagital.** x = mediana de `ImagePositionPatient[0]` en los cortes primero, central y último. En LPS, +x es la izquierda del paciente, así que x > 0 → L.

Es la regla de Owen (`code/eda/09_lateralidad.py`, tabla `preproc/lados_train_2084968.parquet`): AUC 0.991 y 97.7 % de acierto con x > 0 frente a la etiqueta en 2,178 estudios (98.6 % con |x| > 20 mm). Él usa la posición sobre la normal (con el signo corregido, porque su primera tabla, la 2084964, tenía el signo al revés); yo uso la coordenada x directa, que no depende del signo de la normal. El job imprime el acuerdo con la tabla de Owen. En el smoke, 2 de 2 coinciden: una L declarada con x = +97 mm y una R estimada con x = −108 mm.

### Cómo se voltea (por qué no basta el lado)

El corpus Raptor **no reorienta los píxeles**: guarda la matriz tal como viene en el DICOM y ordena los cortes por la posición sobre n = fila × columna de `ImageOrientationPatient` (`rsna/raptor_preproc.py`, líneas 101-132). El preprocesado de Owen sí reorienta a coordenadas del paciente, y por eso a él le basta el lado. Aquí, por ranura:

- **Sagital:** se invierte el orden de los cortes de la ranura si (n_x > 0) XOR (lado = L). En la orientación estándar n_x < 0 (IOP (0,1,0,0,0,−1)), que coincide con la nota de Owen: «la normal sagital apunta a la derecha del paciente».
- **Coronal y axial:** se voltea el ancho si (r_x < 0) XOR (lado = L). En la orientación estándar las columnas crecen hacia +x.
- Las etiquetas **no se tocan**: una rodilla izquierda en espejo es anatómicamente una derecha, y lo medial sigue siendo medial. No es el «espejo con intercambio de etiquetas» que Owen probó sin éxito: no hay aumento, cada estudio tiene una sola vista.
- De paso, las series con orientación no estándar quedan en la estándar. El job cuenta cuántas hay y cuántas son oblicuas (|componente| < 0.5).

**Verificación:** con 2 estudios (una L declarada y una R estimada), deshacer los volteos registrados en `indice.csv` devuelve **exactamente** (bit a bit) el volumen de `rsna/raptor_corpus.py`. La rodilla R estándar no cambia. En la L se voltearon todas sus ranuras.

### Riesgos

- **Inferencia en Kaggle:** el modelo exigiría la misma canonicalización en test: leer `Laterality`, `ImagePositionPatient` e `ImageOrientationPatient` de las series y voltear igual. Son cabeceras baratas, pero es código nuevo en el notebook, y sin él el modelo vería rodillas izquierdas sin unificar.
- **Lado mal estimado:** ~1-2 % de los estudios sin Laterality (≈ 0.5-1 % del total) quedarían en el lado equivocado.
- **Transferencia:** el +0.010-0.020 de Owen es en CNN con 16 cortes por serie y atención entre cortes. CoAtNet con MIL sobre ventanas es otra arquitectura, y el TTA en espejo del 0.957 sugiere que este modelo ya ve rodillas de los dos lados sin problema. La ganancia no está garantizada.

Comando (lanzado como 2085558):

```bash
sbatch -J seeds-lado-f0 --dependency=afterany:2085360 --export=ALL,SEED=42,LADO=1,FOLD0=1 jobs/seeds_coatnet.slurm
```

## Estado

**Encolados el 2026-10-14 con el OK de jcastro** (cuota antes: 545.9 GB de 700 GB):

| Job | JOBID | Qué |
|---|---|---|
| `seeds-s7` | **2085556** | all-data, semilla 7 |
| `seeds-s2026` | **2085557** | all-data, semilla 2026 |
| `seeds-lado-f0` | **2085558** | lado unificado, fold 0, semilla 42 (`LADO=1 FOLD0=1`); su `criterio_swa` frente a effhib es comparable con el piloto |

Los tres con `--dependency=afterany:2085360` y gate. Comandos usados:

```bash
lfs quota -h -g p082 /lustre          # no lanzar si quedan < 20 GiB
sbatch -J seeds-s7    --dependency=afterany:2085360 --export=ALL,SEED=7    jobs/seeds_coatnet.slurm
sbatch -J seeds-s2026 --dependency=afterany:2085360 --export=ALL,SEED=2026 jobs/seeds_coatnet.slurm
```

## Resultados

(pendiente)
