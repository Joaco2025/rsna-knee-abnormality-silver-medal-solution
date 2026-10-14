# Guía de envío: notebook 0.957 + brazo v5 (variante diagnóstica)

**Archivo a importar:** [kaggle/notebooks/0957_mas_v5_diag.ipynb](../../kaggle/notebooks/0957_mas_v5_diag.ipynb)
**Qué es:** el notebook `referencias/0-957-latest-dataset-score-in-one-click.ipynb` con **una celda nueva al final** (la 8.ª, índice 7), después de la celda que escribe `submission.csv`. Las celdas 0 a 6 son idénticas byte a byte al original.
**Qué hace la celda nueva:** mezcla por rangos el brazo v5 sobre `/kaggle/working/submission.csv`, con las familias `eff`, `effhib` y `r34att` (15 modelos, **sin `cnxhib`**) y un peso de **0.10** en los 12 hallazgos. Define ella misma `TIME_BUDGET = 8 h` y `ROOT = DATA`.
**Por qué «diagnóstica»:** empieza con `BRAZO_V5_EXIGIR = True`. Si el brazo se omite o falla, la celda lanza una excepción y el rerun termina en error. Con eso un solo envío responde a «¿cupo el brazo en el tiempo?», porque el log del rerun oculto no se ve.

## 1. Antes de empezar (permisos)

- El paquete v5 es un dataset **privado de Owen** (`v5_cv20_eff_cnxhib_effhib_r34att_paquete.zip`, 1.84 GB). Necesitas acceso de lectura y su autorización para usarlo en un envío del equipo. Detalle en `brazo_v5_plan.md`, §4.
- La alternativa que descarta el plan es subirlo de nuevo desde otra cuenta; no se recomienda.

## 2. Importar en Kaggle

1. En Kaggle: **Create → New Notebook**.
2. En el editor: **File → Import Notebook** y elige `0957_mas_v5_diag.ipynb` (súbelo desde tu equipo; el archivo está en el clúster, así que primero hay que bajarlo, y eso es una transferencia fuera de Yuca que te toca autorizar a ti).
3. Comprueba que aparecen **8 celdas**: 2 de texto, 5 de código del original y la celda nueva que empieza por `# VARIANTE DIAGNÓSTICA`.

## 3. Ajustes del notebook

- **Acelerador:** el mismo que use el 0.957 original. La celda 2 solo pide «a T4 GPU»; el controlador de la etapa 2 se llama `probe_dual` y la celda 3 del 0.940 usaba **2 × T4**. Si Kaggle te deja elegir, usa **GPU T4 ×2**: v5 reparte los estudios entre las GPU y con dos va el doble de rápido. No pude confirmar cuál usó el autor.
- **Internet:** apagado (igual que en el rerun).
- **Persistencia:** ninguna.

## 4. Datasets a adjuntar

Los mismos que usa el 0.957 original (**Add Input**), más el paquete v5. Los nombres salen del código de las celdas 2 y 4; no los verifiqué contra Kaggle ni comprobé que sigan visibles para tu cuenta.

| Dataset | Qué aporta | Seguro |
|---|---|---|
| `rsna-knee-abnormality-detection` (la competencia) | `test.csv`, `test_series.csv`, `test_series/`, `sample_submission.csv` | Sí |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (etapa 1) | Sí (el notebook lo busca primero) |
| `rsna-knee-oai-original384-s42-pair` | misma ruta alternativa para el checkpoint | Solo si falta el anterior |
| `rsna-knee-compact-r7cnxfsss` | `manifest.json` y pesos de los 4 pliegues del R7 | Sí |
| `rsna-orthofoundation-fold0-epoch7-20261004` | `receipt.json`, `cloud_code.zip`, `dependencies.zip`, y `anatomy.pt` si las vistas lo exigen | Sí |
| `rsna-knee-ofl-ssl-weights` | solo si el backbone del manifiesto fuera `ofl_l` (no parece) | Adjúntalo si el original lo traía |
| **Paquete v5 de Owen** | `rsna/` (3 `.py`), `pesos/` (los `.pt`), `wheels/` | Sí; la celda lo busca por contenido |

Notas:

- El notebook falla de forma explícita si falta un dataset obligatorio (`Cannot resolve ...`, `Attach the exact selected weights`). Eso aparece en el log de la ejecución visible, no en el rerun.
- Kaggle fija la versión de cada dataset al adjuntarlo y no la actualiza sola.
- La celda v5 comprueba el sha256 de `rsna/inferir.py`, `modelo.py` y `preproc.py` contra la copia revisada. Si el dataset de Owen tiene otro código, el brazo se omite (y, siendo diagnóstica, el rerun falla).

## 5. Antes de enviar: ejecución visible (sin gastar envío)

1. **Save Version → Save & Run All (Commit)**. Corre todo el notebook sobre el test público (unos pocos estudios) y deja el log a la vista.
2. En el log, busca estas líneas de la celda v5:
   - `[yuca-v5] applied: 15 modelos, W={...}` → el brazo corrió.
   - `[yuca-v5] skipped (...)` o `FAILED (...)` → no corrió, con el motivo. Como la celda es diagnóstica, además la ejecución termina con `RuntimeError: V5_EXIGIR`.
   - La línea `[yuca-v5] {...}` trae `restante_min`, `estimado_min`, `inferencia_min`, `gpus`, `modelos` y `spearman_medio`.
3. En **Output** de la versión guardada deberían estar `brazo_v5/estado.json`, `brazo_v5/v5_sub.csv` y `brazo_v5/submission_antes_v5.csv` (la copia de `submission.csv` antes de mezclar).
4. Imprime también `pip list | grep -i -E "torch|torchvision|opencv|pydicom|numpy|pandas"`. La celda 4 instala las ruedas fijadas de `dependencies.zip`, que desconozco, y v5 usa el `torch`/`torchvision` que haya.
5. Si el commit visible falla por otra razón (dataset que falta, etc.), arréglalo antes de gastar un envío.

Esta ejecución usa pocos estudios, así que **no dice si el brazo cabe en el tiempo del test completo**. Esa pregunta la contesta el envío diagnóstico.

## 6. Enviar y leer el resultado

1. Desde la versión guardada: **Submit to Competition**.
2. El rerun oculto corre sobre el test completo y dura horas. El resultado aparece en la pestaña de envíos:

| Lo que ves | Qué significa | Siguiente paso |
|---|---|---|
| **Una puntuación** | El brazo corrió: se aplicó con 15 modelos y W = 0.10 sobre el test completo. La puntuación es la del notebook con v5. | Compáralo con la puntuación pública del 0.957 sin v5. Para el envío definitivo, pasa a la celda normal (sección 7). |
| **Error de ejecución o de puntuación** | El brazo se omitió o falló, **o** falló otra cosa del notebook. El log oculto no dice cuál. | Contrasta con el commit visible de la sección 5: si allí `applied` salía bien, lo más probable es el tiempo (`restante < necesario + margen`). Mide con `ensayo_tiempo_kaggle.py` y reconsidera. |

El envío cuesta un cupo y da **un solo bit**. No distingue «no cupo» de «un dataset cambió». Por eso la ejecución visible va primero.

## 7. Después del diagnóstico

- Si puntuó, el envío definitivo usa la celda normal (`kaggle/brazo_v5/celda_brazo_v5.py`: sin la línea `BRAZO_V5_EXIGIR = True`). Así, si v5 no cabe, el notebook sigue y publica `submission.csv` como el 0.957.
- Si falló, no envíes la normal sin entender por qué: sin la excepción no te enteras de que v5 no corrió.
- Para cambiar familias o peso sin editar la celda, pon antes una variable (`BRAZO_V5_FAMILIAS = ("eff", "cnxhib", "effhib", "r34att")`) o edita `V5_W` en la celda.
- La mezcla es un peso fijo (0.10), **no se ajustó con etiquetas**. Es una apuesta de diversidad; no hay OOF del R7 reentrenado con el que medirla.

## 8. Qué se verificó y qué no

- **Verificado:** el test de CPU (`kaggle/brazo_v5/test_celda_brazo_v5.py`, etapa sintética de N = 7 y N = 1400, sin `rsna_frame`, con `DATA` en vez de `ROOT` y la etapa `submission.csv`): 335/335 comprobaciones PASS. El notebook generado difiere del original solo por la celda añadida (prefijo y sufijo idénticos byte a byte, celdas 0–6 y metadatos iguales).
- **No verificado:** que el brazo cabe en el tiempo del rerun real (el 0.957 no trae log de tiempos), que las versiones de `torch`/`torchvision` del entorno (tras las ruedas de la celda 4) sirven a v5, el acelerador que usa el original, y que los datasets sigan accesibles desde tu cuenta.
