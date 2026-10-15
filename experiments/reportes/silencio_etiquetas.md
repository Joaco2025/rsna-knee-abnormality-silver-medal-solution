# Silencio en las etiquetas: qué es NaN, cómo lo trata `entrenar.py` y qué predice en los 58 Gold

Fecha: 2026-10-14. Solo lectura (tablas, `rsna/entrenar.py`, `rsna/evaluar_fusion_offline.py`, `runs_cons/auditoria/generar_tabla_candidata.py`). Sin GPU, sin commits. Se ejecutó Python ligero en el login node con `venvs/rsna`.

Tablas auditadas (4,407 estudios cada una; 58 oficiales/Gold + 4,349 no oficiales; las tres tienen los mismos UID, en distinto orden de filas):

| Nombre | Archivo |
|---|---|
| v1 | `preproc/etiquetas_v1_2079203.parquet` (Qwen, informe a objetivo) |
| híbrida | `runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet` |
| v3 | `runs_joako/2084839/etiquetas_v3_distilled_gold.parquet` |

**Advertencia de tamaño.** Gold son 58 estudios. Cada proporción tiene un IC95 de ±0.10 a ±0.25 y los AUC de ±0.10 a ±0.15 (bootstrap de 4,000 réplicas, semilla 0). No se interpreta ninguna diferencia menor a 0.02, y casi ninguna menor a 0.10 es distinguible del azar.

---

## Hallazgos principales

1. **Ninguna de las tres tablas contiene un objetivo 0, 1 ni 0.5 exacto.** El silencio está codificado solo como `t = NaN` con `w = 0`. Hay 0 celdas con `w > 0` y `t` NaN, y 0 celdas con `w = 0` y `t` no NaN, en las tres tablas.
2. **`entrenar.py` enmascara bien el silencio.** El `nan_to_num` de la línea 37 lo convierte en 0, pero la pérdida lo multiplica por `w = 0`, así que no hay gradiente.
3. **El 0.5 "no abordado" de stevenleehans en la híbrida NO quedó como objetivo 0.5.** Entra con peso `w_ext = 0` y desaparece del promedio. La tabla híbrida se reproduce exactamente a partir de `llm_labels_v2.csv`, y NO de `llm_labels_v4_blend.csv`. Eso importa: en v4 el silencio es 0.25, que daría `w = 0.5` y no quedaría enmascarado.
4. **La premisa "lo desconocido nunca se convierte en negativo" solo vale para las celdas 0.5 de Steven.** Donde Qwen v1 calla pero Steven v2 emite un valor distinto de 0.5, la híbrida sí crea objetivos. Son cerca de 590 celdas, casi todas negativas con `w ≈ 0.87`. Además hay 2,807 celdas de Synovitis que son imputaciones por co-ocurrencia con Effusion (sección 2).
5. **El 0.71 vs 0.68 del post se reproduce con `llm_labels_full.csv` (0.711 vs 0.678), no con nuestra v1.** Las proporciones del post 743416 también coinciden exactamente con ese archivo.
6. **Cobertura del OOF.** Synovitis tiene solo 1,206 de 4,349 filas evaluables (27.7%) y Baker's 2,515 (57.8%). Todos los demás hallazgos tienen entre 84.5% y 95.6%. En Gold el silencio es informativo, así que el OOF de Synovitis y Baker's mide una subpoblación seleccionada.

---

## 1. Composición de celdas por tabla y hallazgo

Denominador: 4,407 filas. Dentro de cada tabla, `%NaN` es igual a `%w=0`.

| Hallazgo | v1 %NaN | híbrida %NaN | v3 %NaN |
|---|---:|---:|---:|
| ACL | 7.56 | 7.49 | 7.49 |
| MCL | 7.01 | 7.01 | 7.01 |
| Medial Meniscus | 4.40 | 4.40 | 4.40 |
| Lateral Meniscus | 6.74 | 6.74 | 6.74 |
| Medial OA | 15.14 | 12.93 | 12.93 |
| Lateral OA | 15.18 | 12.93 | 12.93 |
| PF OA | 15.54 | 12.28 | 12.28 |
| Effusion | 7.60 | 7.60 | 7.60 |
| **Synovitis** | **72.07** | 7.67 | 7.67 |
| **Baker's** | **42.27** | 40.59 | 40.59 |
| Contusion | 12.55 | 10.37 | 10.37 |
| Fracture | 13.52 | 11.60 | 11.60 |

- **`% = 0.5 exacto`, `% = 0 exacto` y `% = 1 exacto` valen 0.00 % en las 36 celdas (3 tablas × 12 hallazgos).** Los objetivos son suavizados y nunca llegan a 0 ni a 1 (v1 tiene 62 valores distintos, entre 0.0078 y 0.9768; híbrida 1,218 valores, entre 0.0078 y 0.9884; v3 46,252 valores, entre 0.0078 y 0.9909). Los más cercanos a un extremo (≤ 0.02 o ≥ 0.98) son ACL y MCL en v1: 3,049 y 3,328 celdas con t ≤ 0.02.
- Sobre solo los 4,349 no oficiales, los porcentajes cambian menos de 0.25 puntos (por ejemplo Synovitis v1 72.27 %, Baker's v1 42.17 %).
- **v3 tiene exactamente el mismo patrón de NaN que la híbrida.** La destilación solo cambia valores de celdas ya con objetivo (`f_ = 1` en 4,023 celdas de ACL, 4,016 de Effusion, 2,589 de Baker's, etc.).
- La caída de Synovitis de 72 % a 7.7 % no viene de que Qwen ahora lea más informes. Viene de Steven (sección 2).

---

## 2. Cómo trata `entrenar.py` esas celdas, y qué pasó con el 0.5 de Steven

### Tratamiento en `rsna/entrenar.py`

| Línea | Qué hace | Efecto sobre una celda en silencio (`t = NaN`, `w = 0`) |
|---|---|---|
| [37](../../rsna/entrenar.py#L37) | `self.t = np.nan_to_num(t)` | NaN pasa a 0.0 en el tensor objetivo. Es seguro solo porque esa celda tiene `w = 0`. |
| [38](../../rsna/entrenar.py#L38) | `self.w = w` sin modificar | El peso va tal cual a la pérdida. |
| [168](../../rsna/entrenar.py#L168) | `loss = (BCE(logits, t) * w).sum() / w.sum().clamp(min=1)` | La celda aporta 0 a la pérdida y 0 al gradiente. No es "objetivo 0": es máscara. |
| [114-115](../../rsna/entrenar.py#L114-L115) | Descarta de `train` los estudios con Σw = 0 | Un estudio sin ningún objetivo no entra al entrenamiento. |
| [57-60](../../rsna/entrenar.py#L57-L60) | `macro_auc` con `m = w[:, j] > 0` | El AUC por hallazgo solo usa celdas con `w > 0`. |
| [184](../../rsna/entrenar.py#L184) | `val_int`: `y = (t >= 0.5)` con máscara `w` | Las celdas en silencio no se evalúan. |
| [186](../../rsna/entrenar.py#L186) | Oficial: `macro_auc(y_off, p_off)` sin máscara | Los 58 Gold se evalúan completos con su `y` real. |

Dos matices:

- **La normalización es por lote, no por hallazgo** (`Σw` sobre las 12 columnas). Un hallazgo con muchos NaN (Synovitis en v1, 72 %) aporta proporcionalmente menos gradiente que los demás. No es un error, pero explica por qué el hallazgo con más silencio recibe menos señal.
- **Las celdas "possible" (`w = 0.5`) se pesan por la mitad en el entrenamiento, pero cuentan completas en la evaluación** (`w > 0`) y se binarizan con `t >= 0.5`. En v1 su `t` es constante por hallazgo: MCL 0.53, Contusion 0.53 y Fracture 0.61 (los tres positivos); ACL 0.33, Medial Meniscus 0.23, Lateral Meniscus 0.40, las OA 0.33 a 0.41 y Baker's/Synovitis 0.41 (todos negativos).
- `evaluar_fusion_offline.py` ([112-114](../../rsna/evaluar_fusion_offline.py#L112-L114)) usa la misma regla: `y_oof = nan_to_num(t_v1) >= 0.5`, con `w` de v1.
- Solo se leyó `entrenar.py`. Los demás scripts de entrenamiento (`entrenar_arq.py`, `entrenar_vistas.py`, etc.) no se verificaron.

### La híbrida: qué pasó con el 0.5 de Steven

**Qué CSV externo se usó.** Se reconstruyó la híbrida con `generar_tabla_candidata.py` (modo `blend_50_50`) a partir de cada CSV de `runs_cons/datos/RSNA Knee LLM Report Labels/`:

| CSV externo | Coincide con la híbrida | max abs Δt | max abs Δw |
|---|---|---:|---:|
| **`llm_labels_v2.csv`** | **Sí, exacta** (NaN idénticos) | **0.000** | **0.000** |
| `llm_labels_v4_blend.csv` | No | 0.490 | 0.940 |
| `llm_labels_full.csv` | No | 0.154 | 0.490 |

**Cómo quedó el 0.5.** En `llm_labels_v2.csv` el silencio es `p = 0.5` exacto (10,167 celdas, la segunda cifra más frecuente tras 0.05). El script calcula `w_ext = 2·|p − 0.5|`, o sea 0, y mezcla con `p_ext.fillna(0.5)` ponderado por `w_ext`. El 0.5 queda multiplicado por cero:

- Si Qwen v1 tiene señal, `t = t_Qwen` y `w = w_Qwen`.
- Si Qwen también calla, `t = NaN` y `w = 0`.
- **Resultado: 0 celdas con `t = 0.5` exacto en toda la tabla**, y 0 de las celdas donde Steven decía 0.5.

| Hallazgo | Celdas con ext v2 = 0.5 | → híbrida NaN (w=0) | → híbrida con objetivo de Qwen |
|---|---:|---:|---:|
| ACL | 366 | 330 | 36 |
| MCL | 444 | 309 | 135 |
| Medial Meniscus | 244 | 194 | 50 |
| Lateral Meniscus | 440 | 297 | 143 |
| Medial OA | 896 | 570 | 326 |
| Lateral OA | 1,090 | 570 | 520 |
| PF OA | 763 | 541 | 222 |
| Effusion | 432 | 335 | 97 |
| Synovitis | 418 | 338 | 80 |
| Baker's | 2,126 | 1,789 | 337 |
| Contusion | 1,058 | 457 | 601 |
| Fracture | 1,890 | 511 | 1,379 |

**Riesgo si se regenera con otro CSV.** En `llm_labels_v4_blend.csv` el silencio es **0.25** (10,093 celdas), no 0.5. Con la fórmula `w = 2·|p − 0.5|` eso da `w = 0.5` y objetivo 0.25: quedaría como un negativo débil, sin enmascarar. Cualquier regeneración con v4 debe tratar 0.25 como silencio de forma explícita.

**Dónde la híbrida sí crea objetivos desde el silencio** (celdas no oficiales con silencio en v1 y `w > 0` en la híbrida):

| Hallazgo | Celdas | `t` medio | `t ≥ 0.5` | `w` medio |
|---|---:|---:|---:|---:|
| **Synovitis** | **2,807** | 0.510 | 1,690 | 0.40 |
| PF OA | 143 | 0.254 | 37 | 0.86 |
| Lateral OA | 98 | 0.143 | 13 | 0.87 |
| Medial OA | 96 | 0.079 | 5 | 0.89 |
| Contusion | 96 | 0.062 | 2 | 0.90 |
| Fracture | 84 | 0.072 | 3 | 0.90 |
| Baker's | 74 | 0.148 | 9 | 0.86 |
| ACL | 3 | 0.493 | 2 | 0.61 |

- Las ~590 celdas de OA, Baker's, Contusion y Fracture son, en la práctica, "Steven infirió ausencia donde el informe no dice nada". En Gold esa inferencia resulta correcta (sección 3: 0 positivos en 12 de silencio para Medial OA y Lateral OA, 0/8 para Fracture), pero con n muy pequeño.
- Las 2,807 de Synovitis son la imputación por co-ocurrencia con Effusion descrita en el README de `runs_cons`: son un objetivo derivado de otro hallazgo, no una lectura del informe. **El hallazgo Synovitis de la híbrida y de v3 aprende en parte a predecir Effusion.**

**Gold en v3.** Las 58 filas Gold llevan `w = 4.0` (parámetro `peso_gold`) y `gold_en_entrenamiento = True`, pero su `t` es idéntico al de la híbrida (max abs Δ = 0), no la etiqueta real `y`. `entrenar.py` excluye los oficiales del entrenamiento (líneas 110-111), pero no se verificó qué hacen los scripts que consumen v3. Queda anotado como punto a revisar, no como conclusión.

---

## 3. Gold (58): P(positivo | silencio) y P(positivo | mencionado)

**Definición.** "Silencio" = celda con `w = 0` en v1 (`t = NaN`); "mencionado" = `w > 0`. Positivo = etiqueta oficial `y = 1`. Entre corchetes, IC95 de Wilson.

**Nuestra v1:**

| Hallazgo | n silencio | pos | P(pos \| silencio) [IC95] | n mencionado | pos | P(pos \| mencionado) [IC95] |
|---|---:|---:|---|---:|---:|---|
| ACL | 4 | 0 | 0.00 [0.00, 0.49] | 54 | 24 | 0.44 [0.32, 0.58] |
| MCL | 4 | 0 | 0.00 [0.00, 0.49] | 54 | 9 | 0.17 [0.09, 0.29] |
| Medial Meniscus | 2 | 0 | 0.00 [0.00, 0.66] | 56 | 26 | 0.46 [0.34, 0.59] |
| Lateral Meniscus | 3 | 0 | 0.00 [0.00, 0.56] | 55 | 23 | 0.42 [0.30, 0.55] |
| Medial OA | 12 | 0 | 0.00 [0.00, 0.24] | 46 | 15 | 0.33 [0.21, 0.47] |
| Lateral OA | 12 | 0 | 0.00 [0.00, 0.24] | 46 | 11 | 0.24 [0.14, 0.38] |
| PF OA | 9 | 0 | 0.00 [0.00, 0.30] | 49 | 21 | 0.43 [0.30, 0.57] |
| Effusion | 2 | 0 | 0.00 [0.00, 0.66] | 56 | 35 | 0.63 [0.49, 0.74] |
| **Synovitis** | 33 | 9 | **0.27** [0.15, 0.44] | 25 | 18 | **0.72** [0.52, 0.86] |
| **Baker's** | 29 | 1 | **0.03** [0.01, 0.17] | 29 | 11 | **0.38** [0.23, 0.56] |
| Contusion | 8 | 1 | 0.13 [0.02, 0.47] | 50 | 18 | 0.36 [0.24, 0.50] |
| Fracture | 8 | 0 | 0.00 [0.00, 0.32] | 50 | 18 | 0.36 [0.24, 0.50] |

**Comparación con el post 743416 y con las tablas de Steven.** Aquí "silencio" = celda con `p = 0.5` exacto:

| Hallazgo | Post (sil / men) | `full.csv`: n sil, P(pos \| sil) | `full.csv`: n men, P(pos \| men) | v1 (nuestra) |
|---|---|---|---|---|
| Baker's | 0.03 / 0.44 | 33, 0.03 | 25, 0.44 | 0.03 / 0.38 |
| Medial OA | 0.00 / 0.36 | 16, 0.00 | 42, 0.36 | 0.00 / 0.33 |
| PF OA | 0.21 / 0.41 | 14, 0.21 | 44, 0.41 | 0.00 / 0.43 |
| Synovitis | 0.34 / 0.76 | 41, 0.34 | 17, 0.76 | 0.27 / 0.72 |

- **Las cuatro filas del post se reproducen exactamente con `llm_labels_full.csv`** (Synovitis: 14/41 y 13/17). El post describe la tabla de Steven, no la nuestra.
- Con v1 las proporciones tienen el mismo patrón (el silencio predice negativos), pero los conjuntos de silencio son distintos. En `full.csv` Synovitis es 0.5 en el 83.7 % de las filas; en v1 en el 72.1 %. Para Medial OA, de 880 celdas 0.5 de Steven solo 559 coinciden con el silencio de v1.
- Diferencias entre v1 y el post: PF OA en silencio 0.00 (0/9) vs 0.21 (3/14); eso es una diferencia de 3 casos. Synovitis 0.27 vs 0.34 y Baker's mencionado 0.38 vs 0.44 también quedan dentro de los IC. No hay forma de preferir una definición de silencio con estos n.
- **Lectura.** El silencio es informativo en Synovitis (0.27 vs 0.72) y en Baker's (0.03 vs 0.38), con ICs que casi no se solapan. En los demás hallazgos el silencio es raro (2 a 12 casos) y no hay evidencia suficiente para afirmar nada.

---

## 4. AUC en Gold: "Effusion" como predictor de Synovitis frente a "Synovitis"

Objetivo: `y_Synovitis` en los 58 Gold (27 positivos). IC95 por bootstrap pareado de 4,000 réplicas. `Δ` = AUC(Effusion→Synovitis) − AUC(Synovitis→Synovitis).

| Fuente del predictor | Regla para NaN | AUC Effusion | AUC Synovitis | Δ [IC95 de Δ] | P(Δ>0) |
|---|---|---:|---:|---|---:|
| **Steven `full.csv`** (la del post) | no hay NaN | **0.711** | **0.678** | +0.033 [−0.121, +0.186] | 0.66 |
| Steven `v2.csv` | no hay NaN | 0.711 | 0.790 | −0.079 [−0.174, +0.010] | 0.04 |
| Steven `v4_blend.csv` | no hay NaN | 0.729 | 0.790 | −0.061 [−0.158, +0.028] | 0.09 |
| Nuestra v1 | NaN → 0.5 | 0.651 | 0.622 | +0.029 [−0.156, +0.212] | 0.62 |
| Nuestra v1 | NaN → 0 (como `entrenar.py`) | 0.645 | 0.734 | −0.088 [−0.251, +0.076] | 0.15 |
| Nuestra v1 | solo mencionado, cada una en su subconjunto (nE = 56, nS = 25) | 0.658 | 0.587 | +0.071 (no pareable) | — |
| Nuestra v1 | solo mencionado, intersección (n = 25) | 0.516 | 0.587 | −0.071 (no pareable) | — |
| Híbrida (= v3 en Gold) | NaN → 0.5 | 0.711 | 0.716 | −0.005 [−0.125, +0.120] | 0.47 |
| Referencia: `y_Effusion` oficial → `y_Synovitis` | — | 0.698 | — | — | — |

- **El 0.71 vs 0.68 del post se reproduce con `llm_labels_full.csv` (0.711 vs 0.678).** No es nuestra v1: con v1 y NaN → 0.5 es 0.651 vs 0.622.
- **El orden entre los dos predictores en v1 se invierte según cómo se trate el NaN:** +0.029 con NaN → 0.5 y −0.088 con NaN → 0. Con NaN → 0 gana "Synovitis" porque el propio silencio informa (P(pos | silencio) = 0.27 frente a 0.72). Ese es el efecto de la regla, no de la lectura del informe.
- **Ningún Δ es distinguible de 0** salvo, marginalmente, `v2.csv` (P(Δ>0) = 0.04). Todos los IC de Δ miden ±0.12 a ±0.25. Tampoco hay base para decir que uno de los dos predictores es mejor.
- La etiqueta oficial `y_Effusion` predice `y_Synovitis` con AUC 0.698 (35 positivos de Effusion; P(Syn | Eff=1) = 22/35 = 0.63; P(Syn | Eff=0) = 5/23 = 0.22). El 0.711 derivado del informe es igual de bueno que la etiqueta verdadera, así que la co-ocurrencia es real pero modesta.
- Los AUC de v1 en Gold son optimistas para la parte "mencionado": el mapeo categoría → `t` se estimó en esos mismos 58 estudios (docstring de `tabla_entrenamiento.py`). El README de `runs_cons` cita 0.6225 para Synovitis v1; coincide con la regla NaN → 0.5 (0.622).

---

## 5. Cobertura del OOF actual (4,349 filas)

Las 4,349 filas del OOF (`runs/2084813/oof_*.parquet`) son exactamente los no oficiales (verificado: mismo conjunto de UID). La referencia es v1 (objetivos del OOF según `oof.txt` y `evaluar_fusion_offline.py` [112-114](../../rsna/evaluar_fusion_offline.py#L112-L114)): `y = (t ≥ 0.5)`, evaluando solo celdas con `w > 0`.

| Hallazgo | v1: filas evaluables | % | pos (t≥0.5) | neg | prev. % | de las pos, "possible" | híbrida n (%) | v3 n (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ACL | 4,020 | 92.4 | 569 | 3,451 | 14.2 | 0 | 4,023 (92.5) | 4,023 (92.5) |
| MCL | 4,044 | 93.0 | 324 | 3,720 | 8.0 | **48** | 4,044 (93.0) | 4,044 (93.0) |
| Medial Meniscus | 4,157 | 95.6 | 1,931 | 2,226 | 46.5 | 0 | 4,157 (95.6) | 4,157 (95.6) |
| Lateral Meniscus | 4,055 | 93.2 | 749 | 3,306 | 18.5 | 0 | 4,055 (93.2) | 4,055 (93.2) |
| Medial OA | 3,694 | 84.9 | 1,383 | 2,311 | 37.4 | 0 | 3,790 (87.1) | 3,790 (87.1) |
| Lateral OA | 3,692 | 84.9 | 849 | 2,843 | 23.0 | 0 | 3,790 (87.1) | 3,790 (87.1) |
| PF OA | 3,673 | 84.5 | 1,251 | 2,422 | 34.1 | 0 | 3,816 (87.7) | 3,816 (87.7) |
| Effusion | 4,016 | 92.3 | 2,622 | 1,394 | 65.3 | 0 | 4,016 (92.3) | 4,016 (92.3) |
| **Synovitis** | **1,206** | **27.7** | 714 | 492 | 59.2 | 0 | 4,013 (92.3) | 4,013 (92.3) |
| **Baker's** | **2,515** | **57.8** | 564 | 1,951 | 22.4 | 0 | 2,589 (59.5) | 2,589 (59.5) |
| Contusion | 3,804 | 87.5 | 322 | 3,482 | 8.5 | **42** | 3,900 (89.7) | 3,900 (89.7) |
| Fracture | 3,761 | 86.5 | 369 | 3,392 | 9.8 | **73** | 3,845 (88.4) | 3,845 (88.4) |

Qué significa para el sesgo de la medición:

1. **Synovitis y Baker's se evalúan sobre una subpoblación autoseleccionada.** Solo entran las filas donde el radiólogo escribió algo sobre ese hallazgo: 27.7 % y 57.8 % de las filas. En Gold, esa selección está asociada a la positividad (Synovitis: prevalencia 0.72 entre mencionados frente a 0.27 en silencio y 0.47 global; Baker's: 0.38 frente a 0.03 y 0.21). El test de Kaggle incluye los casos en silencio (en Gold, 57 % para Synovitis y 50 % para Baker's). **El AUC OOF de esos dos hallazgos no se puede extrapolar a Kaggle.** La dirección del sesgo no se puede determinar sin predicciones del modelo sobre las 58 Gold; el OOF las excluye.
2. **Los demás hallazgos tienen cobertura de 84.5 % a 95.6 %** y su silencio en Gold es raro y casi siempre negativo; ahí el sesgo de selección es menor.
3. **Las celdas "possible" inflan los positivos de MCL, Contusion y Fracture** (48/324, 42/322 y 73/369: 15 %, 13 % y 20 % de los positivos), porque su `t` en v1 (0.53, 0.53, 0.61) queda sobre el corte 0.5. En el resto de hallazgos se binarizan como negativos (su `t` queda bajo 0.5).
4. **La referencia es la propia pseudoetiqueta de Qwen.** El OOF mide acuerdo con Qwen v1, no con la verdad (ya reconocido en el README).
5. **Con la híbrida o v3 como referencia, Synovitis subiría a 4,013 filas (92.3 %), pero 2,807 de ellas son imputaciones por co-ocurrencia con Effusion** (sección 2). Esa cobertura no es comparable con medir contra lo que dice el informe.

---

## Limitaciones y lo que no se verificó

- Solo se leyó `entrenar.py`. Los scripts `entrenar_arq.py`, `entrenar_vistas.py`, `entrenar_densenet.py` y el que genera v3 (`destilar_etiquetas_v3.py`) no se revisaron.
- Las comparaciones entre silencio en v1 y en Steven suponen que los UID de `train.csv` y de los CSV externos se corresponden uno a uno (4,407 en ambos, sin duplicados; se usó `drop_duplicates` por UID como en `generar_tabla_candidata.py`).
- Con 58 Gold, ninguna conclusión de las secciones 3 y 4 justifica por sí sola una política (por ejemplo, imputar o enmascarar). Sirven para fijar qué hipótesis merecen la prueba más barata.
- Los scripts de análisis están en el scratchpad de la sesión (`silencio.py`, `silencio2.py`, `silencio3.py`) y no se versionan.
