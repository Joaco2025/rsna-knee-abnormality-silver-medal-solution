# Estado verificado del proyecto (2026-10-19, ~08:00 UTC)

Etiquetas de evidencia: **[recibo]** = score de Kaggle comunicado por jcastro para un envío nuestro; **[autor]** = score que
declara el autor de un notebook público; **[archivo]** = verificado en un archivo con sha256 o en el código;
**[local]** = calculado aquí sobre Gold-58; **[E]** = estimación.

## 1. Envíos y su receta

| Envío | Público | Receta | Artefacto | Evidencia |
|---|---|---|---|---|
| **«0.958»** (copia de goodpjw2008 v25/27) | **0.958, puesto 987, ~6 h** | Raptor 384 (espejo) + Raptor 224 (espejo) al 0.65/0.35 en la mitad Raptor; R6 (4 folds, flip TTA) en la mitad compacta; pesos por hallazgo de heliosli para Raptor (Syn 0.8, Lat OA 0.2, PF OA 0.6, Baker 0.35, resto 0.5); lector 2.5D ConvNeXt-T con pesos v6 ×0.1 | `runs_joako/exp_E_0958_s6/fuentes/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb` (sha `9485a6fc…`) | [recibo] score; [archivo] receta. **Sin log de la ejecución guardado localmente.** |
| **E** | pendiente (en Kaggle desde ~05:30 UTC del 19-oct) | «0.958» ×0.75 + S6 nartaa ×0.25, versión estricta | `runs_joako/exp_E_0958_s6/E_0958_s6_estricto.ipynb` (`043ab37c…`) | [archivo]; prueba de 3 estudios 13/13 |
| **C** | 0.957, ~5 h | 0.956 (R7 + Raptor 384 50/50) + Raptor 224 + S6 al 0.25 | `runs_joako/exp_4brazos_20261018/C_0956_224_s6_estricto.ipynb` (`f058b471…`) | [recibo] |
| B (varshitha) | 0.956 | 0.956 + Raptor 224 | — | [autor]; C − B = S6 → **+0.001 mostrado** |
| v9c2 + rama propia | 0.955 | 0.956 + 25 CNN propias (0.10–0.30) | `runs_joako/exp_0957_v9c2_20261017/` | [recibo]; **nuestras CNN restan** |
| 0.956 | 0.956 | R7 + Raptor 384 | `code_joako/referencias/0-957-…ipynb` | registro del equipo, sin recibo |

Escalera del autor de la base «0.958» (write-up, [autor], todo en el público):

| Paso | Público |
|---|---|
| Raptor 384 solo | 0.949 |
| lector solo | 0.929 |
| Raptor + S6 heliosli | 0.954 |
| Raptor + R7 | 0.956 |
| + Raptor 224 | 0.956 (mejor puesto) |
| R6 flip en lugar de R7 | 0.957 |
| + pesos por hallazgo | 0.958 |
| R7 al 0.4 junto a R6 | 0.958, sin cambio |
| flip también en R7 | 0.957 |

## 2. Componentes disponibles

| Componente | Solo público | Kaggle (T4×2) [E] | Gold-58 [local] | Pesos locales | Etiquetas de entrenamiento |
|---|---|---|---|---|---|
| Raptor 384 (CoAtNet-2) | 0.949 [autor] | ~2 h | macro 0.923 (sin espejo; **inflado**: SWA elegido por Gold) | `code_joako/referencias/rsna_research/pesos/raptor_0949/` | blendjev + OAI en 3 hallazgos |
| Raptor 224 | 0.945 [autor] | ~1 h | 0.916 (ídem) | ídem | ídem |
| R6 (ConvNeXt-S ×4 folds, flip) | — (0.957 con Raptor) | ~2–3 h | no medido | **no** (está en Kaggle) | de heliosli (no públicas) |
| S6 nartaa (ConvNeXt-S, 1 modelo) | 0.950 [autor] | ~0.5–1 h | no medido (Gold limpio: excluido y sin selección) | `runs_joako/exp_s6_semillas/entradas/s6fall_v5/` | blendjev (las mismas que Raptor) |
| Lector goodpjw2008 (ConvNeXt-T ×3) | 0.929 [autor] | ~0.5 h | no medido (Gold limpio) | no | 4 tablas públicas CC0 |
| R7 (ConvNeXt-S ×4) | — | ~3 h (8.7 s/estudio) | no | no | heliosli + fastMRI auxiliar |
| OrthoFoundation `best.pt` (fold 0 ep. 7, 1.2 GB) | ninguno público conocido | desconocido | no | `entradas/orthofoundation_v1/best.pt` | heliosli; val débil 0.9318 (otra métrica) |

## 3. Predicciones y validación disponibles

- **No hay OOF de ningún modelo público.** Raptor, S6 y el lector se entrenaron con todos los datos; los OOF que existen son
  de nuestras CNN propias (0.935 en el público), que no entran en la receta.
- **Gold-58 es el único conjunto puntuado como el leaderboard.** Solo tenemos predicciones de Raptor sobre él
  (`runs_joako/2089458/preds_gold.npz`), y están contaminadas porque el SWA de Raptor se eligió por Gold.
- **Pacientes:** 4,407 `PatientID` distintos en 4,407 estudios (`runs_joako/patchcore_pacientes_2090009`). No hay vía de
  fuga por paciente.
- **Test:** 1,284 estudios ocultos; el público es una fracción desconocida.

## 4. Puntos débiles por hallazgo (Raptor sobre Gold [local], mezcla 65/35)

| Hallazgo | ACL | MCL | Med Men | **Lat Men** | Med OA | **Lat OA** | **PF OA** | Effusion | **Synovitis** | Baker | Contusion | Fracture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.979 | 0.950 | 0.980 | **0.893** | 0.980 | **0.778** | **0.856** | 0.981 | **0.815** | 0.971 | 0.945 | 0.957 |

Los cuatro débiles incluyen los tres que OAI supervisa en Raptor (PF OA, Lateral OA, Synovitis; `training_code/coatnet/RUNS.md`).
Son también los que la base «0.958» ya pondera aparte, con los pesos por hallazgo de heliosli.

## 5. Trabajo en Yuca hoy

| Quién | Qué | Estado |
|---|---|---|
| Owen | 16 jobs `rsna-todo-*`: sus familias propias con todos los datos y 4 semillas, para v10 | usa las 4 GPU del equipo |
| Fer | screening (EMA, espejo, pool espacial, vistas por hallazgo) | ninguno cumple su criterio; SSL AE negativo; 3 recetas en cola |
| Nosotros | job 2092983 (anclas S6) | PENDING por QOSGrpGRES |
