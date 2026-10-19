# Hipótesis, agrupadas por mecanismo

Estado: **REFUTADA hoy** (con dato), **CERRADA antes** (journal o recibos), **ABIERTA** (con mecanismo y prueba mínima) o
**DESCARTADA** (por coste, tiempo o reglas, sin probarla).

## M1. Transformar predicciones existentes (sin modelos nuevos)

| # | Hipótesis | Estado | Evidencia |
|---|---|---|---|
| 1 | Penalización cruzada trauma ↔ degeneración (en Gold se excluyen y en los informes no) | **REFUTADA hoy** | A2: Raptor ya lo captura; Δ ≤ 0 en todas las λ |
| 2 | Préstamo Synovitis←Effusion, Lat OA←Lat Men, PF OA←Med OA | **REFUTADA hoy** | A3: ningún IC por encima de 0; Synovitis y PF OA empeoran |
| 3 | Meta-apilado 12×12 | CERRADA antes | journal, job 2084840 (sobre OOF) |
| 4 | Promedio de probabilidades en lugar de rangos | DESCARTADA | el AUC solo depende del orden; la base ya mezcla rangos por hallazgo |
| 5 | Calibración / temperatura | DESCARTADA | monótona por hallazgo: no cambia el AUC |

## M2. Sustituir en lugar de añadir

| # | Hipótesis | Estado | Notas |
|---|---|---|---|
| 6 | Quitar el lector (1/10) y usar sus ~0.5 h para otra cosa | ABIERTA (menor) | la propia escalera del autor dice que el lector ya no suma |
| 7 | Cambiar R6 por R6 + R7 | CERRADA antes | v26 del autor: 0.958, sin cambio |
| 8 | Raptor 224 en lugar del 384 | DESCARTADA | 224 es peor en Gold (0.916 frente a 0.923) y en el público (0.945 frente a 0.949) |

## M3. Reforzar el componente más diverso por etiquetas

| # | Hipótesis | Estado | Notas |
|---|---|---|---|
| 9 | **Más peso a R6 en los hallazgos débiles** (sus etiquetas difieren de las de Raptor y S6) | ABIERTA | Los pesos de heliosli ya dan Lat OA a R6 (0.8). Con S6, la parte de etiquetas blendjev (Raptor + S6) crece en todos los hallazgos, incluidos los débiles. **Un envío.** |
| 10 | **E-sust: S6 dentro de la mitad Raptor, no encima de todo** | ABIERTA, **la más defendible** | En E: final = 0.75·base + 0.25·S6, así que R6 pierde un 25 % de su peso en todos los hallazgos. En E-sust: mitad Raptor = 0.75·Raptor(384/224) + 0.25·S6, y la mitad R6 y los pesos de heliosli quedan intactos. S6 (0.950, sin OAI) comparte papel y etiquetas con Raptor (0.949): lo sustituye en parte en lugar de diluir al componente más diverso. Mismo tiempo de Kaggle que E. **Un envío** |
| 10b | S6 al 0 en Lat OA y Lat Men, y al 0.25 en el resto | ABIERTA, menor | Variante de #9; queda dominada por #10 |

## M4. Reducir la varianza del tercer lector

| # | Hipótesis | Estado | Notas |
|---|---|---|---|
| 11 | Semillas extra de S6 (D) | ABIERTA, ya preparada | Mecanismo claro (promediar semillas), pero S6 pesa 0.25: efecto ≤ +0.001 [E]. En E caben 1–2 semillas por tiempo. Sirve más al final B |
| 12 | Flip TTA en S6 | **REFUTADA hoy por código** | exige `extent` y entrenamiento con espejo; S6 no tiene ninguno de los dos |

## M5. Familia realmente distinta

| # | Hipótesis | Estado | Notas |
|---|---|---|---|
| 13 | ViT-L OrthoFoundation `best.pt` (heliosli, fold 0) como tercer lector | ABIERTA, cara | La única diversidad de arquitectura real. Sin score público, inferencia sin verificar, coste en T4 desconocido, 4/5 de los datos. Mínimo: inferencia en Gold y en 3 estudios de test, ~0.5 día de ingeniería + 1 GPU-h |
| 14 | Entrenar CoAtNet con la receta exacta de nartaa y sin OAI | DESCARTADA | ~16 épocas de CoAtNet-2 a 384 es más de 1 día de GPU, más la caché a 384; no llega |

**Actualización de H13 (19-oct, metadatos de `best.pt` leídos con mmap):**
- **Qué es:** ViT-L DINOv3 (`vit_large_patch16_dinov3_qkvb`) a 384 px, fold 0, época 6; agregación jerárquica multivista con pool anatómico (SKM-TEA).
- **Etiquetas:** las de heliosli (`own_reports_and_official_gold_only_v1`), distintas de blendjev.
- **Gold:** `reserve_gold: false`, es decir, **se entrenó con los 58 Gold**, así que Gold no sirve para evaluarlo; solo el público.
- **Inferencia:** `cloud_code/training/infer.py` (está en el dataset que E ya adjunta).
- **Tiempo en Kaggle [E]:** ViT-L a 384 px ≈ 190 GFLOP por imagen, ×~160 ventanas por estudio ≈ 30 TFLOP por estudio. En 2×T4, ~0.7–1.5 h para 1,284 estudios. E ya va en ~7 h, así que **no cabe junto a E** sin quitar Raptor 224 o S6.
- **Veredicto:** solo tiene sentido como **sustituto de S6** y si un envío propio muestra ≥ 0.952. Son dos envíos de 6 h o más. Prioridad C; no se construye salvo que E y E-sust fallen.

**Variante de D descartada: «D-boot».** Sería entrenar S6 con la tabla B de Owen (Raptor como maestro), siguiendo la pista
del foro (las etiquetas de informe mezcladas con predicciones de un maestro elevan los modelos únicos a 0.953–0.959). Pero
el maestro es Raptor, que ya está en la mezcla: el modelo resultante se parecería más a Raptor y aportaría menos
diversidad, y además heredaría OAI, así que no serviría para el final B.

## M6. Supervisión distinta para los hallazgos débiles

| # | Hipótesis | Estado | Notas |
|---|---|---|---|
| 15 | Reetiquetar Synovitis y Lat OA con criterio de experto (LLM + imagen) y reentrenar | DESCARTADA por tiempo | es la pista del foro («labels, AI, not reports alone»); más de una semana |
| 16 | Usar los 58 Gold para ajustar pocos parámetros (pesos por hallazgo) | ABIERTA, con riesgo | 12 parámetros con n = 58 sobreajustan; solo valdría si el cambio es grande y coincide con un mecanismo. Necesita predicciones de S6/R6 en Gold |

## M7. Fugas y estructura del test

| # | Hipótesis | Estado | Notas |
|---|---|---|---|
| 17 | Mismo paciente en train y test | **CERRADA** | `PatientID` único por estudio (anonimizado) |
| 18 | Duplicados exactos de imagen entre train y test | DESCARTADA | no se puede comprobar sin el test oculto; el `ExactDuplicateResolver` de heliosli es intra-estudio |

## M8. Ronda «fuera de la caja» del 19-oct (A4–A8 y revisiones)

| # | Hipótesis | Estado | Evidencia |
|---|---|---|---|
| 19 | T4 ociosa → meter otro modelo en paralelo | REFUTADA | log de E: Raptor, R6 y S6 ya usan las 2 GPU |
| 20 | Sesgo de puntuación por centro | REFUTADA | A4: la corrección sin etiquetas empeora (−0.0006 a −0.013) |
| 21 | Metadatos DICOM como predictor | REFUTADA para el LB | A6: +0.0044 frente a etiquetas de informe pero −0.0044 en Gold; capta los hábitos de redacción de cada centro |
| 22 | Casi-duplicados entre estudios | REFUTADA | A6: ningún par con correlación > 0.95; acuerdo de etiquetas igual al azar |
| 23 | **Desatajo de centro en Synovitis (y Fracture)** | ABIERTA, pequeña | A8: +0.001 de macro en Gold con λ 0.1 y P(mejora) 0.76–0.90, pero los hallazgos se eligieron mirando Gold. Candidato a un envío |
| 24 | Contraste (gadolinio) → Synovitis | DESCARTADA | `ContrastBolusAgent` casi siempre vacío o basura («right», «15»); ningún estudio de Gold con contraste |
| 25 | Pesos públicos de la «community stack» (mattiaangeli, antoinegg1) | DESCARTADA a priori | Sin OAI, pero de nivel ~0.94 (la community stack + lector dio 0.944). Añadir modelos de 0.935–0.94 ya restó (v9c2 → 0.955) |
| 26 | Destilación: S6 con etiquetas JEV mezcladas con el maestro (Raptor/R6) | ABIERTA, con GPU | Pista del foro (los modelos únicos del top, de 0.953–0.959, salen así). Bloqueada por las 2 descargas |

## La pregunta del punto ciego

> ¿Qué solución plausible podría superar 0.958 que no estamos considerando por asumir que hay que añadir otro modelo?

Respuesta: **redistribuir el peso por hallazgo hacia el lector con supervisión distinta (R6) en los cuatro hallazgos débiles**.
- La escalera pública del autor muestra que los dos saltos más grandes después de R7 vinieron del componente compacto
  (R6 + flip) y de los **pesos por hallazgo**.
- E añade S6, que comparte las etiquetas de Raptor, y eso diluye a R6 justo donde más diverge.
- Es una sola línea de pesos, sin modelos nuevos y sin coste extra de Kaggle.
- No tenemos datos para ajustar esos pesos: hay que fijarlos a priori y medirlos en el público (#9 y #10).
