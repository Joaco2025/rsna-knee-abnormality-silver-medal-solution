# Análisis técnico

Todo lo marcado [local] se calculó en esta sesión con los scripts de `analisis/` (CPU, segundos, solo lectura).

## 1. Dónde se pierde AUC

En Gold-58, Raptor flojea en **Lateral OA (0.78), Synovitis (0.82), PF OA (0.86) y Lateral Meniscus (0.89)**; el resto
está entre 0.95 y 0.98. Con 12 hallazgos, subir Lateral OA de 0.78 a 0.85 equivale a +0.006 de macro. Ningún otro
hallazgo ofrece ese margen. Cualquier mejora real tiene que salir de estos cuatro.

Prevalencias:

| Hallazgo | Gold-58 (experto) | Etiquetas de informe JEV |
|---|---|---|
| Synovitis | 47 % | 30 % |
| Lateral OA | 19 % | 26 % |
| Fracture | 31 % | 15 % |
| ACL | 41 % | 23 % |

La definición de experto difiere de lo que dicen los informes, y esa diferencia pesa más en los hallazgos débiles.

## 2. Hipótesis de post-procesado probadas hoy y refutadas

**A1 + A2: eje trauma frente a degeneración** (`a1_coocurrencia.json`, `a2_eje.json`).
- En Gold, ACL, MCL, Contusion y Fracture se oponen a OA, Baker's y los meniscos (correlaciones entre −0.20 y −0.35).
  En las etiquetas de informe esa correlación es ≈ 0 o positiva.
- Hipótesis: los modelos no aprendieron esa exclusión y una penalización cruzada la recuperaría.
- **Refutada:** las predicciones de Raptor ya la reproducen, incluso con más fuerza (corr. pred Contusion–Medial OA
  −0.47 frente a −0.33 en Gold). Restar λ·eje empeora el macro (λ 0.3: −0.002 [−0.008, +0.004]; λ 0.5: −0.006).

**A3: préstamo entre hallazgos** (`a3_prestamo.json`). Parejas fijadas por mecanismo clínico, no buscadas:
- Synovitis ← Effusion
- Lateral OA ← Lateral Meniscus
- PF OA ← Medial OA

**Refutada:** ninguna mejora con IC por encima de 0. Synovitis ← Effusion empeora (λ 0.2: −0.006 en la mezcla) y
PF OA ← Medial OA también (P(mejora) ≤ 0.10). La única señal es Lateral OA ← Lateral Meniscus en el 224 (+0.008 con
λ 0.1, IC [−0.007, +0.029]): es ruido con n = 58.

**Conclusión:** junto con el meta-apilado 12×12 ya refutado sobre OOF (journal, job 2084840), los modelos fuertes ya
capturan la estructura entre hallazgos. **La vía de las transformaciones deterministas de predicciones está agotada con
la evidencia disponible.**

## 3. Inferencia: comprobaciones de código

- **Flip TTA en S6: imposible tal cual.** El `--flip-tta` de `compact_infer` v4c (sha `8401812c`, el que usa R6) exige
  modelos entrenados con la columna de extensión (`N_EXTENT > 0`) y con el espejo del trainer v9.4. S6 se entrenó con
  `extent: false` y `aug: crop`, sin espejo (`E_prior_s42_fall.run_config.json`). El código para con `SystemExit`. Además,
  pasarle espejos a un modelo que nunca los vio sería salir de su distribución. **Descartado.**
- **R6 ya va con flip y Raptor ya va con espejo.** No queda TTA geométrica barata sin probar en la receta.
- **Flip en R7** (v23 del autor): 0.957 frente a 0.958 sin él. No ayuda.
- **El lector de la Parte B** pone a cero, sin avisar, una serie que no puede leer (lo documentamos en el LEEME de E).
  Pesa 1/10, así que su efecto es despreciable. Quitarlo solo ahorraría ~0.5 h de Kaggle.
- **Tiempo:** «0.958» tarda 6 h y E ~7 h [E] sobre ~9 h. Hay ~2 h de margen para un componente más, como mucho.

## 4. Diversidad

- **S6 y Raptor comparten etiquetas** (labels_v1_blendjev). Sus errores de etiqueta están correlacionados; S6 aporta
  arquitectura (ConvNeXt), anclas SKM-TEA y ausencia de OAI, pero no supervisión distinta.
- **R6 y el lector usan otras etiquetas** (heliosli y cuatro tablas públicas). Por etiquetas, R6 es el componente más
  complementario, y eso encaja con que el paso de R7 a R6 con flip fue el que más movió el público.
- **El `best.pt` de OrthoFoundation** (ViT-L, DINOv3, fold 0) es el único componente de familia realmente distinta (un
  transformador fundacional, no una CNN), y no lo usa ningún notebook conocido. Pero no hay score público, el coste en T4
  es desconocido (1.2 GB) y solo está entrenado en 4/5 de los datos. Necesita la ruta de inferencia `probe_infer`/mainline
  de heliosli, que no hemos verificado.

## 4b. Dos ideas del modo no convencional, probadas el 19-oct

**¿Hay una T4 ociosa para paralelizar etapas?** Revisado en el log de la prueba de 3 estudios de E (`exp_E_0958_s6/pruebas/fork-…ipynb`).
**Refutada:**
- Raptor 384 y 224 usan 2 dispositivos (`[gpu] done: 3/3 studies on 2 device(s)`), R6 usa 2 workers y S6 2 procesos de GPU.
- Solo el lector usa una GPU, y es corto.
- R6 está limitado por la decodificación de DICOM en CPU (53 s de decodificación frente a 19 s de GPU por estudio, con 3 estudios: no representativo).
- **No se libera tiempo para el ViT-L sin quitar un componente.**

**¿Hay sesgo de puntuación por centro (fabricante × campo)?** A4, `a4_sesgo_centro.py` → `a4_centro.json`. Sobre los OOF de v10 (4,349 estudios, 25 modelos), con 13 grupos de centro sacados de las cabeceras DICOM (job 2093185).

| Corrección | λ | Δ macro | IC95 |
|---|---|---|---|
| Sin etiquetas (aplicable en el test) | 0.25 | −0.0006 | [−0.0009, −0.0001] |
| Sin etiquetas | 1.0 | −0.0134 | — |
| Supervisada con ajuste cruzado | 0.25 | +0.0007 | [+0.0004, +0.0010] |

Los centros difieren en prevalencia real, así que igualar sus puntuaciones quita información. La ganancia supervisada es
mínima, depende del modelo y no se puede trasladar a Raptor ni a R6.
**Refutada como palanca.**

## 4c. Metadatos DICOM, casi-duplicados y «desatajo» (A5–A8, 19-oct)

**Extracción (job 2093187).** De las 24,371 series hay cabecera y de los 4,407 estudios, miniatura. El texto clínico viene
borrado (`StudyDescription`, motivo del estudio). Quedan: sexo, lateralidad, contraste (34.6 % de los estudios), nombres
de secuencia, bobina, fabricante/modelo/campo y parámetros de adquisición.

**A6: los metadatos como predictor.** 145 rasgos, HistGradientBoosting, 5 particiones.

| | Macro | Notas |
|---|---|---|
| Metadatos solos, contra etiquetas de informe | 0.651 | Synovitis 0.755 |
| OOF v10 + metadatos (w elegido anidado), contra informe | **+0.0044** [+0.0036, +0.0053] | |
| Raptor + metadatos al 0.1, contra Gold | **−0.0044** | metadatos solos 0.606; **Synovitis 0.465** |

**Conclusión:** los metadatos predicen **los hábitos de redacción de cada centro**, no la patología. Además, esto
demuestra que **una ganancia en OOF contra etiquetas de informe puede ser un espejismo** frente al test de expertos.

**Casi-duplicados.** Ningún par de estudios tiene correlación de miniatura > 0.95 (percentil 99.9 = 0.93), y el acuerdo de
etiquetas con el vecino es igual al azar (0.695 frente a 0.662). **Refutado.**

**A7/A8: quitar a los modelos el atajo «el centro informa X».** Restar λ·rango(metadatos) a la puntuación de Raptor en Gold:

| Variante | Resultado |
|---|---|
| En los 12 hallazgos | +0.001 con λ ≤ 0.1 (no significativo); empeora con λ ≥ 0.2, porque en artrosis los metadatos llevan señal real |
| Solo Synovitis y Fracture, con modelo lineal portable | **+0.0009 a +0.0012** con λ 0.1 (P(mejora) 0.76–0.90), casi todo de Synovitis (+0.005 a +0.008) |

Esos dos hallazgos se eligieron mirando Gold, así que el resultado es **optimista por multiplicidad**. Es candidato a un
envío, no una prueba. El modelo exportable está en `analisis/desatajo_modelo.json`; para usarlo en Kaggle falta guardar
los nombres de las bobinas y modelos de escáner, que hoy son rasgos por índice.

## 5. El cuello de botella

1. **Validación.** No hay ningún conjunto limpio, grande y puntuado como el test: los OOF usan etiquetas de informe y
   Gold tiene 58 estudios, contaminados para Raptor. Cualquier ajuste de pesos se decide en el público.
2. **Los cuatro hallazgos débiles.** La definición de experto diverge de los informes, y solo un modelo con supervisión
   o datos distintos para esos hallazgos los movería. Ningún recurso disponible en 3 días cumple eso.
3. **El tiempo de Kaggle** limita a un solo componente adicional.
