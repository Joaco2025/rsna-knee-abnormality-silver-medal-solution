# Ensamble Trío Base+D v2 (Kaggle Public Score 0.904)

Documentación técnica integral del ensamble multi-arquitectura con atención focal sobre cortes (*Slice-Attention Multi-Instance Learning*) para la competencia [RSNA Knee Abnormalities Detection](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection).

---

## A. Resumen Ejecutivo

El **Trío Base+D v2** representa la evolución del ensamble multietiqueta del proyecto, integrando por primera vez mecanismos de atención por corte (*Slice Attention*) combinados con representaciones convolucionales residuales y modernas de gran campo receptivo.

### Hitos Clave
* **Kaggle Public Score verificado:** **`0.904`** (Macro ROC-AUC sobre el conjunto de prueba público de la competencia).
* **Ganancia neta:** **`+0.0070`** sobre el ensamble baseline previo (`0.8970`) y **`+0.0290`** sobre el modelo fundacional individual (`0.8750`).
* **Validación Interna (`val_int`, 415 estudios):** **`0.8628`** Macro AUC frente a `0.8574` del baseline 50/50 ($\Delta = +0.0054$, $P(\Delta > 0) = 99.6\%$ en bootstrap pareado de 1,000 réplicas).
* **Fórmula de Fusión:**
  $$P_{\text{final}} = 0.25 \cdot P_{\text{ResNet34 Base}} + 0.25 \cdot P_{\text{ResNet34-D (Slice Att)}} + 0.50 \cdot P_{\text{ConvNeXt-Tiny Base}}$$
* **Notebook de Kaggle Oficial:**
  * **Título:** `RSNA Joako | Trio Base+D v2`
  * **URL:** [`https://www.kaggle.com/code/jqnx25/notebook923d143ff5?scriptVersionId=357922183`](https://www.kaggle.com/code/jqnx25/notebook923d143ff5?scriptVersionId=357922183)
  * **Versión de ejecución:** Script Version ID `357922183`.

---

## B. Historia del Diseño y Evolución del Ensamble

### 1. Contexto de Partida
El proyecto inició con un modelo base único ResNet-34 entrenado sobre proyecciones 2.5D con 4 vistas anatómicas, alcanzando un score público de `0.8750`. Posteriormente, la formulación del primer ensamble pareado (ResNet-34 al 50% + ConvNeXt-Tiny al 50%) demostró que la diversidad arquitectónica es el factor más potente para mitigar el error de generalización en diagnóstico musculoesquelético, elevando la puntuación a `0.8970` (+0.0220).

A pesar del éxito del ensamble de 2 modelos, un cuello de botella fundamental persistía en la agregación espacial/temporal de cortes dentro de cada serie.

### 2. Fundamento Clínico y Físico del Modelo D (Slice Attention)
En una resonancia magnética (RM) de rodilla estándar, cada serie volumétrica contiene entre 20 y 40 cortes que atraviesan la articulación:
* Patologías difusas o articulares amplias (como derrame articular / *Effusion* u osteoartritis avanzada) se manifiestan a lo largo de casi todo el volumen.
* **Patologías focales críticas** (como desgarros radiales o de raíz en meniscos, desinserciones del ligamento cruzado anterior [ACL], contusiones óseas trabeculares o fracturas corticales sutiles) están confinadas con frecuencia a **únicamente 1 a 3 cortes contiguos** de 3 a 4 mm de espesor.

En la arquitectura estándar `KneeNet`, los cortes de una vista se agregan mediante una combinación fija o un promedio uniforme. Este mecanismo sufre de **dilución de señal**: 13 a 15 cortes anatómicamente normales promedian y ocultan la respuesta de los 1 o 2 cortes con patología severa.

Para resolver este desafío, se concibió el **Modelo D (`slice_patho`)**: una arquitectura con atención multi-instancia (*Multiple Instance Learning*, MIL) donde cada una de las 12 patologías aprende una distribución de atención independiente sobre los 16 cortes seleccionados:
$$\alpha_{c, s} = \frac{\exp(w_c^T h_s)}{\sum_{s'=1}^S \exp(w_c^T h_{s'})}, \quad z_c = \sum_{s=1}^S \alpha_{c, s} h_s$$
donde $h_s$ es la representación latente del corte $s$, $w_c$ es el vector de proyección para la patología $c$, y $\alpha_{c, s}$ representa la probabilidad de relevancia diagnóstica del corte $s$ para dicha afección.

### 3. Justificación de la Ponderación Asimétrica 25 / 25 / 50
En lugar de un promedio uniforme ($33.3\% / 33.3\% / 33.3\%$), se implementó la combinación $0.25 / 0.25 / 0.50$ basada en las fortalezas relativas de los backbones:
1. **ConvNeXt-Tiny (50%):** Es el modelo de mayor capacidad representacional global y el especialista indiscutible en lesiones óseas complejas y ligamentos colaterales (Fracture: 0.8426 vs 0.7551; MCL: 0.8043 vs 0.7516). Reducir su peso al 33% deterioraba el desempeño en fracturas (caída de 0.8181 a 0.8071).
2. **ResNet-34 Base (25%):** Proporciona calibración conservadora y alta estabilidad en ligamentos cruzados (ACL: 0.9203).
3. **ResNet-34-D Slice Attention (25%):** Aporta detección focal y sensibilidad en desgarros y contusiones.

Al dividir el 50% correspondiente a la familia ResNet en dos mitades (25% base + 25% con atención), se inyecta la capacidad de resolución focal sin desplazar la calibración de ConvNeXt.

---

## C. Arquitectura Técnica y Flujo de Inferencia

### 1. Miembros del Ensamble

| Componente | Modelo | Arquitectura Base | Mecanismo de Agregación | Checkpoint | Peso |
|:---|:---|:---|:---|:---|:---:|
| **Modelo 1** | ResNet-34 Base | ResNet-34 (ImageNet) | Pooling temporal compartido (`shared/concat`) | Época 7 | **0.25** |
| **Modelo 2** | ResNet-34-D | ResNet-34 (ImageNet) | Atención MIL por patología (`patho/concat`) | Época 6 | **0.25** |
| **Modelo 3** | ConvNeXt-Tiny Base | ConvNeXt-Tiny (ImageNet) | Convolución 7x7 + pooling compartido (`shared/concat`) | Época 5 | **0.50** |

```
                               Estudio RM Multivista
                         (Axial, Sagital, Coronal DICOM)
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │    ResNet-34      │      │    ResNet-34-D    │      │   ConvNeXt-Tiny   │
   │    Base (25%)     │      │  Slice Att (25%)  │      │    Base (50%)     │
   │  Pooling Estándar │      │  Atención MIL 12p │      │ Kernel 7x7 Amplio │
   └────────┬──────────┘      └────────┬──────────┘      └────────┬──────────┘
            │ Probabilidad             │ Probabilidad             │ Probabilidad
            │ $P_1$                    │ $P_2$                    │ $P_3$
            └────────────────────┐     │     ┌────────────────────┘
                                 ▼     ▼     ▼
                     ┌───────────────────────────────────────┐
                     │          Fusión Ponderada             │
                     │  $P = 0.25 P_1 + 0.25 P_2 + 0.50 P_3$ │
                     └───────────────────┬───────────────────┘
                                         ▼
                             Predicciones Finales
                             (12 Macro ROC-AUC)
```

### 2. Pipeline de Inferencia Offline

1. **Lectura y Descompresión Segura:**
   * En el entorno aislado de Kaggle (sin internet), el paquete instala de forma local ruedas preempaquetadas (`wheels/`): `pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`.
   * Manejo robusto de compresión médica: soporta JPEG Baseline, JPEG Lossless (1.2.840.10008.1.2.4.57/70) y JPEG 2000.
2. **Estandarización Geométrica:**
   * Las rebanadas de cada serie se ordenan estrictamente en el espacio 3D por `ImagePositionPatient[2]` normalizado, garantizando continuidad anatómica longitudinal.
   * Se asignan a 4 ranuras anatómicas (`Sagittal FS`, `Sagittal no-FS`, `Coronal FS`, `Axial FS`).
   * Se extraen 16 cortes equidistantes por ranura con recorte central a $288 \times 288$ píxeles y normalización de canales ImageNet.
3. **Inferencia por Lotes con Precisión Mixta:**
   * Evaluación secuencial optimizada con `torch.amp.autocast` en GPU.
   * Memoria `torch.channels_last` para aceleración convolucional.
4. **Fusión Probabilística y Post-Procesamiento:**
   * Aplicación de sigmoide por modelo y combinación lineal ponderada.
   * Control estricto de integridad: verificación de ausencia de valores `NaN`, acotamiento en $[0, 1]$ y correspondencia exacta de 13 columnas con `sample_submission.csv`.

---

## D. Análisis Técnico del Salto 0.897 -> 0.904

El incremento de **`+0.0070`** en la tabla pública de Kaggle refleja principios sólidos de machine learning médico:

1. **Complementariedad Ortogonal:**
   * ResNet-34 y ConvNeXt-Tiny procesan patrones visuales con campos receptivos muy diferentes (convoluciones clásicas 3x3 vs bloques invertidos con kernel 7x7).
   * El Modelo D añade una **dimensión ortogonal de agregación**: no altera la extracción de características 2D básica, sino el mapa de atención a través de la dimensión de rebanada $Z$. Esto produce predicciones descorrelacionadas frente al modelo base ($r \approx 0.89 - 0.91$).
2. **Sensibilidad en Patologías Focales:**
   * En validación interna, las mayores ganancias se concentraron precisamente en diagnósticos caracterizados por alteraciones localizadas:
     * **Baker's Cyst:** `0.9216` $\to$ **`0.9331`** ($\Delta = +0.0115$)
     * **Bone Contusion:** `0.7911` $\to$ **`0.8069`** ($\Delta = +0.0158$)
     * **Lateral Meniscus Tear:** `0.8254` $\to$ **`0.8411`** ($\Delta = +0.0157$)
     * **MCL Sprain/Tear:** `0.7988` $\to$ **`0.8102`** ($\Delta = +0.0114$)
     * **ACL Tear:** `0.8988` $\to$ **`0.9104`** ($\Delta = +0.0116$)
3. **Discrepancia de Escala (+0.0054 en `val_int` vs +0.0070 en Kaggle):**
   * En `val_int`, el cálculo evalúa 415 estudios contra pseudoetiquetas derivadas con LLM.
   * El test set de Kaggle está evaluado contra consensos diagnósticos radiológicos de alta calidad. Cuando un modelo adquiere capacidad focal genuina (como identificar correctamente un pequeño desgarro meniscal o una fractura por avulsión), su ordenamiento relativo en datos limpios mejora con mayor fuerza que en pseudoetiquetas con ruido residual.

---

## E. Comparativa Exhaustiva de Alternativas

### 1. Resumen de Modelos y Ensambles

| Solución / Experimento | Esquema de Fusión | `val_int` Macro AUC | IC 95% Bootstrap | $P(\Delta > 0)$ vs Baseline 50/50 | Kaggle Public Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **ResNet-34 Base** (Owen) | Modelo Individual | 0.8435 | [0.8264, 0.8640] | — | 0.8750 |
| **ConvNeXt-Tiny Base** (Owen) | Modelo Individual | 0.8518 | [0.8333, 0.8716] | — | — |
| **ResNet-34-D** (Joako) | Slice Attention Indiv. | 0.8509* | [0.8320, 0.8690] | — | — |
| **Ensamble 50/50 A+B** | $0.50 \text{ R34} + 0.50 \text{ CNX}$ | 0.8574 | [0.8401, 0.8772] | Baseline | 0.8970 |
| **Trío Uniforme** | $\frac{1}{3} \text{R34} + \frac{1}{3} \text{D} + \frac{1}{3} \text{CNX}$ | 0.8620 | [0.8451, 0.8812] | 98.5% | — |
| **Trío Base+D v2** | **$0.25 \text{ R34} + 0.25 \text{ D} + 0.50 \text{ CNX}$** | **0.8628** | **[0.8462, 0.8821]** | **99.6%** | **0.9040** |

*\*Nota: El AUC individual del Modelo D en validación interna contra las etiquetas benchmark de referencia es 0.8509 (y alcanzó 0.8633 sobre etiquetas de supervisión híbrida).*

### 2. Desglose Comparativo por Patología (`val_int`, 415 estudios)

| Patología | Casos (+) | ResNet34 Base | ConvNeXt Base | Modelo D (Slice Att) | Baseline 50/50 | Trío Uniforme (33/33/33) | **Trío Base+D v2 (25/25/50)** | $\Delta$ vs 50/50 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baker's** | 50 | 0.9014 | 0.9299 | 0.9285 | 0.9216 | 0.9299 | **0.9331** | **+0.0115** |
| **Contusion** | 32 | 0.7565 | 0.8055 | 0.8060 | 0.7911 | 0.8045 | **0.8069** | **+0.0158** |
| **Lateral Meniscus** | 65 | 0.8155 | 0.8213 | 0.8258 | 0.8254 | **0.8435** | 0.8411 | **+0.0157** |
| **MCL** | 37 | 0.7516 | 0.8043 | 0.7714 | 0.7988 | 0.8042 | **0.8102** | **+0.0114** |
| **ACL** | 50 | 0.9203 | 0.8686 | 0.9199 | 0.8988 | **0.9165** | 0.9104 | **+0.0116** |
| **Medial Meniscus** | 183 | 0.9153 | 0.9106 | 0.9045 | 0.9231 | **0.9247** | 0.9244 | **+0.0013** |
| **Lateral OA** | 71 | 0.8406 | 0.8168 | 0.8415 | 0.8385 | **0.8417** | 0.8397 | **+0.0012** |
| **Medial OA** | 127 | 0.8749 | 0.8502 | 0.8672 | 0.8676 | **0.8711** | 0.8688 | **+0.0012** |
| **PF OA** | 112 | 0.8623 | 0.8325 | 0.8548 | 0.8561 | **0.8601** | 0.8564 | **+0.0003** |
| **Effusion** | 243 | 0.9051 | 0.8991 | 0.8990 | 0.9077 | **0.9084** | 0.9080 | **+0.0003** |
| **Synovitis** | 71 | 0.8238 | 0.8398 | 0.8131 | **0.8407** | 0.8327 | 0.8363 | -0.0044 |
| **Fracture** | 34 | 0.7551 | **0.8426** | 0.7794 | **0.8195** | 0.8071 | 0.8181 | -0.0014 |
| **Macro AUC** | — | 0.8435 | 0.8518 | 0.8509 | 0.8574 | 0.8620 | **0.8628** | **+0.0054** |

### 3. Por Qué la Ponderación 25/25/50 Superó al Promedio Uniforme
En la tabla anterior se observa claramente la razón:
* En **Fracture**, ConvNeXt alcanza un sobresaliente `0.8426`. Reducir su contribución a $33\%$ en el promedio uniforme causó una severa caída a `0.8071` (pérdida de más de 120 puntos base de AUC). En cambio, mantener el peso de $50\%$ en el Trío v2 preservó el desempeño en `0.8181`.
* En patologías complejas como **MCL** y **Baker's**, el Trío 25/25/50 superó holgadamente tanto al baseline 50/50 como al promedio uniforme.

---

## F. Limitaciones Conocidas y Gestión de Riesgos

1. **Riesgo de Sobreajuste al Leaderboard Público:**
   * El Public Score de Kaggle evalúa aproximadamente el $33\%$ del test total (~430 estudios), mientras que el Private Score final evaluará el $67\%$ restante (~870 estudios).
   * Para mitigar el riesgo de persecución de ruido, **no se seleccionaron los pesos mediante búsqueda exhaustiva de cuadrícula sobre el test**. La ponderación 25/25/50 fue ratificada por su consistencia teórica y por el remuestreo bootstrap sobre `val_int` ($P = 99.6\%$).
2. **Dependencia Arquitectónica (`modelo_att.py`):**
   * A diferencia del ensamble base que solo requería la definición tradicional de `KneeNet`, el Modelo D requiere la clase `KneeNetAtt` con soporte para el parámetro `agg_slice='patho'`.
   * **Mitigación:** Se empaquetó e incluyó formalmente `rsna/modelo_att.py` en el bundle de Kaggle, y el script `rsna/inferir.py` implementa despacho dinámico de arquitectura leyendo las claves del checkpoint (`agg_slice`, `agg_view`), garantizando compatibilidad retrospectiva total.
3. **Presupuesto Temporal de Inferencia:**
   * Evaluar 3 modelos en lugar de 2 incrementa el tiempo de inferencia por estudio en un ~35%.
   * **Medición observada:** Para 1,300 estudios en GPU P100/T4, el pipeline completo toma aproximadamente **13 a 15 minutos**, lo cual representa únicamente el **$2.6\%$** de la ventana de 9 horas de Kaggle.

---

## G. Reproducibilidad y Manifiesto de Artefactos

### 1. Manifiesto de Checkpoints

| Checkpoint en Paquete | Origen SLURM / Época | Tamaño | SHA-256 | Peso |
|:---|:---:|:---:|:---|:---:|
| `resnet34_job2080214_ep7.pt` | Job 2080214 (Ep. 7) | 86 MB (89,776,448 B) | `9452d38ae4d884f6222dcbd63402d6625f0fef56b7b6f566e60ac27b0337ea01` | 0.25 |
| `resnet34_d_job2084614_ep6.pt` | Job 2084614 (Ep. 6) | 86 MB (89,782,272 B) | `638c9624a32b614b0014138413ae9aaf973f7c33c7178e0d0dead454a90cbf33` | 0.25 |
| `convnext_tiny_job2080215_ep5.pt`| Job 2080215 (Ep. 5) | 113 MB (118,068,302 B) | `0d927cdb84f090333dfec9a4d88947f1d6ada4d3f5b772415a2ccb8a95b11d9d` | 0.50 |

### 2. Estructura del Dataset Privado en Kaggle
El paquete autónomo (`candidato_trio_d_paquete.zip`, 281 MB) contiene:
```text
candidato_trio_d/
├── dataset-metadata.json
├── inferencia.ipynb
├── pesos/
│   ├── convnext_tiny_job2080215_ep5.pt
│   ├── resnet34_d_job2084614_ep6.pt
│   └── resnet34_job2080214_ep7.pt
├── rsna/
│   ├── __init__.py
│   ├── inferir.py
│   ├── modelo.py
│   ├── modelo_att.py
│   └── preproc.py
└── wheels/
    ├── pydicom-3.0.2-py3-none-any.whl
    ├── pylibjpeg-2.1.0-py3-none-any.whl
    ├── pylibjpeg_libjpeg-2.4.0-*.whl
    └── pylibjpeg_openjpeg-2.6.0-*.whl
```

### 3. Comando de Reproducción de Inferencia Local
```bash
python rsna/inferir.py \
    --data /ruta/datos/competencia \
    --split test \
    --checkpoints pesos/resnet34_job2080214_ep7.pt \
                  pesos/resnet34_d_job2084614_ep6.pt \
                  pesos/convnext_tiny_job2080215_ep5.pt \
    --weights 0.25 0.25 0.50 \
    --out submission.csv \
    --batch 4
```

---

## H. Estado Oficial y Próximos Pasos

1. **Estado Oficial de Cierre:**
   * El ensamble **Trío Base+D v2 (Kaggle Public Score 0.904)** queda cerrado y formalmente documentado como la mejor solución verificada del repositorio para esta etapa.
2. **Línea de Trabajo Futura:**
   * En el clúster se han desarrollado exploraciones adicionales que incorporan un cuarto modelo con supervisión híbrida (ConvNeXt-Tiny híbrido, Job 2084609), configurando un ensamble cuádruple denominado **ABCD**.
   * Dicha variante pertenece a una línea de experimentación avanzada que se mantendrá en reserva interna y se abordará en una fase posterior.
