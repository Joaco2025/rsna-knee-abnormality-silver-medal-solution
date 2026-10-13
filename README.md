# RSNA Knee Abnormalities Detection

Repositorio para la competencia de Kaggle [RSNA Knee Abnormalities Detection](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection).

El objetivo es predecir la probabilidad de 12 anomalías y patologías musculoesqueléticas a partir de estudios de resonancia magnética (RM) de rodilla en tres planos anatómicos (axial, coronal y sagital).

---

## 👥 Créditos y Evolución del Proyecto

Este repositorio documenta el trabajo colaborativo estructurado en dos fases:

* **Trabajo original y modelos base (Owen):**
  * Análisis exploratorio de datos (EDA), inventario y control geométrico de las 24,371 series y 819,078 cortes DICOM.
  * Pipeline de derivación de pseudoetiquetas clínicas a partir de 4,407 informes radiológicos en texto libre mediante LLM local (`qwen2.5:72b`), superando la limitación de disponer de solo 58 estudios con etiquetas ground truth completas.
  * Diseño e implementación de la arquitectura de red `KneeNet` en PyTorch.
  * **Entrenamiento de los modelos base individuales:**
    * **ResNet-34** (checkpoint época 7, `best_oficial.pt`), con el que se obtuvo la puntuación baseline de **0.875** en el leaderboard público de Kaggle.
    * **ConvNeXt-Tiny** (checkpoint época 5, `best_oficial.pt`).
* **Integración, evaluación y despliegue de ensambles (Joako / FernyBoy):**
  * Aislamiento determinista de infraestructura personal (rutas de salida en `runs_joako`, logs, canalizaciones SLURM, base de datos de MLflow aislada y caché de GPU).
  * Formulación de la hipótesis del ensamble multiarquitectura (combinación 50/50 de ResNet-34 y ConvNeXt-Tiny para integrar representaciones residuales clásicas y de kernel amplio).
  * **Diseño e implementación del evaluador comparativo pareado** (`rsna/evaluar_ensamble.py` y `jobs/evaluar_ensamble.slurm`), con métricas macro por split, desglose por patología, matrices de correlación y análisis de incertidumbre mediante remuestreo bootstrap (500 réplicas, IC 95%).
  * Verificación empírica local de la mejora del ensamble tanto en validación interna como en el conjunto oficial.
  * **Construcción y validación del paquete autónomo de inferencia offline** (`inferencia.ipynb`, dependencias DICOM offline, validaciones de integridad en `rsna/inferir.py` y smoke test).
  * Generación del primer envío oficial a Kaggle logrando una puntuación de **0.897** (**+0.022** sobre el baseline).
  * **Diseño e integración del ensamble Trío Base+D v2:** incorporación de atención multi-instancia por corte (*Slice Attention* / `KneeNetAtt`) combinada con ponderación asimétrica ($0.25 / 0.25 / 0.50$), alcanzando la puntuación verificada de **0.904** en el leaderboard público de Kaggle (**+0.029** sobre el baseline original). Ver documentación detallada en [`docs/trio_base_d.md`](docs/trio_base_d.md).

---

## 🏆 Resumen de Resultados

### 1. Puntuación Oficial en Kaggle (Test Oculto de la Competencia, ~1,300 estudios)
La evaluación ciega oficial en la plataforma de Kaggle confirmó la generalización de las soluciones:

| Solución | Backbone / Composición | Kaggle Public Score (Macro AUC) | Delta vs Baseline |
|---|---|:---:|:---:|
| **Baseline (Owen)** | ResNet-34 (época 7) | 0.8750 | — |
| **Ensamble Base (Joako)** | ResNet-34 (50%) + ConvNeXt-Tiny (50%) | 0.8970 | +0.0220 |
| **Trío Base+D v2 (Joako)** | **ResNet-34 (25%) + ResNet-34-D Slice Att (25%) + ConvNeXt-Tiny (50%)** | **0.9040** | **+0.0290** (+0.0070 vs Ensamble) |

> Para un análisis exhaustivo del Trío Base+D v2, gráficos, fundamentación médica y comparativa de alternativas, consulta el documento técnico [`docs/trio_base_d.md`](docs/trio_base_d.md).

---

### 2. Evaluación Comparativa Local Pareada

Se evaluaron ambos modelos base y su ensamble sobre dos particiones locales:

* **Validación Interna (`val_int`, 415 estudios en Fold 0):** Casos **nunca vistos** durante el entrenamiento de los pesos. Las etiquetas son pseudoetiquetas derivadas de reportes con LLM, filtradas para casos con evidencia clínica ($t \ge 0.5$ con peso $w > 0$).
* **Conjunto Oficial (`val_oficial`, 58 estudios Gold):** Anotaciones ground truth completas de radiólogos expertos de RSNA.
  > [!WARNING]
  > **Advertencia sobre `val_oficial`:** Este conjunto **no constituye una validación independiente**, ya que estos mismos 58 estudios participaron previamente en la calibración y ajuste de umbrales para la extracción de pseudoetiquetas. Además, su tamaño muestral reducido ($N=58$) amplía la incertidumbre estadística; sus resultados deben interpretarse únicamente como chequeo de coherencia histórica.

| Partición | Métrica / Modelo | ResNet-34 | ConvNeXt-Tiny | Ensamble (50/50) | $\Delta$ Ensamble vs ResNet | Prob. Mejora (Bootstrap) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **`val_int`**<br>(N=415) | **Macro AUC**<br>IC 95% | 0.8435<br>[0.8264, 0.8640] | 0.8518<br>[0.8333, 0.8716] | **0.8574**<br>[0.8401, 0.8772] | **+0.0139**<br>[+0.0074, +0.0201] | **100.0%** (vs ResNet)<br>**97.6%** (vs ConvNeXt) |
| **`val_oficial`**<br>(N=58 Gold) | **Macro AUC**<br>IC 95% | 0.8581<br>[0.8233, 0.8898] | 0.8769<br>[0.8460, 0.9091] | **0.8793**<br>[0.8485, 0.9113] | **+0.0211**<br>[+0.0079, +0.0347] | **100.0%** (vs ResNet)<br>**66.4%** (vs ConvNeXt) |

---

### 3. Detalle por Hallazgo en Validación Interna (`val_int`)
Muestra cómo el ensamble combina la estabilidad de ResNet-34 en ligamentos con la mayor potencia de ConvNeXt-Tiny en patologías óseas y colaterales:

| Patología | Casos (+) / Evaluable | AUC ResNet-34 | AUC ConvNeXt | AUC Ensamble | Ganancia ($\Delta$) | Correlación ($r$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fracture** | 34 / 361 | 0.7551 | 0.8426 | **0.8195** | **+0.0644** | 0.8037 |
| **MCL** | 37 / 389 | 0.7516 | 0.8043 | **0.7988** | **+0.0471** | 0.7448 |
| **Contusion** | 32 / 370 | 0.7565 | 0.8055 | **0.7911** | **+0.0346** | 0.8882 |
| **Baker's** | 50 / 238 | 0.9014 | 0.9299 | **0.9216** | **+0.0202** | 0.8885 |
| **Synovitis** | 71 / 119 | 0.8238 | 0.8398 | **0.8407** | **+0.0169** | 0.9176 |
| **Lateral Meniscus** | 65 / 384 | 0.8155 | 0.8213 | **0.8254** | **+0.0098** | 0.8417 |
| **Medial Meniscus** | 183 / 394 | 0.9153 | 0.9106 | **0.9231** | **+0.0078** | 0.8919 |
| **Effusion** | 243 / 387 | 0.9051 | 0.8991 | **0.9077** | **+0.0026** | 0.9474 |
| **Lateral OA** | 71 / 348 | 0.8406 | 0.8168 | **0.8385** | -0.0021 | 0.9062 |
| **PF OA** | 112 / 351 | 0.8623 | 0.8325 | **0.8561** | -0.0062 | 0.9139 |
| **Medial OA** | 127 / 349 | 0.8749 | 0.8502 | **0.8676** | -0.0073 | 0.9410 |
| **ACL** | 50 / 388 | 0.9203 | 0.8686 | **0.8988** | -0.0214 | 0.8881 |

---

## 🧠 Arquitectura de Inferencia y Ensamble

```
                     Estudio RM (Axial, Coronal, Sagital)
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
                             Predicciones Finales (12 AUC)
```

1. **Entrada Multi-Corte y Multi-Vista:** Para cada estudio se cargan 16 cortes seleccionados a lo largo del volumen en los cuatro planos/secuencias anatómicas (`Sagittal FS`, `Sagittal no-FS`, `Coronal FS`, `Axial FS`) recortados al centro a $288 \times 288$.
2. **Backbones Heterogéneos y Especialización:**
   * **ResNet-34 Base:** Estabilidad residual clásica y alta sensibilidad en ligamentos (ACL: 0.9203).
   * **ResNet-34-D (Slice Attention):** Atención multi-instancia (*MIL*) independiente para cada patología, seleccionando cortes con lesiones focales (Baker's: 0.9285, Contusion: 0.8060).
   * **ConvNeXt-Tiny Base:** Gran capacidad representacional con kernels 7x7, destacando en fracturas (0.8426) y colaterales (0.8043).
3. **Fusión Asimétrica Calibrada:** Ponderación $0.25 / 0.25 / 0.50$ que preserva el peso predominante de ConvNeXt en patologías óseas/ligamentosas mientras divide la familia ResNet para incorporar resolución focal sin distorsionar la calibración global.

---

## 📦 Pipeline de Inferencia Offline para Kaggle

El entorno de submission de Kaggle opera **sin acceso a internet**, con un límite de **9 horas de ejecución** y lectura de series DICOM comprimidas (JPEG Lossless / JPEG 2000).

El paquete de inferencia (`inferencia.ipynb` y `rsna/inferir.py`) está diseñado de forma autónoma:
* **Dependencias Offline:** Instalación local desde ruedas `.whl` (`pydicom`, `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `python-gdcm`).
* **Validación de Integridad:** Comprobaciones dinámicas (`assert labels`, `assert backbone`) para garantizar el orden de clases y arquitecturas soportadas.
* **Optimización de Ejecución:** Lectura y descompresión DICOM directa, ordenamiento espacial por `ImagePositionPatient[2]`, modo de memoria `channels_last` y precisión mixta (`torch.amp.autocast`).

---

## 🛠️ Estructura del Repositorio

```text
├── eda/                          # Exploración de metadatos, cabeceras DICOM y verificación geométrica
├── etiquetas/                    # Extracción de pseudoetiquetas con LLM y consolidación
├── imagenes/                     # Pruebas preliminares con modelos multimodales
├── jobs/                         # Scripts de ejecución por lotes SLURM para cluster HPC
│   ├── entrenar.slurm            # Pipeline de entrenamiento parametrizable
│   ├── inferir.slurm             # Inferencia y generación de predicciones
│   ├── evaluar_ensamble.slurm    # Evaluación comparativa pareada
│   └── ...
├── rsna/                         # Código fuente del modelo y utilidades
│   ├── dataset.py                # DataLoader de tensores volumétricos
│   ├── modelo.py                 # Definición de KneeNet y arquitecturas soportadas
│   ├── entrenar.py               # Bucle de entrenamiento multietiqueta (BCE con pesos)
│   ├── inferir.py                # Motor de inferencia autónomo con aserciones de consistencia
│   └── evaluar_ensamble.py       # Evaluador comparativo pareado y análisis bootstrap
├── inferencia.ipynb              # Notebook de inferencia para envío a Kaggle
└── .gitignore                    # Reglas de exclusión de datos médicos, checkpoints y ruedas
```

---

## 🚀 Requisitos y Reproducibilidad

### Dependencias Principales
* Python $\ge 3.10$
* PyTorch $\ge 2.0$ (con aceleración ROCm o CUDA)
* `pydicom`, `pylibjpeg`, `pandas`, `scikit-learn`, `numpy`

### Ejecución de la Evaluación Pareada
```bash
python rsna/evaluar_ensamble.py \
    --data path/to/preproc/v1_train \
    --labels path/to/labels.parquet \
    --ckpt-resnet path/to/resnet34/best_oficial.pt \
    --ckpt-convnext path/to/convnext/best_oficial.pt \
    --out ./resultados_evaluacion \
    --batch 8
```

---

## 🔮 Trabajo Futuro y Mejoras Experimentales Seguras
1. **Aumentos y TTA Invariantes a Lateralidad:**
   * Jitter sutil de contraste y ventana (window/level), simulando la variabilidad de adquisición entre fabricantes sin alterar la anatomía.
   * Micro-rotaciones en el plano ($\pm 3^\circ$ a $\pm 5^\circ$) y pequeñas traslaciones ($\pm 4-8$ px) sin invertir ejes.
   *(Nota metodológica: Se evitan deliberadamente flips horizontales ingenuos, ya que invierten la anatomía medial/lateral alterando patologías de meniscos y ligamentos).*
2. **Ponderación Óptima por Patología (Weighted Blending):**
   * Optimizar pesos específicos $w_c \in [0, 1]$ por patología mediante regresión logística sobre `val_int` (e.g. mayor peso a ResNet en ACL; mayor a ConvNeXt en Fracture/MCL).
3. **Calibración Post-Hoc de Probabilidades:**
   * Evaluar empíricamente técnicas de calibración como Platt scaling o regresión isotónica por patología para determinar si mejoran la fiabilidad probabilística o si introducen distorsiones en el ordenamiento relativo.
4. **Mecanismo de Atención entre Cortes (Slice Attention):**
   * *Validado e implementado:* La atención MIL por patología sobre cortes (`slice_patho`) demostró ser altamente efectiva en lesiones focales, impulsando el ensamble Trío Base+D v2 a **0.904** en Kaggle (ver [`docs/trio_base_d.md`](docs/trio_base_d.md)). Como extensión futura, se exploran arquitecturas volumétricas 3D directas e inflado temporal de kernels.
5. **Diversidad Arquitectónica Adicional:**
   * Evaluar un backbone Transformer (e.g. Swin Transformer) para enriquecer el ensamble con un sesgo inductivo complementario a las CNNs.
