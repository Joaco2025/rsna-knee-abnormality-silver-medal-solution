# Lecciones Consolidadas del Proyecto RSNA Knee

Este documento recopila reglas empíricas, decisiones metodológicas firmes y restricciones biológicas/técnicas descubiertas en el proyecto. Consulta este archivo antes de proponer cualquier hipótesis o diseño experimental.

---

## 1. Datos y Etiquetas

1. **Estructura de las etiquetas**:
   - De 4,407 estudios, únicamente **58 estudios Gold** cuentan con etiquetas oficiales completas.
   - Los restantes 4,349 estudios tienen informes radiológicos con etiquetas ausentes (`NaN`).
   - **Regla estricta**: Nunca imputar `NaN` como cero (negativo). Los estudios no evaluados o no mencionados deben enmascararse (`w = 0`).
2. **Leakage en `val_oficial`**:
   - Los 58 estudios Gold fueron utilizados para calibrar el mapeo bayesiano de pseudoetiquetas generadas con Qwen2.5:72B (`preproc/etiquetas_v1_2079203.parquet`).
   - **Consecuencia**: `val_oficial` **no es un conjunto independiente** para ajuste de hiperparámetros, selección de checkpoints o pesos de ensamble. Su uso para optimización induce sobreajuste y fuga de datos.
3. **Métrica y conjunto de decisión interno**:
   - La selección de modelos y pesos debe basarse en `val_int` (415 estudios con pseudoetiquetas, con máscaras de ausencia).
   - Siempre reportar por separado:
     - `val_int` macro AUC (415 casos).
     - `val_oficial` macro AUC (58 casos Gold, solo referencia descriptiva).
     - Score público de Kaggle (test oculto, ~33% del conjunto test).

---

## 2. Preprocesamiento e Imágenes DICOM

1. **Tensores 4D**:
   - Dimensiones estándar: `(4, 32, 320, 320)` en tipo `uint8`.
   - Cuatro slots anatómicos:
     1. Sagital FS (Fat-Suppressed)
     2. Sagital no-FS
     3. Coronal FS
     4. Axial FS
2. **Ordenamiento geométrico obligatorio**:
   - Las rebanadas DICOM deben ordenarse por posición geométrica en el espacio 3D (`ImagePositionPatient`), **nunca por nombre de archivo ni por `InstanceNumber` ciego**, ya que los escáneres hospitalarios con frecuencia alteran el orden de adquisición.
3. **Restricción anatómica de Data Augmentation**:
   - **Prohibido realizar flips horizontales ingenuos**: invertir horizontalmente una resonancia de rodilla invierte la semántica medial/lateral (ej. un desgarro de menisco medial pasaría a parecer lateral), destruyendo la correlación anatómica con las patologías.
4. **Dataset crudo en Lustre**:
   - `/lustre/proyectos/p082/kaggle_data` (~570 GB, ~820k archivos) es **estrictamente de solo lectura**. Jamás ejecutar recorridos recursivos (`find`, `du`, `ls -R`) sobre él, ya que satura el servidor de metadatos de Lustre.

---

## 3. Modelos y Ensambles

1. **Rendimiento de Backbones individuales**:
   - **ResNet34** (Job `2080214`, ep 7): `val_int` macro AUC **0.8435**, score público Kaggle **0.875**.
   - **ConvNeXt-Tiny** (Job `2080215`, ep 5): `val_int` macro AUC **0.8518**.
2. **Ensamble Base (50/50)**:
   - Job `2081432`: $P_{\text{ensamble}} = 0.5 \cdot P_{\text{resnet}} + 0.5 \cdot P_{\text{convnext}}$.
   - `val_int` macro AUC **0.8574** (+0.0139 vs ResNet34, +0.0056 vs ConvNeXt).
   - Score público Kaggle verificado: **0.897** (+0.022 de mejora neta sobre el baseline público).
3. **Regla de validación de ensambles**:
   - No adoptar ninguna combinación de pesos $\alpha$ basada únicamente en el máximo puntual sobre los 415 estudios de `val_int`.
   - Exigir siempre **bootstrap pareado (mínimo 500 réplicas, IC 95%)** y corroborar que el intervalo de mejora $\Delta$ se mantenga estrictamente positivo en la gran mayoría de las réplicas.
4. **Supervisión Híbrida y Diversidad de Régimen**:
   - Entrenar ambos backbones (ResNet34 y ConvNeXt-Tiny) con pseudoetiquetas híbridas (`etiquetas_candidatas_v2_hibridas.parquet`) mejora individualmente a los modelos (ResNet34 0.8435 $\to$ 0.8474, ConvNeXt-Tiny 0.8518 $\to$ 0.8582, $\Delta = +0.0064$, $P=91.4\%$) y genera diversidad de sesgo respecto al régimen original.
5. **Ensamble Cuádruple (ABCD)**:
   - Combinar los 4 modelos (ResNet34 base + ConvNeXt base + ResNet34 híbrido + ConvNeXt híbrido al 25% c/u) alcanza `val_int` macro AUC **0.8653** ($\Delta = \mathbf{+0.0079}$ sobre el baseline 50/50, IC 95% `[+0.0032, +0.0124]`, **100% de réplicas bootstrap positivas**).
   - Mejora 11 de 12 patologías sin añadir complejidad arquitectónica (cero riesgo en inferencia de Kaggle).
6. **Módulos de Atención sobre Cortes (MIL) y Ensamble Trío Base+D v2**:
   - **Solución campeona Kaggle 0.904**: La atención MIL por patología sobre cortes (`slice_patho`, job `2084614`) resuelve la dilución de señal en lesiones focales de 1-3 cortes (Baker's, Contusion, desgarros meniscales).
   - **Superioridad de la ponderación asimétrica (25/25/50)**: Mantener un 50% en ConvNeXt-Tiny protege su capacidad especializada en fracturas (0.8426 individual vs 0.8071 en promedio uniforme 33/33/33, una pérdida crítica de >120 pb de AUC). Dividir el 50% restante entre ResNet34 Base (25%) y ResNet34-D (25%) añade sensibilidad focal sin degradar la calibración ni requerir reentrenamientos.
   - **Magnificación en datos limpios**: La ganancia en `val_int` (+0.0054 con pseudoetiquetas ruidosas) se tradujo en un salto aún mayor en Kaggle público (+0.0070, alcanzando 0.9040), demostrando que la atención focal aporta señal diagnóstica real en consensos radiológicos de alta calidad.
7. **Perspectiva hacia Ensamble Cuádruple (ABCD)**:
   - La combinación con un cuarto modelo entrenado con supervisión híbrida (ConvNeXt híbrido) representa la siguiente cota de mejora teórica en validación interna (0.8653, +0.0079), manteniéndose como línea de trabajo futura a publicar en una fase posterior.


---

## 4. Operación en Clúster Yuca (HPC)

1. **Login node (`login-01`)**:
   - Es un recurso compartido de gestión. Prohibido ejecutar entrenamientos, preprocesamientos pesados o procesos persistentes en background (`nohup`, `&`, demonios).
2. **Hardware y entorno**:
   - GPUs: AMD Instinct MI210 (`gfx90a`, 64 GB VRAM) con **ROCm 7.2.x** (no CUDA).
   - PyTorch debe corresponder al build de ROCm del clúster.
3. **SLURM y monitoreo**:
   - Salidas obligatoriamente aisladas en subdirectorios nuevos: `runs_joako/<JOB_ID>/` o `runs/<JOB_ID>/`.
   - Monitoreo pasivo con intervalo $\ge 60$ segundos entre consultas a `squeue`/`sacct`.
   - Si un job está en estado pendiente con `(QOSGrpGRES)`, significa contención de cuota de GPUs en la cuenta `p082`. No reenviar ni cancelar; esperar liberación de recursos.
4. **Tracking de experimentos**:
   - Cero dependencias externas o conexiones SaaS (red UniSon filtrada por Fortinet SSL).
   - Usar siempre MLflow local con base SQLite: `sqlite:////lustre/proyectos/p082/mlflow/mlflow_joako.db` y artefactos en `/lustre/proyectos/p082/mlflow/artifacts_joako`.

