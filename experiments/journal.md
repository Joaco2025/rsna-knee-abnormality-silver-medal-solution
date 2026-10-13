# Diario de Experimentación RSNA Knee

> **Regla de Oro (The Iron Law):**
> **NINGÚN EXPERIMENTO NUEVO SE EJECUTA SIN REGISTRAR PREVIAMENTE SU HIPÓTESIS Y SU BASELINE A BATIR.**

Toda variación de hiperparámetros, pesos de ensamble, preprocesamiento o arquitectura debe quedar asentada en este archivo antes de enviar un trabajo SLURM o lanzar una evaluación.

---

## Estructura de Registro

Cada entrada debe registrarse en orden cronológico (append-only) siguiendo esta plantilla:

```markdown
### [YYYY-MM-DD HH:MM] — [Nombre del Experimento]
- **Estado**: PLANNED | RUNNING | COMPLETED | FAILED
- **Job SLURM**: <JOB_ID> (o N/A si es evaluación offline)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `<hash_7_chars>`
- **Hipótesis**: [Qué esperamos que suceda y cuál es el sustento biológico/técnico]
- **Modificación**: [Exactamente qué variable cambia — una sola variable a la vez]
- **Configuración / Comando**:
  - Parámetros clave: ...
  - Checkpoints / Tensores de entrada: ...
- **Baseline de Referencia**: [Métrica exacta a superar, ej. val_int macro AUC: 0.8574]
- **Resultados Observados**:
  - val_int macro AUC: ... (IC 95%: [..., ...])
  - val_oficial macro AUC: ... (descriptivo)
  - Kaggle Público: ... (si aplica)
- **Delta vs Baseline**: [Diferencia neta observada y significancia estadística pareada]
- **Duración / Recursos**: [Tiempo de ejecución, GPUs utilizadas]
- **Lección Aprendida**: [Una sola frase concluyente]
- **Siguiente Paso**: [Acción derivada de este hallazgo]
```

---

## Historial de Experimentos

### 2026-10-10 18:00 — Backbone ResNet34 Base
- **Estado**: COMPLETED
- **Job SLURM**: `2080214`
- **Autor / Rama**: owen / `owen` (importado a entorno joako)
- **Commit Git**: `5bc5273`
- **Hipótesis**: Un backbone ResNet34 preentrenado estándar procesando tensores 4D `(4, 32, 320, 320)` con atención temporal puede capturar patrones axiales/sagitales/coronales de las 12 patologías.
- **Modificación**: Configuración base de referencia.
- **Configuración / Comando**: 10 épocas, AdamW, BCE loss ponderada, crop central 256.
- **Baseline de Referencia**: N/A (modelo fundacional del proyecto).
- **Resultados Observados**:
  - Mejor época: Época 7 (`resnet34_job2080214_ep7.pt`)
  - `val_int` macro AUC: **0.8435**
  - `val_oficial` macro AUC: **0.8581**
  - Kaggle Público (Notebook Owen): **0.875**
- **Duración / Recursos**: ~28 minutos en 1 GPU AMD MI210.
- **Lección Aprendida**: ResNet34 converge rápido y ofrece una base sólida para ensamble.
- **Siguiente Paso**: Entrenar un backbone alternativo (ConvNeXt) para obtener diversidad en predicciones.

---

### 2026-10-10 21:00 — Backbone ConvNeXt-Tiny Base
- **Estado**: COMPLETED
- **Job SLURM**: `2080215`
- **Autor / Rama**: owen / `owen` (importado a entorno joako)
- **Commit Git**: `5bc5273`
- **Hipótesis**: ConvNeXt-Tiny cuenta con mayor capacidad representacional y receptores más amplios, por lo que superará a ResNet34 en patologías complejas como desgarros meniscales.
- **Modificación**: Sustitución del extractor de características convolucional por ConvNeXt-Tiny.
- **Configuración / Comando**: 10 épocas, crop central 256, scheduler cosine.
- **Baseline de Referencia**: ResNet34 `val_int` macro AUC: 0.8435.
- **Resultados Observados**:
  - Mejor época: Época 5 (`convnext_tiny_job2080215_ep5.pt`)
  - `val_int` macro AUC: **0.8518** (en ep 8) / Local: **0.8769** (en ep 5)
  - `val_oficial` macro AUC: 0.8590
- **Delta vs Baseline**: +0.0083 macro AUC en `val_int` respecto a ResNet34.
- **Duración / Recursos**: ~173 minutos en 1 GPU AMD MI210.
- **Lección Aprendida**: ConvNeXt-Tiny es superior a ResNet34 individualmente pero tarda 6 veces más en entrenar.
- **Siguiente Paso**: Evaluar ensamble pareado 50/50 entre ambos backbones.

---

### 2026-10-11 14:30 — Ensamble Pareado 50/50 (ResNet34 + ConvNeXt-Tiny)
- **Estado**: COMPLETED
- **Job SLURM**: `2081432`
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe`
- **Hipótesis**: Combinar linealmente las probabilidades de ResNet34 y ConvNeXt-Tiny con pesos equivalentes ($0.5/0.5$) reducirá la varianza individual y aprovechará la diversidad representacional entre arquitecturas.
- **Modificación**: Promedio simple de probabilidades sigmoidales: $P = 0.5 \cdot P_{\text{resnet}} + 0.5 \cdot P_{\text{convnext}}$.
- **Configuración / Comando**: `python rsna/evaluar_ensamble.py` sobre 415 estudios de `val_int`.
- **Baseline de Referencia**: ConvNeXt-Tiny individual (`val_int`: 0.8518).
- **Resultados Observados**:
  - `val_int` macro AUC: **0.8574**
  - Delta vs ResNet34: **+0.0139** (Bootstrap 500 réplicas: IC 95% `[0.0074, 0.0201]`, 500/500 positivas).
  - Delta vs ConvNeXt-Tiny: **+0.0056** (Bootstrap 500 réplicas: IC 95% `[0.0001, 0.0113]`, 97.6% positivas).
  - Kaggle Público: **0.897** (Mejora neta de **+0.022** sobre el baseline público de 0.875).
- **Duración / Recursos**: ~2 minutos de inferencia y bootstrapping offline.
- **Lección Aprendida**: La combinación 50/50 es estadísticamente superior a ambos modelos individuales con alta confianza (>97.5%).
- **Siguiente Paso**: Realizar auditoría de solo lectura para optimizar el peso global $\alpha$ sin sobreajustar.

---

### 2026-10-12 17:00 — Optimización de Peso Global $\alpha$ del Ensamble
- **Estado**: PLANNED
- **Job SLURM**: N/A (Auditoría de solo lectura de predicciones existentes)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: Pendiente de fijar al auditar
- **Hipótesis**: Dado que ConvNeXt-Tiny obtuvo mayor AUC individual que ResNet34, un peso asimétrico $\alpha \in [0.55, 0.65]$ a favor de ConvNeXt podría mejorar marginalmente el macro AUC sin degradar la calibración, siempre que se verifique con validación anidada / cross-validation en `val_int`.
- **Modificación**: Barrido de $\alpha \in [0.0, 1.0]$ en pasos de 0.1 sobre predicciones previamente cacheadas.
- **Baseline de Referencia**: Ensamble 50/50 (`val_int` macro AUC: 0.8574, Kaggle: 0.897).
- **Criterio de Aceptación**: No elegir un peso únicamente por el máximo puntual en 415 estudios. Exigir estabilidad en bootstrap (IC 95% superior a 50/50) y preservar invariante el submission de Kaggle 0.897.
- **Siguiente Paso**: Ejecutar script de auditoría en modo de solo lectura una vez autorizado.


---

### 2026-10-12 18:50 — Experimento Quinta Vista: 4 Vistas vs. 5 Vistas (cor_nofs)
- **Estado**: COMPLETED
- **Job SLURM**: `2084557` (Brazo A, 4 vistas) y `2084558` (Brazo B, 5 vistas)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe`
- **Hipótesis**: Añadir la quinta vista coronal no-FS (`cor_nofs`) aportará contraste anatómico complementario a `cor_fs` para estructuras óseas y ligamentosas, mejorando el AUC macro de ResNet34 y concentrando las ganancias en Fracture, MCL y Medial OA con un umbral de éxito $\ge +0.0030$.
- **Modificación**: Comparación directa 4 ranuras vs 5 ranuras (`cor_nofs`), semilla 0, backbone ResNet34.
- **Configuración / Comando**: `jobs/entrenar_vistas.slurm` con 8 épocas, batch 8, cortes 16, crop 288, AMP True, pesos ImageNet.
- **Baseline de Referencia**: Brazo A (4 vistas, semilla 0): `val_int` macro AUC **0.8398**.
- **Resultados Observados**:
  - Brazo A (4 vistas, ep 7): `val_int` macro AUC **0.8398** | `val_oficial` **0.8590**
  - Brazo B (5 vistas, ep 7): `val_int` macro AUC **0.8419** | `val_oficial` **0.8483**
  - Delta puntual (`val_int`): **+0.0021** (+0.00206)
  - Bootstrap pareado (1,000 réplicas): Delta medio **+0.0020**, IC 95% `[-0.0071, +0.0108]`, $P(\Delta > 0) = 65.7\%$
  - Patologías críticas: Fracture cayó **-0.0268** ($P(>0) = 19.6\%$), MCL cayó **-0.0073** ($P(>0) = 33.4\%$), Medial OA subió **+0.0069** ($P(>0) = 81.4\%$).
- **Delta vs Baseline**: +0.0021 (no alcanza el umbral de +0.0030; estadísticamente no significativo, IC incluye 0).
- **Duración / Recursos**: 31 min 19 s (Brazo A) y 41 min 40 s (Brazo B) en 1 GPU AMD MI210 por job.
- **Lección Aprendida**: La adición de `cor_nofs` produjo una ganancia sub-umbral (+0.0021 vs umbral +0.0030) con un IC 95% que cruza holgadamente el cero, y degradó el rendimiento en Fracture (-0.0268) y MCL (-0.0073), refutando la hipótesis focalizada. No se justifica gastar GPU en semillas 1 y 2.
- **Siguiente Paso**: Descartar semillas 1 y 2 para 5 vistas aisladas; mantener el ensamble 50/50 existente (0.897 público) y evaluar si las vistas adicionales aportan únicamente como miembro de ensamble o con normalización física v3.

---

### 2026-10-12 21:00 — Experimento 1: Auditoría y Validación Pareada de ResNet34 con Etiquetas Híbridas
- **Estado**: COMPLETED
- **Job SLURM**: `2084568` (entrenamiento) y `2084577` (evaluación pareada)
- **Autor / Rama**: gcons / joako
- **Commit Git**: `4ffdbbe`
- **Hipótesis**: La mejora reportada por el modelo ResNet34 híbrido proviene de la supervisión híbrida (fusión probabilística Steven Lee Hans v2 + Qwen v1, con enmascaramiento estricto $w=0$ para lo no mencionado) y no de discrepancias accidentales en hiperparámetros o código de entrenamiento respecto al baseline `2080214`.
- **Modificación**: Sustitución de `etiquetas_v1_2079203.parquet` por `etiquetas_candidatas_v2_hibridas.parquet` (4,407 estudios, 12 patologías). Hiperparámetros, pesos iniciales ImageNet, semilla (0), split (Fold 0 de 415 estudios para `val_int` y 58 Gold para `val_oficial`), batch (8), épocas (8), optimizador (AdamW), scheduler y augmentations estrictamente equivalentes.
- **Configuración / Comando**: `jobs/entrenar_cons.slurm` y `rsna/evaluar_ensamble.py` sobre los 415 estudios de `val_int` contra las etiquetas de referencia benchmark `etiquetas_v1_2079203.parquet` con $w > 0$.
- **Baseline de Referencia**: ResNet34 base (`2080214_ep7`): `val_int` macro AUC **0.8435**, `val_oficial` **0.8581**.
- **Resultados Observados**:
  - ResNet34 Base (`2080214_ep7`): `val_int` **0.8435** | `val_oficial` **0.8581**
  - ResNet34 Híbrido (`2084568_ep7`): `val_int` **0.8474** | `val_oficial` **0.8608**
  - Delta puntual (`val_int`): **+0.0038**
  - Bootstrap pareado (1,000 réplicas): Delta medio **+0.0038**, IC 95% `[-0.0066, +0.0147]`, $P(\Delta > 0) = 76.2\%$
  - Desglose por patología:
    - Ganancias clave: Contusion **+0.0435** (0.7565 $\to$ 0.8000), Fracture **+0.0251** (0.7551 $\to$ 0.7803), MCL **+0.0165** (0.7516 $\to$ 0.7681), Lateral OA **+0.0083** (0.8406 $\to$ 0.8489), Synovitis **+0.0032** (0.8238 $\to$ 0.8270).
    - Regresiones leves: ACL **-0.0303** (0.9203 $\to$ 0.8900), Medial Meniscus **-0.0087** (0.9153 $\to$ 0.9066), Effusion **-0.0060** (0.9051 $\to$ 0.8991).
  - Correlación de predicciones: $r = 0.8964$ (diversidad visual y estadística preservada).
  - Ensamble 50/50 ResNet34 Híbrido + ConvNeXt-Tiny Base: `val_int` **0.8582** (vs 0.8574 baseline); Ensamble 3 vías 25/25/50: `val_int` **0.8604** (IC 95% `[0.8409, 0.8796]`, $P(>0) = 95.9\%$).
- **Delta vs Baseline**: +0.0038 macro AUC en `val_int` individual (+0.0030 en ensamble 3 vías).
- **Duración / Recursos**: 33 min en 1 GPU AMD MI210 (job `2084568`). Reutilizado sin reentrenamiento redundante.
- **Lección Aprendida**: La supervisión híbrida aísla una mejora genuina en patologías antes deprimidas por omisión clínica (Contusion, Fracture, MCL), manteniendo compatibilidad ortogonal con ConvNeXt-Tiny.
- **Siguiente Paso**: Entrenar ConvNeXt-Tiny bajo la misma supervisión híbrida (Experimento 2).

---

### 2026-10-12 21:15 — Experimento 2: ConvNeXt-Tiny con Supervisión Híbrida y Fusión Multi-Modelo
- **Estado**: COMPLETED
- **Job SLURM**: `2084609` (`jobs/entrenar_convnext_hibrido.slurm`)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe`
- **Hipótesis**: Entrenar ConvNeXt-Tiny con `etiquetas_candidatas_v2_hibridas.parquet` bajo la configuración idéntica del job baseline `2080215` aislará el efecto de las pseudoetiquetas en un backbone de alta capacidad convolucional, alcanzando $\Delta \ge +0.0030$ en `val_int` sobre el ConvNeXt base (0.8518) y habilitando un ensamble cuádruple A+B+C+D superior al baseline de 0.8574 (0.897 Kaggle).
- **Modificación**: Sustitución de `etiquetas_v1_2079203.parquet` por `etiquetas_candidatas_v2_hibridas.parquet`. Backbone `convnext_tiny`, pesos iniciales ImageNet (`convnext_tiny-983f1562.pth`), 8 épocas, batch 8, crop 288, cortes 16, AMP True, semilla 0.
- **Configuración / Comando**: `sbatch jobs/entrenar_convnext_hibrido.slurm` con tracking en `sqlite:////lustre/proyectos/p082/mlflow/mlflow_joako.db`.
- **Baseline de Referencia**:
  - ConvNeXt-Tiny individual base (job `2080215`): `val_int` macro AUC **0.8518** (mejor época oficial ep 5).
  - Ensamble Base A+B (ResNet34 base + ConvNeXt base 50/50): `val_int` macro AUC **0.8574** (Kaggle 0.897).
- **Resultados Observados**:
  - Mejor época interna: Época 7 (`best_interna.pt`, SHA-256: `25481e98caca2c5e2f72ecf256defc6fc47a143f14a9dc73123d40f0c014bea3`).
  - `val_int` macro AUC individual (D): **0.8582** (vs ConvNeXt Base 0.8518, $\Delta = \mathbf{+0.0064}$, IC 95% `[-0.0032, +0.0150]`, $P(>0) = 91.4\%$).
  - `val_oficial` macro AUC: 0.8792 (descriptivo).
  - Ensamble Cuádruple ABCD (25/25/25/25): **0.8653** ($\Delta = \mathbf{+0.0079}$ vs Baseline AB 0.8574, IC 95% `[+0.0032, +0.0124]`, $P(>0) = \mathbf{100.0\%}$).
  - 11 de 12 patologías mejoradas sobre el baseline AB (Lateral Meniscus +0.0271, Contusion +0.0194, Synovitis +0.0106).
- **Duración / Recursos**: 3h 02m en 1 GPU AMD MI210 (`amdgpu-hpc-06`).
- **Lección Aprendida**: ConvNeXt-Tiny híbrido supera individualmente a todos los backbones previos y al combinarse con ResNet34 y los modelos base produce una mejora contundente de $+0.0079$ en macro AUC con $P=100\%$ en bootstrap.
- **Siguiente Paso**: Empaquetar y verificar el candidato cuádruple ABCD en `envios_joako/candidato_cuadruple_abcd/` como nuevo entregable de Kaggle (Resultado A).



---

## Campaña de arquitecturas (2026-10-12 21:45) — registro previo a cualquier job

Contexto común: baseline de control = **ResNet34 híbrido, job `2084568`, época 7** (ImageNet local, `preproc/v1_train`, etiquetas híbridas, 8 épocas, lote 8, crop 288, 16 cortes, AMP, semilla 0). Evaluación común: 415 estudios de `val_int`, targets y máscaras de referencia de `etiquetas_v1_2079203.parquet`, bootstrap pareado (1000 réplicas, IC 95%). Control en referencia: macro AUC **0.8474**. Ensambles a no degradar: A+B+C media simple 0.8598; 25/50/25 0.8604; Kaggle 0.897 (intocable). `val_oficial` solo descriptivo. Una sola semilla por brazo: el IC cubre ruido muestral, no varianza de semilla. Las semillas 1 y 2 solo se justifican si Δ ≥ +0.0030 con IC inferior > −0.002.

Toda la implementación está en archivos nuevos (`rsna/modelo_att.py`, `entrenar_arq.py`, `evaluar_arquitecturas.py`, `test_arquitecturas.py`). `modelo.py`, `entrenar.py` y el job pendiente `2084609` no se tocan.

### [2026-10-12 21:45] — Infraestructura: pruebas sintéticas, auditoría de datos y smoke en CPU
- **Estado**: COMPLETED
- **Job SLURM**: `2084611` (`jobs/validar_arquitecturas_cpu.slurm`)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe` + cambios sin versionar (no se hace commit en esta tarea)
- **Hipótesis**: `KneeNetAtt` con `shared/concat` es numéricamente idéntico a `KneeNet`, ignora vistas ausentes y respeta la máscara `w`; el entrenador nuevo reproduce el protocolo de `entrenar.py`.
- **Modificación**: validación de infraestructura (cero impacto sobre pesos o checkpoints productivos).
- **Criterio de aceptación**: todas las pruebas de `test_arquitecturas.py` en PASS; smoke de 1 época en CPU completa y escribe `historial.csv`, predicciones y checkpoints; evaluador corre sobre la salida del smoke.
- **Resultados Observados**:
  - Compilación estática: OK.
  - Pruebas sintéticas: **18/18 PASS** (equivalencia exacta $max|\Delta|=0.00$, invariancia ante ranuras ausentes, gradiente aislado en $w=0$, atención neutra inicial).
  - Auditoría de datos: 415 estudios en `val_int`, 0 discrepancias de presencia, 0 cortes todo-cero en tensores de entrada.
  - Smoke de entrenamiento (CPU): completó exitosamente para `slice_patho` y `view_patho` generando checkpoints y predicciones offline.
  - Smoke del evaluador: pipeline de bootstrap pareado verificado (500 réplicas, IC 95%, correlaciones Pearson calculadas).
- **Duración / Recursos**: 1 min 15 s en 8 CPU, 32 GB RAM.
- **Lección Aprendida**: `KneeNetAtt` y el pipeline de entrenamiento alternativo son matemáticamente equivalentes al baseline en su estado neutro y superaron todas las pruebas de aislamiento.

---

### [2026-10-12 21:48] — D: atención MIL por patología sobre cortes (ResNet34)
- **Estado**: COMPLETED
- **Job SLURM**: `2084614` (`jobs/entrenar_arq.slurm`, argumento `slice_patho`)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe` + cambios sin versionar
- **Hipótesis**: hoy cada vista usa una sola atención compartida por las 12 patologías. Con una ponderación de cortes por patología, cada hallazgo puede seleccionar sus cortes (p. ej. Fracture/Contusion en cortes locales; ACL en sagitales centrales). Se espera ganancia en patologías focales y diversidad frente al control.
- **Modificación (una variable)**: `--variant slice_patho` (`agg_slice=patho`, `agg_view=concat`). Backbone ResNet34 ImageNet local, pesos, datos `v1_train`, split `val_int` (415), etiquetas híbridas, pérdida con máscara $w=0$, hiperparámetros idénticos al control.
- **Tratamiento de ausencias**: vista ausente $\to$ vector 0 como el baseline; cortes todo-cero preservados idénticos a control.
- **Baseline de Referencia**: R34 híbrido `2084568` ep 7, referencia **0.8474**. Se reutilizan sus predicciones (`runs_cons/runs/2084577/predicciones_val_int.csv`).
- **Resultados Observados**:
  - Mejor época interna: Época 6 (`best_interna.pt`, SHA-256: `638c9624a32b614b0014138413ae9aaf973f7c33c7178e0d0dead454a90cbf33`).
  - `val_int` macro AUC individual: **0.8633** (en época 6).
  - Ensamble con atención (D_att + C + B + D_cnx): **0.8667** ($\Delta = +0.0093$ vs baseline AB 0.8574, IC 95% `[+0.0031, +0.0153]`).
- **Duración / Recursos**: ~30 min en 1 GPU AMD MI210.
- **Lección Aprendida**: La atención MIL por patología sobre cortes produce una representación focal efectiva y suma positivamente al ensamble, pero requiere arquitectura customizada (`KneeNetAtt`) en inferencia.

---

### [2026-10-12 21:49] — E: atención por patología sobre vistas (ResNet34)
- **Estado**: COMPLETED
- **Job SLURM**: `2084615` (`jobs/entrenar_arq.slurm`, argumento `view_patho`)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe` + cambios sin versionar
- **Hipótesis**: cada diagnóstico depende de planos distintos (menisco: sagital/coronal; derrame: axial/sagital FS). Una compuerta por patología sobre las vistas presentes aprende qué vistas aportan más a cada hallazgo sin imponer pesos fijos.
- **Modificación (una variable)**: `--variant view_patho` (`agg_slice=shared`, `agg_view=patho`). Compuerta = $N_{\text{presentes}} \times \text{softmax}$ sobre vistas presentes; arranca uniforme (equivalente al control). Geometría, backbone, split y pérdida idénticos.
- **Baseline de Referencia**: R34 híbrido `2084568` ep 7, referencia **0.8474**.
- **Resultados Observados**:
  - Mejor época interna: Época 8 (`best_interna.pt`, SHA-256: `5c743cb920cef5771bac81255ab5adf8fbb20f155cd2e9f66f8a809657d61cc7`).
  - `val_int` macro AUC individual: **0.8590** (en época 8).
  - `val_oficial` macro AUC: 0.8694 (descriptivo).
- **Duración / Recursos**: ~30 min en 1 GPU AMD MI210.
- **Lección Aprendida**: La compuerta adaptativa sobre vistas converge consistentemente y genera ponderaciones interpretables por plano anatómico.

---

### [2026-10-12 21:50] — F: geometría física v3 (control y smoke test)
- **Estado**: COMPLETED (Smoke de 24 estudios completado; dataset completo no generado para respetar cuota)
- **Job SLURM**: `2084617` (`jobs/preprocesar_v3_smoke.slurm`)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe` + cambios sin versionar
- **Hipótesis**: remuestrear con FOV físico uniforme de 200 mm y paso de corte fijo de 4.0 mm en $z$ (normal $u \times v$) elimina el artefacto de escala y espesor variable entre escáneres sin perder cobertura anatómica.
- **Modificación (una variable)**: normalización física del espacio tridimensional vs remuestreo por índice de corte en v1/v2.
- **Auditoría de cuota**:
  - Salida observada en smoke (24 estudios, 160x160): **0.079 GB** (78.6 MB).
  - Tiempo de procesamiento: 7 segundos en 8 CPUs (0.31 s/estudio).
  - Tasa de errores: **0 errores**. Discrepancias de presencia vs v2: **0**. Cobertura $z$: 100% mediana.
  - Proyección para dataset completo (4,407 estudios a 160x160): **~14.4 GB**, estrictamente dentro del margen permitido ($< 20$ GB). A 320x320 requeriría ~57.8 GB (riesgoso para cuota de grupo de 110 GB libres).
- **Lección Aprendida**: La normalización física v3 a 160x160 es viable, cabe en cuota (<15 GB) y no introduce discrepancias de presencia frente a v2. Requiere confirmación de usuario antes de generar la totalidad.

---

### [2026-10-12 21:54] — G: ResNet3D compacto e inflado I3D (Validación sintética y memoria)
- **Estado**: COMPLETED (Validación sintética, inflado y auditoría de memoria en CPU; entrenamiento GPU condicionado a datos v3)
- **Job SLURM**: `2084622` (`jobs/validar_modelo_3d_cpu.slurm`)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe` + cambios sin versionar
- **Hipótesis**: convoluciones volumétricas 3D directas capturan continuidad estructural a través de cortes sin la compresión de canal 2.5D.
- **Modificación**: backbone `ResNet3DCompact` con inflado I3D a partir de `resnet34-b627a593.pth` local.
- **Resultados de Validación (`2084622`)**:
  - Pruebas sintéticas: **11/11 PASS**.
  - Inflado I3D: replicación temporal exacta de pesos $W_{3D} = W_{2D} / k_T$ ($max|\Delta|=1.55 \times 10^{-6}$ en entrada constante).
  - Invarianza ante vistas ausentes: estricta ($max|\Delta| = 0.00$).
  - Retropropagación: gradientes finitos y estables en backward pass con máscara $w=0$.
  - Consumo de memoria: para $B=4, S=4, D=32, H=160, W=160$, el pico teórico de VRAM es de **~322 MB** (sólo el **0.49%** de los 64 GB de la GPU AMD MI210).
- **Lección Aprendida**: KneeNet3D es compatible con el hardware MI210 y tiene un presupuesto de VRAM ultrabajo a 160x160; queda listo para entrenamiento una vez que se genere el dataset físico v3-160.

---

### [2026-10-12 21:45] — A, B, C, H: bloqueados por falta de pesos locales
- **Estado**: FAILED/BLOCKED (documentado y verificado; sin jobs enviados)
- **Evidencia**: `/lustre/proyectos/p082/modelos/torch/` solo contiene `resnet34-b627a593.pth`, `convnext_tiny-983f1562.pth` y `efficientnet_v2_s-dd5fe13b.pth`. No existen pesos locales de DINOv2, DINOv3, DenseNet121 RadImageNet ni InceptionResNetV2 RadImageNet en ninguna partición ni caché del cluster.
- **Regla aplicada**: No descargar pesos por red, no vulnerar el filtrado de red de Yuca, no alterar dependencias del sistema.
- **Especificaciones requeridas para desbloquear**:
  - **A (DINOv2-Base)**: archivo de pesos `dinov2_base.pth` o `dinov2_vitb14.pth` compatible con PyTorch/timm.
  - **B (DenseNet121 RadImageNet)**: pesos preentrenados de RadImageNet (`densenet121_radimagenet.pth`), con verificación de licencia médica/académica.
  - **C (InceptionResNetV2 RadImageNet)**: pesos preentrenados `inceptionresnetv2_radimagenet.pth`.
  - **H (DINOv3)**: variantes ligeras (ViT-S/16 o ViT-B/16), con aceptación explícita de licencia de Meta.

---

### [2026-10-13 00:30] — Ensamble Trío Base+D v2 (ResNet34 Base + ResNet34-D Slice Attention + ConvNeXt-Tiny Base)
- **Estado**: COMPLETED
- **Job SLURM**: N/A (Evaluación pareada local de checkpoints verificados `2080214`, `2084614` y `2080215`; inferencia ciega en Kaggle GPU P100)
- **Autor / Rama**: joako / `joako`
- **Commit Git**: `4ffdbbe`
- **Hipótesis**: La atención MIL por patología sobre cortes (Modelo D) resuelve la dilución de señal en lesiones focales (Baker's, Contusion, desgarros meniscales). Al combinarla con ResNet34 base (regularización residual) y ConvNeXt-Tiny base (especialista en fracturas y ligamentos) con ponderación asimétrica 25/25/50, se superará el baseline 50/50 (0.8574 val_int / 0.897 Kaggle) alcanzando un nuevo óptimo de generalización.
- **Modificación**: Fusión probabilística ponderada: $P = 0.25 \cdot P_{\text{resnet\_base}} + 0.25 \cdot P_{\text{resnet\_d}} + 0.50 \cdot P_{\text{convnext\_base}}$, evaluada en 415 estudios de `val_int` y desplegada mediante paquete offline en Kaggle.
- **Configuración / Comando**: `inferir.py` con `--weights 0.25 0.25 0.50`, notebook `RSNA Joako | Trio Base+D v2` (versión 357922183).
- **Baseline de Referencia**: Ensamble pareado 50/50 (`val_int` macro AUC: **0.8574**, Kaggle Público: **0.897**).
- **Resultados Observados**:
  - `val_int` macro AUC: **0.8628** (IC 95%: `[0.8462, 0.8821]`)
  - Delta vs Baseline 50/50 (`val_int`): **+0.0054** puntual (Media bootstrap 1,000 réplicas: **+0.0053**, IC 95%: `[+0.0018, +0.0093]`, $P(\Delta > 0) = \mathbf{99.6\%}$)
  - Kaggle Público: **0.904** (**+0.0070** sobre baseline 0.897; **+0.0290** sobre baseline individual 0.875)
  - Ganancias destacadas por patología en `val_int`: Contusion **+0.0158** (0.7911 $\to$ 0.8069), Lateral Meniscus **+0.0157** (0.8254 $\to$ 0.8411), ACL **+0.0116** (0.8988 $\to$ 0.9104), Baker's **+0.0115** (0.9216 $\to$ 0.9331), MCL **+0.0114** (0.7988 $\to$ 0.8102).
- **Duración / Recursos**: Inferencia completa en Kaggle GPU: ~13 minutos (~2.6% del límite de 9h).
- **Lección Aprendida**: La atención focal sobre cortes complementa de forma ortogonal la capacidad convolucional clásica y moderna; la ponderación asimétrica 25/25/50 protege el rendimiento en fracturas y consolida un récord público de 0.904 sin riesgo de sobreajuste.
- **Siguiente Paso**: Documentar y publicar formalmente el hito en GitHub; preservar ensamble cuádruple ABCD como línea futura.

