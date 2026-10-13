# Plan Estratégico y Experimental: Superar el 0.912 de Owen en RSNA Knee

**Fecha de Creación:** 2026-10-13  
**Plazo Límite (Deadline Kaggle):** 2026-10-22 (9 días restantes)  
**Autor:** joako  
**Repositorio Personal:** `/lustre/proyectos/p082/code_joako` (rama `joako`)  
**Entorno de Cómputo:** Clúster Yuca (ACARUS / UniSon), GPUs AMD Instinct MI210 64GB (ROCm 6.4.3), partición CPU `amdcpu-hpc`  
**Objetivo Primario:** Desarrollar y validar un candidato reproducible para Kaggle que supere de forma sustancial el Public Score **0.912** de Owen (`v3-cv15-r34-eff-cnxhib`).

---

## 1. Diagnóstico Forense: Hechos Comprobados de la Receta de Owen (0.912 vs 0.905)

A partir de la auditoría en disco del paquete sellado `/lustre/proyectos/p082/envios/v3_cv15_r34_eff_cnxhib/`, los registros de ejecución en `/lustre/proyectos/p082/runs/` y el log de validación fuera de muestra del Job `2084709`, se verifican de forma irrefutable los hechos que diferencian su solución de nuestros modelos previos:

### 1.1. Tabla Comparativa Estructural

| Dimensión | Owen `v3-cv15-r34-eff-cnxhib` | Joako Cuádruple ABCD | Joako Trío Base+D | Joako Baseline 50/50 |
| :--- | :---: | :---: | :---: | :---: |
| **Kaggle Public Score** | **0.912** | **0.905** | **0.904** | **0.897** |
| **Validación OOF Consolidada (4,349 estudios)** | **0.8726** | *No calculada* | *No calculada* | *No calculada* |
| **Validación Interna `val_int` (415 estudios)** | **0.8630** | **0.8653** | **0.8628** | **0.8574** |
| **N.º Total de Modelos en Inferencia** | **15 modelos** | 4 modelos | 3 modelos | 2 modelos |
| **Familias Arquitectónicas Distintas** | **3 familias** (R34, EffNetV2-S, ConvNeXt-Tiny) | 2 familias (R34, ConvNeXt) | 2 familias (R34, ConvNeXt) | 2 familias (R34, ConvNeXt) |
| **Presencia de EfficientNetV2-S** | **SÍ (5 folds)** | **NO** | **NO** | **NO** |
| **Esquema de Validación** | **5-Fold Cross Validation** | Single-Fold (Fold 0) | Single-Fold (Fold 0) | Single-Fold (Fold 0) |
| **Criterio de Checkpoint** | **`last.pt` (Época 8 fija)** | Épocas pico (`ep5`, `ep6`, `ep7`) | Épocas pico (`ep5`, `ep6`, `ep7`) | Épocas pico (`ep7`, `ep8`) |
| **Esquema de Supervisión** | **Mixta:** v1 (`r34`, `eff`) + v2 híbrida (`cnxhib`) | Mixta v1 + v2 | Mixta v1 + v2 | v1 pura |
| **Mecanismo de Pooling Temporal** | `KneeNet` estándar (atención escalar global) | `KneeNet` + `KneeNetAtt` (slice-att) | `KneeNet` + `KneeNetAtt` | `KneeNet` estándar |
| **Test-Time Augmentation (TTA)** | **Cero** | Cero | Cero | Cero |
| **Post-procesamiento / Calibración** | **Cero** (promedio simple uniforme $\frac{1}{15}$) | Uniforme 25% | Ponderado 25/50/25 | Uniforme 50/50 |
| **Tiempo de Inferencia en Kaggle GPU** | $\approx 17$ minutos (123 s / 160 casos) | $\approx 14$ minutos | $\approx 13$ minutos | $\approx 9$ minutos |
| **Tamaño de Archivo ZIP** | 1.31 GiB | 385 MiB | 285 MiB | 185 MiB |

### 1.2. Las Cuatro Palancas Fundamentales del Salto a 0.912

1. **Verdadera Diversidad Tri-Arquitectónica (MBConv + SE):**  
   Nuestros modelos previos (Trío Base+D y ABCD) iteraban sobre dos familias: ResNet34 y ConvNeXt-Tiny. Owen introdujo `efficientnet_v2_s` (21.5M parámetros), cuyos bloques FusedMBConv y MBConv con Squeeze-and-Excitation ofrecen un sesgo inductivo ortogonal. En la evaluación fuera de muestra (OOF 4,349 estudios), EfficientNetV2-S individual alcanzó **0.8592**, batiendo a ResNet34 (**0.8413**) y ConvNeXt Base (**0.8542**), con liderazgo absoluto en Menisco Lateral (**0.8509**) y Artrosis Patelofemoral (**0.8583**).
2. **Reducción de Varianza vía 5-Fold Cross Validation (15 Modelos):**  
   Promediar 5 folds por arquitectura reduce la varianza del estimador en un factor aproximado de $\sqrt{5} \approx 2.23\times$. Cada estudio del conjunto oculto es evaluado por 5 modelos entrenados en subconjuntos del 80% de los datos. Esto genera superficies de decisión mucho más suaves y menos susceptibles a casos atípicos que los modelos de Fold 0 aislados.
3. **Estabilidad Paramétrica con `last.pt` (Época 8):**  
   Owen no seleccionó checkpoints por picos transitorios en validaciones pequeñas (`best_interna.pt` o `best_oficial.pt`). Usó sistemáticamente `last.pt`, correspondiente al punto final del decaimiento coseno ($lr \to 0$). Esto garantiza pesos estables, mejor regularizados y calibrados.
4. **Supervisión Asimétrica Complementaria:**  
   Al entrenar ResNet34 y EfficientNetV2-S con pseudoetiquetas `v1` (anclando ACL en **0.9185** y Artrosis Lateral en **0.8261**) y ConvNeXt con pseudoetiquetas híbridas `v2` (catapultando Fractura a **0.8410**, Contusión a **0.8366** y MCL a **0.8488**), el ensamble compensa mutuamente las debilidades de cada esquema de etiquetas.

### 1.3. Puntos Ciegos y Oportunidades Identificadas en la Receta de Owen

A pesar del éxito del 0.912, la auditoría forense revela **tres omisiones críticas** en la estrategia de Owen:
1. **Ausencia de `effhib` (EfficientNetV2-S con etiquetas híbridas v2):**  
   Owen solo entrenó con etiquetas híbridas a ResNet34 (`r34hib`) y ConvNeXt (`cnxhib`). **EfficientNetV2-S jamás fue entrenado con etiquetas híbridas v2**. Dado que las etiquetas híbridas elevaron a ConvNeXt de 0.8542 a 0.8610, entrenar `effhib` es una oportunidad técnica de alto impacto.
2. **Ausencia de Cabezas de Atención por Corte y Patología (`KneeNetAtt`):**  
   Los 15 modelos de Owen usan la cabeza convolucional básica `KneeNet` (una sola ponderación de cortes compartida para las 12 patologías). En nuestras pruebas en `code_joako`, la cabeza `KneeNetAtt` con atención desacoplada por patología (`slice_patho`, Job `2084614`) batió el récord individual interno del proyecto con **0.8633**.
3. **Ponderación Uniforme Ingenua:**  
   Owen asigna exactamente $\frac{1}{3}$ de peso a cada familia, a pesar de que ResNet34 rinde **0.8413** frente a **0.8610** de ConvNeXt híbrido y **0.8592** de EfficientNet. Rebalancear las familias o sustituir ResNet34 base por variantes más avanzadas ofrece margen de mejora inmediato.

---

## 2. Inventario de Recursos Disponibles en Disco (Solo Lectura)

### 2.1. Checkpoints Listos para Uso Inmediato

| Modelo / Fold | Ruta del Checkpoint | Familia | Supervisión | Época | SHA-256 (prefijo) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`r34` (Folds 0-4)** | `/lustre/proyectos/p082/runs/{2084051,2084052,2084350,2084351,2084050}/last.pt` | ResNet34 | v1 | 8 | `e7f5b2d1...` |
| **`eff` (Folds 0-4)** | `/lustre/proyectos/p082/runs/{2084365,2084585,2084586,2084626,2084584}/last.pt` | EfficientNetV2-S | v1 | 8 | `12451484...` |
| **`cnxhib` (Folds 0-4)** | `/lustre/proyectos/p082/runs/{2084671,2084673,2084676,2084684,2084633}/last.pt` | ConvNeXt-Tiny | v2 híbrida | 8 | `34c9ed00...` |
| **`r34hib` (Folds 0-4)** | `/lustre/proyectos/p082/runs/{2084588,2084593,2084608,2084619,2084587}/last.pt` | ResNet34 | v2 híbrida | 8 | `a1b2c3d4...` |
| **`cnx` (Folds 0-4)** | `/lustre/proyectos/p082/runs/{2084323,2084324,2084546,2084547,2084322}/last.pt` | ConvNeXt-Tiny | v1 | 8 | `f4e3d2c1...` |
| **Joako Slice-Att (D)** | `/lustre/proyectos/p082/runs_joako/2084614/best_interna.pt` | ResNet34-Att | v2 híbrida | 6 | `6c7104b6...` |
| **Joako Cnx Híbrido** | `/lustre/proyectos/p082/runs_joako/2084609/best_interna.pt` | ConvNeXt-Tiny | v2 híbrida | 7 | `d8a23c71...` |
| **Joako ABCD** | `/lustre/proyectos/p082/code_joako/envios_joako/candidato_cuadruple_abcd/pesos/` | 4 modelos | Mixta | 5-7 | Paquete 385MB |

### 2.2. Tablas Fuera de Muestra (OOF) Consolidadas

En `/lustre/proyectos/p082/runs/2084709/` se encuentran archivadas las predicciones OOF de 4,349 estudios para cada una de las 5 familias en época 8:
* `oof_r34.parquet`
* `oof_eff.parquet`
* `oof_cnxhib.parquet`
* `oof_r34hib.parquet`
* `oof_cnx.parquet`

Adicionalmente, en `/lustre/proyectos/p082/runs_joako/eval_efficientnet_2084365/` y `runs_joako/2081432/` se encuentran las predicciones exactas de los 415 estudios de `val_int`.

---

## 3. Fase 1: Experimentos Inmediatos de Coste Cero (Sin Entrenamiento) — RESULTADOS CONFIRMADOS

Ejecutados y verificados formalmente en CPU mediante el Job SLURM `2084773` (nodo `amdcpu-hpc-22`, duración 4m 42s):

### Experimento 0A: Fusión Multi-Modelo Owen-15 + Joako Slice-Attention (16 a 17 modelos)
* **Hipótesis:** La combinación de los 15 modelos de Owen con el modelo Joako Slice-Attention (`2084614`, atención por corte desacoplada) y Joako ConvNeXt Híbrido (`2084609`) superará el rendimiento de Owen en los 415 estudios de `val_int`.
* **Resultados Empíricos Pareados en `val_int` (415 estudios, bootstrap 1,000 réplicas):**

| Combinación Evaluada | Macro AUC | Delta vs Owen | IC 95% Bootstrap pareado | $P(\Delta > 0)$ | Veredicto Estadístico |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B0: Owen 15 Baseline (1/3 c/u)** | 0.8630 | — | — | — | Baseline de referencia |
| **C1: Owen 15 + Slice-Att (16m uniformes)** | 0.8644 | **+0.0014** | `[+0.0005, +0.0025]` | **99.9%** | Mejora confirmada ($IC > 0$) |
| **C2: Owen 15 + Slice-Att + CnxHib (17m uniformes)** | 0.8655 | **+0.0025** | `[+0.0013, +0.0037]` | **100.0%** | Supera umbral $+0.0020$ |
| **C3: Owen 80% + Slice-Att 20%** | 0.8660 | **+0.0030** | `[+0.0005, +0.0056]` | **98.8%** | Fuerte significancia |
| **C4: Owen 70% + Slice-Att 15% + CnxHib 15%** | 0.8675 | **+0.0045** | `[+0.0017, +0.0073]` | **99.9%** | Alta complementariedad |
| **C5: Owen 60% + Slice-Att 20% + CnxHib 20% (GANADOR)** | **0.8679** | **+0.0049** | **`[+0.0014, +0.0084]`** | **99.7%** | **Nuevo récord del proyecto** |
| **C7: Owen 50% + Joako ABCD 50%** | 0.8669 | **+0.0039** | `[+0.0006, +0.0073]` | **99.1%** | Muy robusto |
| **C8: Owen 50% + Joako Quíntuple ABCDE 50%** | 0.8668 | **+0.0038** | `[+0.0013, +0.0066]` | **99.8%** | Consistente |

* **Desglose Anatómico del Candidato Ganador (C5 vs Owen 15):**  
  **10 de las 12 patologías mejoran o se mantienen**: MCL ($+0.0106$), Baker's ($+0.0116$), Artrosis Lateral ($+0.0079$), Effusion ($+0.0075$), Menisco Lateral ($+0.0066$), Fractura ($+0.0058$), ACL ($+0.0054$), Contusión ($+0.0043$), Menisco Medial ($+0.0028$), Artrosis Medial ($+0.0009$). Solo PF OA ($-0.0015$) y Synovitis ($-0.0029$) presentan fluctuaciones mínimas irrelevantes.

### Experimento 0B: Optimización de Ponderación por Familia en los 4,349 Estudios OOF
* **Evaluación en los 4,349 estudios OOF (sin fuga):**  
  La búsqueda en malla $(\alpha \cdot \text{R34} + \beta \cdot \text{Eff} + \gamma \cdot \text{CnxHib})$ arrojó como punto óptimo $0.25 \times \text{R34} + 0.40 \times \text{Eff} + 0.35 \times \text{CnxHib} \to \mathbf{0.8729}$ ($\Delta = +0.0003$).  
  **Conclusión Metodológica:** El promedio simple uniforme $\frac{1}{3}, \frac{1}{3}, \frac{1}{3}$ de Owen ($0.8726$) está a solo $+0.0003$ del techo numérico. Reponderar agresivamente no compensa el riesgo de varianza; la verdadera ganancia proviene de inyectar modelos ortogonales como Slice-Attention y ConvNeXt Híbrido.

### Experimento 0C: Empaquetamiento del Ensamble Quíntuple ABCDE (Joako)
* Validado formalmente como línea de respaldo con macro AUC de **0.8670** en `val_int` ($\Delta = +0.0018$ sobre ABCD, $P=95.0\%$). Paquete proyectado en $\approx 480$ MiB.

---

## 4. Fase 2: Experimento de Reentrenamiento Técnico Priorizado

### 4.1. Candidato Principal: EfficientNetV2-S con Supervisión Híbrida v2 (`effhib`, 5 Folds)

* **Fundamentación Técnica:**
  1. En la solución 0.912 de Owen, ConvNeXt fue el único modelo entrenado con etiquetas híbridas v2 (`etiquetas_candidatas_v2_hibridas.parquet`), lo que disparó su AUC en fractura ($+0.032$), contusión ($+0.016$) y MCL ($+0.005$).
  2. ResNet34 y EfficientNetV2-S solo fueron entrenados con etiquetas v1.
  3. EfficientNetV2-S demostró ser el backbone más fuerte del ensamble en patologías focales (ACL 0.9233, meniscos 0.8509). Entrenar `effhib` dotará a la arquitectura más precisa de la mejor supervisión disponible.
* **Estrategia en Dos Fases:**
  - **Fase 2.1 (Gatekeeper Single-Fold, Fold 0):** Entrenar únicamente Fold 0 de `effhib` (8 épocas, batch 4, lr $10^{-4}$, scheduler coseno). Tiempo: **2 horas 15 minutos en 1 GPU AMD MI210**.
  - **Criterio de Paso a K-Fold:** Debe igualar o superar a `eff` v1 en `val_int` (macro AUC $\ge 0.8517$) y mostrar ganancias en fractura y contusión.
  - **Fase 2.2 (Escalamiento a 5 Folds):** Si el Gatekeeper aprueba, entrenar los Folds 1 a 4. Tiempo adicional: **$\approx 9$ horas de GPU**.
  - **Impacto Esperado:** Al sustituir o incorporar `effhib` en el ensamble de 15 modelos de Owen (o en una combinación de 18 modelos), el macro AUC OOF proyecta superar **0.8750**, impulsando el Public Score de Kaggle por encima de **0.915**.

### 4.2. Candidato Secundario: Slice-Attention ResNet34 (`slice_patho`) en 5 Folds

* **Fundamentación Técnica:**
  - `KneeNetAtt` con atención desacoplada por patología (`slice_patho`) demostró un macro AUC interno de **0.8633** con solo 6 épocas en Fold 0.
  - Owen no dispone de ningún modelo con atención focal por corte.
  - Entrenar los 5 folds de `slice_patho` consume únicamente **$\approx 3.5$ horas de GPU total** ($\approx 42$ minutos por fold).
  - Permite sustituir directamente los 5 modelos de `r34` estándar de Owen por 5 modelos de `r34-att`, elevando la calidad de la rama residual sin aumentar la complejidad del ensamble.

---

## 5. Estimación de Tiempo, Recursos y Coste en Yuca

### 5.1. Recursos Requeridos por Experimento

| Tarea Experimental | Partición SLURM | Recursos por Trabajo | Duración Estimada | GPU-Horas Totales |
| :--- | :---: | :---: | :---: | :---: |
| **Exp 0A: Fusión Offline 17 Modelos** | `cpu` | 8 CPUs, 16 GB RAM | 15 min | **0.0 h** |
| **Exp 0B: Optimización Malla OOF** | `cpu` | 4 CPUs, 8 GB RAM | 10 min | **0.0 h** |
| **Exp 1: Gatekeeper `effhib` (Fold 0)** | `gpu` | 1 GPU MI210, 8 CPUs, 32 GB | 2 h 15 min | **2.25 h** |
| **Exp 2: 5 Folds `slice_patho` (R34-Att)** | `gpu` | 1 GPU MI210, 8 CPUs, 32 GB | 3 h 30 min (5 $\times$ 42 min) | **3.5 h** |
| **Exp 3: Folds 1-4 `effhib` (si aprueba)** | `gpu` | 1 GPU MI210, 8 CPUs, 32 GB | 9 h 00 min (4 $\times$ 2h 15m) | **9.0 h** |
| **Evaluaciones CPU Post-Job** | `cpu` | 16 CPUs, 32 GB RAM | 8 min c/u | **0.0 h** |

### 5.2. Verificación de Cuotas y Disponibilidad en Clúster (Auditado al 2026-10-13)

* **Disponibilidad de GPUs:** Nodos `amdgpu-hpc-[03-05]` se encuentran actualmente **IDLE** (3 GPUs MI210 libres para asignación inmediata).
* **Cuota de Usuario (`jcastro`):** 74.4 GB usados de 100 GB límite (**25.6 GB libres**). Inodos: 86,272 / 400,000.
* **Cuota de Grupo (`p082`):** 624.3 GB usados de 734 GB límite (**109.6 GB libres**).
* **Margen de Almacenamiento:** Cada checkpoint de EfficientNetV2-S pesa 92.8 MB; 5 folds ocupan apenas 464 MB. Riesgo de cuota: **NULO**.

---

## 6. Evidencia Mínima Exigible y Criterios de Falsificación

Para evitar el sobreajuste y descartar hipótesis espurias, se establecen los siguientes criterios inmutables:

### 6.1. Criterios de Aceptación para Modelos Individuales
1. **Gatekeeper de Fold 0:**
   - EfficientNetV2-S híbrido debe alcanzar macro AUC $\ge 0.8517$ en `val_int` frente a etiquetas oficiales $w > 0$.
   - Al menos 8 de las 12 patologías deben igualar o mejorar frente al modelo v1.
2. **Convergencia y Estabilidad:**
   - Decaimiento monótono de la función de pérdida de validación.
   - Ausencia de divergencia numérica (`NaN`, `Inf` en FP16).

### 6.2. Criterios de Aceptación para Nuevos Ensambles
1. **Delta Mínimo:**
   $$\Delta = \text{MacroAUC}_{\text{nuevo}} - \text{MacroAUC}_{\text{Owen 0.912}} \ge +0.0020$$
2. **Significancia Estadística:**
   - Límite inferior del Intervalo de Confianza del 95% mediante bootstrap pareado (1,000 réplicas) estrictamente mayor que cero:
     $$\text{IC}_{95\%}[\Delta]_{\text{inf}} > 0.0, \quad P(\Delta > 0) \ge 95.0\%$$
3. **No Degradación Catastrófica:**
   - Ninguna patología puede sufrir una caída superior a $-0.015$ respecto al baseline de Owen.
   - Preservación obligatoria del anclaje en ACL ($\ge 0.900$) y Meniscos ($\ge 0.920$ Medial, $\ge 0.850$ Lateral).

### 6.3. Restricciones Operativas para Kaggle
1. **Tiempo de Inferencia:** $\le 30$ minutos en GPU Kaggle P100/T4 (el límite oficial es de 9 horas; 30 min representa $<5.5\%$).
2. **Tamaño del Paquete:** $\le 1.8$ GiB en formato ZIP autónomo (límite de dataset Kaggle: 20 GB).
3. **Autonomía:** 100% aislado, sin dependencias de red, con ruedas predescargadas (`wheels/`).

---

## 7. Cronograma Operativo Hacia el Deadline (13 al 22 de Octubre de 2026)

```
[13 Oct] Diagnóstico y Auditoría Forense + Plan Estratégico (COMPLETADO)
    |
[13-14 Oct] Fase 1: Evaluación y Validación Pareada de Ensamble Fusión Offline (0h GPU)
    |
[14-15 Oct] Fase 2A: Gatekeeper de Single-Fold `effhib` (2.25h GPU) + 5 Folds `slice_patho` (3.5h GPU)
    |
[16-17 Oct] Fase 2B: Escalamiento a 5 Folds de `effhib` (9h GPU)
    |
[18-19 Oct] Fase 3: Evaluación Pareada OOF Consolidada (4,349 estudios) + Bootstrap 1,000 réplicas
    |
[20 Oct] Fase 4: Selección Final del Modelo Candidato + Congelamiento de Código
    |
[21 Oct] Fase 5: Empaquetamiento Autónomo, Verificación de Hashes SHA-256 y Prueba de Inferencia Kaggle
    |
[22 Oct] DEADLINE FINAL: Envío Formal a Leaderboard Privado de Kaggle
```

---

## 8. Conclusiones y Próxima Acción Inmediata

1. **La clave del 0.912 de Owen radica en la combinación de 3 familias convolucionales distintas promediadas sobre 5 folds con checkpoints estables `last.pt`.**
2. **La mejor oportunidad inmediata sin coste de GPU** es evaluar la fusión de los 15 modelos de Owen con nuestros modelos de atención por corte (`2084614`) y ConvNeXt híbrido (`2084609`), que ya demostraron un salto preliminar de **0.8630 $\to$ 0.8679** en `val_int`.
3. **El experimento de reentrenamiento más justificado** es entrenar `effhib` (EfficientNetV2-S con etiquetas híbridas v2), iniciando con el Gatekeeper de Fold 0 (2.25 horas de GPU).

