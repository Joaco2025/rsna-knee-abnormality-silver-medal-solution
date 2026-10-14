# RSNA Knee — qué se ha intentado, dónde está cada cosa y qué queda en cola

Estado al 13 de octubre de 2026, 20:00. Cierre de envíos: 22 de octubre, 23:59 UTC.
**Mejor puntuación en la tabla pública de Kaggle: 0.918.**

Rutas relativas a `/lustre/proyectos/p082/`. El código de este repositorio vive en `code/`.

## Cómo leer los números

- **Kaggle**: AUC en la tabla pública. Hay 5 envíos al día y diferencias de 0.001 son ruido.
- **Oficial (58)**: AUC contra las etiquetas reales de los 58 estudios que las traen. Pocos casos (margen de unos ±0.03) y no es independiente: se usaron para calibrar los objetivos.
- **Fuera de muestra**: validación cruzada de 5 particiones. Cada uno de los 4,349 estudios sin etiqueta oficial lo predice un modelo que no lo vio, y se mide contra los objetivos derivados de los informes. Salvo que se diga otra cosa, la referencia es nuestra tabla original (`etiquetas_v1`). Mide el acuerdo con etiquetas derivadas, no con la verdad.
- "Oficial" en las tablas de validación cruzada es el promedio de los cinco modelos de las particiones.

La validación local ha predicho bien Kaggle: anticipaba +0.012 a +0.014 del ensamble de 15 modelos sobre el envío de 0.897 y fueron +0.015.

## Envíos a Kaggle

| Fecha | Envío | Autor | Contenido | Kaggle | Fuera de muestra | Oficial (58) |
|---|---|---|---|---|---|---|
| 11 oct | `notebook000579244b` v2 | Owen | ResNet-34, una partición, época 7 | 0.875 | 0.8435 (415 estudios) | 0.858 |
| 12 oct | RSNA Joako Ensemble v1 | Joako | ResNet-34 + ConvNeXt-Tiny, media | 0.897 | 0.8574 (415) | 0.879 |
| 12 oct | `cons_v3_submit` | Cons | Los dos anteriores + ResNet-34 con tabla híbrida | 0.898 | 0.8599 (415) | 0.881 |
| 13 oct | Trío Base+D v2 | Joako | ResNet-34 (25 %) + ResNet-34 con atención por hallazgo (25 %) + ConvNeXt-Tiny (50 %) | 0.904 | 0.8628 (415) | — |
| 13 oct | `cuadruple_abcd_corregido` | Joako | Cuádruple A+B+C+D | 0.905 | — | — |
| 13 oct | **`v3_cv15_r34_eff_cnxhib`** | Owen | 5 particiones de ResNet-34 + 5 de EfficientNetV2-S + 5 de ConvNeXt-Tiny con tabla híbrida, media | **0.912** | 0.8726 | 0.893 |
| 13 oct | **`v5_cv20_eff_cnxhib_effhib_r34att`** | Owen | 5 particiones de EfficientNetV2-S + 5 de ConvNeXt-Tiny híbrida + 5 de EfficientNetV2-S híbrida + 5 de ResNet-34 híbrida con atención por hallazgo, media | **0.918** | 0.8759 | 0.901 |

Paquetes: `envios/v1_resnet34/`, `envios/v3_cv15_r34_eff_cnxhib/` y su `_paquete.zip`; los de Joako en `code_joako/envios_joako/`; el de Cons en `runs_cons/envio/`. Todos los envíos están anotados en MLflow (corridas `envio-*`).

**Armado y sin enviar:** `envios/v4_cv20_r34_eff_cnxhib_effhib_paquete.zip`: como el v5 pero con el ResNet-34 original en vez del de atención por hallazgo (0.8756 y 0.8985). Lo supera el v5 y no hace falta subirlo.

La validación local ha anticipado las dos mejoras: +0.012/+0.014 local y +0.015 en Kaggle del v3; +0.003/+0.008 local y +0.006 en Kaggle del v5.

## Etiquetas

| Intento | Resultado | Dónde |
|---|---|---|
| Extraer los 12 hallazgos de los informes con `qwen2.5:72b` (estado y gravedad) | Base original. Acuerdo del 83 % con las etiquetas oficiales | `etiquetas/extraer.py`, `etiquetas/consolidar.py`; salida en `runs/2079118/` |
| Lo mismo con MedGemma 27B de texto | Mismo 83 %: el límite es el informe, no el lector | `etiquetas/prueba_ollama.py` |
| Etiquetar mirando imágenes con MedGemma 1.5 4B | No sirve: AUC 0.61–0.66 | `imagenes/prueba_medgemma.py` |
| Objetivos graduados (probabilidad por estado y gravedad) con peso; "no mencionado" sin peso | En uso. Tabla `preproc/etiquetas_v1_2079203.parquet` | `rsna/tabla_entrenamiento.py` |
| **Tabla híbrida de Cons**: Qwen + etiquetas públicas de otros participantes | **Gana y es la tabla base.** Misma red y particiones: ResNet-34 0.8464 frente a 0.8413; ConvNeXt-Tiny 0.8610 frente a 0.8542; EfficientNetV2-S 0.8644 frente a 0.8592. En oficiales +0.006 de media | `runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet`, `runs_cons/auditoria/` |
| Rellenar "no mencionado" con predicciones fuera de muestra | Solo respaldado en sinovitis (AUC 0.77 en las 33 celdas sin mención de los oficiales; Baker 0.32). Sin entrenar todavía | `rsna/tabla_corregida.py`, `jobs/tabla_corregida.slurm` |
| Bajar el peso de las celdas que el modelo contradice (mezcla de gaussianas o umbral) | Descartado: dos de cada tres celdas marcadas tenían la etiqueta correcta | mismo script |
| Reglas entre hallazgos o árbol sobre la extracción de Qwen | Sin mejora: 0.801 y 0.765 frente a 0.794 (58 oficiales, dejando uno fuera) | `rsna/evaluar_reglas_etiquetas.py` |

## Modelos

Arquitectura (`rsna/modelo.py`): por estudio, 4 series (sagital con y sin supresión grasa, coronal, axial); de cada una, 16 cortes con sus dos vecinos como canales; red 2D preentrenada en ImageNet; atención entre cortes; concatenación de las series y 12 salidas. Entrenamiento en `rsna/entrenar.py`, lanzado con `jobs/entrenar.slurm`. Todas las redes parten de pesos de ImageNet (`modelos/torch/`).

Validación cruzada de 5 particiones, 8 épocas, última época. Las carpetas de cada corrida están en `runs/<job>/` (pesos `last.pt`, `historial.csv`, `predicciones.npz`).

| Grupo | Fuera de muestra | Oficial (58) | Corridas (partición 0 a 4) |
|---|---|---|---|
| ResNet-34 | 0.8413 | 0.860 | 2084051, 2084052, 2084350, 2084351, 2084050 |
| ConvNeXt-Tiny | 0.8542 | 0.881 | 2084323, 2084324, 2084546, 2084547, 2084322 |
| EfficientNetV2-S (lote 4, tasa 1e-4) | 0.8592 | 0.886 | 2084365, 2084585, 2084586, 2084626, 2084584 |
| ResNet-34, tabla híbrida | 0.8464 | 0.866 | 2084588, 2084593, 2084608, 2084619, 2084587 |
| ConvNeXt-Tiny, tabla híbrida | 0.8610 | 0.887 | 2084671, 2084673, 2084676, 2084684, 2084633 |
| EfficientNetV2-S, tabla híbrida | 0.8644 | 0.893 | 2084674, 2084683, 2084689, 2084694, 2084634 |
| ResNet-34, híbrida, **atención por hallazgo** | 0.8502 | 0.877 | 2084756, 2084769, 2084771, 2084783, 2084722 |
| ResNet-34, híbrida, desde preentrenamiento sin etiquetas | 0.8495 | 0.868 | 2084726, 2084728, 2084742, 2084743, 2084713 |

Otras pruebas, todas con ResNet-34 en 5 particiones salvo indicación:

| Intento | Resultado | Dónde |
|---|---|---|
| EfficientNetV2-S con lote 8 | Agota los 64 GB de la GPU; con lote 4 cabe | job 2084361 |
| Más resolución (tensores a 384, recorte 352) | No mejora: 0.8403 frente a 0.8413 | corridas 2084405, 2084541, 2084554, 2084559, 2084364 |
| Aumentación fuerte (giro, escala, desplazamiento, gamma, ruido, borrado, apagar una serie), 12 épocas | No mejora: 0.8427 | `--aug fuerte --quitar-serie`; 2084650, 2084652, 2084653, 2084657, 2084625 |
| Lo anterior + mezcla de estudios | Peor: 0.8381 | `--mixup`; 2084654, 2084658, 2084659, 2084666, 2084628 |
| Lo anterior + consistencia entre versiones | No mejora: 0.8424 | `--consistencia`; 2084664, 2084665, 2084667, 2084672, 2084629 |
| **Atención sobre los cortes propia de cada hallazgo** (modelo D de Joako) | **Mejora**: +0.004 fuera de muestra y +0.010 en oficiales; menisco lateral 0.79 → 0.87 | `--atencion hallazgo` |
| Preentrenamiento sin etiquetas sobre nuestros cortes (MoCo v3 desde ImageNet, 300 pasadas, 9 h) | Aporta poco: +0.003 y +0.002. La pérdida se estanca desde la pasada 50 | `rsna/preentrenar_moco.py`, `jobs/preentrenar_moco.slurm`; pesos en `runs/2084630/` |
| Quinta vista, coronal sin supresión grasa (Joako, una partición) | +0.002 no significativo | `code_joako/rsna/entrenar_vistas.py` |
| EfficientNetV2-S con 5 series a 320 sin recorte (tensores de Joako) | Sin mejora en lo que va: partición 0 con 0.8635 frente a 0.8762 del de 4 series (interna contra la tabla híbrida) | arreglo 2084715, en curso |

## Combinación de modelos e inferencia

| Intento | Resultado | Dónde |
|---|---|---|
| Media de probabilidades de redes y particiones | Lo que se envía. De 2 a 15 modelos: 0.897 → 0.912 en Kaggle | `rsna/inferir.py`, `kaggle/inferencia.ipynb` |
| Elegir la combinación con las predicciones fuera de muestra | Las diez mejores combinaciones quedan a menos de 0.001 entre sí | `rsna/evaluar_oof.py`; resultados en `runs/2084709/`, `runs/2084786/`, `runs/2084787/` |
| Promedio de rangos | Peor que la media | `rsna/evaluar_conjunto.py` |
| Pesos por modelo y por hallazgo (regresión logística) | No mejora: 0.8453 frente a 0.8462 | `rsna/evaluar_apilado.py` |
| Segunda etapa con los 12 hallazgos (regresión logística, gradient boosting, bosque aleatorio) | Peor: 0.8442, 0.8344 y 0.8403 frente a 0.8462 | mismo script |
| Usar la salida de derrame para predecir sinovitis | No: 0.737 frente a 0.765 de la salida propia; mezclarlas no cambia el AUC macro | `rsna/evaluar_cruzado.py` |
| TTA con desplazamiento de cortes (Joako) | +0.0011, no significativo | `code_joako/rsna/evaluar_tta.py` |
| Tiempo de inferencia | 15 modelos: 0.77 s por estudio en una MI210 (unos 17 min para 1,300 estudios) | `jobs/inferir.slurm`; `runs/2084712/` |

## Descartado sin probar, y por qué

| Idea | Motivo |
|---|---|
| Subclases por agrupamiento (4S-DT, k-means, DBSCAN, espectral) | Hay supervisión para los 12 hallazgos; agrupar no crea etiquetas |
| Clasificadores de scikit-learn sobre las imágenes | Trabajan sobre tablas; sobre las salidas de la red ya se midió |
| Optuna | Para pesos de ensamble, ver arriba. Para hiperparámetros, cada medida fiable cuesta 2.5 horas-GPU con un tope de 4 GPU |
| Tareas de pretexto (coloreado, rompecabezas) | No aplican a imágenes de un canal ni a anatomía que no se puede desordenar |
| Repetir el preentrenamiento sin etiquetas con otras redes | 9 horas de GPU por red para +0.003 |

## En cola o corriendo (13 de octubre, 15:15)

El proyecto tiene un tope de 4 GPU simultáneas; el orden lo decide el planificador.

| Job | Qué es | Estado |
|---|---|---|
| 2084727 | EfficientNetV2-S con **6 series** a 320, tabla híbrida, 5 particiones (tensores `preproc/v3_6s_train`) | 2 corriendo, 3 en cola; ~3.5 h por partición |
| 2084715 | EfficientNetV2-S con 5 series a 320 (tensores de Joako) | 2 terminadas, 1 corriendo, 2 en cola; candidato a cancelar |
| 2084801 | EfficientNetV2-S con atención por hallazgo, tabla híbrida, 5 particiones | en cola |
| 2084802 | ConvNeXt-Tiny con atención por hallazgo, tabla híbrida, 5 particiones | en cola |
| 2084790 | Ensayo de inferencia del paquete de 20 modelos | en cola |
| 2084782 | `rsna-eff-hib0`, de Joako | en cola |

## Pendiente

1. Enviar el paquete de 20 modelos cuando pase su ensayo de inferencia.
2. Si la atención por hallazgo se confirma en EfficientNetV2-S y ConvNeXt-Tiny, armar el ensamble con esas versiones.
3. Decidir con las 6 series si se rehace todo con ellas.
4. Rellenar la sinovitis no mencionada con las predicciones fuera de muestra del ensamble y reentrenar.
5. Preentrenamiento sin etiquetas con la aumentación ganadora (propuesta de Fernando): sin objeto por ahora, porque ninguna aumentación ganó y el preentrenamiento aporta poco.
6. Sin probar: más peso a los positivos en hallazgos raros, red recurrente entre cortes, red 3D (R3D-18), pseudoetiquetar el test dentro del notebook (falta confirmar que las reglas lo permiten).
7. Elegir los dos envíos finales antes del 22 de octubre.

## Mapa de archivos

### Código (`code/`, rama `owen` de `FernyBoy/RSNA-Knee`)

| Archivo | Para qué |
|---|---|
| `eda/` | Exploración y verificación del dataset; salidas en `code/data/eda/` (sin versionar) |
| `etiquetas/extraer.py`, `consolidar.py`, `evaluar_extraccion.py` | Extracción de etiquetas de los informes con un modelo de lenguaje local |
| `rsna/preproc.py`, `preprocesar.py` | De DICOM a tensores de tamaño fijo; 4 o 6 series, lado configurable |
| `rsna/tabla_entrenamiento.py` | De la extracción a objetivos graduados con peso |
| `rsna/tabla_corregida.py` | Relleno y desconfianza con predicciones fuera de muestra |
| `rsna/modelo.py` | Red, aumentación y preparación de los cortes |
| `rsna/entrenar.py` | Entrenamiento por partición; guarda pesos, historial y predicciones por época |
| `rsna/preentrenar_moco.py` | Preentrenamiento sin etiquetas |
| `rsna/evaluar_oof.py` | Validación cruzada: cada grupo y sus combinaciones |
| `rsna/evaluar_apilado.py`, `evaluar_cruzado.py`, `evaluar_reglas_etiquetas.py`, `evaluar_conjunto.py` | Pruebas de segunda etapa, relaciones entre hallazgos y promedios |
| `rsna/inferir.py`, `kaggle/inferencia.ipynb` | Inferencia de DICOM a `submission.csv`; lo que corre en Kaggle |
| `rsna/registrar_mlflow.py`, `registrar_envio.py`, `registrar_lote.py`, `registros/*.json` | Anotación en MLflow de corridas, envíos y resúmenes |
| `jobs/*.slurm` | Un job por cada script; todo lo pesado se lanza desde aquí. `jobs/armar_envio.slurm` arma el paquete de un envío |
| `logs/` | Salida de cada job (sin versionar) |

Opciones de `jobs/entrenar.slurm` por variable de entorno: `KNEE_LABELS` (tabla de objetivos), `KNEE_PREPROC` (carpeta de tensores), `KNEE_WEIGHTS` (pesos de partida). Como arreglo (`--array=0-4%2` con `--folds 5`) cada tarea toma su partición.

### Datos y resultados

| Ruta | Qué es |
|---|---|
| `kaggle_data/` | Dataset de la competencia. Solo lectura |
| `preproc/v1_train/` | Tensores de 4 series a 320 (44 GB): los que usan todos los modelos enviados |
| `preproc/v3_6s_train/` | Tensores de 6 series a 320 (48 GB) |
| `preproc_joako/v2_train/` | Tensores de 5 series de Joako |
| `preproc/etiquetas_v1_2079203.parquet` | Tabla de objetivos original |
| `runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet` | Tabla híbrida de Cons, la base actual |
| `runs/<job>/` | Salida de cada job de Owen |
| `envios/` | Paquetes para Kaggle |
| `mlflow/mlflow.db` | Seguimiento de experimentos (experimento `rsna-knee`; etiquetas `autor`, `grupo`, `tipo`) |
| `modelos/torch/` | Pesos de ImageNet de las tres redes |
| `code_joako/`, `runs_joako/` | Código, bitácora (`experiments/journal.md`) y resultados de Joako |
| `runs_cons/` | Auditoría de etiquetas, entrenamiento y envío de Cons |
| `_to_delete/2026-10-13/` | 66 GB apartados para borrar (tensores de 384, pruebas, pesos de experimentos descartados); ver su `LEEME.txt` |

## Reglas que hay que respetar

- Los datos de la competencia no salen del cluster: ni informes, ni imágenes, ni identificadores de estudio, ni etiquetas derivadas, en GitHub o servicios externos.
- El proyecto tiene 700 GB de cuota y un tope de 4 GPU simultáneas, compartidos por todo el equipo.
