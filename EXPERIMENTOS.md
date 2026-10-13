# Qué se ha intentado y cómo resultó

Estado al 12 de octubre de 2026, 21:00. Cierre de envíos: 22 de octubre.
Mejor puntuación en la tabla pública de Kaggle: **0.898** (objetivo: pasar de 0.900).

Cómo leer los números:

- **Kaggle**: AUC en la tabla pública. Es la única medida contra verdad con muchos casos, pero hay 5 envíos al día y diferencias de 0.001 son ruido.
- **Oficial (58)**: AUC contra las etiquetas reales de los 58 estudios que las traen. Pocos casos (margen de unos ±0.03) y no es independiente: se usaron para calibrar los objetivos y para elegir época en los envíos actuales.
- **Interna**: AUC contra los objetivos derivados de los informes, en estudios que el modelo no vio. Hasta el 12 de octubre era una sola partición de 415 estudios; desde entonces, validación cruzada de 5 particiones sobre 4,349 estudios ("fuera de muestra"). Mide el acuerdo con las etiquetas derivadas, no con la verdad.

## Envíos a Kaggle

| Envío | Autor | Contenido | Kaggle | Interna (415) | Oficial (58) |
|---|---|---|---|---|---|
| v1 | Owen | ResNet-34, época 7 | 0.875 | 0.8435 | 0.858 |
| Joako v1 | Joako | ResNet-34 ép. 7 + ConvNeXt-Tiny ép. 5, media | 0.897 | 0.8574 | 0.879 |
| Cons v3 | Cons | Los dos anteriores + ResNet-34 con etiquetas híbridas, media | 0.898 | 0.8599 | 0.881 |

## Etiquetas

| Intento | Resultado |
|---|---|
| Extraer los 12 hallazgos de los informes con `qwen2.5:72b` (estado y gravedad, no solo sí/no) | Funciona y es la base de todo. Acuerdo del 83 % con las etiquetas oficiales. Bien en ligamentos y menisco medial (91–95 %); mal en derrame, contusión y sinovitis (69–72 %). |
| Lo mismo con un modelo médico (MedGemma 27B de texto) | Mismo 83 %, cuatro veces más rápido. El límite es lo que dice el informe, no quién lo lee. |
| Etiquetar mirando las imágenes con MedGemma 1.5 4B sin ajuste | No sirve: AUC de 0.61 a 0.66. Algo ve en derrame y sinovitis, nada en ligamentos y meniscos. |
| Objetivos graduados en vez de 0/1; "no mencionado" sin peso | En uso. Deja sin objetivo el 72 % de los estudios en sinovitis y el 42 % en Baker. |
| Tabla híbrida: Qwen + etiquetas públicas de otros participantes (Cons) | Prometedor, sin demostrar. Las etiquetas públicas coinciden mejor con los 58 oficiales (0.887 frente a 0.820). El ResNet-34 reentrenado sube el ensamble de 0.8574 a 0.8582 al medirlo con la misma tabla; el "+0.019" que figura en su informe compara tablas distintas. En Kaggle, +0.001. **Prueba en curso** en 5 particiones (arreglo 2084587). |

## Modelos y entrenamiento

Arquitectura común (`rsna/modelo.py`): cuatro series por estudio (sagital con y sin supresión grasa, coronal, axial), cada corte con sus dos vecinos por una red 2D preentrenada, atención entre cortes y concatenación de las cuatro series.

| Intento | Resultado |
|---|---|
| ResNet-34, 5 particiones | Interna fuera de muestra 0.8413. Cada modelo suelto da 0.818–0.855 en los oficiales; el promedio de los cinco, 0.860. Promediar particiones ayuda. |
| ConvNeXt-Tiny, 5 particiones | Cuatro terminadas: interna 0.847–0.858, oficial 0.850–0.875. Seis veces más lenta que ResNet-34 (2 h 45 min por partición). |
| EfficientNetV2-S | Con lote 8 agota los 64 GB de la GPU. Con lote 4 y tasa 1e-4 cabe: partición 0 con interna 0.853 y oficial 0.860, la mejor red suelta en esa partición. Resto de particiones en curso (arreglo 2084584). |
| Más resolución (tensores a 384, recorte 352) con ResNet-34 | **No mejora**: 0.8403 frente a 0.8413 fuera de muestra, 0.857 frente a 0.860 en oficiales, con 1.5 veces el cómputo. |
| Quinta vista, coronal sin supresión grasa (Joako) | **No mejora**: +0.002 con intervalo que incluye el cero; fractura empeora. |
| Más épocas | Sin indicios de que ayude: ResNet-34 se aplana entre las épocas 7 y 8; en EfficientNetV2-S la pérdida sigue bajando pero el AUC se detiene en la época 4 (sobreajuste leve a las etiquetas). |

## Combinación de modelos e inferencia

| Intento | Resultado |
|---|---|
| Media de probabilidades de dos redes distintas | La mayor mejora hasta ahora: de 0.875 a 0.897 en Kaggle. |
| Promedio de rangos en vez de probabilidades | Peor: 0.8588 frente a 0.8609 interna, 0.871 frente a 0.875 oficial. |
| Pesos distintos por modelo (Cons, 5 combinaciones) | Diferencias de 0.0005 sobre 415 estudios; no distinguible del ruido. |
| Pesos por modelo y por hallazgo aprendidos (regresión logística) | **No mejora**: 0.8453 frente a 0.8462 del promedio. |
| Segunda etapa con los 12 hallazgos a la vez (regresión logística, gradient boosting, bosque aleatorio) | **Peor**: 0.8442, 0.8344 y 0.8403 frente a 0.8462. Las co-ocurrencias entre hallazgos ya las aprovecha la red. Medido con dos grupos de ResNet-34; se repetirá con las tres redes. |
| TTA con desplazamiento de cortes (Joako) | +0.0011, intervalo que incluye el cero. |

## Descartado sin probar, y por qué

| Idea | Motivo |
|---|---|
| Preentrenamiento autosupervisado (MICLe, DINO, SimCLR) | 4,407 estudios y diez días; lo más fácil de aprender es "mismo paciente y mismo equipo". No hay código ni pesos en el cluster. |
| Subclases por agrupamiento (4S-DT, k-means, DBSCAN, espectral) | Hay supervisión para los 12 hallazgos; agrupar no crea etiquetas. La estructura entre hallazgos no aportó en la segunda etapa. |
| Clasificadores de scikit-learn (SVC, vecinos, Naive Bayes) | Trabajan sobre tablas; aquí solo podrían usarse sobre las salidas de la red, y eso ya se midió. |
| Optuna | Para pesos de ensamble, ver arriba. Para hiperparámetros, cada medida fiable cuesta 2.5 horas-GPU y el proyecto tiene un tope de 4 GPU simultáneas. |
| Entrenar primero con las etiquetas manuales y añadir después las derivadas | Solo hay 58 estudios con etiquetas manuales. |
| Volteo horizontal como aumento | Solo sería válido como espejo coherente en los tres planos; no se ha implementado y el TTA aporta poco. |

## Sin probar todavía, por orden de interés

1. Reentrenar con etiquetas corregidas usando las predicciones fuera de muestra: rellenar las celdas "no mencionado" y bajar el peso de las que el modelo contradice con fuerza.
2. Reentrenar ConvNeXt-Tiny y EfficientNetV2-S con la tabla híbrida, si gana la prueba en curso.
3. Penalización de consistencia entre versiones alteradas de la misma imagen (diferencias entre equipos y hospitales).
4. Más peso a los positivos en los hallazgos raros y peores (MCL, contusión, fractura: 8–10 % de positivos, AUC de 0.77–0.79). Con AUC como métrica se espera poco; la pérdida focal tiene el riesgo de concentrarse en etiquetas erróneas.
5. Red recurrente (GRU) entre cortes en vez de la atención actual.
6. Red 3D preentrenada en video (R3D-18) como miembro distinto del ensamble.
7. Pseudoetiquetar el test dentro del notebook. Falta confirmar que las reglas lo permiten.

## Incidencias operativas

- El proyecto tiene un tope de GPU simultáneas (4). Mientras corre una validación cruzada, el resto del equipo no puede usar GPU.
- MLflow rechazaba las métricas por hallazgo por el apóstrofo de "Baker's"; corregido el 12 de octubre. Las corridas anteriores no las tienen.
- `preproc/v2_384_train` (55 GB) solo sirvió para la prueba de resolución.
- Todas las corridas y los envíos están en MLflow (experimento `rsna-knee`), con etiquetas `autor` y `grupo`. Lo registrado a mano está en `registros/`.
