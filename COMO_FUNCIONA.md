# Cómo revisa el modelo cada estudio

Explicación para el equipo, sin matemáticas. Para los resultados de cada experimento, ver
[EXPERIMENTOS.md](EXPERIMENTOS.md).

Primero una aclaración: no son radiografías. Cada estudio es una **resonancia magnética de rodilla**:
varias *series* (secuencias tomadas en distintos planos y con distinto contraste), y cada serie es
una pila de 20 a 40 *cortes*, imágenes en escala de grises de una rebanada de la rodilla. Un estudio
típico trae 5 series y unos 180 cortes.

Para cada estudio el modelo devuelve 12 números entre 0 y 1: la probabilidad de cada hallazgo (rotura
de LCA, de LCM, de menisco medial o lateral, artrosis medial, lateral o patelofemoral, derrame,
sinovitis, quiste de Baker, contusión ósea y fractura). Kaggle mide qué tan bien ordenan esas
probabilidades a los estudios con y sin hallazgo (AUC), no si aciertan un sí o un no.

## 1. Preparar el estudio (`rsna/preproc.py`)

```
DICOM del estudio (≈180 cortes en 5 series)
   │
   ├─ elegir 4 series: sagital con supresión grasa, sagital sin ella, coronal y axial
   ├─ ordenar los cortes por su posición en el espacio (no por nombre de archivo)
   ├─ girarlos a una orientación común (mismo arriba, abajo, delante y detrás para todos)
   ├─ ajustar el brillo de cada serie: lo más oscuro y lo más claro de esa serie van a 0 y 255
   └─ llevar cada serie a 32 cortes de 320 × 320 píxeles
   ▼
un bloque de 4 × 32 × 320 × 320 por estudio (unos 13 MB), guardado una sola vez en preproc/
```

Esto se hace igual al entrenar en el cluster y al predecir en Kaggle, con el mismo código.

**El lado de la rodilla.** Una rodilla izquierda es el espejo de una derecha: lo que en una queda a
la izquierda de la imagen (por ejemplo, el menisco medial), en la otra queda a la derecha. Hasta el
13 de octubre el modelo veía las dos mezcladas y tenía que adivinar cuál era cuál. Ahora el lado se
lee de la cabecera DICOM o, si falta, de la posición de la rodilla en el escáner (acierta el 98 %), y
las izquierdas se voltean para que todas queden como derechas. Fue la mejora más grande que hemos
tenido con un solo cambio (+0.01 a +0.02 de AUC).

## 2. Mirar cada corte (`rsna/modelo.py`)

```
de cada serie se toman 16 de los 32 cortes; cada uno va con sus dos vecinos como si fueran
los tres canales de color de una foto (así la red ve un poco de profundidad)
   │
   ▼
red de imágenes preentrenada (ResNet-34, ConvNeXt-Tiny o EfficientNetV2-S), que ya aprendió a
reconocer bordes, texturas y formas con millones de fotos (ImageNet) y que aquí se reentrena
   │
   ▼
cada corte se resume en una lista de ~500 a 1,300 números que describe lo que contiene
```

Son 4 series × 16 cortes = **64 imágenes por estudio**.

## 3. Decidir qué cortes importan

No todos los cortes sirven para todo: el LCA se ve en unos pocos cortes sagitales del centro, y el
derrame en otros. Una pequeña red de **atención** asigna un peso a cada corte, y el resumen de cada
serie es el promedio pesado de sus cortes. En los modelos con "atención por hallazgo" cada uno de
los 12 hallazgos tiene sus propios pesos: el modelo puede mirar unos cortes para el menisco y otros
para el derrame.

## 4. Juntar las series y dar las 12 probabilidades

Los resúmenes de las 4 series se ponen uno junto a otro y pasan por dos capas finales que dan 12
números; cada uno se convierte en probabilidad entre 0 y 1.

```
estudio → 4 series → 64 cortes → 64 resúmenes → atención → 4 resúmenes de serie → 12 probabilidades
```

## 5. Con qué aprende (`rsna/tabla_entrenamiento.py`, tabla híbrida de Cons)

Solo 58 de los 4,407 estudios de entrenamiento traen las 12 etiquetas oficiales. Para el resto
usamos el **informe del radiólogo**: un modelo de lenguaje lo lee y dice, por cada hallazgo, si está
presente, posible, ausente o no mencionado, y con qué gravedad. Eso se convierte en un objetivo
graduado ("presente y grave" vale casi 1; "ausente", casi 0) con un peso: lo que el informe no
menciona no se usa para enseñar. Hoy usamos la tabla híbrida de Cons, que combina nuestra lectura
con etiquetas públicas de otros participantes y se parece más a las oficiales.

Los 58 oficiales nunca se usan para entrenar: son nuestra referencia contra la verdad.

## 6. Cómo sabemos si mejora

- **Validación cruzada de 5 particiones**: el modelo se entrena 5 veces, cada vez dejando fuera un
  quinto distinto de los estudios. Así cada estudio recibe una predicción de un modelo que nunca lo
  vio ("fuera de muestra"), y se compara contra las etiquetas derivadas de los informes.
- **Los 58 oficiales**: cada modelo los predice sin haberlos visto; es la medida contra la verdad,
  aunque con pocos casos.
- Hasta ahora, lo que mejora en estas dos medidas ha mejorado en Kaggle en una cantidad parecida.

## 7. Juntar muchos modelos (ensamble)

Cada red se entrena en las 5 particiones, y se promedian las probabilidades de todos los modelos.
Redes distintas se equivocan en casos distintos y el promedio se equivoca menos: así pasamos de 0.875
(un modelo) a 0.920 (20 modelos de cuatro grupos).

## 8. En Kaggle (`rsna/inferir.py`, `kaggle/inferencia.ipynb`)

El notebook no tiene internet: se le sube como dataset un paquete con el código, los pesos de los
modelos y las librerías para leer DICOM comprimido. Lee cada estudio de test, hace exactamente la
misma preparación (incluido detectar el lado), pasa el estudio por todos los modelos, promedia y
escribe `submission.csv`. Con 20 modelos tarda alrededor de una hora de las 9 permitidas.

Las celdas de `kaggle/mezcla_con_publicos.py` y `kaggle/mezcla_ramas.py` permiten sumar nuestra
rama a notebooks públicos y mezclar las predicciones por rangos.

## Qué falla todavía

En los 58 oficiales el ensamble está por encima de 0.94 en LCA, LCM, Baker, menisco medial,
artrosis medial y derrame, pero alrededor de 0.78–0.80 en **artrosis lateral y sinovitis**. En los
dos casos el límite parece estar en las etiquetas: el informe exagera la artrosis lateral y casi
nunca menciona la sinovitis. Lo que más ha ayudado ahí es mezclar el objetivo del informe con lo que
el propio ensamble ve en la imagen (destilación).
