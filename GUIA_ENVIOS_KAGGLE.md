# Guía: de los modelos entrenados en Yuca a un envío en Kaggle

Para cualquiera del equipo que no haya hecho un envío todavía. Son cinco pasos: elegir modelos,
armar el paquete, probarlo en el cluster, subirlo a Kaggle y enviar.

## 0. Cómo funciona un envío en esta competencia

- Es una competencia **de código**: no se sube un CSV, se sube un **notebook** de Kaggle. Kaggle lo vuelve a
  correr sobre el test oculto (~1,300 estudios) y puntúa el `submission.csv` que deja.
- El notebook corre **sin internet**, con un límite de **9 horas**. Todo lo que necesita (nuestro código, los pesos y
  las librerías para leer DICOM comprimido) tiene que estar adjunto como un **dataset privado** de Kaggle: eso es el
  "paquete".
- Hay **5 envíos al día** y, antes del **22 de octubre a las 23:59 UTC**, hay que **seleccionar 2** como finales.

```
Yuca: runs/<job>/last.pt  ──►  envios/<nombre>_paquete.zip  ──►  dataset privado en Kaggle
                                                                        │
                                       notebook de Kaggle (inferencia.ipynb) ◄┘
                                                │
                                         submission.csv ──► puntuación
```

## 1. Elegir qué modelos van

Cada experimento entrena una red en **5 particiones**; cada partición es una corrida con su carpeta
`runs/<job>/` y sus pesos `last.pt`. Un paquete junta varios grupos de 5.

- Los grupos, sus identificadores y su puntuación están en `code/EXPERIMENTOS.md` y en MLflow (experimento
  `rsna-knee`, etiqueta `grupo`).
- Para decidir qué combinación conviene, se mide antes de gastar un envío:

```bash
cd /lustre/proyectos/p082/code
sbatch --export=ALL,KNEE_LABELS=/lustre/proyectos/p082/runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet \
    jobs/evaluar_oof.slurm r34lados=2084986,2085001,2085007,2085014,2084973 efflados=2085028,2085029,2085063,2085064,2084975
```

  Deja en `logs/rsna-evaluar-oof_<job>.out` una tabla de combinaciones ordenada de mejor a peor ("interna" = fuera de
  muestra sobre 4,349 estudios; "oficial" = los 58 estudios con etiqueta real). Hasta ahora, lo que sube en esas dos
  columnas ha subido en Kaggle en una cantidad parecida.

## 2. Armar el paquete

```bash
cd /lustre/proyectos/p082/code
sbatch jobs/armar_envio.slurm v8_mi_nombre \
    r34destil=2085149,2085150,2085156,2085157,2085146 \
    efflados=2085028,2085029,2085063,2085064,2084975
```

- El primer argumento es el nombre del envío; después, un `grupo=job,job,...` por cada grupo.
- Copia `rsna/inferir.py`, `rsna/modelo.py` y `rsna/preproc.py` **tal como están en ese momento**, el `last.pt` de
  cada corrida (renombrado `grupo_job<id>.pt`), las librerías (`wheels/`) y un `MANIFIESTO.txt`.
- Resultado: la carpeta `envios/v8_mi_nombre/` y el archivo `envios/v8_mi_nombre_paquete.zip`. Si el nombre ya existe,
  el job se detiene sin sobrescribir.
- Tamaño orientativo: ~90 MB por ResNet-34, ~85 MB por EfficientNetV2-S, ~115 MB por ConvNeXt-Tiny.
- Cada modelo guarda cómo se entrenó (lado unificado, recorte, series...), y `inferir.py` lo respeta solo. Se pueden
  mezclar modelos con opciones distintas en un mismo paquete.

## 3. Probarlo en el cluster antes de subirlo

```bash
cd /lustre/proyectos/p082/code
# en GPU: 160 estudios de entrenamiento; mide el tiempo y valida el formato
sbatch -J rsna-inferir-v8 jobs/inferir.slurm train 160 /lustre/proyectos/p082/envios/v8_mi_nombre/pesos/*.pt
# si la cola de GPU está llena: en CPU con pocos estudios (solo valida que corre y el formato)
sbatch -p cpu --gres=none --cpus-per-task=32 --mem=64G --time=01:00:00 -J rsna-inferir-cpu \
    jobs/inferir.slurm train 6 /lustre/proyectos/p082/envios/v8_mi_nombre/pesos/*.pt
```

En `logs/<nombre>_<job>.out` hay que ver:

- `columnas iguales a sample_submission: True`, `valores fuera de [0,1]: 0`, `NaN: 0`, `ilegibles con 0.5: 0`;
- el número de líneas `modelo ...` igual al número de pesos del paquete;
- en GPU, `a este ritmo, 1,300 estudios tardarían N min`. En Kaggle tarda unas 3–4 veces más (GPU T4 y 4 núcleos).
  Con 25 modelos son ~1.5 h, lejos de las 9 h.

## 4. Subir el paquete a Kaggle

Primero, traer el `.zip` a tu computadora (son pesos y código nuestros: no llevan datos de la competencia):

```bash
scp TU_USUARIO@yuca.unison.mx:/lustre/proyectos/p082/envios/v8_mi_nombre_paquete.zip .
```

Luego, en kaggle.com:

1. **Datasets → New Dataset**. Arrastra el `.zip` (Kaggle lo descomprime solo).
2. Ponle un título reconocible (p. ej. `rsna-knee-p082-v8`) y déjalo **Private**. **Create**.
3. Espera a que termine de procesarse (un par de minutos para 2 GB).

Para una versión nueva del mismo paquete se puede usar **New Version** en ese dataset, pero es más claro crear uno
nuevo por envío.

## 5. Notebook y envío

### Opción A: solo nuestro ensamble

1. **Code → New Notebook → File → Import Notebook** y sube `code/kaggle/inferencia.ipynb` (está en el repo de GitHub,
   rama `owen`).
2. En el panel derecho:
   - **Add Input** → busca la competencia *RSNA Knee Abnormality Detection* y añádela.
   - **Add Input** → *Your Datasets* → añade tu paquete.
   - **Session options → Accelerator: GPU T4 x2**.
   - **Internet: off** (obligatorio en esta competencia).
3. **Run All** para comprobar que no falla. Con el test público (3 estudios) tarda unos minutos y debe imprimir
   `submission.csv correcto`.
4. **Save Version → Save & Run All (Commit)**. Espera a que la versión termine en verde.
5. En la página de la competencia: **Submit Prediction** → elige tu notebook y esa versión → archivo
   `submission.csv` → **Submit**. Kaggle lo vuelve a correr con el test oculto; la puntuación aparece en
   *My Submissions* después de 1–2 horas.

El notebook encuentra solo el paquete y los datos, así que no hay que tocar rutas.

### Opción B: nuestro ensamble como tercera rama del notebook público de 0.957

Es lo que más puede subir la puntuación. Usa `envios/notebooks/0957_mas_yuca.ipynb` (el notebook público sin
cambios más tres celdas nuestras al final; el código de esas celdas está en `code/kaggle/mezcla_con_publicos.py`).

1. Lo más sencillo: en Kaggle, abre el notebook público ya enviado por Joaquín → **Copy & Edit**. Así hereda todas sus
   entradas (Raptor, R7, librerías).
2. Pega al final las dos celdas de `code/kaggle/mezcla_con_publicos.py` (cada `# %%` es una celda).
3. **Add Input** → tu paquete. GPU T4 x2, internet off.
4. Igual que en la opción A: Run All, Save & Run All, Submit.

Las celdas nuevas solo corren nuestra rama si queda tiempo bajo las 9 h. Si no hay tiempo o algo falla, envían la
mezcla pública tal cual: el notebook nunca termina en error. En el registro, busca
`tiempo restante para nuestra rama` y, al final, `mezcla pública + propia` o `solo la mezcla pública`.

## 6. Después del envío

- Apunta la puntuación en `code/EXPERIMENTOS.md` (tabla de envíos) y en MLflow: crea un `registros/AAAA-MM-DD_<nombre>.json`
  como los que ya hay y corre `sbatch jobs/registrar_lote.slurm registros/AAAA-MM-DD_<nombre>.json`.
- Avisa en el grupo qué enviaste, para no gastar dos envíos en lo mismo.
- Antes del cierre, en *My Submissions*, marca los **2 envíos finales**. Tiene que hacerlo alguien del equipo; valen
  para todos.

## Problemas típicos

| Síntoma | Causa | Arreglo |
|---|---|---|
| `No se encuentra el paquete` | El dataset no está añadido como entrada | **Add Input** en el notebook |
| `No se encuentran los datos de la competencia` | Falta la competencia como entrada | **Add Input** → la competencia |
| Error al instalar `pydicom` o `pylibjpeg` | El paquete no trae `wheels/` | Volver a armarlo con `jobs/armar_envio.slurm` |
| El envío falla por tiempo | Demasiados modelos o notebook público lento | Paquete con menos modelos, o la opción B (tiene tope) |
| Puntuación peor de lo esperado | Combinación no medida | Medir antes con `jobs/evaluar_oof.slurm` |

## Reglas que no hay que romper

- **Nada de datos de la competencia fuera del cluster**: ni informes, ni imágenes, ni tablas con `StudyInstanceUID`.
  Los paquetes solo llevan código, pesos y librerías.
- No se comparten paquetes ni código con otros equipos.
- Los tokens de Kaggle (`kaggle.json`) no se suben al cluster ni a GitHub.
