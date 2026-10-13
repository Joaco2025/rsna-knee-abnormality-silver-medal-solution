# RSNA Knee Abnormalities Detection — proyecto p082

Código del equipo para la competencia de Kaggle
[RSNA Knee Abnormalities Detection](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection),
desarrollado en el cluster Yuca (ACARUS, Universidad de Sonora).

**Estado al 13 de octubre de 2026:** el circuito completo funciona (etiquetas derivadas, preprocesamiento,
entrenamiento con validación cruzada, inferencia en Kaggle) y la mejor puntuación pública es **0.912**.

**Qué se ha intentado, con qué resultado, dónde está cada archivo y qué hay en cola: ver
[EXPERIMENTOS.md](EXPERIMENTOS.md).** Las secciones "Qué está hecho" y "Qué falta" de más abajo describen
la etapa inicial del proyecto (exploración y etiquetas) y no se han actualizado.

## La competencia en una página

| | |
|---|---|
| Tarea | Para cada estudio de RM de rodilla, dar una probabilidad para cada uno de 12 hallazgos |
| Hallazgos | ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| Métrica | AUC ROC promediada sobre los 12 hallazgos. Solo importa el orden de las probabilidades, no su calibración |
| Envío | Notebook de Kaggle, máximo 9 h, sin internet; se permiten modelos preentrenados públicos |
| Test | Unos 1,300 estudios, solo imágenes: en test no hay informe |
| Límite de inscripción y de equipos | 15 de octubre de 2026, 23:59 UTC |
| Límite de envío final | 22 de octubre de 2026, 23:59 UTC |
| Envíos | 5 por día, 2 finales; equipos de hasta 5 personas |

## El problema central: casi no hay etiquetas

De los 4,407 estudios de entrenamiento, solo 58 traen las 12 etiquetas. Los otros 4,349 traen
únicamente el informe radiológico en texto libre, en al menos nueve idiomas. Además, los 58
etiquetados no son representativos: tienen de media 4.1 hallazgos por estudio, frente a unos 2.7 en
el resto.

La estrategia es derivar etiquetas de los informes con un modelo de lenguaje local, entrenar con
ellas un modelo de imagen y reservar los 58 estudios oficiales para validar.

## Qué está hecho

### 1. Exploración y verificación del dataset (`eda/`)

- 4,407 estudios, 24,371 series y 819,078 archivos DICOM; los CSV y el disco coinciden.
- Barrido de las 819,078 cabeceras: ningún archivo ilegible.
- Verificación por serie: ninguna con cortes repetidos, espaciado irregular ni geometría
  inconsistente entre cortes.
- Adquisición muy heterogénea: cinco fabricantes, 1.5 T y 3 T, 221 tamaños de matriz, píxel de
  0.07 a 1.17 mm. Las intensidades crudas no son comparables entre equipos.
- Todos los estudios tienen los tres planos. `Fluid_Sensitive` y `Fat_Suppression` son idénticas en
  entrenamiento, pero Kaggle advierte que en test pueden diferir.

### 2. Etiquetas derivadas de los informes (`etiquetas/`)

- **Piloto con los 58 estudios oficiales:** leyendo solo el informe se coincide con la etiqueta
  oficial en el 84 % de las decisiones. El resto no es ruido al azar: las etiquetas oficiales
  aplican umbrales que el informe no declara, y parte de la información no está en el texto.
- **Extracción completa:** `qwen2.5:72b` (local, con Ollama) leyó los 4,407 informes sin errores.
  Para cada hallazgo devuelve un estado (presente, posible, ausente, no mencionado) y una gravedad.
- **Calidad medida en los 58:** acuerdo del 83 % con las etiquetas oficiales.

| Se deriva bien del informe | Se deriva mal |
|---|---|
| ACL y MCL (95 %), menisco medial (91 %), artrosis patelofemoral y Baker's (88 %) | Derrame (72 %), contusión (71 %), sinovitis (69 %) |

- **Un modelo médico no mejora el acierto.** MedGemma 27B de texto obtiene el mismo 83 %, aunque
  es cuatro veces más rápido (5 s por informe con cuatro en paralelo, frente a 21 s).
- La sinovitis no se menciona en el 72 % de los informes; el quiste de Baker, en el 42 %.

### 3. Pruebas con un modelo que mira las imágenes (`imagenes/`)

MedGemma 1.5 4B, sin ningún ajuste, sobre las resonancias de los 58 estudios oficiales:

- Sus respuestas de sí o no están sesgadas y no sirven como etiquetas.
- Sus probabilidades sí ordenan parcialmente los casos: AUC medio de 0.61 a 0.66 según el plano.
- Donde más ve es en derrame (hasta 0.89) y sinovitis (hasta 0.82), justo donde el texto flojea.
  En ligamentos y meniscos queda cerca del azar.
- Darle todos los cortes del estudio no mejora y tarda cuatro veces más.

Ese 0.66 queda como referencia a superar por un modelo entrenado.

## Qué falta

1. **Envío trivial** a Kaggle, para comprobar el circuito completo.
2. **Entorno de PyTorch para ROCm**, siguiendo la receta de `/shared/ejemplos_scripts/pytorch/`.
3. **Preprocesamiento de imágenes:** ordenar los cortes por posición geométrica, normalizar la
   intensidad por serie, llevar a un tamaño común y guardar en un formato rápido de leer.
4. **Tabla de entrenamiento** a partir de la extracción: valores graduados según estado y gravedad,
   con "no mencionado" como dato faltante.
5. **Primer modelo de imagen**, validado con los 58 estudios oficiales.
6. **Notebook de inferencia** para Kaggle, que debe leer DICOM comprimido (JPEG Lossless,
   JPEG 2000), que en entrenamiento no aparece pero Kaggle anuncia para test.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `eda/` | Exploración: CSV, inventario de disco, cabeceras DICOM, verificación de series. Ver `eda/README.md` |
| `etiquetas/` | Extracción de etiquetas con Ollama, evaluación contra las oficiales y consolidación |
| `imagenes/` | Prueba de MedGemma 1.5 sobre las resonancias |
| `jobs/` | Scripts de SLURM. Todo lo que tarda o lee muchos archivos se ejecuta desde aquí |
| `.claude/` | Skill `yuca-hpc-ops` y guardarraíl para trabajar con Claude Code en el cluster |

Jobs principales, lanzados desde `code/`:

```bash
sbatch jobs/cabeceras.slurm                      # barrido de cabeceras DICOM
sbatch jobs/verificar_series.slurm               # geometría y consistencia por serie
sbatch jobs/extraer_etiquetas.slurm              # extracción sobre los 58 etiquetados y su evaluación
sbatch jobs/consolidar_etiquetas.slurm           # tabla única de la extracción y su revisión
sbatch jobs/prueba_imagenes_medgemma.slurm 0     # MedGemma 1.5 sobre las imágenes de los 58
```

Cada job escribe en `/lustre/proyectos/p082/runs/<jobid>/` y guarda ahí el commit con el que corrió.

## Dónde está cada cosa en el cluster

| Ruta bajo `/lustre/proyectos/p082/` | Qué es |
|---|---|
| `kaggle_data/` | Dataset de la competencia. **Solo lectura** |
| `code/` | Este repositorio |
| `runs/<jobid>/` | Salidas de los jobs |
| `runs/2079118/` | Extracción consolidada de Qwen para los 4,407 estudios |
| `modelos/` | Modelos propios del proyecto (Ollama y GGUF) |
| `/lustre/proyectos/ollama-models/` | Modelos compartidos del cluster, incluido `qwen2.5:72b` |

## Reglas que hay que respetar

- **Los datos no salen del cluster.** La regla 2.4.b de la competencia y la licencia MIRA de RSNA
  prohíben transmitir o publicar cualquier parte del dataset. No se suben a GitHub ni a servicios
  externos informes, imágenes, identificadores de estudio ni etiquetas derivadas. El `.gitignore`
  excluye por eso todos los CSV, JSONL, DICOM y arrays.
- **Sin compartir código ni datos con otros equipos** de la competencia.
- **Las etiquetas a mano solo sobre entrenamiento.** Etiquetar a mano datos de test está prohibido.
- **Nada de desactivar la verificación de certificados.** La red de la universidad intercepta
  parte del tráfico HTTPS (Hugging Face entre otros). Lo que no se puede descargar desde el cluster
  se baja fuera, se verifica su SHA-256 y se sube; `jobs/registrar_gguf_ollama.slurm` hace esa
  comprobación.

## Requisitos

- Python 3.12 de `/apps/miniconda3` con pandas, pyarrow, pydicom, matplotlib y scikit-learn.
- Módulos `rocm/7.2.2` y `ollama/0.34.0` para los jobs con modelos de lenguaje.
- Cuenta de SLURM `p082`; particiones `cpu` y `gpu` (AMD Instinct MI210).
