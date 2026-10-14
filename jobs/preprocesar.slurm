#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-preprocesar
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Convierte cada estudio DICOM en un tensor (4, 32, 320, 320) uint8: sagital FS, sagital no FS,
# coronal FS y axial FS, en orientación canónica y con intensidad normalizada por serie.
# El lote completo de train son 4,407 archivos de 13 MB, unos 58 GB en total.
# Lanzar desde code/:  sbatch jobs/preprocesar.slurm [VERSION] [LIMITE] [train|test] [LADO] [SERIES]
#   VERSION: nombre de la carpeta de salida bajo preproc/ (por defecto v1)
#   LIMITE:  número de estudios, para pruebas; 0 = todos
#   LADO:    lado en píxeles del tensor (por defecto 320); a 384 el lote de train son unos 83 GB
# Es reanudable: relanzarlo con la misma VERSION continúa donde quedó.
set -euo pipefail

VERSION=${1:-v1}
LIMIT=${2:-0}
SPLIT=${3:-train}
SIDE=${4:-320}
NSERIES=${5:-4}      # 6 añade coronal y axial no sensibles a líquido

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/preproc/${VERSION}_${SPLIT}"
RUN="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT" "$RUN"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) salida=$OUT limite=$LIMIT split=$SPLIT lado=$SIDE"
echo "commit=$(git rev-parse --short HEAD)" | tee "$RUN/commit.txt" | tee -a "$OUT/commits.txt"
lfs quota -h -g p082 /lustre | sed -n 3p

# --- Entorno -----------------------------------------------------------------
PY="$PROJ/venvs/rsna/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # un hilo por proceso: paraleliza el Pool
export OPENCV_FOR_THREADS_NUM=1

# --- Trabajo -----------------------------------------------------------------
# Opciones extra de preprocesar.py (p. ej. --crop-mm 140 --z-min 0.02 --z-max 0.98) en la variable KNEE_PREPROC_OPTS.
"$PY" rsna/preprocesar.py --out "$OUT" --split "$SPLIT" --workers "${SLURM_CPUS_PER_TASK:?}" --limit "$LIMIT" --size "$SIDE" --series "$NSERIES" --qa 12 ${KNEE_PREPROC_OPTS:-}

du -sh "$OUT"
echo "fin=$(date -Is)"
