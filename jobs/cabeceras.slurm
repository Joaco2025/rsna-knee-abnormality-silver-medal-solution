#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-cabeceras
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Barrido completo de cabeceras DICOM (~820 mil archivos, sin decodificar píxeles) y su resumen.
# Lanzar desde code_joako/:  sbatch jobs/cabeceras.slurm [train|test]
set -euo pipefail

SPLIT=${1:-train}

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
export KNEE_EDA_OUT="$OUT"
mkdir -p "$OUT"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) split=$SPLIT"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python

# --- Trabajo -----------------------------------------------------------------
PARQUET="$OUT/cabeceras_${SPLIT}_todos.parquet"
"$PY" eda/03_extraer_cabeceras.py --split "$SPLIT" --per-series 0 \
    --workers "${SLURM_CPUS_PER_TASK:?}" --out "$PARQUET"
"$PY" eda/04_resumen_cabeceras.py "$PARQUET" --split "$SPLIT"

echo "fin=$(date -Is)"
