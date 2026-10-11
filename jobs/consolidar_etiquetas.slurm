#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-consolidar-etiquetas
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Junta las extracciones de varios jobs en una tabla por estudio y resume si tienen sentido.
# Solo lee los extraccion.jsonl de runs/ y train.csv; no usa GPU ni modelos.
# Lanzar desde code/:  sbatch jobs/consolidar_etiquetas.slurm [JOBID ...]
set -euo pipefail

if [[ $# -gt 0 ]]; then EXTRACCIONES=("$@"); else EXTRACCIONES=(2078806 2078913 2078914); fi

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) extracciones=${EXTRACCIONES[*]}"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

ENTRADAS=()
for id in "${EXTRACCIONES[@]}"; do
    f="$PROJ/runs/$id/extraccion.jsonl"
    [[ -r "$f" ]] || { echo "No existe $f" >&2; exit 1; }
    ENTRADAS+=("$f")
done

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python

# --- Trabajo -----------------------------------------------------------------
"$PY" etiquetas/consolidar.py --out "$OUT" "${ENTRADAS[@]}"

echo "fin=$(date -Is)"
