#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-tabla-entrenamiento
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Convierte la extracción consolidada de los informes en la tabla de objetivos de entrenamiento.
# Lanzar desde code/:  sbatch jobs/tabla_entrenamiento.slurm [JOBID_CONSOLIDACION] [VERSION]
set -euo pipefail

CONSOLIDACION=${1:-2079118}
VERSION=${2:-v1}

PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
RUN="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$RUN" "$PROJ/preproc"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) extraccion=$CONSOLIDACION version=$VERSION"
echo "commit=$(git rev-parse --short HEAD)" | tee "$RUN/commit.txt"

"$PROJ/venvs/rsna/bin/python" rsna/tabla_entrenamiento.py "$PROJ/runs/$CONSOLIDACION/extraccion_consolidada.parquet" \
    --out "$RUN/etiquetas_${VERSION}.parquet"
# La tabla vigente se publica con un nombre fijo junto a los tensores; cada job conserva la suya en runs/.
cp "$RUN/etiquetas_${VERSION}.parquet" "$PROJ/preproc/etiquetas_${VERSION}_${SLURM_JOB_ID}.parquet"
echo "fin=$(date -Is)"
