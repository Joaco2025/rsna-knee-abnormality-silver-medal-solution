#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-reglas-etiquetas
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Mide en los 58 estudios oficiales si mirar los otros hallazgos del informe mejora la etiqueta de cada uno.
# Lanzar desde code/:  sbatch jobs/evaluar_reglas_etiquetas.slurm [JOBID_CONSOLIDACION]
set -euo pipefail

PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
cd "$PROJ/code"
echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) extraccion=${1:-2079118} commit=$(git rev-parse --short HEAD)"
"$PROJ/venvs/rsna/bin/python" rsna/evaluar_reglas_etiquetas.py "$PROJ/runs/${1:-2079118}/extraccion_consolidada.parquet"
echo "fin=$(date -Is)"
