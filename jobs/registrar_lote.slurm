#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-registrar-lote
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Anota en MLflow un lote de registros (rsna/registrar_lote.py): corridas del equipo, envíos, resúmenes.
# Lanzar desde code/:  sbatch jobs/registrar_lote.slurm registros/ARCHIVO.json
set -euo pipefail
[[ $# -eq 1 && -f "$1" ]] || { echo "indica el archivo JSON del lote" >&2; exit 1; }

PROJ=/lustre/proyectos/p082
cd "$PROJ/code"
echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) lote=$1 commit=$(git rev-parse --short HEAD)"
export MLFLOW_DISABLE_AGENT_HINT=1
"$PROJ/venvs/rsna/bin/python" rsna/registrar_lote.py "$1"
echo "fin=$(date -Is)"
