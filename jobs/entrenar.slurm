#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-entrenar
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Entrena el modelo de imagen (rsna/entrenar.py) con los tensores preprocesados y la tabla de objetivos.
# Lanzar desde code/:  sbatch jobs/entrenar.slurm [OPCIONES de entrenar.py...]
#   p. ej.  sbatch jobs/entrenar.slurm --epochs 8 --backbone resnet34
#   prueba: sbatch --time=00:30:00 jobs/entrenar.slurm --limit 240 --epochs 1
# Los checkpoints y el historial quedan en runs/<jobid>/; las métricas, además, en MLflow
# (base de datos SQLite en mlflow/mlflow.db, la misma que abre la aplicación de Open OnDemand).
set -euo pipefail

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
DATA="$PROJ/preproc/v1_train"
LABELS=$(ls -t "$PROJ"/preproc/etiquetas_v1_*.parquet | head -n 1)     # la tabla de objetivos más reciente
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"                                     # directorio nuevo por job
mkdir -p "$OUT" "$PROJ/mlflow/artifacts" "$PROJ/cache/miopen"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) datos=$DATA objetivos=$LABELS opciones=$*"
export GIT_COMMIT=$(git rev-parse --short HEAD)
echo "commit=$GIT_COMMIT" | tee "$OUT/commit.txt"

# --- Entorno -----------------------------------------------------------------
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
PY="$PROJ/venvs/rsna/bin/python"
# MIOpen compila y ajusta las convoluciones la primera vez; se guarda para que los siguientes jobs no lo repitan.
export MIOPEN_USER_DB_PATH="$PROJ/cache/miopen" MIOPEN_CUSTOM_CACHE_DIR="$PROJ/cache/miopen"
export MLFLOW_DISABLE_AGENT_HINT=1
# Pesos preentrenados de torchvision según la red pedida (resnet34 por defecto).
BACKBONE=resnet34
for ((i = 1; i <= $#; i++)); do [[ "${!i}" == "--backbone" ]] && { j=$((i + 1)); BACKBONE="${!j}"; }; done
WEIGHTS=$(ls "$PROJ"/modelos/torch/"${BACKBONE}"-*.pth 2>/dev/null | head -n 1 || true)
echo "red=$BACKBONE pesos=${WEIGHTS:-ninguno}"

# --- Trabajo -----------------------------------------------------------------
"$PY" rsna/entrenar.py --data "$DATA" --labels "$LABELS" --out "$OUT" --mlflow "sqlite:///$PROJ/mlflow/mlflow.db" --mlflow-artifacts "$PROJ/mlflow/artifacts" \
    --run-name "job-$SLURM_JOB_ID" ${WEIGHTS:+--weights "$WEIGHTS"} "$@"

echo "fin=$(date -Is)"
