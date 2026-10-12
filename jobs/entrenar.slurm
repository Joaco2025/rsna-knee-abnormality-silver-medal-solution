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
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Entrena el modelo de imagen (rsna/entrenar.py) con los tensores preprocesados y la tabla de objetivos.
# Lanzar desde code_joako/:  sbatch jobs/entrenar.slurm [OPCIONES de entrenar.py...]
#   p. ej.  sbatch jobs/entrenar.slurm --epochs 8 --backbone resnet34
#   prueba: sbatch --time=00:30:00 jobs/entrenar.slurm --limit 240 --epochs 1
# Las entradas DATA y LABELS pueden sobreescribirse por variable de entorno; por defecto usan las fijas oficiales.
# Los checkpoints y el historial quedan en runs_joako/<jobid>/; las métricas en mlflow/mlflow_joako.db.
set -euo pipefail

# --- Rutas y Entradas Deterministas ------------------------------------------
PROJ=/lustre/proyectos/p082

DATA=${DATA:-"$PROJ/preproc/v1_train"}
LABELS=${LABELS:-"$PROJ/preproc/etiquetas_v1_2079203.parquet"}

[[ -d "$DATA" ]] || { echo "ERROR: DATA no existe o no es un directorio: $DATA" >&2; exit 1; }
[[ -r "$DATA" ]] || { echo "ERROR: DATA no tiene permisos de lectura: $DATA" >&2; exit 1; }
[[ -f "$LABELS" ]] || { echo "ERROR: LABELS no existe o no es un archivo regular: $LABELS" >&2; exit 1; }
[[ -r "$LABELS" ]] || { echo "ERROR: LABELS no tiene permisos de lectura: $LABELS" >&2; exit 1; }

OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"                                     # directorio nuevo por job
ARTIFACTS="$PROJ/mlflow/artifacts_joako"
CACHE_MIOPEN="$PROJ/cache/miopen_joako"

mkdir -p "$OUT" "$ARTIFACTS" "$CACHE_MIOPEN"
cd "$PROJ/code_joako"

ORIGEN_DATA="personal"
[[ "$DATA" == "$PROJ/preproc/"* ]] && ORIGEN_DATA="compartido-owen (solo lectura)"
ORIGEN_LABELS="personal"
[[ "$LABELS" == "$PROJ/preproc/"* ]] && ORIGEN_LABELS="compartido-owen (solo lectura)"

echo "=== ENTRADAS DETERMINISTAS ===" | tee "$OUT/inputs.txt"
echo "DATA:   $DATA [Origen: $ORIGEN_DATA]" | tee -a "$OUT/inputs.txt"
echo "LABELS: $LABELS [Origen: $ORIGEN_LABELS]" | tee -a "$OUT/inputs.txt"
echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) opciones=$*" | tee -a "$OUT/inputs.txt"
export GIT_COMMIT=$(git rev-parse --short HEAD)
echo "commit=$GIT_COMMIT" | tee "$OUT/commit.txt"

# --- Entorno -----------------------------------------------------------------
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
PY="$PROJ/venvs/rsna/bin/python"
# MIOpen compila y ajusta las convoluciones la primera vez en caché personal aislada.
export MIOPEN_USER_DB_PATH="$CACHE_MIOPEN" MIOPEN_CUSTOM_CACHE_DIR="$CACHE_MIOPEN"
export MLFLOW_DISABLE_AGENT_HINT=1
# Pesos preentrenados de torchvision según la red pedida (resnet34 por defecto).
BACKBONE=resnet34
for ((i = 1; i <= $#; i++)); do [[ "${!i}" == "--backbone" ]] && { j=$((i + 1)); BACKBONE="${!j}"; }; done
WEIGHTS=$(ls "$PROJ"/modelos/torch/"${BACKBONE}"-*.pth 2>/dev/null | head -n 1 || true)
echo "red=$BACKBONE pesos=${WEIGHTS:-ninguno}"

# --- Trabajo -----------------------------------------------------------------
"$PY" rsna/entrenar.py --data "$DATA" --labels "$LABELS" --out "$OUT" --mlflow "sqlite:///$PROJ/mlflow/mlflow_joako.db" --mlflow-artifacts "$ARTIFACTS" \
    --run-name "job-$SLURM_JOB_ID" ${WEIGHTS:+--weights "$WEIGHTS"} "$@"

echo "fin=$(date -Is)"
