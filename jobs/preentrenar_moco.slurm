#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-moco
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Continúa el preentrenamiento de la red 2D sobre nuestros cortes, sin etiquetas (rsna/preentrenar_moco.py).
# Lanzar desde code/:  sbatch jobs/preentrenar_moco.slurm [OPCIONES de preentrenar_moco.py...]
#   prueba: sbatch --time=00:20:00 jobs/preentrenar_moco.slurm --limit 200 --epochs 2
# Los pesos quedan en runs/<jobid>/<red>-moco.pth; para usarlos:
#   sbatch --export=ALL,KNEE_WEIGHTS=runs/<jobid>/resnet34-moco.pth jobs/entrenar.slurm ...
set -euo pipefail

PROJ=/lustre/proyectos/p082
DATA="${KNEE_PREPROC:-$PROJ/preproc/v1_train}"
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT" "$PROJ/cache/miopen"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) datos=$DATA opciones=$*"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
export MIOPEN_USER_DB_PATH="$PROJ/cache/miopen" MIOPEN_CUSTOM_CACHE_DIR="$PROJ/cache/miopen"

BACKBONE=resnet34
for ((i = 1; i <= $#; i++)); do [[ "${!i}" == "--backbone" ]] && { j=$((i + 1)); BACKBONE="${!j}"; }; done
WEIGHTS=$(ls "$PROJ"/modelos/torch/"${BACKBONE}"-*.pth 2>/dev/null | head -n 1 || true)
echo "red=$BACKBONE pesos de partida=${WEIGHTS:-ninguno}"

"$PROJ/venvs/rsna/bin/python" rsna/preentrenar_moco.py --data "$DATA" --out "$OUT" ${WEIGHTS:+--weights "$WEIGHTS"} "$@"
echo "fin=$(date -Is)"
