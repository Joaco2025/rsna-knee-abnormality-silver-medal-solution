#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-evaluar-conjunto
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Mide varios modelos entrenados y sus promedios sobre las dos validaciones, para decidir qué enviar.
# Lanzar desde code/:  sbatch jobs/evaluar_conjunto.slurm CHECKPOINT CHECKPOINT [...]
set -euo pipefail
[[ $# -ge 1 ]] || { echo "indica al menos un checkpoint" >&2; exit 1; }

PROJ=/lustre/proyectos/p082
DATA="$PROJ/preproc/v1_train"
LABELS=$(ls -t "$PROJ"/preproc/etiquetas_v1_*.parquet | head -n 1)
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT" "$PROJ/cache/miopen"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) modelos=$*"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
export MIOPEN_USER_DB_PATH="$PROJ/cache/miopen" MIOPEN_CUSTOM_CACHE_DIR="$PROJ/cache/miopen"

"$PROJ/venvs/rsna/bin/python" rsna/evaluar_conjunto.py --data "$DATA" --labels "$LABELS" --out "$OUT" \
    --workers "$((SLURM_CPUS_PER_TASK - 2))" "$@"
echo "fin=$(date -Is)"
