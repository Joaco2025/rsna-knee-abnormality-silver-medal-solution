#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-eval-ensamble
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Evaluación comparativa pareada: ResNet34 vs ConvNeXt-Tiny vs Ensamble (50/50).
# Evalúa sobre los mismos estudios de validación (val_int: 415 casos, val_oficial: 58 casos).
# Todas las entradas se leen en modo de solo lectura.
# Lanzar desde code_joako/:  sbatch jobs/evaluar_ensamble.slurm [CKPT_RESNET] [CKPT_CONVNEXT]
set -euo pipefail

PROJ=/lustre/proyectos/p082

# Entradas compartidas (estrictamente solo lectura)
DATA="$PROJ/preproc/v1_train"
LABELS="$PROJ/preproc/etiquetas_v1_2079203.parquet"
CKPT_RES=${1:-"$PROJ/runs/2080214/best_oficial.pt"}
CKPT_CNV=${2:-"$PROJ/runs/2080215/best_oficial.pt"}

[[ -d "$DATA" && -r "$DATA" ]] || { echo "ERROR: DATA no existe o sin permiso de lectura: $DATA" >&2; exit 1; }
[[ -f "$LABELS" && -r "$LABELS" ]] || { echo "ERROR: LABELS no existe o sin permiso de lectura: $LABELS" >&2; exit 1; }
[[ -f "$CKPT_RES" && -r "$CKPT_RES" ]] || { echo "ERROR: Checkpoint ResNet no existe o sin permiso de lectura: $CKPT_RES" >&2; exit 1; }
[[ -f "$CKPT_CNV" && -r "$CKPT_CNV" ]] || { echo "ERROR: Checkpoint ConvNeXt no existe o sin permiso de lectura: $CKPT_CNV" >&2; exit 1; }

# Salida personal aislada
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
CACHE_MIOPEN="$PROJ/cache/miopen_joako"
mkdir -p "$OUT" "$CACHE_MIOPEN"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is)"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"
echo "entradas:" | tee -a "$OUT/commit.txt"
echo "  DATA=$DATA" | tee -a "$OUT/commit.txt"
echo "  LABELS=$LABELS" | tee -a "$OUT/commit.txt"
echo "  CKPT_RES=$CKPT_RES" | tee -a "$OUT/commit.txt"
echo "  CKPT_CNV=$CKPT_CNV" | tee -a "$OUT/commit.txt"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
PY="$PROJ/venvs/rsna/bin/python"
export MIOPEN_USER_DB_PATH="$CACHE_MIOPEN" MIOPEN_CUSTOM_CACHE_DIR="$CACHE_MIOPEN"

"$PY" rsna/evaluar_ensamble.py \
    --data "$DATA" \
    --labels "$LABELS" \
    --ckpt-resnet "$CKPT_RES" \
    --ckpt-convnext "$CKPT_CNV" \
    --out "$OUT" \
    --batch 8 \
    --workers "$((SLURM_CPUS_PER_TASK - 2))"

echo "fin=$(date -Is)"

