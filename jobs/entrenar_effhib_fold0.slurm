#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-eff-hib0
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Entrenamiento Piloto Fold 0: EfficientNetV2-S con etiquetas candidatas híbridas v2 (effhib)
# Protocolo riguroso: 5 folds (partición 0/5), lr 1e-4, batch 4, crop 288, slices 16, 8 épocas, semilla 0.
set -euo pipefail

PROJ=/lustre/proyectos/p082

DATA="$PROJ/preproc/v1_train"
LABELS="$PROJ/runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet"
WEIGHTS="$PROJ/modelos/torch/efficientnet_v2_s-dd5fe13b.pth"

[[ -d "$DATA" ]] || { echo "ERROR: DATA no existe: $DATA" >&2; exit 1; }
[[ -r "$DATA" ]] || { echo "ERROR: DATA no tiene permisos de lectura: $DATA" >&2; exit 1; }
[[ -f "$LABELS" ]] || { echo "ERROR: LABELS no existe: $LABELS" >&2; exit 1; }
[[ -r "$LABELS" ]] || { echo "ERROR: LABELS no tiene permisos de lectura: $LABELS" >&2; exit 1; }
[[ -f "$WEIGHTS" ]] || { echo "ERROR: WEIGHTS no existe: $WEIGHTS" >&2; exit 1; }

OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
ARTIFACTS="$PROJ/mlflow/artifacts_joako"
CACHE_MIOPEN="$PROJ/cache/miopen_joako"

mkdir -p "$OUT" "$ARTIFACTS" "$CACHE_MIOPEN" "$PROJ/code_joako/logs"
cd "$PROJ/code_joako"

echo "=== PILOTO FOLD 0: EFFICIENTNETV2-S CON ETIQUETAS HIBRIDAS V2 (effhib) ===" | tee "$OUT/inputs.txt"
echo "Job ID: $SLURM_JOB_ID | Nodo: $(hostname) | Inicio: $(date -Is)" | tee -a "$OUT/inputs.txt"
echo "DATA:    $DATA" | tee -a "$OUT/inputs.txt"
echo "LABELS:  $LABELS" | tee -a "$OUT/inputs.txt"
echo "WEIGHTS: $WEIGHTS" | tee -a "$OUT/inputs.txt"
echo "OUT:     $OUT" | tee -a "$OUT/inputs.txt"

export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "commit=$GIT_COMMIT" | tee "$OUT/commit.txt"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
PY="$PROJ/venvs/rsna/bin/python"

export MIOPEN_USER_DB_PATH="$CACHE_MIOPEN"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_MIOPEN"
export MLFLOW_DISABLE_AGENT_HINT=1

"$PY" rsna/entrenar.py \
    --data "$DATA" \
    --labels "$LABELS" \
    --out "$OUT" \
    --mlflow "sqlite:///$PROJ/mlflow/mlflow_joako.db" \
    --mlflow-artifacts "$ARTIFACTS" \
    --run-name "effhib_fold0_job-$SLURM_JOB_ID" \
    --backbone efficientnet_v2_s \
    --weights "$WEIGHTS" \
    --folds 5 \
    --fold 0 \
    --batch 4 \
    --lr 1e-4 \
    --crop 288 \
    --slices 16 \
    --epochs 8 \
    --seed 0

echo "Entrenamiento finalizado: $(date -Is)"
