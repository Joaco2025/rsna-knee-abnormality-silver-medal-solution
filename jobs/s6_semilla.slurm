#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-semilla
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%A_%a.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%A_%a.err

# D paso 6 (semillas S6): una semilla nueva de la receta S6 de nartaa por tarea del arreglo (todos los datos, 16 épocas,
# bf16, argv de run_all_alldata.sh sin cambios salvo rutas, --seed-base y --px).  Sin OAI.
# Entradas: ANCLAS (carpeta del job de anclas), INIT (convnext_small fb_in22k_ft_in1k) y FOLDS (folds_v0.json), los tres
# con sha256 fijado por nartaa (se comprueba antes de entrenar).
# Datos: la caché de píxeles se renderiza UNA vez por nodo en /scratch (renderizar_px.py; la primera tarea del nodo la
# hace y las demás esperan el candado) en lugar de decodificar los DICOM de Lustre en cada época.
# Salida: runs_joako/<arreglo>_<tarea>/train/models/E_prior_s<semilla>_fall.pt (~200 MB) + ckpt para reanudar.
# Reanudable: si se corta por tiempo, reenviar con REANUDAR=<carpeta de salida> continúa desde el último epoch.
# Ver runs_joako/exp_s6_semillas/LEEME.md.
set -euo pipefail
PROJ=/lustre/proyectos/p082
EXP="$PROJ/runs_joako/exp_s6_semillas"
S6="$EXP/S6F0"
PY="$PROJ/venvs/rsna/bin/python"
SEMILLAS=(${SEMILLAS:-7 2026 1234})
SEED=${SEMILLAS[${SLURM_ARRAY_TASK_ID:?}]}
ANC="${ANCLAS:?definir ANCLAS=runs_joako/<job de anclas>/anchors}"
INIT="$EXP/entradas/convnext_small/model.safetensors"
FOLDS="$EXP/entradas/hpo_assets/folds_v0.json"
OUT="${REANUDAR:-$PROJ/runs_joako/${SLURM_ARRAY_JOB_ID:?}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$OUT"
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID:?}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
echo "job=$SLURM_JOB_ID tarea=$SLURM_ARRAY_TASK_ID semilla=$SEED nodo=$(hostname) inicio=$(date -Is)" | tee -a "$OUT/info.txt"
sha256sum "$S6"/*.py | tee "$OUT/codigo.sha256"

# entradas fijadas por nartaa
echo "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9  $INIT" | sha256sum -c --strict
echo "12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae  $FOLDS" | sha256sum -c --strict

# caché de píxeles por nodo (una sola vez; candado para tareas en el mismo nodo)
SCR="/scratch/$USER/s6px_v4"
mkdir -p "$SCR"
(
  flock -w 21600 9 || { echo "candado de la caché ocupado > 6 h"; exit 75; }
  if [ ! -f "$SCR/px/index.json" ]; then
    "$PY" -u "$S6/renderizar_px.py" --pool "$S6/data/pool_ok.json" --out "$SCR/px" --anchors "$ANC" \
      --tmp "/tmp/s6px_${SLURM_JOB_ID}" --workers 14 2>&1 | tee "$OUT/render.log"
  else
    echo "caché existente en $(hostname):$SCR/px"
  fi
) 9>"$SCR/.lock"
df -h "$SCR" | tee -a "$OUT/info.txt"

"$PY" -B -u "$S6/train_s6f0_all.py" \
  --pool "$S6/data/pool_ok.json" --folds-json "$FOLDS" --targets "$S6/data/targets_s6f0.csv" --out "$OUT/train" --arms E_prior \
  --backbone convnext_small --init "$INIT" --seeds 1 --seed-base "$SEED" --folds 5 --only-folds 0 \
  --expect-train 4263 --expect-holdout 0 --all-data \
  --epochs 16 --batch 2 --train-windows 64 --size 320 --views v4 --crop 288 --slices 3 --phys-mm 3.5 \
  --spatial-attn --grad-ckpt --lr 1e-4 --head-lr 1e-3 --wd 0.05 --aug crop --train-sampling uniform --ema 0 \
  --workers 8 --loader-timeout 600 --max-fail-frac 0.005 --log-every 200 --work-tmp "/tmp/s6f0_work_${SLURM_JOB_ID}" \
  --resume --no-final-predict --i-have-user-authorisation-to-finetune \
  --anchors "$ANC" --px "$SCR/px" 2>&1 | tee -a "$OUT/train.log"
ls -la "$OUT/train/models" | tee -a "$OUT/info.txt"
echo "fin=$(date -Is)"
