#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-gold-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# BORRADOR — NO ENVIAR sin autorización de jcastro (nivel 2: GPU) y SOLO cuando el job 2093337 haya terminado
# (necesita sus checkpoints y no debe competir por la cuota de GPU del grupo mientras entrena).
# H1 + H5 de docs/hipotesis_ensemble_2026-10-19.md: S6 original + destilados s42/s7 sobre los 58 Gold, eval_stride 1 y 3.
# Primero humo con 3 estudios; si pasa, los 58.  Solo lee kaggle_data y runs_joako; escribe en runs_joako/<job>/.
# Estimación: decodificación + anclas de 58 estudios ×2 strides + 3 modelos ConvNeXt-S → < 30 min en una MI210.
set -euo pipefail
: "${CONFIRMADO:?Borrador: exporta CONFIRMADO=1 solo con autorización explícita de jcastro}"
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
R="$PROJ/code_joako/research/ensemble_hipotesis_2026-10-19"
D42="$PROJ/runs_joako/2093337_0/train/models/E_prior_s42_fall.pt"
D7="$PROJ/runs_joako/2093337_1/train/models/E_prior_s7_fall.pt"
test -f "$D42" && test -f "$D7" || { echo "faltan los checkpoints destilados de 2093337"; exit 2; }
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID:?}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
OUT="$PROJ/runs_joako/${SLURM_JOB_ID}"
"$PY" -u "$R/s6_gold_infer.py" --out "$OUT/humo" --d42 "$D42" --d7 "$D7" --variantes 1:288 --limite 3
"$PY" -u "$R/s6_gold_infer.py" --out "$OUT/gold" --d42 "$D42" --d7 "$D7" --variantes 1:288,3:288,1:320
echo "fin=$(date -Is)"
