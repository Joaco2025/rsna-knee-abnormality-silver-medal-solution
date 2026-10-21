#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-omer-gold
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Tras 2108250 (Omer puro) y 2108251 (Omer mix): S6 en Gold-58 (stride 1 y 3) con s42 + d42 + d7 + los S6-Omer que existan,
# criterio preregistrado (evaluar_omer.py) y carpeta de dataset de Kaggle con los checkpoints Omer renombrados (s2042, s3042).
# Nivel 2 (GPU): solo con confirmación explícita de jcastro.
set -euo pipefail
: "${CONFIRMADO:?exporta CONFIRMADO=1 solo con autorización explícita de jcastro}"
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
R="$PROJ/code_joako/research/ensemble_hipotesis_2026-10-19"
D42="$PROJ/runs_joako/2093337_0/train/models/E_prior_s42_fall.pt"
D7="$PROJ/runs_joako/2093337_1/train/models/E_prior_s7_fall.pt"
OP="$PROJ/runs_joako/2108250_0/train/models/E_prior_s42_fall.pt"
OM="$PROJ/runs_joako/2108251_0/train/models/E_prior_s42_fall.pt"
ARGS=()
if [ -f "$OP" ]; then ARGS+=(--omer-puro "$OP"); fi
if [ -f "$OM" ]; then ARGS+=(--omer-mix "$OM"); fi
if [ ${#ARGS[@]} -eq 0 ]; then echo "ningún checkpoint Omer"; exit 2; fi
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID:?}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
OUT="$PROJ/runs_joako/${SLURM_JOB_ID}"
"$PY" -u "$R/s6_gold_infer.py" --out "$OUT/gold" --d42 "$D42" --d7 "$D7" "${ARGS[@]}" --variantes 1:288,3:288
"$PY" -I -B "$R/evaluar_omer.py" "$OUT/gold" | tee "$OUT/veredicto.txt"
DS="$PROJ/runs_joako/exp_Omer/ds_s6_omer"
mkdir "$DS"
if [ -f "$OP" ]; then cp "$OP" "$DS/E_prior_s2042_fall.pt"; fi
if [ -f "$OM" ]; then cp "$OM" "$DS/E_prior_s3042_fall.pt"; fi
(cd "$DS" && sha256sum E_prior_s*_fall.pt | tee SHA256SUMS)
echo "fin=$(date -Is)"
