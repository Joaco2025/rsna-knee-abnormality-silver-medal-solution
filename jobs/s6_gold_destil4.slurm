#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-destil4-gold
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

# Tras el job 2099380 (destilados s314 y s2718): S6 en Gold-58 con s42 + d42 + d7 + d314 + d2718 (stride 1 y 3), criterio
# preregistrado (evaluar_destil4.py) y carpeta del dataset de Kaggle con los 2 checkpoints nuevos renombrados (s1314, s12718).
# Nivel 2 (GPU): solo con confirmación explícita de jcastro.  Salida: runs_joako/$SLURM_JOB_ID/ y exp_Dual_S6_dsj/ds_s6_destilados_v2/.
set -euo pipefail
: "${CONFIRMADO:?exporta CONFIRMADO=1 solo con autorización explícita de jcastro}"
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
R="$PROJ/code_joako/research/ensemble_hipotesis_2026-10-19"
D42="$PROJ/runs_joako/2093337_0/train/models/E_prior_s42_fall.pt"
D7="$PROJ/runs_joako/2093337_1/train/models/E_prior_s7_fall.pt"
D314="$PROJ/runs_joako/2099380_0/train/models/E_prior_s314_fall.pt"
D2718="$PROJ/runs_joako/2099380_1/train/models/E_prior_s2718_fall.pt"
for f in "$D42" "$D7" "$D314" "$D2718"; do test -f "$f" || { echo "falta checkpoint: $f"; exit 2; }; done
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID:?}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
OUT="$PROJ/runs_joako/${SLURM_JOB_ID}"
"$PY" -u "$R/s6_gold_infer.py" --out "$OUT/gold" --d42 "$D42" --d7 "$D7" --d314 "$D314" --d2718 "$D2718" --variantes 1:288,3:288
"$PY" -I -B "$R/evaluar_destil4.py" "$OUT/gold" | tee "$OUT/veredicto.txt"
DS="$PROJ/runs_joako/exp_Dual_S6_dsj/ds_s6_destilados_v2"
mkdir "$DS"
cp "$D314" "$DS/E_prior_s1314_fall.pt"
cp "$D2718" "$DS/E_prior_s12718_fall.pt"
(cd "$DS" && sha256sum E_prior_s1314_fall.pt E_prior_s12718_fall.pt | tee SHA256SUMS)
echo "fin=$(date -Is)"
