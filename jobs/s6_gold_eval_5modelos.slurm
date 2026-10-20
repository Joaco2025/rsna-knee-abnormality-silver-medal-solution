#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-5m-gold
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

# Evaluación en Gold-58 de los 5 checkpoints de S6 en una sola pasada por variante:
#   - s42 original (nartaa, JEV puro)
#   - s2026 y s1234 (job 2098694, JEV puro, semillas independientes)
#   - d42 y d7 (job 2093337, destilados 0.5 JEV + 0.5 Raptor)
# Variantes de geometría/muestreo: 1:288 (estándar), 3:288 (stride 3), 1:320 (FOV completo).
# NO ENVIAR sin confirmación explícita de jcastro (nivel 2: GPU) y SOLO tras finalizar el job 2098694.
set -euo pipefail
: "${CONFIRMADO:?Borrador: exporta CONFIRMADO=1 solo con autorización explícita de jcastro}"
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
R="$PROJ/code_joako/research/ensemble_hipotesis_2026-10-19"
D42="$PROJ/runs_joako/2093337_0/train/models/E_prior_s42_fall.pt"
D7="$PROJ/runs_joako/2093337_1/train/models/E_prior_s7_fall.pt"
S2026="$PROJ/runs_joako/2098694_0/train/models/E_prior_s2026_fall.pt"
S1234="$PROJ/runs_joako/2098694_1/train/models/E_prior_s1234_fall.pt"
for f in "$D42" "$D7" "$S2026" "$S1234"; do
  test -f "$f" || { echo "falta checkpoint: $f"; exit 2; }
done
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID:?}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
OUT="$PROJ/runs_joako/${SLURM_JOB_ID}"
"$PY" -u "$R/s6_gold_infer.py" --out "$OUT/humo" --d42 "$D42" --d7 "$D7" --s2026 "$S2026" --s1234 "$S1234" --variantes 1:288 --limite 3
"$PY" -u "$R/s6_gold_infer.py" --out "$OUT/gold" --d42 "$D42" --d7 "$D7" --s2026 "$S2026" --s1234 "$S1234" --variantes 1:288,3:288,1:320
"$PY" -I -B "$R/evaluar_s6_variantes.py" "$OUT/gold"
echo "fin=$(date -Is)"

