#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-eval-fusion
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

set -euo pipefail

PROJ=/lustre/proyectos/p082
OUT="$PROJ/runs_joako/eval_fusion_offline"
mkdir -p "$OUT" logs

echo "=== EVALUACIÓN DE FUSIÓN OFFLINE 0A / 0B ==="
echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is)"
echo "commit=$(git rev-parse --short HEAD)"

PY="$PROJ/venvs/rsna/bin/python"

"$PY" rsna/evaluar_fusion_offline.py --out "$OUT"

echo "fin=$(date -Is)"
