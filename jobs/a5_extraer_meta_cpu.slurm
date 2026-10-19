#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-a5-meta
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# A5 (research/score_0958): cabeceras de texto (1 imagen por serie, sin píxeles) + 1 miniatura sagital por estudio de train.
# Solo lectura de kaggle_data; salida pequeña (< 100 MB) en runs_joako/<job>/.
set -euo pipefail
PY=/lustre/proyectos/p082/venvs/rsna/bin/python
OUT="/lustre/proyectos/p082/runs_joako/${SLURM_JOB_ID:?}"
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PY" -I research/score_0958/analisis/a5_extraer_meta.py --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:?}"
