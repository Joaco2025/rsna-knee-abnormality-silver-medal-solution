#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-a6-meta
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# A6 (research/score_0958): metadatos DICOM y casi-duplicados (solo lectura; salida a6.json).
set -euo pipefail
PY=/lustre/proyectos/p082/venvs/rsna/bin/python
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$PY" -I research/score_0958/analisis/a6_meta_y_duplicados.py /lustre/proyectos/p082/runs_joako/2093187
