#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-a4-centro
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# A4 (research/score_0958): sesgo de puntuación por centro sobre los OOF de v10 (solo lectura; salida a4_centro.json).
set -euo pipefail
PY=/lustre/proyectos/p082/venvs/rsna/bin/python
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$PY" -I research/score_0958/analisis/a4_sesgo_centro.py /lustre/proyectos/p082/runs_joako/2093185/demografia/demografia_cabeceras.csv
