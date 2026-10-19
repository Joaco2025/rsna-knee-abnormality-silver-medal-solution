#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-dsj-exportar
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -D /lustre/proyectos/p082/runs_joako/exp_Esust_dsj
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# E-sust-dsj: entrena y exporta el modelo de metadatos del desatajo de Synovitis (desatajo_synovitis.json) y lo comprueba
# en Gold-58. Solo lectura de entradas; escribe un JSON pequeño en runs_joako/exp_Esust_dsj/.
set -euo pipefail
PY=/lustre/proyectos/p082/venvs/rsna/bin/python
export OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
"$PY" -E -s exportar_modelo_meta.py
