#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-probar-dinov2
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Prueba en CPU del brazo DINOv2 para el notebook 0.957 (kaggle/brazo_dinov2/): la celda con inferencia simulada
# (N=7 y N=1400), el mecanismo real de subproceso con un script falso y, con --preproceso, el preproceso REAL extraido
# del 0.940 sobre 64 estudios de train (cronometrado con 4 CPU, como los 4 vCPU de Kaggle). Sin GPU ni red.
# NO prueba DINOv2 real: en Yuca falta transformers y los pesos no estan en el cluster. Lanzar desde code_joako/.
set -euo pipefail

PROJ=/lustre/proyectos/p082
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
mkdir -p "$OUT"
PY="$PROJ/venvs/rsna/bin/python"
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) cpus=$SLURM_CPUS_PER_TASK"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"
sha256sum kaggle/brazo_dinov2/celda_para_pegar_0957.py kaggle/brazo_dinov2/celda_para_pegar_0957_diag.py \
    kaggle/brazo_dinov2/script_dinov2.py kaggle/brazo_dinov2/generar_brazo_dinov2.py \
    kaggle/brazo_dinov2/celda_brazo_dinov2.py kaggle/brazo_dinov2/test_celda_brazo_dinov2.py | tee "$OUT/sha_archivos.txt"

"$PY" -I kaggle/brazo_dinov2/generar_brazo_dinov2.py --comprobar

"$PY" -I kaggle/brazo_dinov2/test_celda_brazo_dinov2.py "$OUT/casos" --preproceso 64

echo "fin=$(date -Is)"
