#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-diag-bn
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Diagnóstico de BatchNorm del piloto CoAtNet (2085360), solo CPU. Ver experiments/reportes/bug_coatnet.md.
# 1. Corpus native384 de 186 estudios (58 Gold + 128 de train del fold 0) en el /scratch del nodo.
# 2. rsna/diag_bn_eval.py: Gold del swa.pt con BN normal, por estudio y recalibrada (variantes A-D).
# Salida: runs_joako/<job>/ (resumen.json y preds_gold.npz, < 1 MB). El corpus se borra al salir.
set -euo pipefail

PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
SCR="/scratch/${USER:?}/diag_bn_${SLURM_JOB_ID:?}"
CKPT="$PROJ/runs_joako/2085360/swa.pt"
[[ -r "$CKPT" ]] || { echo "ERROR: no se puede leer $CKPT" >&2; exit 1; }
mkdir -p "$OUT" "$SCR"
trap 'rm -rf -- "${SCR:?}"' EXIT

{
    echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) commit=$(git rev-parse --short HEAD)"
    git status --porcelain rsna/ jobs/ | grep -E "diag_bn|entrenar_coatnet|raptor_" || true
    sha256sum rsna/diag_bn_eval.py rsna/entrenar_coatnet.py rsna/raptor_corpus.py rsna/raptor_preproc.py
} | tee "$OUT/inputs.txt"
cp rsna/diag_bn_eval.py "$OUT/"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TIMM_FUSED_ATTN=0
"$PY" -I rsna/diag_bn_eval.py --ckpt "$CKPT" --corpus x --out x --listar-uids "$SCR/uids.txt"
"$PY" -I rsna/raptor_corpus.py --out "$SCR/corpus" --procesos 16 --uids "$SCR/uids.txt"

export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
"$PY" -I rsna/diag_bn_eval.py --ckpt "$CKPT" --corpus "$SCR/corpus" --out "$OUT"
echo "fin=$(date -Is)"
