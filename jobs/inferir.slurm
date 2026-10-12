#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-inferir
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Ensayo en el cluster de la inferencia que correrá en Kaggle: de los DICOM a submission.csv.
# Lanzar desde code/:  sbatch jobs/inferir.slurm SPLIT LIMITE CHECKPOINT [CHECKPOINT...]
#   p. ej.  sbatch jobs/inferir.slurm test 0 /lustre/proyectos/p082/runs/<jobid>/best_oficial.pt
#           sbatch jobs/inferir.slurm train 200 ...   (mide el tiempo por estudio con más casos)
set -euo pipefail

SPLIT=${1:?indica el split: test o train}
LIMIT=${2:?indica el límite de estudios (0 = todos)}
shift 2
[[ $# -ge 1 ]] || { echo "falta al menos un checkpoint" >&2; exit 1; }

PROJ=/lustre/proyectos/p082
DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT" "$PROJ/cache/miopen"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) split=$SPLIT limite=$LIMIT modelos=$*"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
PY="$PROJ/venvs/rsna/bin/python"
export MIOPEN_USER_DB_PATH="$PROJ/cache/miopen" MIOPEN_CUSTOM_CACHE_DIR="$PROJ/cache/miopen"

"$PY" rsna/inferir.py --data "$DATA" --split "$SPLIT" --limit "$LIMIT" --checkpoints "$@" \
    --out "$OUT/submission.csv" --workers "$((SLURM_CPUS_PER_TASK - 2))"

# Comprobación del formato frente al archivo de ejemplo, sin mostrar identificadores.
"$PY" - "$OUT/submission.csv" "$DATA/sample_submission.csv" <<'EOF'
import sys
import pandas as pd
sub, ref = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
print("columnas iguales a sample_submission:", list(sub.columns) == list(ref.columns))
v = sub.iloc[:, 1:]
print("filas:", len(sub), "| valores fuera de [0,1]:", int(((v < 0) | (v > 1)).sum().sum()), "| NaN:", int(v.isna().sum().sum()))
print("probabilidad media por hallazgo:", {c: round(float(v[c].mean()), 3) for c in v.columns})
EOF
echo "fin=$(date -Is)"
