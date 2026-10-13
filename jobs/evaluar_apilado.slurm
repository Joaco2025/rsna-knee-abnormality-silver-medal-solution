#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-evaluar-apilado
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Mide si una segunda etapa (regresión logística, gradient boosting, bosque) mejora al promedio simple.
# Lanzar desde code/:  sbatch jobs/evaluar_apilado.slurm r34=JOB,JOB,JOB,JOB,JOB cnx=JOB,JOB,JOB,JOB,JOB
# Cada JOB es el identificador de una corrida de entrenar.slurm (su carpeta en runs/).
set -euo pipefail
[[ $# -ge 1 ]] || { echo "indica al menos un grupo nombre=JOB,JOB,..." >&2; exit 1; }

PROJ=/lustre/proyectos/p082
LABELS="${KNEE_LABELS:-$(ls -t "$PROJ"/preproc/etiquetas_v1_*.parquet | head -n 1)}"   # contra qué objetivos se mide
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) grupos=$*"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u

# De nombre=JOB,JOB a nombre=runs/JOB,runs/JOB
GROUPS_ARG=()
for g in "$@"; do
    GROUPS_ARG+=("${g%%=*}=$PROJ/runs/$(sed "s#,#,$PROJ/runs/#g" <<<"${g#*=}")")
done

"$PROJ/venvs/rsna/bin/python" rsna/evaluar_apilado.py --labels "$LABELS" --out "$OUT" "${GROUPS_ARG[@]}"
echo "fin=$(date -Is)"
