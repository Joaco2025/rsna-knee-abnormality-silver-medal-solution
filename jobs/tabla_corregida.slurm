#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-tabla-corregida
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Corrige la tabla de objetivos con las predicciones fuera de muestra (rsna/tabla_corregida.py):
# rellena las celdas que el informe no menciona y baja el peso de las que los modelos contradicen.
# Lanzar desde code/:  sbatch jobs/tabla_corregida.slurm r34=JOB,JOB,JOB,JOB,JOB cnx=JOB,... [-- OPCIONES]
#   tabla base distinta de la v1 más reciente: sbatch --export=ALL,KNEE_LABELS=RUTA jobs/tabla_corregida.slurm ...
# La tabla queda en preproc/etiquetas_corregida_<jobid>.parquet. No la toma ningún job por defecto:
# para entrenar con ella, sbatch --export=ALL,KNEE_LABELS=esa_ruta jobs/entrenar.slurm ...
set -euo pipefail
[[ $# -ge 1 ]] || { echo "indica al menos un grupo nombre=JOB,JOB,..." >&2; exit 1; }

PROJ=/lustre/proyectos/p082
LABELS="${KNEE_LABELS:-$(ls -t "$PROJ"/preproc/etiquetas_v1_*.parquet | head -n 1)}"
OUT="$PROJ/preproc/etiquetas_corregida_${SLURM_JOB_ID:?}.parquet"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) base=$LABELS argumentos=$*"
echo "commit=$(git rev-parse --short HEAD)"

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u

# De nombre=JOB,JOB a nombre=runs/JOB,runs/JOB; lo que venga tras "--" se pasa tal cual.
ARGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    ARGS+=("${1%%=*}=$PROJ/runs/$(sed "s#,#,$PROJ/runs/#g" <<<"${1#*=}")")
    shift
done
[[ "${1:-}" == "--" ]] && shift

"$PROJ/venvs/rsna/bin/python" rsna/tabla_corregida.py --labels "$LABELS" --out "$OUT" "${ARGS[@]}" "$@"
echo "fin=$(date -Is)"
