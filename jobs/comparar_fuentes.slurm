#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-comparar-fuentes
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Compara en los 58 oficiales las fuentes de etiquetas derivadas (rsna/comparar_fuentes.py).
# Lanzar desde code/:  sbatch jobs/comparar_fuentes.slurm nombre=RUTA ...   (tablas .parquet o CSV externos)
set -euo pipefail
[[ $# -ge 1 ]] || { echo "indica al menos una fuente nombre=RUTA" >&2; exit 1; }
PROJ=/lustre/proyectos/p082
cd "$PROJ/code"
echo "job=$SLURM_JOB_ID inicio=$(date -Is) commit=$(git rev-parse --short HEAD)"
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
"$PROJ/venvs/rsna/bin/python" rsna/comparar_fuentes.py "$@"
echo "fin=$(date -Is)"
