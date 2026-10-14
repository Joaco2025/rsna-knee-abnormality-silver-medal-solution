#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-lateralidad
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Mide si se puede saber el lado (izquierda o derecha) de cada estudio a partir de las cabeceras DICOM.
# Lanzar desde code/:  sbatch jobs/lateralidad.slurm
set -euo pipefail
PROJ=/lustre/proyectos/p082
cd "$PROJ/code"
echo "job=$SLURM_JOB_ID inicio=$(date -Is) commit=$(git rev-parse --short HEAD)"
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
"$PROJ/venvs/rsna/bin/python" eda/09_lateralidad.py data/eda/dicom/cabeceras_train_1xserie_px.parquet --out "$PROJ/preproc/lados_train_${SLURM_JOB_ID}.parquet"
echo "fin=$(date -Is)"
