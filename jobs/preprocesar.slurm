#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-preprocesar
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Convierte cada estudio DICOM en un tensor (4, 32, 320, 320) uint8: sagital FS, sagital no FS,
# coronal FS y axial FS, en orientación canónica y con intensidad normalizada por serie.
# El lote completo de train son 4,407 archivos de 13 MB, unos 58 GB en total.
# Lanzar desde code_joako/:  sbatch jobs/preprocesar.slurm [VERSION] [LIMITE] [train|test]
#   VERSION: nombre de la carpeta de salida bajo preproc_joako/ (por defecto v1)
#   LIMITE:  número de estudios, para pruebas; 0 = todos
# Es reanudable: relanzarlo con la misma VERSION continúa donde quedó.
set -euo pipefail

VERSION=${1:-v1}
LIMIT=${2:-0}
SPLIT=${3:-train}

# --- Rutas y Aislamiento -----------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA

# Las nuevas salidas personales residen estrictamente en preproc_joako/
PREPROC_ROOT="$PROJ/preproc_joako"
OUT="$PREPROC_ROOT/${VERSION}_${SPLIT}"
RUN="$PROJ/runs_joako/${SLURM_JOB_ID:?}"

# Salvaguarda activa: impedir cualquier escritura en el preproc/ compartido de Owen
if [[ "$OUT" == "$PROJ/preproc/"* && "$OUT" != "$PROJ/preproc_joako/"* ]]; then
    echo "ERROR DE SEGURIDAD: La ruta de salida '$OUT' apunta al almacenamiento compartido de Owen." >&2
    echo "Las salidas personales deben generarse bajo '$PROJ/preproc_joako/'." >&2
    exit 1
fi

mkdir -p "$OUT" "$RUN"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) salida=$OUT limite=$LIMIT split=$SPLIT"
echo "commit=$(git rev-parse --short HEAD)" | tee "$RUN/commit.txt" | tee -a "$OUT/commits.txt"
lfs quota -h -g p082 /lustre | sed -n 3p

# --- Entorno -----------------------------------------------------------------
PY="$PROJ/venvs/rsna/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # un hilo por proceso: paraleliza el Pool
export OPENCV_FOR_THREADS_NUM=1

# --- Trabajo -----------------------------------------------------------------
"$PY" rsna/preprocesar.py --out "$OUT" --split "$SPLIT" --workers "${SLURM_CPUS_PER_TASK:?}" --limit "$LIMIT" --qa 12

du -sh "$OUT"
echo "fin=$(date -Is)"
