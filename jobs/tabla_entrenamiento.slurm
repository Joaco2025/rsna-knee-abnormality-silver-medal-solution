#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-tabla-entrenamiento
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Convierte la extracción consolidada de los informes en la tabla de objetivos de entrenamiento.
# Lanzar desde code_joako/:  sbatch jobs/tabla_entrenamiento.slurm [JOBID_CONSOLIDACION|RUTA] [VERSION]
set -euo pipefail

CONSOLIDACION=${1:-2079118}
VERSION=${2:-v1}

PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA

# Resolución unívoca de entrada
resolver_consolidacion() {
    local ref="$1"
    if [[ -f "$ref" ]]; then
        echo "$ref"
        return 0
    fi
    local f_joako="$PROJ/runs_joako/$ref/extraccion_consolidada.parquet"
    local f_owen="$PROJ/runs/$ref/extraccion_consolidada.parquet"
    local h_joako=0; local h_owen=0
    [[ -r "$f_joako" ]] && h_joako=1
    [[ -r "$f_owen" ]] && h_owen=1

    if [[ $h_joako -eq 1 && $h_owen -eq 1 ]]; then
        echo "ERROR: Ambigüedad para consolidación '$ref': existe tanto en runs_joako ($f_joako) como en runs ($f_owen). Especifica la ruta completa." >&2
        return 2
    elif [[ $h_joako -eq 1 ]]; then
        echo "INFO: Consolidación '$ref' resuelta en runs_joako: $f_joako (personal)" >&2
        echo "$f_joako"
        return 0
    elif [[ $h_owen -eq 1 ]]; then
        echo "INFO: Consolidación '$ref' resuelta en runs: $f_owen (compartido Owen, solo lectura)" >&2
        echo "$f_owen"
        return 0
    else
        echo "ERROR: No se encontró 'extraccion_consolidada.parquet' para '$ref' en runs_joako ni en runs." >&2
        return 1
    fi
}

ENTRADA=$(resolver_consolidacion "$CONSOLIDACION")
[[ -r "$ENTRADA" ]] || exit 1

RUN="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
PREPROC_DEST="$PROJ/preproc_joako"
mkdir -p "$RUN" "$PREPROC_DEST"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) entrada=$ENTRADA version=$VERSION"
echo "commit=$(git rev-parse --short HEAD)" | tee "$RUN/commit.txt"

"$PROJ/venvs/rsna/bin/python" rsna/tabla_entrenamiento.py "$ENTRADA" \
    --out "$RUN/etiquetas_${VERSION}.parquet"
# La tabla vigente se publica en el almacén personal preproc_joako/; cada job conserva la suya en runs_joako/.
cp "$RUN/etiquetas_${VERSION}.parquet" "$PREPROC_DEST/etiquetas_${VERSION}_${SLURM_JOB_ID}.parquet"
echo "fin=$(date -Is)"
