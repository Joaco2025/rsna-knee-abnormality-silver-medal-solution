#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-verificar-series
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Verifica geometría y consistencia interna de cada serie a partir del parquet de cabeceras.
# No abre DICOM: solo lee el parquet que dejó jobs/cabeceras.slurm.
# Lanzar desde code_joako/:  sbatch jobs/verificar_series.slurm [JOBID_DEL_BARRIDO|RUTA] [train|test]
set -euo pipefail

BARRIDO=${1:-2078579}
SPLIT=${2:-train}

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA

resolver_barrido() {
    local ref="$1"
    local split="$2"
    if [[ -f "$ref" ]]; then
        echo "$ref"
        return 0
    fi
    local f_joako="$PROJ/runs_joako/$ref/cabeceras_${split}_todos.parquet"
    local f_owen="$PROJ/runs/$ref/cabeceras_${split}_todos.parquet"
    local h_joako=0; local h_owen=0
    [[ -r "$f_joako" ]] && h_joako=1
    [[ -r "$f_owen" ]] && h_owen=1

    if [[ $h_joako -eq 1 && $h_owen -eq 1 ]]; then
        echo "ERROR: Ambigüedad para barrido '$ref': existe tanto en runs_joako ($f_joako) como en runs ($f_owen). Especifica la ruta completa." >&2
        return 2
    elif [[ $h_joako -eq 1 ]]; then
        echo "INFO: Barrido '$ref' resuelto en runs_joako: $f_joako (personal)" >&2
        echo "$f_joako"
        return 0
    elif [[ $h_owen -eq 1 ]]; then
        echo "INFO: Barrido '$ref' resuelto en runs: $f_owen (compartido Owen, solo lectura)" >&2
        echo "$f_owen"
        return 0
    else
        echo "ERROR: No se encontró 'cabeceras_${split}_todos.parquet' para '$ref' en runs_joako ni en runs." >&2
        return 1
    fi
}

PARQUET=$(resolver_barrido "$BARRIDO" "$SPLIT")
[[ -r "$PARQUET" ]] || exit 1

OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) entrada=$PARQUET"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python

# --- Trabajo -----------------------------------------------------------------
"$PY" eda/08_verificar_series.py "$PARQUET" --out "$OUT" --split "$SPLIT"

echo "fin=$(date -Is)"
