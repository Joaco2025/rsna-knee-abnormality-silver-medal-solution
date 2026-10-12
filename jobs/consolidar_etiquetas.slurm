#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-consolidar-etiquetas
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Junta las extracciones de varios jobs en una tabla por estudio y resume si tienen sentido.
# Solo lee los extraccion.jsonl de runs_joako/ o runs/ y train.csv; no usa GPU ni modelos.
# Lanzar desde code_joako/:  sbatch jobs/consolidar_etiquetas.slurm [JOBID|RUTA ...]
set -euo pipefail

if [[ $# -gt 0 ]]; then
    EXTRACCIONES=("$@")
else
    echo "INFO: Usando extracciones de referencia por defecto (Owen): 2078806 2078913 2078914"
    EXTRACCIONES=(2078806 2078913 2078914)
fi

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) extracciones=${EXTRACCIONES[*]}"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# Resolución unívoca de cada extracción
resolver_extraccion() {
    local ref="$1"
    if [[ -f "$ref" ]]; then
        echo "$ref"
        return 0
    fi
    local f_joako="$PROJ/runs_joako/$ref/extraccion.jsonl"
    local f_owen="$PROJ/runs/$ref/extraccion.jsonl"
    local h_joako=0; local h_owen=0
    [[ -r "$f_joako" ]] && h_joako=1
    [[ -r "$f_owen" ]] && h_owen=1

    if [[ $h_joako -eq 1 && $h_owen -eq 1 ]]; then
        echo "ERROR: Ambigüedad para extracción '$ref': existe tanto en runs_joako ($f_joako) como en runs ($f_owen). Especifica la ruta completa." >&2
        return 2
    elif [[ $h_joako -eq 1 ]]; then
        echo "INFO: Extracción '$ref' resuelta en runs_joako: $f_joako (personal)" >&2
        echo "$f_joako"
        return 0
    elif [[ $h_owen -eq 1 ]]; then
        echo "INFO: Extracción '$ref' resuelta en runs: $f_owen (compartido Owen, solo lectura)" >&2
        echo "$f_owen"
        return 0
    else
        echo "ERROR: No se encontró 'extraccion.jsonl' para '$ref' en runs_joako ni en runs." >&2
        return 1
    fi
}

ENTRADAS=()
for id in "${EXTRACCIONES[@]}"; do
    f=$(resolver_extraccion "$id")
    [[ -r "$f" ]] || exit 1
    ENTRADAS+=("$f")
done

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python

# --- Trabajo -----------------------------------------------------------------
"$PY" etiquetas/consolidar.py --out "$OUT" "${ENTRADAS[@]}"

echo "fin=$(date -Is)"
