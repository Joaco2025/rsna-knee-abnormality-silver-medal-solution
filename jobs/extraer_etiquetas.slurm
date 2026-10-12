#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-extraer-etiquetas
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Extrae estado y gravedad de los 12 hallazgos de los informes con un modelo local (Ollama) y,
# si el subconjunto incluye estudios etiquetados, evalúa contra las etiquetas oficiales.
# Ollama arranca DENTRO del job, escucha solo en 127.0.0.1 y muere al terminar.
# Por defecto procesa solo los 58 estudios etiquetados (unos 30 minutos).
# Lanzar desde code_joako/:  sbatch jobs/extraer_etiquetas.slurm [labeled|unlabeled|all] [MODELO] [PARTE/N] [JSONL_PREVIO|JOBID...]
# Para los ~4,349 no etiquetados (unos 21 s por informe) repartir en dos jobs y ampliar el tiempo:
#   sbatch --time=16:00:00 jobs/extraer_etiquetas.slurm unlabeled qwen2.5:72b 0/2
#   sbatch --time=16:00:00 jobs/extraer_etiquetas.slurm unlabeled qwen2.5:72b 1/2
# Si un job se corta, relanzarlo pasando su extraccion.jsonl o job ID como JSONL_PREVIO para no repetir informes.
set -euo pipefail

SUBSET=${1:-labeled}
MODEL=${2:-qwen2.5:72b}
SHARD=${3:-0/1}
PREVIOS=("${@:4}")
PARALELO=${PARALELO:-1}        # informes en paralelo; se pasa con: sbatch --export=ALL,PARALELO=4 ...

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code_joako"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) subset=$SUBSET modelo=$MODEL parte=$SHARD paralelo=$PARALELO"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

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
        echo "ERROR: Ambigüedad para previo '$ref': existe tanto en runs_joako ($f_joako) como en runs ($f_owen). Especifica la ruta completa." >&2
        return 2
    elif [[ $h_joako -eq 1 ]]; then
        echo "INFO: Previo '$ref' resuelto en runs_joako: $f_joako (personal)" >&2
        echo "$f_joako"
        return 0
    elif [[ $h_owen -eq 1 ]]; then
        echo "INFO: Previo '$ref' resuelto en runs: $f_owen (compartido Owen, solo lectura)" >&2
        echo "$f_owen"
        return 0
    else
        echo "ERROR: No se encontró 'extraccion.jsonl' para '$ref' en runs_joako ni en runs." >&2
        return 1
    fi
}

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/7.2.2 ollama/0.34.0
set -u

# Los modelos propios del proyecto (hf.co/... o p082/...) viven en $PROJ/modelos/ollama; el resto,
# en la carpeta compartida del cluster, que es de SOLO LECTURA.
case "$MODEL" in
    hf.co/*|p082/*) export OLLAMA_MODELS="$PROJ/modelos/ollama" ;;
    *)       export OLLAMA_MODELS=/lustre/proyectos/ollama-models ;;
esac
export OLLAMA_NOPRUNE=1                                   # no intentar limpiar blobs ajenos
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"   # puerto propio del job, solo local
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PARALLEL="$PARALELO"                    # informes simultáneos sobre una sola copia del modelo
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

# --- Servidor (vive solo mientras dure el job) ---------------------------------
ollama serve >"$OUT/ollama_serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 60); do
    ollama list >/dev/null 2>&1 && break
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "ollama serve terminó antes de responder; ver $OUT/ollama_serve.log" >&2; exit 1; }
    sleep 2
done

# --- Trabajo -----------------------------------------------------------------
PREVIOS_RES=()
for p in "${PREVIOS[@]}"; do
    f=$(resolver_extraccion "$p")
    [[ -r "$f" ]] || exit 1
    PREVIOS_RES+=("$f")
done

"$PY" etiquetas/extraer.py --host "$OLLAMA_HOST" --model "$MODEL" --subset "$SUBSET" --shard "$SHARD" --workers "$PARALELO" \
    --out "$OUT" ${PREVIOS_RES[@]:+--skip-from "${PREVIOS_RES[@]}"}
ollama ps

if [[ "$SUBSET" != unlabeled ]]; then
    "$PY" etiquetas/evaluar_extraccion.py "$OUT/extraccion.jsonl" --lector etiquetas/piloto_58/prediccion_lector.csv
fi

echo "fin=$(date -Is)"
