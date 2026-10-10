#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-prueba-qwen
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Prueba de humo: arranca Ollama DENTRO del job, carga un modelo de la carpeta compartida en una
# MI210 y le pasa unos pocos informes. Comprueba que usa la GPU y mide la velocidad.
# El servidor escucha solo en 127.0.0.1 y muere al terminar el job.
# Lanzar desde code/:  sbatch jobs/prueba_qwen.slurm [MODELO]
set -euo pipefail

MODEL=${1:-qwen2.5:72b}

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) modelo=$MODEL"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/7.2.2 ollama/0.34.0
set -u

export OLLAMA_MODELS=/lustre/proyectos/ollama-models      # carpeta compartida, SOLO LECTURA
export OLLAMA_NOPRUNE=1                                   # no intentar limpiar blobs ajenos
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"   # puerto propio del job, solo local
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_NUM_PARALLEL=1
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-}"
rocm-smi --showmeminfo vram || true

# --- Servidor (vive solo mientras dure el job) ---------------------------------
ollama serve >"$OUT/ollama_serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 60); do
    ollama list >/dev/null 2>&1 && break
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "ollama serve terminó antes de responder; ver $OUT/ollama_serve.log" >&2; exit 1; }
    sleep 2
done
ollama --version
ollama list

# --- Trabajo -----------------------------------------------------------------
"$PY" etiquetas/prueba_ollama.py --host "$OLLAMA_HOST" --model "$MODEL" --out "$OUT"

# --- Evidencia de que se usó la GPU ---------------------------------------------
ollama ps
rocm-smi --showmeminfo vram || true

echo "fin=$(date -Is)"
