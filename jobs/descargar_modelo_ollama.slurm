#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-descargar-modelo
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Descarga un modelo GGUF de Hugging Face al almacén de modelos del proyecto, con Ollama.
# El servidor de Ollama vive solo dentro del job y escucha en 127.0.0.1.
# No usa token: solo sirve para repositorios públicos sin restricción de acceso.
# Lanzar desde code/:  sbatch jobs/descargar_modelo_ollama.slurm [REFERENCIA]
#   REFERENCIA con el formato de Ollama: hf.co/<usuario>/<repositorio>:<cuantización>
set -euo pipefail

MODEL=${1:-hf.co/unsloth/medgemma-27b-text-it-GGUF:Q8_0}

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export OLLAMA_MODELS="$PROJ/modelos/ollama"      # almacén de modelos del proyecto (persistente)
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"                # log del servidor de este job
mkdir -p "$OUT" "$OLLAMA_MODELS"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) modelo=$MODEL destino=$OLLAMA_MODELS"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"
lfs quota -h -g p082 /lustre | tail -2

# --- Red: comprobar la salida a Hugging Face antes de empezar (verificando el certificado) ------
curl -sS -m 30 -o /dev/null -w "huggingface.co http=%{http_code}\n" https://huggingface.co/api/models/unsloth/medgemma-27b-text-it-GGUF \
    || { echo "El nodo no llega a huggingface.co o el certificado no verifica; no se descarga nada." >&2; exit 1; }

# --- Entorno -----------------------------------------------------------------
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load ollama/0.34.0
set -u
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"   # puerto propio del job, solo local
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
ollama pull "$MODEL" 2>&1 | tr '\r' '\n' | awk 'NR % 200 == 1 || /success|error|verifying|writing/'
ollama list
ollama show "$MODEL" || true
du -sh "$OLLAMA_MODELS"

echo "fin=$(date -Is)"
