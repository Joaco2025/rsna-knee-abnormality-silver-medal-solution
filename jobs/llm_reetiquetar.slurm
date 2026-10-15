#!/bin/bash
#SBATCH -A p082
#SBATCH -J llm-reetiquetar
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --deadline=2026-10-19T23:59:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Reetiquetado de informes con 3 lectores LLM locales (Ollama, módulo del sistema; nada instalado).
# Ver experiments/reportes/llm_reetiquetado.md.
# 1. Fase gold: los 58 Gold con qwen3.5:27b, qwen2.5:72b y llama3.3:70b -> gold_eval.json y criterio.
# 2. Fase resto (solo si el consenso PASA en Gold): los 4,349 no Gold, lector a lector, hasta 9.3 h.
#    Reanudable: las respuestas van a runs_joako/llm_reetiquetado_cache/<lector>.jsonl.
# 3. runs_joako/<job>/etiquetas_llm.parquet con todo lo leído (Gold + lo que haya del resto).
# Ollama arranca DENTRO del job en 127.0.0.1 (puerto propio) y muere al terminar. Modelos de solo lectura.
set -euo pipefail

FASES="${FASES:-gold,resto}"   # solo Gold:  sbatch --time=01:30:00 --export=ALL,FASES=gold jobs/llm_reetiquetar.slurm
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
CACHE="$PROJ/runs_joako/llm_reetiquetado_cache"
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
mkdir -p "$OUT/codigo" "$CACHE"
cp rsna/llm_reetiquetar.py jobs/llm_reetiquetar.slurm "$OUT/codigo/"
{
    echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) (código sin commitear, ver codigo/)"
    sha256sum rsna/llm_reetiquetar.py
} | tee "$OUT/inputs.txt"
lfs quota -h -g p082 /lustre 2>/dev/null | tail -2 | tee -a "$OUT/inputs.txt" || true

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/7.2.2 ollama/0.34.0
set -u
rocm-smi --showmeminfo vram --showpids 2>&1 | tail -15 | tee -a "$OUT/inputs.txt" || true

export OLLAMA_MODELS=/lustre/proyectos/ollama-models       # almacén compartido: SOLO LECTURA
export OLLAMA_NOPRUNE=1                                    # no intentar limpiar blobs ajenos
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"
export OLLAMA_KEEP_ALIVE=30m OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=4
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

ollama serve >"$OUT/ollama_serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
for _ in $(seq 60); do
    ollama list >/dev/null 2>&1 && break
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "ollama serve terminó antes de responder; ver $OUT/ollama_serve.log" >&2; exit 1; }
    sleep 2
done
ollama --version | tee -a "$OUT/inputs.txt"
ollama list | tee -a "$OUT/inputs.txt"

"$PY" -I rsna/llm_reetiquetar.py \
    --host "$OLLAMA_HOST" \
    --lectores qwen3.5:27b,qwen2.5:72b,llama3.3:70b \
    --out "$OUT" \
    --cache "$CACHE" \
    --workers 4 \
    --horas 9.3 \
    --fases "$FASES" \
    --parar-si-falla

ollama ps || true
du -sh "$OUT" "$CACHE" | tee -a "$OUT/inputs.txt"
echo "fin=$(date -Is)"
