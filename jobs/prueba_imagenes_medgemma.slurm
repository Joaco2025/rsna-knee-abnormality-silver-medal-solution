#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-imagenes-medgemma
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Prueba sin ajuste de MedGemma 1.5 4B (con visión) sobre las resonancias de los estudios etiquetados.
# Verifica los SHA-256 de los archivos subidos a mano, arranca el llama-server que trae Ollama DENTRO
# del job (solo 127.0.0.1) y le pasa los cortes axiales. Las imágenes no salen del nodo.
# Lanzar desde code/:  sbatch jobs/prueba_imagenes_medgemma.slurm [N_ESTUDIOS] [por_hallazgo|json] [PLANOS] [OPCIONES...]
#   N_ESTUDIOS: 4 por defecto (prueba corta); 0 = los 58 etiquetados (ampliar --time).
set -euo pipefail

LIMIT=${1:-4}
MODE=${2:-por_hallazgo}
PLANES=${3:-Axial}
EXTRA=("${@:4}")      # opciones adicionales para el script, p. ej. --negative-for "Effusion,Baker's" --exclude-first 4

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code"

MODEL="$PROJ/modelos/gguf/medgemma-1.5-4b-it-Q8_0.gguf"
MMPROJ="$PROJ/modelos/gguf/mmproj-F16.gguf"
SHA_MODEL=10c7b9a0d8027c0c151e2050156376f5ed9d4b437494eae81d9cdb81e9b50219    # publicados por Hugging Face
SHA_MMPROJ=f45f0f750587494e8f976d952cb396dfaa3158662120e2f9c224fc11f3882c83

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) estudios=$LIMIT modo=$MODE planos=$PLANES"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# --- Integridad --------------------------------------------------------------
for par in "$MODEL:$SHA_MODEL" "$MMPROJ:$SHA_MMPROJ"; do
    f=${par%%:*}; esperado=${par##*:}
    [[ -r "$f" ]] || { echo "No existe $f" >&2; exit 1; }
    calc=$(sha256sum "$f" | cut -d' ' -f1)
    if [[ "$calc" != "$esperado" ]]; then
        echo "HASH: $(basename "$f") NO coincide (calculado $calc)" | tee /dev/stderr
        exit 1
    fi
    echo "HASH: $(basename "$f") coincide"
done

# --- Entorno -----------------------------------------------------------------
PY=/apps/miniconda3/bin/python
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/7.2.2
set -u
LLAMA_DIR=/apps/ollama/0.34.0/lib/ollama                 # motor llama.cpp que trae Ollama 0.34.0
export LD_LIBRARY_PATH="$LLAMA_DIR:$LLAMA_DIR/rocm_v7_2:${LD_LIBRARY_PATH:-}"
export GGML_BACKEND_PATH="$LLAMA_DIR/rocm_v7_2/libggml-hip.so"   # backend de GPU (ROCm/HIP)
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
PORT=$((20000 + SLURM_JOB_ID % 20000))                   # puerto propio del job, solo local

# --- Servidor (vive solo mientras dure el job) ---------------------------------
# Sin puntos de control ni caché en RAM: con contextos de ~22 mil tokens por estudio agotaban la
# memoria del job (OOM en el job 2079151). La reutilización del prefijo dentro del slot se mantiene.
echo "--- dispositivos que ve el motor:"
"$LLAMA_DIR/llama-server" --list-devices 2>&1 | tail -n 4 || true
"$LLAMA_DIR/llama-server" --model "$MODEL" --mmproj "$MMPROJ" --host 127.0.0.1 --port "$PORT" \
    --no-webui --offline -c 32768 -np 1 -ngl 999 --jinja --ctx-checkpoints 0 --cache-ram 0 \
    >"$OUT/llama_server.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 150); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "llama-server terminó antes de responder:" >&2; tail -n 30 "$OUT/llama_server.log" >&2; exit 1; }
    sleep 2
done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "llama-server no respondió a tiempo" >&2; tail -n 30 "$OUT/llama_server.log" >&2; exit 1; }
echo "--- memoria de GPU con el modelo cargado:"
rocm-smi --showmeminfo vram 2>/dev/null | grep -i "used" || true
echo "--- carga del modelo:"
grep -E "offloaded|using device|mmproj|clip|n_ctx  |chat template|vision" "$OUT/llama_server.log" | cut -c1-160 | head -14 || true

# --- Trabajo -----------------------------------------------------------------
"$PY" imagenes/prueba_medgemma.py --host "127.0.0.1:$PORT" --out "$OUT" --limit "$LIMIT" --mode "$MODE" --planes "$PLANES" ${EXTRA[@]:+"${EXTRA[@]}"}

echo "--- tiempos del servidor (últimas consultas):"
grep -E "prompt eval time|       eval time|image.*encod|encoding image" "$OUT/llama_server.log" | tail -n 6 | cut -c1-160 || true
echo "--- memoria del servidor al terminar: $(ps -o rss= -p "$SERVE_PID" 2>/dev/null | awk '{printf "%.1f GB", $1/1048576}')"
echo "fin=$(date -Is)"
