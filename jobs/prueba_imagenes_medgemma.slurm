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
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Prueba sin ajuste de MedGemma 1.5 4B (con visión) sobre las resonancias de los estudios etiquetados.
# Verifica los SHA-256 de los archivos subidos a mano, usa el llama-server que trae Ollama DENTRO
# del job (solo 127.0.0.1, reiniciado en cada estudio) y le pasa los cortes axiales. Las imágenes no salen del nodo.
# Lanzar desde code_joako/:  sbatch jobs/prueba_imagenes_medgemma.slurm [N_ESTUDIOS] [por_hallazgo|json] [PLANOS] [OPCIONES...]
#   N_ESTUDIOS: 4 por defecto (prueba corta); 0 = los 58 etiquetados (ampliar --time).
set -euo pipefail

LIMIT=${1:-4}
MODE=${2:-por_hallazgo}
PLANES=${3:-Axial}
EXTRA=("${@:4}")      # opciones adicionales para el script, p. ej. --negative-for "Effusion,Baker's" --exclude-first 4

# --- Rutas -------------------------------------------------------------------
PROJ=/lustre/proyectos/p082
export KNEE_DATA="$PROJ/kaggle_data/competitions/rsna-knee-abnormality-detection"   # SOLO LECTURA
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"                                                 # directorio nuevo por job
mkdir -p "$OUT"
cd "$PROJ/code_joako"

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

# --- Trabajo -----------------------------------------------------------------
# El script arranca el servidor y lo reinicia en cada estudio: su memoria crece ~0.2 GB por corte
# recibido (22 GB tras 3 estudios en el job 2079152) y el modelo recarga en un par de segundos.
echo "--- dispositivos que ve el motor:"
"$LLAMA_DIR/llama-server" --list-devices 2>&1 | tail -n 2 || true
"$PY" imagenes/prueba_medgemma.py --host "127.0.0.1:$PORT" --out "$OUT" --limit "$LIMIT" --mode "$MODE" --planes "$PLANES" \
    --llama-server "$LLAMA_DIR/llama-server" --model "$MODEL" --mmproj "$MMPROJ" --restart-every 1 ${EXTRA[@]:+"${EXTRA[@]}"}

echo "fin=$(date -Is)"
