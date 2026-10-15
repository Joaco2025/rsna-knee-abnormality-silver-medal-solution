#!/bin/bash
#SBATCH -A p082
#SBATCH -J seeds-coatnet
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --deadline=2026-10-19T23:59:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Semillas ALL-DATA del CoAtNet propio, receta idéntica al piloto 2085360. Ver experiments/reportes/seeds_coatnet.md.
#   sbatch -J seeds-s7    --dependency=afterany:2085360 --export=ALL,SEED=7    jobs/seeds_coatnet.slurm
#   sbatch -J seeds-s2026 --dependency=afterany:2085360 --export=ALL,SEED=2026 jobs/seeds_coatnet.slurm
#   variante opcional:  --export=ALL,SEED=<s>,LADO=1   (corpus de lado unificado, rsna/seeds_corpus_lado.py)
#   medición pareada:   --export=ALL,SEED=42,LADO=1,FOLD0=1   (fold 0 con OOF, comparable con el piloto)
# 0. GATE: si runs_joako/2085360/resumen.json no existe o no dice PASA (criterio_swa), termina sin entrenar (< 1 min).
# 1. Corpus native384 (~62 GB) en el /scratch local del nodo. Nada en Lustre.
# 2. 16 épocas all-data, top-3 por Gold (sesgado) y SWA. Guarda last_fp16.pt y swa_fp16.pt (~146 MB c/u).
set -euo pipefail

SEED="${SEED:?define SEED con --export}"
LADO="${LADO:-0}"
FOLD0="${FOLD0:-0}"     # 1: fold 0 en vez de all-data (para medir el lado unificado contra el piloto con OOF)
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
PILOTO="$PROJ/runs_joako/2085360/resumen.json"

mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID seed=$SEED lado=$LADO nodo=$(hostname) inicio=$(date -Is)" | tee "$OUT/gate.txt"
if ! "$PY" -I rsna/seeds_gate.py "$PILOTO" | tee -a "$OUT/gate.txt"; then
    echo "gate cerrado: no se entrena. fin=$(date -Is)" | tee -a "$OUT/gate.txt"
    exit 0
fi

SCR="/scratch/${USER:?}/seeds_${SEED}_${SLURM_JOB_ID:?}"
LABELS="$PROJ/runs_joako/etiquetas_coatnet_20261014"
WEIGHTS="$PROJ/code_joako/referencias/rsna_research/pesos/coatnet_rmlp_2_rw_384/model.safetensors"
for f in "$LABELS/etiquetas_blendjev.parquet" "$LABELS/etiquetas_blend.parquet" "$WEIGHTS" \
         "$PROJ/preproc/etiquetas_v1_2079203.parquet"; do
    [[ -r "$f" ]] || { echo "ERROR: no se puede leer $f" >&2; exit 1; }
done

mkdir -p "$OUT/codigo" "$SCR"
trap 'rm -rf -- "${SCR:?}"' EXIT        # el corpus de /scratch nunca sobrevive al job

# El código no está commiteado (regla de la sesión): copia y sha256 junto a la salida.
cp rsna/seeds_entrenar_coatnet.py rsna/seeds_gate.py rsna/seeds_corpus_lado.py rsna/raptor_corpus.py \
   rsna/raptor_preproc.py jobs/seeds_coatnet.slurm "$OUT/codigo/"
export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
{
    echo "commit=$GIT_COMMIT (código sin commitear, ver codigo/)"
    sha256sum rsna/seeds_entrenar_coatnet.py rsna/seeds_corpus_lado.py rsna/raptor_corpus.py rsna/raptor_preproc.py \
              "$LABELS"/etiquetas_*.parquet "$WEIGHTS" "$PILOTO"
} | tee "$OUT/inputs.txt"

LIBRE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc '0-9')
echo "/scratch libre: ${LIBRE_GB} GB" | tee -a "$OUT/inputs.txt"
(( LIBRE_GB >= 90 )) || { echo "ERROR: /scratch con menos de 90 GB libres; el corpus ocupa ~62 GB" >&2; exit 1; }
lfs quota -h -g p082 /lustre 2>/dev/null | tail -2 | tee -a "$OUT/inputs.txt" || true

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
rocm-smi --showmeminfo vram --showpids 2>&1 | tail -15 | tee -a "$OUT/inputs.txt" || true

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MLFLOW_DISABLE_AGENT_HINT=1
export TIMM_FUSED_ATTN=0
export MIOPEN_USER_DB_PATH="$SCR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$SCR/miopen"
mkdir -p "$SCR/miopen"

echo "=== corpus (lado=$LADO): $(date -Is) ==="
if [[ "$LADO" == 1 ]]; then
    CORPUS_TIPO=native384dense-v10-lado
    "$PY" -I rsna/seeds_corpus_lado.py --out "$SCR/corpus" --procesos 16
else
    CORPUS_TIPO=native384dense-v10
    "$PY" -I rsna/raptor_corpus.py --out "$SCR/corpus" --procesos 16
fi
cp "$SCR/corpus/indice.csv" "$OUT/indice_corpus.csv"
du -sh "$SCR/corpus" | tee -a "$OUT/inputs.txt"

if [[ "$FOLD0" == 1 ]]; then DATOS=(--folds 5 --fold 0); TAG=f0; else DATOS=(--all-data); TAG=alldata; fi
echo "=== entrenamiento $TAG seed=$SEED: $(date -Is) ==="
export OMP_NUM_THREADS=4
"$PY" -I rsna/seeds_entrenar_coatnet.py \
    --corpus "$SCR/corpus" \
    --corpus-tipo "$CORPUS_TIPO" \
    --out "$OUT" \
    "${DATOS[@]}" \
    --seed "$SEED" \
    --labels "$LABELS/etiquetas_blendjev.parquet" \
    --juez-blend "$LABELS/etiquetas_blend.parquet" \
    --weights "$WEIGHTS" \
    --ref-resumen "$PILOTO" \
    --workers 8 \
    --max-horas 11.3 \
    --mlflow "sqlite:///$PROJ/mlflow/mlflow_joako.db" \
    --mlflow-artifacts "$PROJ/mlflow/artifacts_joako" \
    --run-name "coatnet_${TAG}_s${SEED}_lado${LADO}_job-$SLURM_JOB_ID"

du -sh "$OUT" | tee -a "$OUT/inputs.txt"
echo "fin=$(date -Is)"
