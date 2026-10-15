#!/bin/bash
#SBATCH -A p082
#SBATCH -J ft0949
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Fine-tuning del checkpoint público 0.949 (Raptor, HEREDA OAI: solo Final A). Ver experiments/reportes/ft0949.md.
# Modo por variable de entorno:  sbatch -J ft0949-v1 --export=ALL,MODO=v1 jobs/ft0949_entrenar.slurm
#   v1: recorte central del 80 % (307 px) reescalado a 224     v2: recorte central de 320 px, nativo
# 1. Construye el corpus native384dense-v10 (~62 GB) en el /scratch local del nodo. Nada en Lustre.
# 2. 5 épocas all-data, lr 1e-5 / 2e-4, OneCycle 10 %. Gold solo como diagnóstico; Spearman frente a la base.
# 3. Guarda en runs_joako/<job>/: last_fp16.pt y prom3_fp16.pt (~146 MB c/u), historial, preds, resumen.json.
set -euo pipefail

MODO="${MODO:?define MODO=v1 o MODO=v2 con --export}"
[[ "$MODO" == v1 || "$MODO" == v2 ]] || { echo "ERROR: MODO=$MODO" >&2; exit 1; }
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
SCR="/scratch/${USER:?}/ft0949_${MODO}_${SLURM_JOB_ID:?}"
LABELS="$PROJ/runs_joako/etiquetas_coatnet_20261014/etiquetas_blendjev.parquet"
CKPT="$PROJ/code_joako/referencias/rsna_research/pesos/raptor_0949/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"

for f in "$LABELS" "$CKPT" "$PROJ/preproc/etiquetas_v1_2079203.parquet"; do
    [[ -r "$f" ]] || { echo "ERROR: no se puede leer $f" >&2; exit 1; }
done

mkdir -p "$OUT/codigo" "$SCR"
trap 'rm -rf -- "${SCR:?}"' EXIT        # el corpus de /scratch nunca sobrevive al job

# El código no está commiteado (regla de la sesión): se guarda una copia y su sha256 junto a la salida.
cp rsna/ft0949_entrenar.py rsna/entrenar_coatnet.py rsna/raptor_corpus.py rsna/raptor_preproc.py \
   jobs/ft0949_entrenar.slurm "$OUT/codigo/"
export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
{
    echo "job=$SLURM_JOB_ID modo=$MODO nodo=$(hostname) inicio=$(date -Is) commit=$GIT_COMMIT (código sin commitear, ver codigo/)"
    sha256sum rsna/ft0949_entrenar.py rsna/entrenar_coatnet.py rsna/raptor_corpus.py rsna/raptor_preproc.py "$LABELS"
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

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export TIMM_FUSED_ATTN=0                                   # la única atención medida sin fallos (bench v2)
export MIOPEN_USER_DB_PATH="$SCR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$SCR/miopen"
mkdir -p "$SCR/miopen"

echo "=== corpus: $(date -Is) ==="
"$PY" -I rsna/raptor_corpus.py --out "$SCR/corpus" --procesos 16
cp "$SCR/corpus/indice.csv" "$OUT/indice_corpus.csv"
du -sh "$SCR/corpus" | tee -a "$OUT/inputs.txt"

echo "=== fine-tuning $MODO: $(date -Is) ==="
export OMP_NUM_THREADS=4
"$PY" -I rsna/ft0949_entrenar.py \
    --corpus "$SCR/corpus" \
    --out "$OUT" \
    --modo "$MODO" \
    --ckpt "$CKPT" \
    --labels "$LABELS" \
    --workers 8 \
    --max-horas 5.5

du -sh "$OUT" | tee -a "$OUT/inputs.txt"
echo "fin=$(date -Is)"
