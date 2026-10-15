#!/bin/bash
#SBATCH -A p082
#SBATCH -J diag-coatnet
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

# Diagnóstico del CoAtNet (piloto 2085360): 6 épocas en el fold 0 (md5 % 5), OneCycle reescalado a 6 épocas,
# UNA sola cosa cambiada respecto al piloto. Ver experiments/reportes/bug_coatnet.md.
#   sbatch -J diag-<modo> --export=ALL,MODO=<modo> jobs/diag_coatnet.slurm
#   base6    piloto sin cambios (solo 6 épocas): referencia para el efecto del reescalado del OneCycle
#   bn8      micro-lote de 8 estudios × acumulación 2 (BatchNorm ve 8 estudios; lote efectivo y pasos iguales)
#   accum8   acumulación 8 en vez de 16 (lote efectivo 8, el doble de pasos de optimizador)
#   dd44     corpus de 44 cortes a 336 px con máscaras (el de dreaddevelopment), reescalado a 384
#   control  entrenamiento de dreaddevelopment portado casi literal (rsna/entrenar_coatnet_dd.py) sobre dd44
# Corpus en el /scratch del nodo (borrado al salir). A Lustre: historial, predicciones y last_fp16.pt (~146 MB).
set -euo pipefail

MODO="${MODO:?define MODO=base6|bn8|accum8|dd44|control con --export}"
case "$MODO" in base6|bn8|accum8|dd44|control) ;; *) echo "ERROR: MODO=$MODO" >&2; exit 1 ;; esac
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
SCR="/scratch/${USER:?}/diag_${MODO}_${SLURM_JOB_ID:?}"
LABELS="$PROJ/runs_joako/etiquetas_coatnet_20261014"
WEIGHTS="$PROJ/code_joako/referencias/rsna_research/pesos/coatnet_rmlp_2_rw_384/model.safetensors"

for f in "$LABELS/etiquetas_blendjev.parquet" "$LABELS/etiquetas_blend.parquet" "$WEIGHTS" \
         "$PROJ/preproc/etiquetas_v1_2079203.parquet"; do
    [[ -r "$f" ]] || { echo "ERROR: no se puede leer $f" >&2; exit 1; }
done

mkdir -p "$OUT/codigo" "$SCR"
trap 'rm -rf -- "${SCR:?}"' EXIT

# Código sin commitear (regla de la sesión): copia y sha256 junto a la salida.
cp rsna/entrenar_coatnet.py rsna/entrenar_coatnet_dd.py rsna/dd44_corpus.py rsna/raptor_corpus.py \
   rsna/raptor_preproc.py jobs/diag_coatnet.slurm "$OUT/codigo/"
{
    echo "job=$SLURM_JOB_ID modo=$MODO nodo=$(hostname) inicio=$(date -Is) commit=$(git rev-parse --short HEAD) (código sin commitear, ver codigo/)"
    sha256sum rsna/entrenar_coatnet.py rsna/entrenar_coatnet_dd.py rsna/dd44_corpus.py rsna/raptor_corpus.py \
              rsna/raptor_preproc.py "$LABELS"/etiquetas_*.parquet
} | tee "$OUT/inputs.txt"

LIBRE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc '0-9')
echo "/scratch libre: ${LIBRE_GB} GB" | tee -a "$OUT/inputs.txt"
(( LIBRE_GB >= 90 )) || { echo "ERROR: /scratch con menos de 90 GB libres" >&2; exit 1; }
lfs quota -h -g p082 /lustre 2>/dev/null | tail -2 | tee -a "$OUT/inputs.txt" || true

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MLFLOW_DISABLE_AGENT_HINT=1
export TIMM_FUSED_ATTN=0
export MIOPEN_USER_DB_PATH="$SCR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$SCR/miopen"
mkdir -p "$SCR/miopen"

echo "=== corpus: $(date -Is) ==="
if [[ "$MODO" == dd44 || "$MODO" == control ]]; then
    "$PY" -I rsna/dd44_corpus.py --out "$SCR/corpus" --procesos 16
else
    "$PY" -I rsna/raptor_corpus.py --out "$SCR/corpus" --procesos 16
fi
cp "$SCR/corpus/indice.csv" "$OUT/indice_corpus.csv"
du -sh "$SCR/corpus" | tee -a "$OUT/inputs.txt"

echo "=== entrenamiento $MODO: $(date -Is) ==="
export OMP_NUM_THREADS=4
COMUN=(--corpus "$SCR/corpus" --out "$OUT" --labels "$LABELS/etiquetas_blendjev.parquet"
       --juez-blend "$LABELS/etiquetas_blend.parquet" --weights "$WEIGHTS" --workers 8)
case "$MODO" in
    base6)   "$PY" -I rsna/entrenar_coatnet.py "${COMUN[@]}" --epochs 6 --topk 1 --pesos-fp16 --max-horas 5.5 ;;
    bn8)     "$PY" -I rsna/entrenar_coatnet.py "${COMUN[@]}" --epochs 6 --topk 1 --pesos-fp16 --max-horas 5.5 \
                 --micro-bs 8 --accum 2 --grad-ckpt ;;
    accum8)  "$PY" -I rsna/entrenar_coatnet.py "${COMUN[@]}" --epochs 6 --topk 1 --pesos-fp16 --max-horas 5.5 \
                 --accum 8 ;;
    dd44)    "$PY" -I rsna/entrenar_coatnet.py "${COMUN[@]}" --epochs 6 --topk 1 --pesos-fp16 --max-horas 5.5 \
                 --corpus-tipo dd44 ;;
    control) "$PY" -I rsna/entrenar_coatnet_dd.py "${COMUN[@]}" --epochs 6 --grad_ckpt ;;
esac
du -sh "$OUT"
echo "fin=$(date -Is)"
