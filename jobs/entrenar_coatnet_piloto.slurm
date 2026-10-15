#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-coatnet-f0
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Piloto CoAtNet, receta del hilo 746792 (#53) sin OAI, partición 0 de 5 (md5 % 5). Ver
# experiments/reportes/receta_coatnet.md. Criterio fijado antes: OOF fold 0 con juez v1 >= effhib + 0.005
# (effhib fold 0 = 0.8628) y sin NaN ni divergencia en fp16. El Gold se reporta y elige las épocas del SWA.
#
# 1. Construye el corpus native384 (~62 GB) en el /scratch local del nodo con 16 procesos. Nada en Lustre.
# 2. Entrena 16 épocas (fp16 + GradScaler, atención sin fusionar) y evalúa Gold y OOF en cada época.
# 3. Guarda en runs_joako/<job>/: historial, predicciones por época, resumen con el criterio, last.pt y
#    swa.pt (~300 MB cada uno). El corpus se borra al salir, también si el job falla.
set -euo pipefail

PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
SCR="/scratch/${USER:?}/coatnet_f0_${SLURM_JOB_ID:?}"
LABELS="$PROJ/runs_joako/etiquetas_coatnet_20261014"
WEIGHTS="$PROJ/code_joako/referencias/rsna_research/pesos/coatnet_rmlp_2_rw_384/model.safetensors"

for f in "$LABELS/etiquetas_blendjev.parquet" "$LABELS/etiquetas_blend.parquet" "$WEIGHTS" \
         "$PROJ/preproc/etiquetas_v1_2079203.parquet" "$PROJ/runs/2084813/oof_effhib.parquet"; do
    [[ -r "$f" ]] || { echo "ERROR: no se puede leer $f" >&2; exit 1; }
done

mkdir -p "$OUT" "$SCR"
trap 'rm -rf -- "${SCR:?}"' EXIT        # el corpus de /scratch nunca sobrevive al job

export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
{
    echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) commit=$GIT_COMMIT"
    git status --porcelain rsna/ jobs/
    sha256sum rsna/entrenar_coatnet.py rsna/raptor_corpus.py rsna/raptor_preproc.py rsna/etiquetas_coatnet.py \
              "$LABELS"/etiquetas_*.parquet "$WEIGHTS"
} | tee "$OUT/inputs.txt"

LIBRE_GB=$(df -BG --output=avail /scratch | tail -1 | tr -dc '0-9')
echo "/scratch libre: ${LIBRE_GB} GB" | tee -a "$OUT/inputs.txt"
(( LIBRE_GB >= 90 )) || { echo "ERROR: /scratch con menos de 90 GB libres; el corpus ocupa ~62 GB" >&2; exit 1; }

set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
rocm-smi --showmeminfo vram --showpids 2>&1 | tail -15 | tee -a "$OUT/inputs.txt" || true

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MLFLOW_DISABLE_AGENT_HINT=1
export TIMM_FUSED_ATTN=0                                   # la única atención medida sin fallos (bench v2)
export MIOPEN_USER_DB_PATH="$SCR/miopen" MIOPEN_CUSTOM_CACHE_DIR="$SCR/miopen"
mkdir -p "$SCR/miopen"

echo "=== corpus: $(date -Is) ==="
"$PY" -I rsna/raptor_corpus.py --out "$SCR/corpus" --procesos 16
cp "$SCR/corpus/indice.csv" "$OUT/indice_corpus.csv"
du -sh "$SCR/corpus" | tee -a "$OUT/inputs.txt"

echo "=== entrenamiento: $(date -Is) ==="
export OMP_NUM_THREADS=4
"$PY" -I rsna/entrenar_coatnet.py \
    --corpus "$SCR/corpus" \
    --out "$OUT" \
    --labels "$LABELS/etiquetas_blendjev.parquet" \
    --juez-blend "$LABELS/etiquetas_blend.parquet" \
    --weights "$WEIGHTS" \
    --folds 5 --fold 0 \
    --workers 8 \
    --max-horas 11.3 \
    --mlflow "sqlite:///$PROJ/mlflow/mlflow_joako.db" \
    --mlflow-artifacts "$PROJ/mlflow/artifacts_joako" \
    --run-name "coatnet_f0_job-$SLURM_JOB_ID"

echo "fin=$(date -Is)"
