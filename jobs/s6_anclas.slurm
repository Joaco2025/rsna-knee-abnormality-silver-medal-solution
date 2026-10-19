#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-anclas
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# D paso 4 (semillas S6): anclas SKM-TEA de los 4,321 estudios del pool de nartaa (4,263 train + 58 gold), con su
# precompute_anchors.py sin cambios y la anatomy.pt de heliosli (sha verificado).  Lee kaggle_data (solo lectura).
# Primero 40 estudios + paridad contra los sha256 de nartaa; si los 40 renderizan sin error, sigue con el resto
# (el script reanuda: no repite los 40).  Salida: runs_joako/<job>/anchors/ (~4,321 JSON pequeños, < 100 MB).
# Ver runs_joako/exp_s6_semillas/LEEME.md.
set -euo pipefail
PROJ=/lustre/proyectos/p082
EXP="$PROJ/runs_joako/exp_s6_semillas"
S6="$EXP/S6F0"
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
mkdir -p "$OUT/anchors"
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID:?}"   # caché MIOpen local (Lustre la corrompe)
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
TMPW="/tmp/s6anc_${SLURM_JOB_ID:?}"
echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is)"
sha256sum "$S6"/*.py | tee "$OUT/codigo.sha256"

# 1) humo: 40 estudios, 4 workers
"$PY" -u "$S6/precompute_anchors.py" --order "$S6/data/pool_ok.json" --out "$OUT/anchors" --tmp "$TMPW" \
  --workers 4 --limit 40 2>&1 | tee "$OUT/humo.log"
python3 -I "$EXP/herramientas/paridad_anclas.py" "$OUT/anchors" | tee "$OUT/paridad_humo.json"
if ! grep -q '"n_errors": 0' "$OUT/humo.log"; then
  echo "HUMO_FALLA: hay estudios con error en los primeros 40; se para aquí"; exit 3
fi

# 2) completo (reanuda)
"$PY" -u "$S6/precompute_anchors.py" --order "$S6/data/pool_ok.json" --out "$OUT/anchors" --tmp "$TMPW" \
  --workers 12 2>&1 | tee "$OUT/completo.log"
python3 -I "$EXP/herramientas/paridad_anclas.py" "$OUT/anchors" | tee "$OUT/paridad.json"
echo "fin=$(date -Is)"
