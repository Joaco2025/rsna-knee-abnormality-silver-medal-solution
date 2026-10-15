#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-probar-coatnet
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Prueba de cableado del piloto CoAtNet, solo CPU: construye en /scratch el corpus de 6 estudios de train,
# 4 del fold 0 y 4 Gold, corre rsna/entrenar_coatnet.py --smoke (2 épocas, k=4, k_eval=6, acumulación 4)
# hasta el final (SWA, criterio frente a effhib, guardado) y borra el corpus. No mide calidad.
# Salida: runs_joako/<job>/ (unos MB: dos .pt de ~300 MB cada uno se borran al final, se guarda su tamaño).
set -euo pipefail

PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
SCR="/scratch/${USER:?}/coatnet_smoke_${SLURM_JOB_ID:?}"
mkdir -p "$OUT" "$SCR"
trap 'rm -rf -- "${SCR:?}"' EXIT        # el corpus de /scratch nunca sobrevive al job

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is)"
{ echo "commit=$(git rev-parse --short HEAD)"; git status --porcelain rsna/ jobs/; \
  sha256sum rsna/entrenar_coatnet.py rsna/raptor_corpus.py rsna/etiquetas_coatnet.py; } | tee "$OUT/commit.txt"
df -h /scratch | tail -1

export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 TIMM_FUSED_ATTN=0

"$PY" -I - "$SCR/uids.txt" <<'EOF'
import hashlib, sys, pandas as pd
v1 = pd.read_parquet("/lustre/proyectos/p082/preproc/etiquetas_v1_2079203.parquet")
f = v1.StudyInstanceUID.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 5)
sel = (v1[~v1.oficial & (f != 0)].StudyInstanceUID.head(6).tolist() + v1[~v1.oficial & (f == 0)].StudyInstanceUID.head(4).tolist()
       + v1[v1.oficial].StudyInstanceUID.head(4).tolist())
open(sys.argv[1], "w").write("\n".join(sel))
print(len(sel), "estudios de prueba")
EOF

"$PY" -I rsna/raptor_corpus.py --out "$SCR/corpus" --procesos 8 --uids "$SCR/uids.txt"
"$PY" -I rsna/entrenar_coatnet.py --corpus "$SCR/corpus" --out "$OUT/smoke" --smoke --workers 2

ls -la "$OUT/smoke" "$OUT/smoke/preds" | tee "$OUT/listado.txt"
rm -f -- "${OUT:?}/smoke/last.pt" "${OUT:?}/smoke/swa.pt"   # pesos de prueba sin valor; no ocupar cuota
echo "fin=$(date -Is)"
