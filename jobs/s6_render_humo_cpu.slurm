#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-s6-render-humo
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Humo del render de la caché de S6 (renderizar_px.py corregido): 30 estudios en el /tmp del nodo, reserva exacta por el
# manifiesto de anclas y paridad px_sha1 con el render de referencia. Solo lectura de kaggle_data; no escribe en Lustre
# salvo el log.
set -euo pipefail
PY=/lustre/proyectos/p082/venvs/rsna/bin/python
S6=/lustre/proyectos/p082/runs_joako/exp_s6_semillas/S6F0
ANC=/lustre/proyectos/p082/runs_joako/2092983/anchors
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
D="/tmp/s6render_humo_${SLURM_JOB_ID:?}"
"$PY" -u "$S6/renderizar_px.py" --pool "$S6/data/pool_ok.json" --out "$D/px" --anchors "$ANC" --manifiesto "$ANC/manifest.jsonl" \
  --tmp "$D/work" --workers 8 --limit 30
"$PY" -I -c "
import json, numpy as np
d = '$D/px'
i = json.load(open(d + '/index.json')); px = np.load(d + '/px.u8.npy', mmap_mode='r')
print('index done', len(i['done']), 'errores', len(i['errors']), 'paridad', i['paridad_px_sha1'], 'px', px.shape)
"
