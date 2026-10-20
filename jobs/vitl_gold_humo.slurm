#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-vitl-gold
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH -D /lustre/proyectos/p082/code_joako
#SBATCH -o /lustre/proyectos/p082/code_joako/logs/%x_%j.out
#SBATCH -e /lustre/proyectos/p082/code_joako/logs/%x_%j.err

# Ruta 2 (hail mary): ViT-L de OrthoFoundation (heliosli, best.pt fold 0 ep. 7) con su infer.py SIN CAMBIOS sobre los 58 Gold.
# Para qué: (1) que el código corre offline con nuestro entorno, (2) segundos por estudio en MI210 (para estimar la T4),
# (3) predicciones en Gold para medir cuánto se parece a R6 (el ViT-L vio 48 de los 58 Gold: su AUC NO es evaluable).
# Lee: runs_joako/exp_s6_semillas/entradas/orthofoundation_v1 (solo lectura), kaggle_data (solo lectura, por enlaces).
# Escribe: runs_joako/$SLURM_JOB_ID/ (raíz falsa con ~10.5k enlaces + caché nativa + salida; estimado < 5 GB).
# Nivel 2 (GPU): enviar solo con confirmación explícita de jcastro.
set -euo pipefail
: "${CONFIRMADO:?exporta CONFIRMADO=1 solo con autorización explícita de jcastro}"
PROJ=/lustre/proyectos/p082
PY="$PROJ/venvs/rsna/bin/python"
ENT="$PROJ/runs_joako/exp_s6_semillas/entradas/orthofoundation_v1"
OUT="$PROJ/runs_joako/${SLURM_JOB_ID:?}"
mkdir -p "$OUT/pesos"
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "inicio=$(date -Is) nodo=$(hostname)"
"$PY" -I "$PROJ/runs_joako/exp_ViTL/raiz_gold.py" "$OUT/gold58_comp"
# anatomy.pt y kneexnet/ juntos (anatomy_runtime resuelve el enlace de anatomy.pt y busca kneexnet/ en su carpeta REAL:
# por eso anatomy.pt se copia, no se enlaza; en Kaggle el dataset de heliosli ya trae kneexnet/ junto a anatomy.pt)
cp "$ENT/anatomy.pt" "$OUT/pesos/anatomy.pt"
ln -s "$ENT/dependencies/kneexnet" "$OUT/pesos/kneexnet"
# copia del código de heliosli + el default.yaml de ultralytics 8.4.166 (heliosli lo vendorizó sin los archivos que no son .py;
# los 230 .py son idénticos al wheel oficial).  Igual que en el notebook: se escribe en la COPIA, la entrada no se toca.
cp -r "$ENT/cloud_code" "$OUT/code"
Y="$OUT/code/training/vendor_runtime/kneexnet/ultralytics/cfg/default.yaml"
cp "$PROJ/runs_joako/exp_ViTL/default.yaml" "$Y"
echo "285fa6ff485cb5d556ed216b966f0952a391c89a651901c49b246fda18cf0108  $Y" | sha256sum -c -
cd "$OUT/code"
/usr/bin/time -v "$PY" -u -m training.infer --data "$OUT/gold58_comp" --checkpoint "$ENT/best.pt" \
  --output "$OUT/vitl" --anatomy-weights "$OUT/pesos/anatomy.pt" --workers 6 2>&1 | tee "$OUT/vitl_log.txt"
echo "fin=$(date -Is)"
