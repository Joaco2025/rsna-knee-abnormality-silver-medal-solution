#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-armar-envio
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Arma el paquete que se sube a Kaggle como dataset: código de inferencia, pesos y librerías.
# Lanzar desde code/:  sbatch jobs/armar_envio.slurm NOMBRE grupo=JOB,JOB,... [grupo=JOB,...]
#   toma runs/<JOB>/last.pt de cada corrida (última época, sin elegir con los 58 oficiales).
# Deja envios/NOMBRE/ (rsna/, pesos/, wheels/, inferencia.ipynb, MANIFIESTO.txt) y envios/NOMBRE_paquete.zip.
# No sobrescribe: si envios/NOMBRE ya existe, se detiene.
set -euo pipefail
NAME=${1:?indica el nombre del envío}
shift
[[ $# -ge 1 ]] || { echo "indica al menos un grupo=JOB,JOB,..." >&2; exit 1; }

PROJ=/lustre/proyectos/p082
DEST="$PROJ/envios/$NAME"
[[ ! -e "$DEST" && ! -e "${DEST}_paquete.zip" ]] || { echo "ya existe $DEST; usa otro nombre" >&2; exit 1; }
cd "$PROJ/code"
mkdir -p "$DEST/rsna" "$DEST/pesos" "$DEST/wheels"

echo "job=$SLURM_JOB_ID inicio=$(date -Is) envío=$NAME commit=$(git rev-parse --short HEAD)"
git rev-parse --short HEAD > "$DEST/commit.txt"
cp rsna/inferir.py rsna/modelo.py rsna/preproc.py "$DEST/rsna/"
cp kaggle/inferencia.ipynb "$DEST/"
cp "$PROJ"/envios/v2_r34_cnx/wheels/*.whl "$DEST/wheels/"

{ echo "envío $NAME, armado el $(date -Is) con el commit $(git rev-parse --short HEAD)"; echo "promedio simple de las probabilidades de todos los modelos"; } > "$DEST/MANIFIESTO.txt"
for g in "$@"; do
    group=${g%%=*}
    for job in ${g#*=}; do :; done
    IFS=, read -ra jobs <<<"${g#*=}"
    for job in "${jobs[@]}"; do
        src="$PROJ/runs/$job/last.pt"
        [[ -f "$src" ]] || { echo "falta $src" >&2; exit 1; }
        cp "$src" "$DEST/pesos/${group}_job${job}.pt"
        echo "$group job $job  sha256=$(sha256sum "$src" | cut -c1-16)  $(grep -o 'partición=[0-9]/[0-9]*' logs/*_"$job".out | head -n 1 | sed 's/.*://')" >> "$DEST/MANIFIESTO.txt"
    done
done
cat "$DEST/MANIFIESTO.txt"

cd "$PROJ/envios"
"$PROJ/venvs/rsna/bin/python" -m zipfile -c "${NAME}_paquete.zip" "$NAME/rsna" "$NAME/pesos" "$NAME/wheels"
ls -la "${NAME}_paquete.zip"; du -sh "$DEST"
echo "fin=$(date -Is)"
