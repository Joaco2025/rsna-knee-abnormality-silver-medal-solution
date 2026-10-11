#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-registrar-gguf
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Verifica el SHA-256 de un archivo GGUF subido a mano y, solo si coincide, lo registra en el
# almacén de Ollama del proyecto. No usa red. Ollama vive solo dentro del job, en 127.0.0.1.
# Lanzar desde code/:  sbatch jobs/registrar_gguf_ollama.slurm [GGUF] [SHA256] [NOMBRE]
set -euo pipefail

PROJ=/lustre/proyectos/p082
GGUF=${1:-$PROJ/modelos/gguf/medgemma-27b-text-it-Q8_0.gguf}
SHA=${2:-ef02e1b21e9833ef7cdab1cce0ac7b8280b825a9fefb0a7f42f7b03e9e46133a}   # publicado por Hugging Face
NAME=${3:-p082/medgemma-27b-text:q8_0}

# --- Rutas -------------------------------------------------------------------
export OLLAMA_MODELS="$PROJ/modelos/ollama"      # almacén de modelos del proyecto (persistente)
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT" "$OLLAMA_MODELS"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is) archivo=$GGUF nombre=$NAME"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# --- Integridad --------------------------------------------------------------
[[ -r "$GGUF" ]] || { echo "No existe $GGUF" >&2; exit 1; }
echo "tamaño=$(stat -c %s "$GGUF") bytes"
CALC=$(sha256sum "$GGUF" | cut -d' ' -f1)
echo "sha256 calculado: $CALC"
echo "sha256 esperado:  $SHA"
if [[ "$CALC" != "$SHA" ]]; then
    echo "RESULTADO: el hash NO coincide; no se registra el modelo." | tee /dev/stderr
    exit 1
fi
echo "RESULTADO: el hash coincide."

# --- Entorno -----------------------------------------------------------------
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load ollama/0.34.0
set -u
export OLLAMA_HOST="127.0.0.1:$((20000 + SLURM_JOB_ID % 20000))"   # puerto propio del job, solo local
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

# --- Servidor (vive solo mientras dure el job) ---------------------------------
ollama serve >"$OUT/ollama_serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 60); do
    ollama list >/dev/null 2>&1 && break
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "ollama serve terminó antes de responder; ver $OUT/ollama_serve.log" >&2; exit 1; }
    sleep 2
done

# --- Registro ----------------------------------------------------------------
# Plantilla de conversación de Gemma 3 (el mensaje de sistema va dentro del turno de usuario).
cat >"$OUT/Modelfile" <<EOF
FROM $GGUF
TEMPLATE """{{- range \$i, \$_ := .Messages }}
{{- \$last := eq (len (slice \$.Messages \$i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if \$last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not \$last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""
PARAMETER stop <end_of_turn>
EOF
ollama create "$NAME" -f "$OUT/Modelfile" 2>&1 | tr '\r' '\n' | awk 'NR % 50 == 1 || /success|error/'
ollama list
ollama show "$NAME"
du -sh "$OLLAMA_MODELS"

echo "fin=$(date -Is)"
