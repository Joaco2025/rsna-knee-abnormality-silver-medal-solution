#!/bin/bash
#SBATCH -A p082
#SBATCH -J rsna-prueba-entorno
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Comprueba que el entorno del proyecto ve la GPU MI210 y que entrena e infiere en ella.
# Lanzar desde code/:  sbatch jobs/prueba_entorno.slurm
set -euo pipefail

PROJ=/lustre/proyectos/p082
OUT="$PROJ/runs/${SLURM_JOB_ID:?}"
mkdir -p "$OUT"
cd "$PROJ/code"

echo "job=$SLURM_JOB_ID nodo=$(hostname) inicio=$(date -Is)"
echo "commit=$(git rev-parse --short HEAD)" | tee "$OUT/commit.txt"

# --- Entorno -----------------------------------------------------------------
set +u
type module >/dev/null 2>&1 || source /usr/share/Modules/init/bash
module load rocm/6.4.3
set -u
PY="$PROJ/venvs/rsna/bin/python"

# --- Trabajo -----------------------------------------------------------------
"$PY" - <<'EOF'
import time
import torch, torchvision, numpy, pandas, pydicom, sklearn, timm, cv2, PIL

print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| numpy", numpy.__version__,
      "| pandas", pandas.__version__, "| pydicom", pydicom.__version__, "| timm", timm.__version__)
print("GPU disponible:", torch.cuda.is_available(), "| dispositivos:", torch.cuda.device_count())
assert torch.cuda.is_available(), "PyTorch no ve ninguna GPU"
dev = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0), "| memoria total GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

# Entrenamiento mínimo: una red convolucional estándar, con y sin precisión mixta.
model = torchvision.models.resnet18(num_classes=12).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(32, 3, 256, 256, device=dev)
y = torch.rand(32, 12, device=dev)
for amp in (False, True):
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"precisión mixta={amp}: {20 * 32 / (time.time() - t0):.0f} imágenes/s, pérdida {loss.item():.4f}")
assert torch.isfinite(loss), "la pérdida no es finita"

# Convolución 3D, por si el modelo final trabaja con volúmenes.
c3 = torch.nn.Conv3d(1, 8, 3, padding=1).to(dev)
print("conv3d:", tuple(c3(torch.randn(2, 1, 32, 128, 128, device=dev)).shape))

# Lectura de DICOM comprimido: los decodificadores deben estar instalados.
from pydicom.pixels import get_decoder
for name in ("JPEGLosslessSV1", "JPEG2000Lossless", "JPEGBaseline8Bit"):
    try:
        dec = get_decoder(getattr(pydicom.uid, name))
        print(f"decodificador {name}: disponible={dec.is_available}")
    except Exception as exc:
        print(f"decodificador {name}: no se pudo consultar ({type(exc).__name__})")
print("memoria de GPU usada GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))
print("ENTORNO OK")
EOF

echo "fin=$(date -Is)"
