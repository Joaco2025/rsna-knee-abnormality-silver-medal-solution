# Celda final para un notebook de Kaggle que corre varias ramas (p. ej. rsna-versia-7, las ramas
# DINOv2/DINOv3/RadImageNet de bend-the-knee-to-the-dinosaurs y nuestra rama de Yuca) y mezcla sus
# predicciones por rangos.
#
# Cada rama escribe su submission.csv; justo después de cada una hay que renombrarlo para que la
# siguiente no lo pise, p. ej.:
#     shutil.move("/kaggle/working/submission.csv", "/kaggle/working/rama_versia7.csv")
# Nuestra rama (kaggle/mezcla_con_publicos.py, celda 1) ya escribe /kaggle/working/yuca_sub.csv.

import glob, os
import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Peso de cada rama. Una rama cuyo archivo no exista se ignora (y los demás pesos se reescalan).
RAMAS = {
    "/kaggle/working/rama_versia7.csv": 0.55,   # R7 ConvNeXt-Small + Raptor CoAtNet (0.956 público)
    "/kaggle/working/rama_dinos.csv": 0.30,     # DINOv2 + DINOv3 + RadImageNet + CoAtNet ResGated (0.941-0.949)
    "/kaggle/working/yuca_sub.csv": 0.15,       # nuestra rama
}
# Ajuste por hallazgo para nuestra rama: multiplica su peso. Cero donde flojea frente a las públicas.
AJUSTE_YUCA = {"ACL": 1.2, "Baker's": 1.2, "Medial OA": 1.1, "MCL": 1.0, "Medial Meniscus": 1.0, "Effusion": 1.0,
               "Fracture": 0.7, "Contusion": 0.5, "PF OA": 0.3, "Lateral Meniscus": 0.2, "Lateral OA": 0.0, "Synovitis": 0.0}

hits = [d for n in range(1, 6) for d in glob.glob("/kaggle/input/" + "*/" * n + "test.csv")]
DATA = os.path.dirname(hits[0])
ids = pd.read_csv(f"{DATA}/test.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID

ranks, weights = [], []
for path, peso in RAMAS.items():
    if not os.path.exists(path):
        print(f"(sin {path}: rama omitida)")
        continue
    sub = pd.read_csv(path, dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID")
    sub = sub.set_index("StudyInstanceUID").reindex(ids)
    # Estudios que la rama no pudo leer (0.5 en todo) o que le faltan: esa rama no vota en ellos.
    falta = sub[LABELS].isna().any(axis=1) | (sub[LABELS] == 0.5).all(axis=1)
    r = sub[LABELS].rank(method="average", pct=True)
    w = pd.DataFrame(peso, index=sub.index, columns=LABELS)
    if path.endswith("yuca_sub.csv"):
        w = w * pd.Series(AJUSTE_YUCA)
    w[falta.to_numpy()] = 0.0
    print(f"{os.path.basename(path)}: peso {peso}, estudios sin predicción {int(falta.sum())}")
    ranks.append(r.fillna(0.0)); weights.append(w)

assert ranks, "ninguna rama dejó su archivo"
num = sum(r * w for r, w in zip(ranks, weights))
den = sum(weights)
den[den == 0] = 1.0                       # si nadie votó (no debería pasar), queda 0 y no rompe
final = (num / den).rank(method="average", pct=True).reset_index()

ref = pd.read_csv(f"{DATA}/sample_submission.csv")
final = final[list(ref.columns)]
assert len(final) == len(ids) and np.isfinite(final[LABELS].to_numpy()).all()
final.to_csv("/kaggle/working/submission.csv", index=False)
print("mezcla escrita:", final.shape)
