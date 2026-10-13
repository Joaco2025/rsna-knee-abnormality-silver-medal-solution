#!/usr/bin/env python
"""¿Predice la salida de un hallazgo la etiqueta de otro mejor que la suya propia?

Con el promedio de los grupos indicados, calcula en los 58 estudios oficiales el AUC de cada
etiqueta real frente a cada una de las 12 salidas, y prueba mezclas de rangos de la salida propia
con la de otro hallazgo (p. ej. sinovitis con derrame). Son 58 casos: márgenes de unos ±0.1.

Uso:  python rsna/evaluar_cruzado.py --labels TABLA.parquet r34=RUN0,... eff=RUN0,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_oof import load_group  # noqa: E402
from modelo import LABELS  # noqa: E402

SHORT = ["ACL", "MCL", "MenM", "MenL", "OA-M", "OA-L", "OA-PF", "Derr", "Sinov", "Baker", "Contu", "Fract"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+")
    ap.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args()
    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    groups = [load_group([Path(d) for d in spec.split("=", 1)[1].split(",")]) for spec in args.grupos]
    uid = groups[0][1].index
    p = np.mean([g[1].loc[uid, int(g[2][0])].to_numpy() for g in groups], axis=0)
    y = table.loc[uid, [f"y_{c}" for c in LABELS]].to_numpy(int)
    r = np.stack([rankdata(p[:, k]) for k in range(12)], axis=1)

    auc = np.array([[roc_auc_score(y[:, j], p[:, k]) for k in range(12)] for j in range(12)])
    print(f"grupos: {' '.join(g.split('=')[0] for g in args.grupos)}; estudios oficiales: {len(uid)}")
    print("\n-- AUC de cada etiqueta real (fila) con cada salida del modelo (columna)")
    print(f"{'':<17}" + "".join(f"{s:>7}" for s in SHORT))
    for j, c in enumerate(LABELS):
        print(f"{c:<17}" + "".join(f"{auc[j, k]:>7.3f}" for k in range(12)))

    print("\n-- Salida propia frente a la mejor salida ajena y a su mezcla de rangos (peso de la ajena)")
    print(f"{'hallazgo':<17}{'propia':>8}{'mejor ajena':>20}{'25 %':>8}{'50 %':>8}{'75 %':>8}")
    for j, c in enumerate(LABELS):
        k = int(np.argmax(np.where(np.arange(12) == j, -1, auc[j])))
        mix = [roc_auc_score(y[:, j], (1 - a) * r[:, j] + a * r[:, k]) for a in (0.25, 0.5, 0.75)]
        print(f"{c:<17}{auc[j, j]:>8.3f}{SHORT[k] + f' {auc[j, k]:.3f}':>20}" + "".join(f"{m:>8.3f}" for m in mix))
    macro = [np.mean([auc[j, j] for j in range(12)])]
    syn, eff = LABELS.index("Synovitis"), LABELS.index("Effusion")
    for a in (0.25, 0.5, 0.75, 1.0):
        s = [roc_auc_score(y[:, j], (1 - a) * r[:, j] + a * r[:, eff]) if j == syn else auc[j, j] for j in range(12)]
        macro.append(np.mean(s))
    print("\nAUC macro: sin cambios {:.4f}; sinovitis mezclada con derrame al 25/50/75/100 %: {:.4f} {:.4f} {:.4f} {:.4f}".format(*macro))


if __name__ == "__main__":
    main()
