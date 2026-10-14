#!/usr/bin/env python
"""¿Ayuda una medida directa del líquido a predecir derrame y sinovitis?

En las series sensibles a líquido, el líquido articular es lo más brillante de la imagen. Como cada
serie se normalizó por sus percentiles (0.5 y 99.5 -> 0 y 255), contar los píxeles muy brillantes es
una medida aproximada del volumen de líquido (lo mismo que hace R7 con umbral 229). Para cada estudio
se mide la fracción de vóxeles >= UMBRAL en cada ranura sensible a líquido, en el volumen entero y en
la mitad central del volumen. Luego se mide:
  - el AUC de cada medida sola contra derrame y sinovitis (fuera de muestra y en los 58 oficiales);
  - si una regresión logística con [salida del ensamble, log de la medida] mejora a la salida sola
    (validación cruzada de 5 partes sobre los estudios fuera de muestra; los oficiales, con el ajuste completo).

Uso:  python rsna/evaluar_liquido.py --data DIR_NPY --labels TABLA.parquet grupo=RUN0,... [más grupos]
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_oof import load_group  # noqa: E402
from modelo import LABELS  # noqa: E402

UMBRAL = 229
SLOTS = {0: "sag_fs", 2: "cor_fs", 3: "ax_fs"}
DATA = Path(".")


def measure(uid: str) -> list[float]:
    arr = np.load(DATA / f"{uid}.npy", mmap_mode="r")
    out = []
    for k in SLOTS:
        v = np.asarray(arr[k])
        if not v.any():
            out += [np.nan, np.nan]
            continue
        d, h, w = v.shape
        c = v[d // 4: 3 * d // 4, h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        out += [float((v >= UMBRAL).mean()), float((c >= UMBRAL).mean())]
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    DATA = args.data

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    groups = [load_group([Path(d) for d in g.split("=", 1)[1].split(",")]) for g in args.grupos]
    uid_int, uid_off = groups[0][0].index, groups[0][1].index
    p_int = np.mean([g[0].loc[uid_int, int(g[2][0])].to_numpy() for g in groups], axis=0)
    p_off = np.mean([g[1].loc[uid_off, int(g[2][0])].to_numpy() for g in groups], axis=0)
    uids = list(uid_int) + list(uid_off)
    with Pool(args.workers) as pool:
        feats = np.array(pool.map(measure, uids, chunksize=16))
    names = [f"{n}_{r}" for n in SLOTS.values() for r in ("todo", "centro")]
    f_int, f_off = feats[: len(uid_int)], feats[len(uid_int):]

    t = np.nan_to_num(table.loc[uid_int, [f"t_{c}" for c in LABELS]].to_numpy())
    w = table.loc[uid_int, [f"w_{c}" for c in LABELS]].to_numpy() > 0
    y_int = (t >= 0.5).astype(int)
    y_off = table.loc[uid_off, [f"y_{c}" for c in LABELS]].to_numpy(int)

    def auc(y, s):
        m = ~np.isnan(s)
        return roc_auc_score(y[m], s[m]) if 0 < y[m].sum() < m.sum() else float("nan")

    def design(p, f):
        f = np.where(np.isnan(f), np.nanmedian(f_int[:, : f.shape[1]], axis=0), f)
        pc = np.clip(p, 1e-5, 1 - 1e-5)
        return np.column_stack([np.log(pc / (1 - pc))[:, None], np.log(f + 1e-4)])

    print(f"estudios fuera de muestra: {len(uid_int)}; oficiales: {len(uid_off)}; umbral {UMBRAL}")
    for c in ("Effusion", "Synovitis", "Baker's"):
        j = LABELS.index(c)
        m = w[:, j]
        print(f"\n-- {c}: AUC de cada medida sola (interna / oficial)")
        print(f"   salida del ensamble: {auc(y_int[m, j], p_int[m, j]):.4f} / {auc(y_off[:, j], p_off[:, j]):.4f}")
        for k, n in enumerate(names):
            print(f"   {n:<14}{auc(y_int[m, j], f_int[m, k]):.4f} / {auc(y_off[:, j], f_off[:, k]):.4f}")
        for cols, label in ((list(range(len(names))), "todas las medidas"), ([0, 1], "solo sagital")):
            xi, xo = design(p_int[:, j], f_int[:, cols]), design(p_off[:, j], f_off[:, cols])
            yi = y_int[m, j]
            cv = np.zeros(m.sum())
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(xi[m], yi):
                cv[te] = LogisticRegression(C=1.0, max_iter=2000).fit(xi[m][tr], yi[tr]).predict_proba(xi[m][te])[:, 1]
            full = LogisticRegression(C=1.0, max_iter=2000).fit(xi[m], yi)
            print(f"   ensamble + {label}: {auc(yi, cv):.4f} / {auc(y_off[:, j], full.predict_proba(xo)[:, 1]):.4f}")


if __name__ == "__main__":
    main()
