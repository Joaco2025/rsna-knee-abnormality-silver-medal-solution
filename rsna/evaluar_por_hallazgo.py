#!/usr/bin/env python
"""Desglose por hallazgo de un ensamble y dos preguntas que se responden sin reentrenar:

  grupos   ¿conviene que cada hallazgo promedie solo un subconjunto de los grupos (incluir o excluir)?
  época    ¿algún hallazgo rinde mejor en una época intermedia que en la última?

Los primeros grupos indicados (--base N) forman el ensamble de referencia; todos son candidatos
para la selección. Para no premiar la casualidad, la elección de subconjunto o de época se hace
con cuatro quintos de los estudios fuera de muestra y se mide en el quinto restante; la columna
oficial elige con todos ellos y mide en los 58 oficiales.

Uso:  python rsna/evaluar_por_hallazgo.py --labels TABLA.parquet --base 4 eff=... cnxhib=... effhib=... r34att=... [más grupos]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_oof import load_group  # noqa: E402
from modelo import LABELS  # noqa: E402


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--base", type=int, default=0, help="cuántos de los primeros grupos forman el ensamble de referencia; 0 = todos")
    args = ap.parse_args()

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    names = [g.split("=", 1)[0] for g in args.grupos]
    groups = [load_group([Path(d) for d in g.split("=", 1)[1].split(",")]) for g in args.grupos]
    nb = args.base or len(groups)
    uid_int, uid_off = groups[0][0].index, groups[0][1].index
    last = [int(g[2][0]) for g in groups]
    n_ep = min(last)
    # (grupos, épocas contadas desde el final, estudios, 12): índice 0 = última época
    pi = np.stack([[g[0].loc[uid_int, e - k].to_numpy() for k in range(n_ep)] for g, e in zip(groups, last)])
    po = np.stack([[g[1].loc[uid_off, e - k].to_numpy() for k in range(n_ep)] for g, e in zip(groups, last)])

    t = np.nan_to_num(table.loc[uid_int, [f"t_{c}" for c in LABELS]].to_numpy(np.float64))
    w = table.loc[uid_int, [f"w_{c}" for c in LABELS]].to_numpy(np.float64) > 0
    y_int = (t >= 0.5).astype(int)
    y_off = table.loc[uid_off, [f"y_{c}" for c in LABELS]].to_numpy(int)
    fold = np.array([int(u[-6:].replace(".", "0")) % 5 for u in uid_int])
    J = range(len(LABELS))

    print(f"objetivos: {args.labels.name}; referencia: {' + '.join(names[:nb])}; candidatos: {' '.join(names)}")

    # 1. Desglose del ensamble de referencia.
    bi, bo = pi[:nb, 0].mean(0), po[:nb, 0].mean(0)
    a_int = [auc(y_int[w[:, j], j], bi[w[:, j], j]) for j in J]
    a_off = [auc(y_off[:, j], bo[:, j]) for j in J]
    print("\n-- 1. Ensamble de referencia por hallazgo (ordenado de peor a mejor en oficiales)")
    print(f"{'hallazgo':<17}{'positivos':>10}{'con objetivo':>13}{'interna':>9}{'oficial':>9}{'pos. oficiales':>15}")
    for j in np.argsort(a_off):
        print(f"{LABELS[j]:<17}{int(y_int[w[:, j], j].sum()):>10}{int(w[:, j].sum()):>13}{a_int[j]:>9.4f}{a_off[j]:>9.4f}{int(y_off[:, j].sum()):>15}")
    print(f"{'MEDIA':<40}{np.mean(a_int):>9.4f}{np.mean(a_off):>9.4f}")

    # 2. Subconjunto de grupos por hallazgo.
    subsets = [c for k in range(1, len(groups) + 1) for c in combinations(range(len(groups)), k)]
    si = np.stack([pi[list(c), 0].mean(0) for c in subsets])       # (subconjuntos, estudios, 12)
    so = np.stack([po[list(c), 0].mean(0) for c in subsets])
    print("\n-- 2. Elegir subconjunto de grupos por hallazgo")
    print(f"{'hallazgo':<17}{'ref. int':>9}{'elegido int':>12}{'ref. of':>9}{'elegido of':>11}  subconjunto elegido con todos los estudios")
    sel_int, sel_off = [], []
    for j in J:
        m = w[:, j]
        cv = np.zeros(m.sum())
        f, yy = fold[m], y_int[m, j]
        for k in range(5):
            tr, te = f != k, f == k
            best = int(np.argmax([auc(yy[tr], s[m, j][tr]) for s in si]))
            cv[te] = si[best][m, j][te]
        best = int(np.argmax([auc(yy, s[m, j]) for s in si]))
        sel_int.append(auc(yy, cv)); sel_off.append(auc(y_off[:, j], so[best][:, j]))
        print(f"{LABELS[j]:<17}{a_int[j]:>9.4f}{sel_int[-1]:>12.4f}{a_off[j]:>9.4f}{sel_off[-1]:>11.4f}  {' + '.join(names[i] for i in subsets[best])}")
    print(f"{'MEDIA':<17}{np.mean(a_int):>9.4f}{np.mean(sel_int):>12.4f}{np.mean(a_off):>9.4f}{np.mean(sel_off):>11.4f}")

    # 3. Época por hallazgo (ensamble de referencia, todas las redes en la misma distancia al final).
    ei, eo = pi[:nb].mean(0), po[:nb].mean(0)                       # (épocas desde el final, estudios, 12)
    print("\n-- 3. AUC por hallazgo según la época (0 = última, 1 = penúltima, ...): interna / oficial")
    show = min(5, n_ep)
    print(f"{'hallazgo':<17}" + "".join(f"{'-' + str(k) if k else 'última':>16}" for k in range(show)) + f"{'elegida int':>13}{'elegida of':>12}")
    ep_int, ep_off = [], []
    for j in J:
        m = w[:, j]
        f, yy = fold[m], y_int[m, j]
        per = [(auc(yy, ei[k][m, j]), auc(y_off[:, j], eo[k][:, j])) for k in range(n_ep)]
        cv = np.zeros(m.sum())
        for k in range(5):
            tr, te = f != k, f == k
            best = int(np.argmax([auc(yy[tr], ei[e][m, j][tr]) for e in range(n_ep)]))
            cv[te] = ei[best][m, j][te]
        best = int(np.argmax([p[0] for p in per]))
        ep_int.append(auc(yy, cv)); ep_off.append(per[best][1])
        print(f"{LABELS[j]:<17}" + "".join(f"{p[0]:>9.4f}/{p[1]:.4f}" for p in per[:show]) + f"{ep_int[-1]:>13.4f}{ep_off[-1]:>12.4f}")
    macro = [(np.mean([auc(y_int[w[:, j], j], ei[k][w[:, j], j]) for j in J]), np.mean([auc(y_off[:, j], eo[k][:, j]) for j in J])) for k in range(show)]
    print(f"{'MEDIA':<17}" + "".join(f"{a:>9.4f}/{b:.4f}" for a, b in macro) + f"{np.mean(ep_int):>13.4f}{np.mean(ep_off):>12.4f}")


if __name__ == "__main__":
    main()
