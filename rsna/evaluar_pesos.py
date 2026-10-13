#!/usr/bin/env python
"""Tres comprobaciones sin reentrenar, sobre las predicciones fuera de muestra de varios grupos:

  épocas    ¿mejora promediar las predicciones de las tres últimas épocas frente a usar la última?
  pesos     ¿mejora un peso por grupo (no negativo, suma 1) frente a la media simple?
  hallazgo  ¿mejora un juego de pesos distinto para cada hallazgo?

Los pesos se ajustan minimizando la entropía cruzada contra los objetivos (con su peso) en cuatro
quintos de los estudios y se miden en el quinto restante, de modo que la columna interna no está
medida sobre los mismos estudios con que se ajustó. La columna oficial ajusta con todos los
estudios fuera de muestra y mide en los 58 oficiales.

Uso:  python rsna/evaluar_pesos.py --labels TABLA.parquet r34=RUN0,... eff=RUN0,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar import macro_auc  # noqa: E402
from evaluar_oof import load_group  # noqa: E402
from modelo import LABELS  # noqa: E402


def fit_simplex(p: np.ndarray, t: np.ndarray, w: np.ndarray, shrink: float) -> np.ndarray:
    """Pesos en el símplex para mezclar p (grupos, casos) hacia t con pesos w; `shrink` tira hacia la media."""
    g = len(p)

    def loss(z):
        a = np.exp(z - z.max()); a /= a.sum()
        q = np.clip(a @ p, 1e-6, 1 - 1e-6)
        ce = -(w * (t * np.log(q) + (1 - t) * np.log(1 - q))).sum() / w.sum()
        return ce + shrink * ((a - 1 / g) ** 2).sum()

    z = minimize(loss, np.zeros(g), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 4000}).x
    a = np.exp(z - z.max())
    return a / a.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--encoger", type=float, default=0.05, help="fuerza con que los pesos se acercan a la media simple")
    args = ap.parse_args()

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    names = [g.split("=", 1)[0] for g in args.grupos]
    groups = [load_group([Path(d) for d in g.split("=", 1)[1].split(",")]) for g in args.grupos]
    uid_int, uid_off = groups[0][0].index, groups[0][1].index
    last = [int(g[2][0]) for g in groups]
    p_int = np.stack([g[0].loc[uid_int, e].to_numpy() for g, e in zip(groups, last)])        # (grupos, estudios, 12)
    p_off = np.stack([g[1].loc[uid_off, e].to_numpy() for g, e in zip(groups, last)])
    p3_int = np.stack([np.mean([g[0].loc[uid_int, k].to_numpy() for k in range(e - 2, e + 1)], axis=0) for g, e in zip(groups, last)])
    p3_off = np.stack([np.mean([g[1].loc[uid_off, k].to_numpy() for k in range(e - 2, e + 1)], axis=0) for g, e in zip(groups, last)])

    t = np.nan_to_num(table.loc[uid_int, [f"t_{c}" for c in LABELS]].to_numpy(np.float64))
    w = table.loc[uid_int, [f"w_{c}" for c in LABELS]].to_numpy(np.float64)
    y_int = (t >= 0.5).astype(int)
    y_off = table.loc[uid_off, [f"y_{c}" for c in LABELS]].to_numpy(int)
    fold = np.array([int(u[-6:].replace(".", "0")) % 5 for u in uid_int])     # reparto fijo cualquiera para medir los pesos

    rows = {}
    rows["media simple, última época"] = (p_int.mean(0), p_off.mean(0))
    rows["media simple, 3 últimas épocas"] = (p3_int.mean(0), p3_off.mean(0))

    # Un peso por grupo.
    cv = np.zeros_like(p_int[0])
    for k in range(5):
        tr, te = fold != k, fold == k
        a = fit_simplex(p_int[:, tr].reshape(len(groups), -1), t[tr].ravel(), w[tr].ravel(), args.encoger)
        cv[te] = np.tensordot(a, p_int[:, te], axes=1)
    a_all = fit_simplex(p_int.reshape(len(groups), -1), t.ravel(), w.ravel(), args.encoger)
    rows["un peso por grupo"] = (cv, np.tensordot(a_all, p_off, axes=1))

    # Un juego de pesos por hallazgo.
    cv, off, per_label = np.zeros_like(p_int[0]), np.zeros_like(p_off[0]), []
    for j in range(len(LABELS)):
        for k in range(5):
            tr, te = fold != k, fold == k
            cv[te, j] = fit_simplex(p_int[:, tr, j], t[tr, j], w[tr, j], args.encoger) @ p_int[:, te, j]
        a = fit_simplex(p_int[:, :, j], t[:, j], w[:, j], args.encoger)
        per_label.append(a)
        off[:, j] = a @ p_off[:, :, j]
    rows["pesos por hallazgo"] = (cv, off)

    print(f"objetivos: {args.labels.name}; grupos: {' '.join(names)}; estudios: {len(uid_int)}; oficiales: {len(uid_off)}")
    print(f"\n{'método':<34}{'interna':>9}{'oficial':>9}")
    for name, (pi, po) in rows.items():
        print(f"{name:<34}{macro_auc(y_int, pi, w)[0]:>9.4f}{macro_auc(y_off, po)[0]:>9.4f}")
    print("\n-- Peso por grupo ajustado con todos los estudios")
    print("".join(f"{n:>10}" for n in names)); print("".join(f"{v:>10.2f}" for v in a_all))
    print("\n-- Pesos por hallazgo")
    print(f"{'hallazgo':<17}" + "".join(f"{n:>10}" for n in names))
    for c, a in zip(LABELS, per_label):
        print(f"{c:<17}" + "".join(f"{v:>10.2f}" for v in a))


if __name__ == "__main__":
    main()
