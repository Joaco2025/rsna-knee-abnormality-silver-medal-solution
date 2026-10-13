#!/usr/bin/env python
"""Mide si una segunda etapa sobre las salidas de los modelos mejora al promedio simple.

Entrada: las predicciones fuera de muestra de cada grupo (como en evaluar_oof.py). Para cada
hallazgo se entrena un modelo pequeño que recibe las 12 salidas de todos los grupos, de modo que
puede aprovechar las co-ocurrencias (p. ej. derrame y sinovitis). Se comparan:
  promedio      media de los grupos (lo que se envía hoy)
  rl_propia     regresión logística solo con las salidas de ese mismo hallazgo (pesos por modelo)
  rl_todas      regresión logística con las salidas de los 12 hallazgos
  gbm           gradient boosting por histogramas con las 12 salidas
  bosque        bosque aleatorio con las 12 salidas
La columna interna es validación cruzada de la segunda etapa sobre los estudios fuera de muestra
(objetivos derivados); la oficial entrena con todos ellos y mide en los 58 oficiales.

Uso:  python rsna/evaluar_apilado.py --labels TABLA.parquet --out DIR r34=RUN0,... cnx=RUN0,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_oof import load_group  # noqa: E402
from modelo import LABELS  # noqa: E402

MODELS = {
    "rl_propia": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    "rl_todas": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    "gbm": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0),
    "bosque": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=20, n_jobs=-1, random_state=0),
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    groups = {}
    for spec in args.grupos:
        name, dirs = spec.split("=", 1)
        groups[name] = load_group([Path(d) for d in dirs.split(",")])
    epoch = int(min(g[2][0] for g in groups.values()))
    uid_int = next(iter(groups.values()))[0].index
    uid_off = next(iter(groups.values()))[1].index
    p_int = np.stack([g[0].loc[uid_int, epoch].to_numpy() for g in groups.values()])   # (grupos, estudios, 12)
    p_off = np.stack([g[1].loc[uid_off, epoch].to_numpy() for g in groups.values()])
    x_int, x_off = logit(p_int).transpose(1, 0, 2), logit(p_off).transpose(1, 0, 2)     # (estudios, grupos, 12)

    t = table.loc[uid_int, [f"t_{c}" for c in LABELS]].to_numpy(np.float32)
    w = table.loc[uid_int, [f"w_{c}" for c in LABELS]].to_numpy(np.float32)
    y_int = (np.nan_to_num(t) >= 0.5).astype(int)
    y_off = table.loc[uid_off, [f"y_{c}" for c in LABELS]].to_numpy(int)

    names = ["promedio", *MODELS]
    res = {(kind, n): [] for kind in ("interna", "oficial") for n in names}
    for j, c in enumerate(LABELS):
        m = w[:, j] > 0
        y = y_int[m, j]
        res["interna", "promedio"].append(roc_auc_score(y, p_int[:, m, j].mean(axis=0)))
        res["oficial", "promedio"].append(roc_auc_score(y_off[:, j], p_off[:, :, j].mean(axis=0)))
        for n, make in MODELS.items():
            cols = (lambda x: x[:, :, j]) if n == "rl_propia" else (lambda x: x.reshape(len(x), -1))
            xa, xo = cols(x_int)[m], cols(x_off)
            cv = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(xa, y):
                cv[te] = make().fit(xa[tr], y[tr], **({} if n.startswith("rl") else {"sample_weight": w[m, j][tr]})).predict_proba(xa[te])[:, 1]
            res["interna", n].append(roc_auc_score(y, cv))
            res["oficial", n].append(roc_auc_score(y_off[:, j], make().fit(xa, y).predict_proba(xo)[:, 1]))

    lines = [f"grupos: {', '.join(groups)}; época {epoch}; estudios fuera de muestra: {len(uid_int)}; oficiales: {len(uid_off)}"]
    for kind in ("interna", "oficial"):
        lines += ["", f"-- AUC {kind} por hallazgo", f"{'hallazgo':<17}" + "".join(f"{n:>11}" for n in names)]
        for j, c in enumerate(LABELS):
            lines.append(f"{c:<17}" + "".join(f"{res[kind, n][j]:>11.4f}" for n in names))
        lines.append(f"{'MEDIA':<17}" + "".join(f"{np.mean(res[kind, n]):>11.4f}" for n in names))
    print("\n".join(lines), flush=True)
    (args.out / "apilado.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
