#!/usr/bin/env python
"""¿Mejora la etiqueta de un hallazgo si, además de lo que el informe dice de él, se mira lo que
dice de los otros once? Se mide en los 58 estudios oficiales, dejando uno fuera cada vez.

Para cada hallazgo se comparan tres formas de pasar de la extracción (estado y gravedad) a una
probabilidad de etiqueta oficial positiva:
  propia   regresión logística solo con la categoría de ese hallazgo (equivale a la tabla actual)
  todas    regresión logística con las categorías de los 12 hallazgos
  arbol    árbol de decisión de profundidad 3 con las categorías de los 12 hallazgos
Con 58 casos los intervalos son anchos (unos ±0.1 por hallazgo); sirve para ver tendencias.

Uso:  python rsna/evaluar_reglas_etiquetas.py EXTRACCION.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tabla_entrenamiento import DATA, LABELS, PRIOR, category  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extraccion", type=Path)
    args = ap.parse_args()

    ext = pd.read_parquet(args.extraccion).set_index("StudyInstanceUID")
    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str}, usecols=["StudyInstanceUID", *LABELS]).set_index("StudyInstanceUID")
    gold = train.loc[train[LABELS].notna().all(axis=1), LABELS].astype(int)
    cats = pd.DataFrame({c: category(ext[f"{c}_status"], ext[f"{c}_severity"]) for c in LABELS}).loc[gold.index]

    # Por hallazgo: una puntuación ordinal (el valor de partida de la categoría) y si se menciona o no.
    score = cats.apply(lambda s: s.map(PRIOR)).astype(float)
    said = score.notna().astype(float)
    feats = pd.concat([score.fillna(0.3).add_prefix("v_"), said.add_prefix("m_")], axis=1)

    models = {
        "propia": lambda: LogisticRegression(C=1.0, max_iter=1000),
        "todas": lambda: LogisticRegression(C=0.3, max_iter=1000),
        "arbol": lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=0),
    }
    print(f"estudios oficiales: {len(gold)}")
    print(f"{'hallazgo':<17}{'positivos':>10}{'sin mención':>12}" + "".join(f"{n:>9}" for n in models))
    rows = []
    for c in LABELS:
        y = gold[c].to_numpy()
        aucs = []
        for name, make in models.items():
            x = (feats[[f"v_{c}", f"m_{c}"]] if name == "propia" else feats).to_numpy()
            p = np.zeros(len(y))
            for tr, te in LeaveOneOut().split(x):
                p[te] = make().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
            aucs.append(roc_auc_score(y, p))
        rows.append(aucs)
        print(f"{c:<17}{int(y.sum()):>10}{int((said[c] == 0).sum()):>12}" + "".join(f"{a:>9.3f}" for a in aucs))
    print(f"{'MEDIA':<39}" + "".join(f"{a:>9.3f}" for a in np.mean(rows, axis=0)))


if __name__ == "__main__":
    main()
