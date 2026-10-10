#!/usr/bin/env python
"""Compara etiquetas extraídas de los informes con las oficiales de train.csv.

Entrada: un CSV con StudyInstanceUID y una columna por hallazgo (nombres de train.csv) cuyos
valores son `p` (el informe lo afirma), `a` (lo niega) o `n` (no lo menciona). Solo `p` cuenta
como positivo. Sirve para el piloto leído a mano y para la salida de un modelo.

Uso:  python etiquetas/comparar_piloto.py etiquetas/piloto_58/prediccion_lector.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prediccion", type=Path)
    args = ap.parse_args()

    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    labels = [c for c in train.columns if c != "Report"]
    gold = train.loc[train[labels].notna().all(axis=1), labels].astype(int)
    pred = pd.read_csv(args.prediccion, dtype=str).set_index("StudyInstanceUID")[labels]
    common = gold.index.intersection(pred.index)
    gold, pred = gold.loc[common], pred.loc[common]
    P = (pred == "p").astype(int)
    n = len(common)

    print(f"{n} estudios etiquetados en común")
    print(f"{'etiqueta':<17}{'pos.of':>7}{'pos.inf':>8}{'VP':>4}{'FP':>4}{'FN':>4}{'VN':>4}{'sens':>7}{'prec':>7}{'acuerdo':>9}"
          "   FN: informe lo niega / no lo menciona")
    total = [0, 0, 0, 0]
    for col in labels:
        tp = int(((P[col] == 1) & (gold[col] == 1)).sum())
        fp = int(((P[col] == 1) & (gold[col] == 0)).sum())
        fn = int(((P[col] == 0) & (gold[col] == 1)).sum())
        tn = int(((P[col] == 0) & (gold[col] == 0)).sum())
        total = [t + v for t, v in zip(total, (tp, fp, fn, tn))]
        fn_a = int(((pred[col] == "a") & (gold[col] == 1)).sum())
        fn_n = int(((pred[col] == "n") & (gold[col] == 1)).sum())
        sens = tp / (tp + fn) if tp + fn else float("nan")
        prec = tp / (tp + fp) if tp + fp else float("nan")
        print(f"{col:<17}{tp + fn:>7}{tp + fp:>8}{tp:>4}{fp:>4}{fn:>4}{tn:>4}{sens:>7.2f}{prec:>7.2f}{(tp + tn) / n:>9.2f}   {fn_a} / {fn_n}")
    tp, fp, fn, tn = total
    print(f"{'TOTAL':<17}{tp + fn:>7}{tp + fp:>8}{tp:>4}{fp:>4}{fn:>4}{tn:>4}{tp / (tp + fn):>7.2f}{tp / (tp + fp):>7.2f}{(tp + tn) / (n * len(labels)):>9.2f}")
    print(f"\nEstudios con las {len(labels)} etiquetas iguales: {int((P.values == gold.values).all(axis=1).sum())} de {n}")


if __name__ == "__main__":
    main()
