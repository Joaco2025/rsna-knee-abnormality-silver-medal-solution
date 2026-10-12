#!/usr/bin/env python
"""Convierte la extracción de los informes (estado + gravedad) en objetivos de entrenamiento.

Como la métrica de la competencia es AUC, no hace falta decidir un corte por hallazgo: cada celda
recibe un valor entre 0 y 1 que expresa cuán probable es que la etiqueta oficial sea positiva dado
lo que dice el informe. Ese valor se estima en los 58 estudios con etiqueta oficial, por hallazgo y
por categoría (estado/gravedad), y se suaviza hacia la tasa conjunta de todos los hallazgos porque
hay pocos casos. "No mencionado" queda sin objetivo (peso 0).

Los 58 estudios oficiales conservan además su etiqueta real y se reservan para validar.

Uso:  python rsna/tabla_entrenamiento.py EXTRACCION.parquet --out etiquetas_v1.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Valor de partida por categoría, usado solo para suavizar donde no hay casos oficiales.
PRIOR = {"present/severe": 0.90, "present/moderate": 0.80, "present/unspecified": 0.70, "present/mild": 0.45,
         "present/na": 0.70, "possible": 0.40, "absent": 0.05}
K_POOL, K_PRIOR = 4.0, 4.0     # fuerza del suavizado, en "casos equivalentes"
WEIGHT = {"present": 1.0, "absent": 1.0, "possible": 0.5, "not_mentioned": 0.0}


def category(status: pd.Series, severity: pd.Series) -> pd.Series:
    cat = status.where(status != "present", "present/" + severity)
    return cat.where(~cat.str.startswith("present/") | cat.isin(PRIOR), "present/unspecified")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("extraccion", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ext = pd.read_parquet(args.extraccion).set_index("StudyInstanceUID")
    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = train.loc[train[LABELS].notna().all(axis=1), LABELS].astype(int)
    assert set(gold.index) <= set(ext.index), "faltan estudios oficiales en la extracción"

    cats = pd.DataFrame({c: category(ext[f"{c}_status"], ext[f"{c}_severity"]) for c in LABELS})
    g_cats = cats.loc[gold.index]

    # Tasa conjunta por categoría (todos los hallazgos), suavizada hacia el valor de partida.
    pooled = {}
    for cat, prior in PRIOR.items():
        m = (g_cats == cat).to_numpy()
        pooled[cat] = (gold.to_numpy()[m].sum() + K_PRIOR * prior) / (m.sum() + K_PRIOR)

    out = pd.DataFrame(index=ext.index)
    out["oficial"] = out.index.isin(gold.index)
    print(f"estudios: {len(out):,}; con etiqueta oficial: {int(out.oficial.sum())}")
    print("\n-- Objetivo asignado por hallazgo y categoría (entre paréntesis, casos oficiales: positivos/total)")
    order = ["present/severe", "present/moderate", "present/unspecified", "present/mild", "possible", "absent"]
    print(f"{'hallazgo':<17}" + "".join(f"{c.replace('present/', 'pres/'):>19}" for c in order))
    for c in LABELS:
        mapping, cells = {}, []
        for cat in PRIOR:
            m = (g_cats[c] == cat).to_numpy()
            n, pos = int(m.sum()), int(gold[c].to_numpy()[m].sum())
            mapping[cat] = (pos + K_POOL * pooled[cat]) / (n + K_POOL)
            if cat in order:
                cells.append(f"{mapping[cat]:.2f} ({pos}/{n})")
        print(f"{c:<17}" + "".join(f"{x:>19}" for x in cells))
        status = ext[f"{c}_status"]
        out[f"t_{c}"] = cats[c].map(mapping).astype("float32")           # NaN en "no mencionado"
        out[f"w_{c}"] = status.map(WEIGHT).fillna(0).astype("float32")
        out.loc[out[f"w_{c}"] == 0, f"t_{c}"] = np.nan
        out[f"y_{c}"] = gold[c].reindex(out.index).astype("float32")     # etiqueta oficial, solo en los 58

    print(f"\n{'conjunto':<17}" + "".join(f"{c.replace('present/', 'pres/'):>19}" for c in order))
    print(f"{'(todos)':<17}" + "".join(f"{pooled[c]:>19.2f}" for c in order))

    un = out[~out.oficial]
    print("\n-- En los estudios sin etiqueta oficial (los de entrenamiento)")
    print(f"{'hallazgo':<17}{'con objetivo%':>14}{'objetivo medio':>16}{'objetivo>=0.5 %':>17}")
    for c in LABELS:
        has = un[f"w_{c}"] > 0
        print(f"{c:<17}{has.mean() * 100:>14.1f}{un.loc[has, f't_{c}'].mean():>16.3f}{(un.loc[has, f't_{c}'] >= 0.5).mean() * 100:>17.1f}")
    n_obj = sum((un[f"w_{c}"] > 0).astype(int) for c in LABELS)
    print(f"hallazgos con objetivo por estudio: media {n_obj.mean():.1f}; estudios sin ningún objetivo: {int((n_obj == 0).sum())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(args.out, index=False)
    print(f"\ntabla escrita en {args.out}")


if __name__ == "__main__":
    main()
