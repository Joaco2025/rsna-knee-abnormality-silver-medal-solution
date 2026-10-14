#!/usr/bin/env python
"""Compara, en los 58 estudios oficiales, cuánto se parece a la verdad cada fuente de etiquetas
derivadas de los informes: AUC por hallazgo de cada tabla (y de CSV externos) contra la etiqueta
oficial. Las celdas sin objetivo cuentan como 0.5, para que todas las fuentes se midan sobre los
mismos 58 casos. Con 58 casos los márgenes son de unos ±0.1 por hallazgo.

Uso:  python rsna/comparar_fuentes.py nombre=TABLA.parquet ... [nombre=EXTERNO.csv ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo import LABELS  # noqa: E402

ALIAS = {"acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
         "medial_oa": "Medial OA", "lateral_oa": "Lateral OA", "pf_oa": "PF OA", "effusion": "Effusion",
         "synovitis": "Synovitis", "bakers": "Baker's", "baker": "Baker's", "bakers_cyst": "Baker's",
         "contusion": "Contusion", "fracture": "Fracture"}


def load(path: Path, gold: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    """Objetivos de los 58 oficiales (NaN donde no hay) y fracción de celdas con objetivo por hallazgo."""
    if path.suffix == ".parquet":
        tab = pd.read_parquet(path).set_index("StudyInstanceUID").loc[gold]
        t = pd.DataFrame({c: tab[f"t_{c}"].where(tab[f"w_{c}"] > 0) for c in LABELS})
    else:
        tab = pd.read_csv(path)
        ren = {}
        for col in tab.columns:
            low = col.strip().lower().replace(" ", "_").replace("'", "")
            if low in ALIAS:
                ren[col] = ALIAS[low]
            elif low in ("studyinstanceuid", "study_instance_uid", "study_uid", "uid"):
                ren[col] = "StudyInstanceUID"
        tab = tab.rename(columns=ren).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(gold)
        t = tab[LABELS].astype(float)
        t = t.where(t != 0.5)          # 0.5 es "sin información" en las etiquetas públicas
    return t, t.notna().mean()


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    ref = pd.read_parquet(next(Path(p) for _, p in specs if p.endswith(".parquet"))).set_index("StudyInstanceUID")
    gold = ref.index[ref.oficial]
    y = ref.loc[gold, [f"y_{c}" for c in LABELS]].to_numpy(int)
    data = {name: load(Path(p), gold) for name, p in specs}
    print(f"estudios oficiales: {len(gold)}")
    print("\n-- AUC contra la etiqueta oficial (entre paréntesis, % de los 58 con objetivo)")
    print(f"{'hallazgo':<17}" + "".join(f"{n:>17}" for n in data))
    means = {n: [] for n in data}
    for j, c in enumerate(LABELS):
        cells = []
        for n, (t, cov) in data.items():
            a = roc_auc_score(y[:, j], t[c].fillna(0.5).to_numpy())
            means[n].append(a)
            cells.append(f"{a:.3f} ({cov[c] * 100:3.0f} %)")
        print(f"{c:<17}" + "".join(f"{x:>17}" for x in cells))
    print(f"{'MEDIA':<17}" + "".join(f"{np.mean(v):>10.3f}{'':>7}" for v in means.values()))


if __name__ == "__main__":
    main()
