#!/usr/bin/env python
"""Evalúa la validación cruzada a partir de los predicciones.npz que deja entrenar.py.

Cada grupo es una red entrenada en las K particiones. Para cada grupo y para su promedio:
  - interna: AUC de las predicciones fuera de muestra (cada estudio lo predice el modelo que no lo
    vio) contra los objetivos derivados binarizados en 0.5. Usa todos los estudios sin etiqueta
    oficial, no solo un 10 %.
  - oficial: AUC en los 58 estudios oficiales del promedio de los K modelos, que es lo que se envía.

Uso:  python rsna/evaluar_oof.py --labels TABLA.parquet --out DIR r34=RUN0,RUN1,... cnx=RUN0,RUN1,...
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar import macro_auc  # noqa: E402
from modelo import LABELS  # noqa: E402


def load_group(dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Predicciones fuera de muestra y oficiales de un grupo, por época: columnas (época, hallazgo)."""
    runs = [np.load(d / "predicciones.npz") for d in dirs]
    folds = sorted(int(r["fold"]) for r in runs)
    assert folds == list(range(int(runs[0]["folds"]))), f"particiones incompletas o repetidas: {folds}"
    epochs = min(r["p_interna"].shape[0] for r in runs)
    cols = pd.MultiIndex.from_product([range(1, epochs + 1), LABELS])

    def frame(p, uid):   # (épocas, estudios, hallazgos) -> una fila por estudio
        return pd.DataFrame(p[:epochs].transpose(1, 0, 2).reshape(len(uid), -1), index=uid, columns=cols)

    oof = pd.concat([frame(r["p_interna"], r["uid_interna"]) for r in runs])
    assert oof.index.is_unique, "un estudio aparece en dos particiones"
    off = sum(frame(r["p_oficial"], r["uid_oficial"]) for r in runs) / len(runs)
    return oof, off, np.array([epochs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+", help="nombre=dir0,dir1,... con las corridas de cada partición")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    groups = {}
    for spec in args.grupos:
        name, dirs = spec.split("=", 1)
        groups[name] = load_group([Path(d) for d in dirs.split(",")])
    epochs = int(min(g[2][0] for g in groups.values()))
    uid_int = next(iter(groups.values()))[0].index
    uid_off = next(iter(groups.values()))[1].index
    assert all(set(g[0].index) == set(uid_int) for g in groups.values()), "los grupos no cubren los mismos estudios"

    t = table.loc[uid_int, [f"t_{c}" for c in LABELS]].to_numpy(np.float32)
    w = table.loc[uid_int, [f"w_{c}" for c in LABELS]].to_numpy(np.float32)
    y_int = (np.nan_to_num(t) >= 0.5).astype(int)
    y_off = table.loc[uid_off, [f"y_{c}" for c in LABELS]].to_numpy(int)

    def score(names: tuple[str, ...], epoch: int):
        p_int = np.mean([groups[n][0].loc[uid_int, epoch].to_numpy() for n in names], axis=0)
        p_off = np.mean([groups[n][1].loc[uid_off, epoch].to_numpy() for n in names], axis=0)
        return macro_auc(y_int, p_int, w), macro_auc(y_off, p_off)

    lines = [f"estudios fuera de muestra: {len(uid_int)}; oficiales: {len(uid_off)}; épocas: {epochs}", "",
             "-- AUC macro por época (interna fuera de muestra / oficial con el promedio de las particiones)",
             f"{'época':<7}" + "".join(f"{n:>18}" for n in groups)]
    for e in range(1, epochs + 1):
        cells = [score((n,), e) for n in groups]
        lines.append(f"{e:<7}" + "".join(f"{a[0]:>10.4f} /{b[0]:>6.4f}" for a, b in cells))

    combos = [c for k in range(1, len(groups) + 1) for c in combinations(groups, k)]
    res = {c: score(c, epochs) for c in combos}
    lines += ["", f"-- Última época ({epochs}): AUC por hallazgo, interna fuera de muestra",
              f"{'hallazgo':<17}{'casos+':>7}{'total':>7}" + "".join(f"{'+'.join(c):>14}" for c in combos)]
    for j, c in enumerate(LABELS):
        m = w[:, j] > 0
        lines.append(f"{c:<17}{int(y_int[m, j].sum()):>7}{int(m.sum()):>7}" + "".join(f"{res[k][0][1][j]:>14.4f}" for k in combos))
    lines.append(f"{'MEDIA':<31}" + "".join(f"{res[k][0][0]:>14.4f}" for k in combos))
    lines += ["", "-- Última época: AUC por hallazgo en los 58 oficiales (promedio de las particiones)",
              f"{'hallazgo':<17}" + "".join(f"{'+'.join(c):>14}" for c in combos)]
    for j, c in enumerate(LABELS):
        lines.append(f"{c:<17}" + "".join(f"{res[k][1][1][j]:>14.4f}" for k in combos))
    lines.append(f"{'MEDIA':<17}" + "".join(f"{res[k][1][0]:>14.4f}" for k in combos))

    print("\n".join(lines), flush=True)
    (args.out / "oof.txt").write_text("\n".join(lines) + "\n")
    # Predicciones fuera de muestra de la última época, para el autoentrenamiento y los pesos del conjunto.
    for n, (oof, _, _) in groups.items():
        oof[epochs].loc[uid_int].rename_axis("StudyInstanceUID").reset_index().to_parquet(args.out / f"oof_{n}.parquet", index=False)


if __name__ == "__main__":
    main()
