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
    ap.add_argument("--max-combo", type=int, default=0, help="tamaño máximo de las combinaciones de grupos; 0 = todas")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    groups = {}
    for spec in args.grupos:
        name, dirs = spec.split("=", 1)
        groups[name] = load_group([Path(d) for d in dirs.split(",")])
    last = {n: int(g[2][0]) for n, g in groups.items()}       # cada grupo se mide en su última época
    uid_int = next(iter(groups.values()))[0].index
    uid_off = next(iter(groups.values()))[1].index
    assert all(set(g[0].index) == set(uid_int) for g in groups.values()), "los grupos no cubren los mismos estudios"

    t = table.loc[uid_int, [f"t_{c}" for c in LABELS]].to_numpy(np.float32)
    w = table.loc[uid_int, [f"w_{c}" for c in LABELS]].to_numpy(np.float32)
    y_int = (np.nan_to_num(t) >= 0.5).astype(int)
    y_off = table.loc[uid_off, [f"y_{c}" for c in LABELS]].to_numpy(int)

    def score(names: tuple[str, ...], epoch: int | None = None):
        p_int = np.mean([groups[n][0].loc[uid_int, epoch or last[n]].to_numpy() for n in names], axis=0)
        p_off = np.mean([groups[n][1].loc[uid_off, epoch or last[n]].to_numpy() for n in names], axis=0)
        return macro_auc(y_int, p_int, w), macro_auc(y_off, p_off)

    lines = [f"objetivos: {args.labels.name}; estudios fuera de muestra: {len(uid_int)}; oficiales: {len(uid_off)}", "",
             "-- AUC macro por época (interna fuera de muestra / oficial con el promedio de las particiones)",
             f"{'época':<7}" + "".join(f"{n:>18}" for n in groups)]
    for e in range(1, max(last.values()) + 1):
        cells = [score((n,), e) if e <= last[n] else None for n in groups]
        lines.append(f"{e:<7}" + "".join(f"{c[0][0]:>10.4f} /{c[1][0]:>6.4f}" if c else f"{'':>18}" for c in cells))

    top = args.max_combo or len(groups)
    combos = [c for k in range(1, top + 1) for c in combinations(groups, k)]
    res = {c: score(c) for c in combos}
    lines += ["", "-- Combinaciones (media de probabilidades; cada grupo en su última época), de mejor a peor interna",
              f"{'interna':>8}{'oficial':>9}  combinación"]
    for c in sorted(combos, key=lambda c: -res[c][0][0]):
        lines.append(f"{res[c][0][0]:>8.4f}{res[c][1][0]:>9.4f}  {' + '.join(c)}")

    best = max((c for c in combos if len(c) > 1), key=lambda c: res[c][0][0], default=None)
    cols = [(n,) for n in groups] + ([best] if best else [])
    heads = [c[0] if len(c) == 1 else "mejor" for c in cols]
    for kind, k in (("interna fuera de muestra", 0), ("en los 58 oficiales (promedio de las particiones)", 1)):
        lines += ["", f"-- AUC por hallazgo, {kind}", f"{'hallazgo':<17}" + "".join(f"{h:>11}" for h in heads)]
        for j, c in enumerate(LABELS):
            lines.append(f"{c:<17}" + "".join(f"{res[col][k][1][j]:>11.4f}" for col in cols))
        lines.append(f"{'MEDIA':<17}" + "".join(f"{res[col][k][0]:>11.4f}" for col in cols))
    if best:
        lines += ["", f"mejor = {' + '.join(best)}"]

    print("\n".join(lines), flush=True)
    (args.out / "oof.txt").write_text("\n".join(lines) + "\n")
    # Predicciones fuera de muestra de la última época, para el autoentrenamiento y los pesos del conjunto.
    for n, (oof, _, _) in groups.items():
        oof[last[n]].loc[uid_int].rename_axis("StudyInstanceUID").reset_index().to_parquet(args.out / f"oof_{n}.parquet", index=False)


if __name__ == "__main__":
    main()
