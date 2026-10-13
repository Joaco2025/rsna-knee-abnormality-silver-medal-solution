#!/usr/bin/env python
"""Corrige una tabla de objetivos con las predicciones fuera de muestra de los modelos.

El "maestro" de cada estudio es el promedio de las predicciones de los modelos que no lo vieron
(los mismos grupos que usa evaluar_oof.py). Con él se hacen dos cosas, por hallazgo:
  relleno      las celdas sin objetivo (el informe no menciona el hallazgo) reciben como objetivo
               la probabilidad del maestro, con un peso reducido;
  desconfianza en las celdas con objetivo se mide la discrepancia |objetivo - maestro| y se ajusta
               una mezcla de dos gaussianas; a las celdas del componente discrepante se les baja
               el peso en proporción a su probabilidad de pertenecer a él.
Los objetivos que vienen del informe no se cambian, solo su peso. Los 58 estudios oficiales no se
tocan: sirven para comprobar, contra la verdad, si el relleno ordena bien los casos y si las celdas
marcadas como sospechosas fallan más que las demás.

Ojo al medir después: el maestro de un estudio se entrenó con las demás particiones, así que la
validación interna de un modelo entrenado con esta tabla queda algo favorecida. La referencia
limpia son los 58 oficiales.

Uso:  python rsna/tabla_corregida.py --labels BASE.parquet --out NUEVA.parquet r34=RUN0,... cnx=RUN0,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluar_oof import load_group  # noqa: E402
from modelo import LABELS  # noqa: E402


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grupos", nargs="+", help="nombre=dir0,dir1,... con las corridas de cada partición")
    ap.add_argument("--labels", type=Path, required=True, help="tabla base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--peso-relleno", type=float, default=0.5, help="peso de las celdas rellenadas; 0 = no rellenar")
    ap.add_argument("--peso-minimo", type=float, default=0.2, help="fracción del peso que conserva una celda claramente discrepante; 1 = no desconfiar")
    ap.add_argument("--umbral", type=float, default=0.0, help="si es > 0, en vez de la mezcla se marca como sospechosa toda celda con discrepancia mayor que este valor")
    args = ap.parse_args()

    table = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    groups = [load_group([Path(d) for d in spec.split("=", 1)[1].split(",")]) for spec in args.grupos]
    epoch = int(min(g[2][0] for g in groups))
    uid_int, uid_off = groups[0][0].index, groups[0][1].index
    teacher = pd.DataFrame(np.mean([g[0].loc[uid_int, epoch].to_numpy() for g in groups], axis=0), index=uid_int, columns=LABELS)
    teacher_off = pd.DataFrame(np.mean([g[1].loc[uid_off, epoch].to_numpy() for g in groups], axis=0), index=uid_off, columns=LABELS)
    print(f"base: {args.labels.name}; maestro: {len(groups)} grupos, época {epoch}; estudios corregidos: {len(uid_int)}; oficiales: {len(uid_off)}")

    out = table.copy()
    rows, checks, gaps = [], [], []
    for c in LABELS:
        t, w = table.loc[uid_int, f"t_{c}"], table.loc[uid_int, f"w_{c}"]
        p = teacher[c]
        has = w > 0

        # Desconfianza: mezcla de dos gaussianas sobre la discrepancia de las celdas con objetivo.
        gap = (t[has] - p[has]).abs().to_numpy()[:, None]
        gmm = GaussianMixture(2, random_state=0).fit(gap)
        noisy = int(np.argmax(gmm.means_.ravel()))

        def suspicion(g: np.ndarray) -> np.ndarray:
            if args.umbral > 0:
                return (g > args.umbral).astype(float)
            return gmm.predict_proba(g[:, None])[:, noisy]

        s = suspicion(gap.ravel())
        w_new = w.astype("float64")
        w_new[has] = w[has] * (1 - (1 - args.peso_minimo) * s)

        # Relleno de las celdas sin objetivo.
        t_new = t.astype("float64")
        if args.peso_relleno > 0:
            t_new[~has] = p[~has]
            w_new[~has] = args.peso_relleno
        out.loc[uid_int, f"t_{c}"] = t_new.astype("float32")
        out.loc[uid_int, f"w_{c}"] = w_new.astype("float32")
        rows.append((c, int((~has).sum()), float(p[~has].mean()) if (~has).any() else float("nan"),
                     float((p[~has] >= 0.5).mean() * 100) if (~has).any() else float("nan"),
                     float(gmm.means_.ravel()[1 - noisy]), float(gmm.means_.ravel()[noisy]),
                     float((s > 0.5).mean() * 100), float(w_new[has].sum() / w[has].sum() * 100)))

        # Comprobación en los 58 oficiales (el maestro tampoco los vio; aquí hay verdad).
        y = table.loc[uid_off, f"y_{c}"].to_numpy(int)
        t_o, w_o, p_o = table.loc[uid_off, f"t_{c}"].to_numpy(), table.loc[uid_off, f"w_{c}"].to_numpy(), teacher_off[c].to_numpy()
        h = w_o > 0
        s_o = suspicion(np.abs(t_o[h] - p_o[h]))
        wrong = np.abs(t_o[h] - y[h]) > 0.5          # el objetivo del informe cae del lado contrario a la verdad
        flag = s_o > 0.5
        gaps.append(np.stack([np.abs(t_o[h] - p_o[h]), wrong], axis=1))
        checks.append((c, int((~h).sum()), auc(y[~h], p_o[~h]) if (~h).sum() > 1 else float("nan"),
                       int(flag.sum()), int(wrong[flag].sum()), int((~flag).sum()), int(wrong[~flag].sum())))

    print("\n-- Corrección en los estudios de entrenamiento")
    print(f"{'hallazgo':<17}{'rellenadas':>11}{'obj.medio':>10}{'>=0.5 %':>9}{'discr.normal':>13}{'discr.sosp.':>12}{'sospechosas%':>13}{'peso restante%':>15}")
    for r in rows:
        print(f"{r[0]:<17}{r[1]:>11}{r[2]:>10.3f}{r[3]:>9.1f}{r[4]:>13.3f}{r[5]:>12.3f}{r[6]:>13.1f}{r[7]:>15.1f}")

    print("\n-- Comprobación contra la verdad en los 58 oficiales")
    print(f"{'hallazgo':<17}{'sin objetivo':>13}{'AUC maestro ahí':>17}{'sospechosas':>12}{'de ellas mal':>13}{'no sosp.':>10}{'de ellas mal':>13}")
    tot = np.zeros(4, int)
    for c, n0, a, nf, wf, nn, wn in checks:
        print(f"{c:<17}{n0:>13}{a:>17.3f}{nf:>12}{wf:>13}{nn:>10}{wn:>13}")
        tot += [nf, wf, nn, wn]
    print(f"{'TOTAL':<17}{'':>30}{tot[0]:>12}{tot[1]:>13}{tot[2]:>10}{tot[3]:>13}")
    if tot[0] and tot[2]:
        print(f"objetivos del lado equivocado: {tot[1] / tot[0] * 100:.0f} % entre las sospechosas, {tot[3] / tot[2] * 100:.0f} % entre las demás")

    gaps = np.concatenate(gaps)
    print("\n-- En los 58 oficiales, según el umbral de discrepancia (todos los hallazgos juntos)")
    print(f"{'umbral':<8}{'por encima':>11}{'mal %':>7}{'por debajo':>12}{'mal %':>7}")
    for u in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        m = gaps[:, 0] > u
        print(f"{u:<8}{int(m.sum()):>11}{gaps[m, 1].mean() * 100 if m.any() else float('nan'):>7.0f}{int((~m).sum()):>12}{gaps[~m, 1].mean() * 100:>7.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(args.out, index=False)
    print(f"\ntabla escrita en {args.out}")


if __name__ == "__main__":
    main()
