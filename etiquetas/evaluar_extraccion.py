#!/usr/bin/env python
"""Evalúa una extracción (extraer.py) contra las etiquetas oficiales de los estudios etiquetados.

Para cada hallazgo prueba cuatro cortes entre positivo y negativo y muestra el acuerdo de cada uno:
  A  presente o posible, cualquier gravedad
  B  presente, cualquier gravedad
  C  presente y gravedad distinta de leve (moderada, grave o sin especificar)
  D  presente y gravedad moderada o grave

OJO: el "mejor corte" se elige y se mide sobre los mismos estudios, así que su acuerdo es optimista.

Uso:  python etiquetas/evaluar_extraccion.py DIR/extraccion.jsonl [--lector etiquetas/piloto_58/prediccion_lector.csv]
Salida: DIR/evaluacion.txt (además de imprimirlo)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from extraer import DATA, FINDINGS

CUTS = {
    "A": lambda st, sv: st in ("present", "possible"),
    "B": lambda st, sv: st == "present",
    "C": lambda st, sv: st == "present" and sv != "mild",
    "D": lambda st, sv: st == "present" and sv in ("moderate", "severe"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--lector", type=Path, default=None, help="CSV p/a/n de una lectura de referencia")
    args = ap.parse_args()

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    records, n_err = {}, 0
    for line in args.jsonl.read_text().splitlines():
        rec = json.loads(line)
        if "error" in rec:
            n_err += 1
        else:
            records[rec["StudyInstanceUID"]] = rec  # si un estudio se repitió, vale la última respuesta correcta

    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    cols = list(FINDINGS.values())
    gold = train.loc[train[cols].notna().all(axis=1), cols].astype(int)
    common = [uid for uid in gold.index if uid in records]
    gold = gold.loc[common]
    n = len(common)
    say(f"== {args.jsonl}: {len(records)} respuestas válidas, {n_err} líneas con error, {n} estudios etiquetados evaluados")
    if n == 0:
        raise SystemExit("No hay estudios etiquetados en la extracción.")
    say(f"modelo={records[common[0]].get('model')} prompt={records[common[0]].get('prompt_version')}")

    status = pd.DataFrame({col: [records[u]["parsed"][key]["status"] for u in common] for key, col in FINDINGS.items()}, index=common)
    severity = pd.DataFrame({col: [records[u]["parsed"][key]["severity"] for u in common] for key, col in FINDINGS.items()}, index=common)

    def positives(col: str, cut: str) -> pd.Series:
        return pd.Series([CUTS[cut](st, sv) for st, sv in zip(status[col], severity[col])], index=common).astype(int)

    say(f"\n-- Acuerdo (%) con la etiqueta oficial según el corte; n = {n}")
    say(f"{'hallazgo':<17}{'pos.of':>7}" + "".join(f"{c:>7}" for c in CUTS) + f"{'mejor':>7}{'sens':>7}{'prec':>7}")
    totals = {c: 0 for c in CUTS}
    best_total = 0
    for col in cols:
        g = gold[col]
        agree = {c: int((positives(col, c) == g).sum()) for c in CUTS}
        for c in CUTS:
            totals[c] += agree[c]
        best = max(CUTS, key=lambda c: agree[c])
        best_total += agree[best]
        p = positives(col, best)
        tp, fp, fn = int(((p == 1) & (g == 1)).sum()), int(((p == 1) & (g == 0)).sum()), int(((p == 0) & (g == 1)).sum())
        sens = tp / (tp + fn) * 100 if tp + fn else float("nan")
        prec = tp / (tp + fp) * 100 if tp + fp else float("nan")
        say(f"{col:<17}{int(g.sum()):>7}" + "".join(f"{agree[c] / n * 100:>7.0f}" for c in CUTS)
            + f"{best:>7}{sens:>7.0f}{prec:>7.0f}")
    cells = n * len(cols)
    say(f"{'TOTAL':<17}{int(gold.values.sum()):>7}" + "".join(f"{totals[c] / cells * 100:>7.1f}" for c in CUTS))
    say(f"Con el mejor corte de cada hallazgo (elegido sobre estos mismos estudios): {best_total / cells * 100:.1f} %")

    say("\n-- Qué dice el modelo en los positivos y negativos oficiales (estado/gravedad: nº de estudios)")
    for col in cols:
        combo = status[col] + "/" + severity[col]
        for value, name in [(1, "oficial 1"), (0, "oficial 0")]:
            counts = combo[gold[col] == value].value_counts()
            say(f"{col:<17}{name}: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    if args.lector and args.lector.exists():
        lector = pd.read_csv(args.lector, dtype=str).set_index("StudyInstanceUID")
        both = [u for u in common if u in lector.index]
        say(f"\n-- Acuerdo (%) con la lectura de referencia ({args.lector.name}), corte B; n = {len(both)}")
        tot = 0
        for col in cols:
            same = int((positives(col, "B").loc[both] == (lector.loc[both, col] == "p").astype(int)).sum())
            tot += same
            say(f"{col:<17}{same / len(both) * 100:>6.0f}")
        say(f"{'TOTAL':<17}{tot / (len(both) * len(cols)) * 100:>6.1f}")

    (args.jsonl.parent / "evaluacion.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
