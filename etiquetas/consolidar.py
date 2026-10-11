#!/usr/bin/env python
"""Junta las extracciones (extraer.py) en una tabla por estudio y revisa si tienen sentido.

No decide las etiquetas finales: deja estado y gravedad por hallazgo, y un resumen para detectar
problemas (idiomas mal leídos, informes sin contenido, respuestas inconsistentes).

Uso:  python etiquetas/consolidar.py --out DIR RUN1/extraccion.jsonl RUN2/extraccion.jsonl ...
Salidas en DIR:
  extraccion_consolidada.parquet / .csv   una fila por estudio: <hallazgo>_status y <hallazgo>_severity
  consolidacion_resumen.txt               el resumen que también se imprime
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from evaluar_extraccion import CUTS
from extraer import DATA, FINDINGS

# Corte tentativo por hallazgo: el que más acertó en los 58 estudios oficiales (job 2078806).
# Elegido y medido sobre los mismos estudios, así que es orientativo.
TENTATIVE_CUT = {
    "ACL": "C", "MCL": "C", "Medial Meniscus": "B", "Lateral Meniscus": "B", "Medial OA": "D",
    "Lateral OA": "D", "PF OA": "C", "Effusion": "A", "Synovitis": "A", "Baker's": "C",
    "Contusion": "D", "Fracture": "A",
}

STOPWORDS = {
    "en": "the of and is are with no in there to without at seen on normal",
    "es": "el la los las de del que en y con se por un una no sin es al como",
    "nl": "de het een van en met is geen niet in op bij ook er zijn aan",
    "de": "der die das und mit ist kein keine nicht im des ein eine von zu",
    "fr": "le la les des et du un une est pas avec dans au sans sur de",
    "pt": "o a os as do da dos das e com sem não em um uma na no ao",
    "it": "il lo la gli le di del della e con non in un una si al senza",
    "tr": "ve bir ile için bu olarak izlendi izlenmiştir mevcuttur normaldir yoktur düzeyinde",
    "hr": "je se u na i te su bez od uz kao koji nalazi područja znakova",
}
STOPWORDS = {k: set(v.split()) for k, v in STOPWORDS.items()}
SCRIPTS = {"GREEK": "el", "CYRILLIC": "cirílico"}


def guess_language(text: str) -> str:
    """Heurística por alfabeto y palabras frecuentes; orientativa, no un detector real."""
    text = str(text).lower()
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return "indet."
    non_latin = [unicodedata.name(c, "").split()[0] for c in letters if ord(c) > 0x24F]
    if len(non_latin) > 0.3 * len(letters):
        script = max(set(non_latin), key=non_latin.count)
        return SCRIPTS.get(script, script.lower())
    words = re.findall(r"[^\W\d_]+", text)
    scores = sorted(((sum(w in sw for w in words), lang) for lang, sw in STOPWORDS.items()), reverse=True)
    (best, lang), (second, _) = scores[0], scores[1]
    return lang if best >= 3 and best > second else "indet."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    rows, n_err, n_lines = {}, 0, 0
    for path in args.jsonl:
        for line in path.read_text().splitlines():
            n_lines += 1
            rec = json.loads(line)
            if "error" in rec:
                n_err += 1
                continue
            row = {"StudyInstanceUID": rec["StudyInstanceUID"], "model": rec.get("model"),
                   "prompt_version": rec.get("prompt_version"), "done_reason": rec.get("done_reason"),
                   "eval_count": rec.get("eval_count"), "origen": path.parent.name}
            for key, col in FINDINGS.items():
                row[f"{col}_status"] = rec["parsed"][key]["status"]
                row[f"{col}_severity"] = rec["parsed"][key]["severity"]
            rows[rec["StudyInstanceUID"]] = row  # si un estudio aparece dos veces, vale la última respuesta
    table = pd.DataFrame(rows.values())

    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str})
    cols = list(FINDINGS.values())
    train["etiquetado"] = train[cols].notna().all(axis=1)
    train["n_chars"] = train.Report.fillna("").str.len()
    train["idioma"] = train.Report.map(guess_language)
    table = table.merge(train[["StudyInstanceUID", "etiquetado", "n_chars", "idioma"]], on="StudyInstanceUID", how="left")

    say(f"== {n_lines:,} líneas leídas de {len(args.jsonl)} archivos; {n_err} con error; {len(table):,} estudios distintos")
    missing = set(train.StudyInstanceUID) - set(table.StudyInstanceUID)
    extra = table.etiquetado.isna().sum()
    say(f"Estudios de train.csv sin extracción: {len(missing)}; extracciones que no están en train.csv: {int(extra)}")
    say(f"Modelos: {table.model.value_counts().to_dict()}  prompts: {table.prompt_version.value_counts().to_dict()}")
    say(f"Respuestas no terminadas con 'stop': {int((table.done_reason != 'stop').sum())}; "
        f"tokens de salida: mediana {table.eval_count.median():.0f}, máximo {table.eval_count.max():.0f}")
    say(f"Etiquetados: {int(table.etiquetado.sum())}; sin etiqueta: {int((table.etiquetado == False).sum())}")  # noqa: E712

    status = table[[f"{c}_status" for c in cols]].set_axis(cols, axis=1)
    severity = table[[f"{c}_severity" for c in cols]].set_axis(cols, axis=1)

    say("\n-- Estado por hallazgo (% de los estudios)")
    dist = pd.DataFrame({c: status[c].value_counts(normalize=True) for c in cols}).T.fillna(0) * 100
    dist = dist.reindex(columns=["present", "possible", "absent", "not_mentioned"]).fillna(0)
    say(dist.round(1).to_string())

    say("\n-- Gravedad cuando el estado es 'presente' (% de los presentes)")
    sev = pd.DataFrame({c: severity[c][status[c] == "present"].value_counts(normalize=True) for c in cols}).T.fillna(0) * 100
    sev = sev.reindex(columns=["mild", "moderate", "severe", "unspecified", "na"]).fillna(0)
    say(sev.round(1).to_string())
    bad = int(((status.isin(["absent", "not_mentioned"])) & (severity != "na")).values.sum()) \
        + int(((status.isin(["present", "possible"])) & (severity == "na")).values.sum())
    say(f"Combinaciones incoherentes (gravedad en ausentes, o 'na' en presentes): {bad} de {status.size:,}")

    def positives(col: str, cut: str) -> pd.Series:
        return pd.Series([CUTS[cut](st, sv) for st, sv in zip(status[col], severity[col])], index=table.index)

    gold = train.loc[train.etiquetado].set_index("StudyInstanceUID")[cols].astype(int)
    lab = table.etiquetado == True  # noqa: E712
    say("\n-- Tasa de positivos (%): oficial en los 58, y Qwen con el corte tentativo")
    say(f"{'hallazgo':<17}{'corte':>6}{'oficial(58)':>13}{'Qwen(58)':>10}{'Qwen(sin etiq.)':>17}{'presentes B(sin etiq.)':>24}")
    pos_tent = pd.DataFrame(index=table.index)
    for c in cols:
        p = positives(c, TENTATIVE_CUT[c])
        pos_tent[c] = p
        b = positives(c, "B")
        say(f"{c:<17}{TENTATIVE_CUT[c]:>6}{gold[c].mean() * 100:>13.1f}{p[lab].mean() * 100:>10.1f}"
            f"{p[~lab].mean() * 100:>17.1f}{b[~lab].mean() * 100:>24.1f}")

    n_pos = pos_tent.sum(axis=1)
    say(f"\nHallazgos positivos por estudio (corte tentativo): media {n_pos.mean():.2f}, mediana {n_pos.median():.0f}, "
        f"máximo {int(n_pos.max())}; estudios sin ningún positivo: {int((n_pos == 0).sum()):,} ({(n_pos == 0).mean():.1%})")
    say(f"En los 58 oficiales: media {gold.sum(axis=1).mean():.2f} positivos por estudio según la etiqueta oficial, "
        f"{n_pos[lab].mean():.2f} según Qwen")
    not_m = (status == "not_mentioned").sum(axis=1)
    say(f"Estudios con los 12 hallazgos 'no mencionados': {int((not_m == 12).sum())}; con 9 o más: {int((not_m >= 9).sum())}")

    say("\n-- Por idioma estimado del informe (heurística)")
    by = table.assign(n_pos=n_pos, no_menc=not_m, presentes=(status == "present").sum(axis=1)).groupby("idioma").agg(
        estudios=("n_pos", "size"), caracteres=("n_chars", "median"), presentes=("presentes", "mean"),
        positivos=("n_pos", "mean"), no_mencionados=("no_menc", "mean"))
    say(by.sort_values("estudios", ascending=False).round(2).to_string())

    say("\n-- Por longitud del informe")
    bins = pd.cut(table.n_chars, [0, 300, 600, 1000, 1500, 2500, 100000])
    bl = table.assign(n_pos=n_pos, no_menc=not_m).groupby(bins, observed=True).agg(
        estudios=("n_pos", "size"), positivos=("n_pos", "mean"), no_mencionados=("no_menc", "mean"))
    say(bl.round(2).to_string())

    # Informes idénticos deben dar la misma extracción (temperatura 0): mide la consistencia del modelo.
    rep = train.set_index("StudyInstanceUID").Report.map(lambda t: " ".join(str(t).split()))
    key_cols = [c for c in table.columns if c.endswith("_status") or c.endswith("_severity")]
    sig = table.set_index("StudyInstanceUID")[key_cols].astype(str).agg("|".join, axis=1)
    groups = pd.DataFrame({"rep": rep.reindex(sig.index), "sig": sig}).groupby("rep").sig.agg(["size", "nunique"])
    dup = groups[groups["size"] > 1]
    say(f"\n-- Consistencia: {len(dup)} informes aparecen repetidos en {int(dup['size'].sum())} estudios; "
        f"en {int((dup['nunique'] == 1).sum())} de esos {len(dup)} grupos la extracción es idéntica")

    table.to_parquet(args.out / "extraccion_consolidada.parquet", index=False)
    table.to_csv(args.out / "extraccion_consolidada.csv", index=False)
    (args.out / "consolidacion_resumen.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSalidas en {args.out}")


if __name__ == "__main__":
    main()
