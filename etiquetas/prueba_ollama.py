#!/usr/bin/env python
"""Prueba de humo: ¿un modelo local servido por Ollama extrae las 12 etiquetas de un informe?

Envía unos pocos informes a un servidor Ollama ya arrancado (dentro del mismo job), pide una
respuesta con esquema JSON fijo y mide tiempos. NO es una evaluación: con tan pocos informes
solo sirve para comprobar que el modelo carga en la GPU, respeta el esquema y a qué velocidad va.

Uso (dentro de un job, con `ollama serve` escuchando):
  python etiquetas/prueba_ollama.py --host 127.0.0.1:11434 --model qwen2.5:72b --out DIR
Salidas en DIR: prueba_respuestas.jsonl (una línea por informe) y prueba_resumen.txt
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))

# clave del esquema -> (columna de train.csv, definición tomada de la descripción de Kaggle)
FINDINGS = {
    "acl": ("ACL", "anterior cruciate ligament injury"),
    "mcl": ("MCL", "medial collateral ligament injury"),
    "medial_meniscus": ("Medial Meniscus", "medial meniscus tear"),
    "lateral_meniscus": ("Lateral Meniscus", "lateral meniscus tear"),
    "medial_oa": ("Medial OA", "osteoarthritis of the medial tibiofemoral compartment"),
    "lateral_oa": ("Lateral OA", "osteoarthritis of the lateral tibiofemoral compartment"),
    "pf_oa": ("PF OA", "patellofemoral osteoarthritis"),
    "effusion": ("Effusion", "joint effusion / excess fluid"),
    "synovitis": ("Synovitis", "inflammation of the joint lining"),
    "bakers": ("Baker's", "Baker's cyst"),
    "contusion": ("Contusion", "bone contusion / bone bruise"),
    "fracture": ("Fracture", "fracture"),
}
STATES = ["present", "absent", "not_mentioned"]

SYSTEM = (
    "You assist a musculoskeletal radiologist. You are given the text of a knee MRI radiology "
    "report, which may be written in any language. For each of twelve findings decide whether the "
    "report states that it is present, states that it is absent, or does not mention it. Use only "
    "what the report says; do not infer findings the report does not describe. For each finding "
    "also copy the shortest passage of the report, in its original language, that supports your "
    "decision; use an empty string when the finding is not mentioned."
)
SCHEMA = {
    "type": "object",
    "properties": {
        key: {
            "type": "object",
            "properties": {"state": {"type": "string", "enum": STATES}, "quote": {"type": "string"}},
            "required": ["state", "quote"],
        }
        for key in FINDINGS
    },
    "required": list(FINDINGS),
}


def build_prompt(report: str) -> str:
    defs = "\n".join(f"- {key}: {definition}" for key, (_, definition) in FINDINGS.items())
    return f"Findings to assess:\n{defs}\n\nReport:\n\"\"\"\n{report.strip()}\n\"\"\""


def chat(host: str, model: str, report: str, timeout: int) -> dict:
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 1500},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(report)}],
    }
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # El servidor es local al job: nunca pasar por un proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def pick_reports(n_labeled: int, n_non_latin: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str})
    cols = [c for c, _ in FINDINGS.values()]
    labeled = df[df[cols].notna().all(axis=1)].sample(n_labeled, random_state=0)
    # Informes en alfabeto no latino (griego, cirílico...) para ver que el modelo los entiende.
    non_latin = df[cols[0]].isna() & df.Report.map(lambda t: sum(ord(c) > 0x36F for c in str(t)) > 0.3 * len(str(t)))
    extra = df[non_latin].sample(min(n_non_latin, int(non_latin.sum())), random_state=0)
    return pd.concat([labeled, extra])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-labeled", type=int, default=4)
    ap.add_argument("--n-non-latin", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900, help="segundos por informe (el primero incluye la carga del modelo)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reports = pick_reports(args.n_labeled, args.n_non_latin)
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    say(f"modelo={args.model} host={args.host} informes={len(reports)}")
    rows = []
    with open(args.out / "prueba_respuestas.jsonl", "w") as fh:
        for i, rec in enumerate(reports.itertuples(index=False), 1):
            record = {"StudyInstanceUID": rec.StudyInstanceUID, "model": args.model, "n_chars": len(rec.Report)}
            t0 = time.time()
            try:
                resp = chat(args.host, args.model, rec.Report, args.timeout)
                record["wall_s"] = round(time.time() - t0, 2)
                record["raw"] = resp.get("message", {}).get("content", "")
                for k in ["load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration", "done_reason"]:
                    record[k] = resp.get(k)
                record["parsed"] = json.loads(record["raw"])
                record["schema_ok"] = all(record["parsed"].get(k, {}).get("state") in STATES for k in FINDINGS)
            except Exception as exc:  # la prueba debe informar del fallo, no abortar
                record["wall_s"] = round(time.time() - t0, 2)
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            rows.append(record)

            say(f"\n[{i}/{len(reports)}] {rec.StudyInstanceUID}  ({record['n_chars']} caracteres, {record['wall_s']} s)")
            if "error" in record:
                say(f"  ERROR: {record['error']}")
                continue
            tok_s = record["eval_count"] / record["eval_duration"] * 1e9 if record.get("eval_duration") else float("nan")
            say(f"  carga={(record['load_duration'] or 0) / 1e9:.1f} s  entrada={record['prompt_eval_count']} tokens  "
                f"salida={record['eval_count']} tokens  {tok_s:.1f} tokens/s  esquema_ok={record['schema_ok']}  fin={record['done_reason']}")
            gold_known = pd.notna(getattr(rec, "ACL"))
            say(f"  {'hallazgo':<18}{'modelo':<15}{'oficial':<9}cita")
            for key, (col, _) in FINDINGS.items():
                item = record["parsed"].get(key, {})
                gold = reports.loc[reports.StudyInstanceUID == rec.StudyInstanceUID, col].iloc[0]
                gold_txt = ("1" if gold == 1 else "0") if gold_known else "-"
                say(f"  {key:<18}{str(item.get('state')):<15}{gold_txt:<9}{str(item.get('quote', ''))[:90]!r}")

    ok = [r for r in rows if "error" not in r]
    say(f"\n== Resumen: {len(ok)}/{len(rows)} respuestas sin error, "
        f"{sum(r['schema_ok'] for r in ok)}/{len(ok)} con esquema válido")
    warm = [r for r in ok if (r.get("load_duration") or 0) < 2e9]  # sin contar la carga inicial del modelo
    if warm:
        mean_s = sum(r["wall_s"] for r in warm) / len(warm)
        tok = sum(r["eval_count"] for r in warm) / sum(r["eval_duration"] for r in warm) * 1e9
        say(f"Con el modelo ya cargado: {mean_s:.1f} s por informe de media, {tok:.1f} tokens/s de salida")
        say(f"Extrapolación a 4,407 informes en una GPU: {mean_s * 4407 / 3600:.1f} horas "
            f"(estimación gruesa a partir de {len(warm)} informes)")
    (args.out / "prueba_resumen.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
