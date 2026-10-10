#!/usr/bin/env python
"""Extrae de cada informe el estado y la gravedad de los 12 hallazgos con un modelo local (Ollama).

No decide la etiqueta final: guarda lo que el informe dice (estado + gravedad) para poder elegir
después dónde cortar entre positivo y negativo (ver evaluar_extraccion.py).

Uso (dentro de un job, con `ollama serve` escuchando):
  python etiquetas/extraer.py --host 127.0.0.1:11434 --model qwen2.5:72b --subset labeled --out DIR
Salida: DIR/extraccion.jsonl, una línea por informe. Es reanudable: los estudios ya presentes se saltan.
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
PROMPT_VERSION = "v2-estado-gravedad"

# clave del esquema -> columna de train.csv
FINDINGS = {
    "acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA", "lateral_oa": "Lateral OA", "pf_oa": "PF OA", "effusion": "Effusion",
    "synovitis": "Synovitis", "bakers": "Baker's", "contusion": "Contusion", "fracture": "Fracture",
}
STATUS = ["present", "possible", "absent", "not_mentioned"]
SEVERITY = ["mild", "moderate", "severe", "unspecified", "na"]

SYSTEM = """You extract structured findings from knee MRI radiology reports. The report may be written in any language. Use only what the report states; never infer a finding the report does not describe.

For each of the twelve findings return:
- status:
  "present" - the report states the finding.
  "possible" - the report raises it as suspected, possible, "cannot be excluded", "rule out", or as one option of a differential diagnosis.
  "absent" - the report states that the structure is normal or that the finding is not there. A general statement such as "ligaments intact", "menisci normal" or "cartilage normal" counts for every finding it covers.
  "not_mentioned" - the report says nothing that bears on the finding.
- severity: only when status is "present" or "possible"; otherwise "na". Use "unspecified" when the report gives no grade, size or qualifier.

Findings and severity scales:
- acl, mcl: injury of the anterior cruciate / medial collateral ligament (sprain, partial tear, complete tear, avulsion). Mucoid degeneration, a ganglion cyst or fluid next to an intact ligament is not an injury: "absent". mild = grade 1, sprain, low-grade or interstitial partial tear; moderate = grade 2, partial tear; severe = high-grade partial tear, complete tear or rupture.
- medial_meniscus, lateral_meniscus: meniscal tear. Degeneration, intrasubstance signal that does not reach the surface, extrusion alone or a discoid shape alone is not a tear: "absent". mild = small, incomplete or low-grade tear; moderate = a definite tear that is not displaced; severe = complex, root, bucket-handle or displaced tear.
- medial_oa, lateral_oa, pf_oa: osteoarthritis or degenerative cartilage disease of the medial tibiofemoral, lateral tibiofemoral or patellofemoral compartment (chondropathy, chondromalacia, chondrosis, cartilage thinning, fissures, ulcers or defects, osteophytes, joint space narrowing). "Tricompartmental" applies to all three. An acute traumatic osteochondral lesion alone is not osteoarthritis. mild = grade 1-2, superficial, low-grade, incipient; moderate = grade 3, deep partial-thickness, moderate; severe = grade 4, full-thickness cartilage loss, advanced.
- effusion: joint effusion. "Physiologic" or "no significant" fluid is "absent". mild = small, minimal, slight, trace; moderate = moderate or unqualified "some"; severe = large, massive.
- synovitis: synovitis or synovial thickening, proliferation or hypertrophy. mild, moderate, severe as worded.
- bakers: Baker's (popliteal) cyst. mild = small, trace or under 2 cm; moderate = 2 to 4 cm; severe = over 4 cm.
- contusion: bone contusion, bone bruise or traumatic bone marrow edema. Marrow edema that the report attributes to degeneration, overload or an adjacent fracture line only is not a contusion: "absent". mild, moderate, severe by the extent the report words.
- fracture: any fracture, including subchondral, insufficiency, impaction, avulsion and osteochondral fractures. mild = subchondral, insufficiency, trabecular or impaction fracture without displacement; moderate = non-displaced cortical or avulsion fracture; severe = displaced, depressed or comminuted fracture."""

SCHEMA = {
    "type": "object",
    "properties": {
        key: {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": STATUS},
                           "severity": {"type": "string", "enum": SEVERITY}},
            "required": ["status", "severity"],
        }
        for key in FINDINGS
    },
    "required": list(FINDINGS),
}


def chat(host: str, model: str, report: str, timeout: int) -> dict:
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 700},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"Report:\n\"\"\"\n{' '.join(str(report).split())}\n\"\"\""}],
    }
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    # El servidor es local al job: nunca pasar por un proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def valid(parsed: dict) -> bool:
    return all(parsed.get(k, {}).get("status") in STATUS and parsed.get(k, {}).get("severity") in SEVERITY
               for k in FINDINGS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    ap.add_argument("--model", default="qwen2.5:72b")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", default="labeled", choices=["labeled", "unlabeled", "all"])
    ap.add_argument("--limit", type=int, default=0, help="máximo de informes a procesar en esta ejecución; 0 = todos")
    ap.add_argument("--timeout", type=int, default=900, help="segundos por informe (el primero incluye la carga del modelo)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "extraccion.jsonl"

    df = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str})
    labeled = df[list(FINDINGS.values())].notna().all(axis=1)
    if args.subset == "labeled":
        df = df[labeled]
    elif args.subset == "unlabeled":
        df = df[~labeled]

    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if "error" not in rec:
                done.add(rec["StudyInstanceUID"])
    todo = df[~df.StudyInstanceUID.isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"modelo={args.model} prompt={PROMPT_VERSION} subset={args.subset} ya_hechos={len(done)} por_hacer={len(todo)}", flush=True)

    t_start, n_ok, n_bad, gen_tokens, gen_ns = time.time(), 0, 0, 0, 0
    with open(path, "a") as fh:
        for i, rec in enumerate(todo.itertuples(index=False), 1):
            record = {"StudyInstanceUID": rec.StudyInstanceUID, "model": args.model,
                      "prompt_version": PROMPT_VERSION, "n_chars": len(str(rec.Report))}
            t0 = time.time()
            try:
                resp = chat(args.host, args.model, rec.Report, args.timeout)
                record["raw"] = resp.get("message", {}).get("content", "")
                for k in ["load_duration", "prompt_eval_count", "eval_count", "eval_duration", "done_reason"]:
                    record[k] = resp.get(k)
                record["parsed"] = json.loads(record["raw"])
                if not valid(record["parsed"]):
                    raise ValueError("la respuesta no cumple el esquema")
                n_ok += 1
                gen_tokens += record["eval_count"] or 0
                gen_ns += record["eval_duration"] or 0
            except Exception as exc:  # un informe problemático no debe detener el lote
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
                n_bad += 1
            record["wall_s"] = round(time.time() - t0, 2)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if i <= 3 or i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {record['wall_s']} s  salida={record.get('eval_count')} tokens  "
                      f"{'ERROR ' + record['error'] if 'error' in record else 'ok'}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n== {n_ok} correctos, {n_bad} con error, {elapsed / 60:.1f} min en total", flush=True)
    if n_ok:
        print(f"{elapsed / (n_ok + n_bad):.1f} s por informe incluida la carga; "
              f"{gen_tokens / n_ok:.0f} tokens de salida por informe; "
              f"{gen_tokens / gen_ns * 1e9 if gen_ns else float('nan'):.1f} tokens/s", flush=True)
    elif len(todo):
        raise SystemExit("Ningún informe se procesó correctamente; revisar el servidor de Ollama.")


if __name__ == "__main__":
    main()
