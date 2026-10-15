#!/usr/bin/env python
"""Reetiquetado de los informes de train.csv con 3 lectores LLM locales (Ollama) y validación en los 58 Gold.

Documentación y criterio: experiments/reportes/llm_reetiquetado.md.

Por informe y lector, el modelo devuelve para cada uno de los 12 hallazgos:
  status  present / possible / absent / not_addressed   ("no abordado" explícito)
  p       entero 0-100: probabilidad de que un radiólogo que ve las imágenes marque el hallazgo
Salida JSON forzada con el `format` (esquema) de Ollama, temperatura 0, semilla 0.

Fases (en este orden, dentro de un mismo job):
  1. gold   los 58 estudios Gold con cada lector -> gold_eval.json (AUC por hallazgo y macro, IC bootstrap,
            comparación pareada con v1). Si el consenso no cumple el criterio y --parar-si-falla, termina aquí.
  2. resto  los 4,349 no Gold, lector a lector (orden de --lectores), hasta --deadline. Reanudable: cada
            respuesta va a <cache>/<lector>.jsonl y una corrida posterior salta lo ya hecho.
Al final (o con --solo-agregar) escribe etiquetas_llm.parquet con lo que haya.

Las definiciones de los hallazgos son las del extractor de Owen (code/etiquetas/extraer.py, v2-estado-gravedad),
con "not_mentioned" renombrado a "not_addressed" y la gravedad sustituida por la probabilidad p.

Uso:  python rsna/llm_reetiquetar.py --host 127.0.0.1:PUERTO --out DIR --cache DIR [--mock]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA = Path(os.environ.get("KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))
V1 = Path("/lustre/proyectos/p082/preproc/etiquetas_v1_2079203.parquet")
PROMPT_VERSION = "llm3-p-na-v1"
FINDINGS = {
    "acl": "ACL", "mcl": "MCL", "medial_meniscus": "Medial Meniscus", "lateral_meniscus": "Lateral Meniscus",
    "medial_oa": "Medial OA", "lateral_oa": "Lateral OA", "pf_oa": "PF OA", "effusion": "Effusion",
    "synovitis": "Synovitis", "bakers": "Baker's", "contusion": "Contusion", "fracture": "Fracture",
}
LAB = list(FINDINGS.values())
STATUS = ["present", "possible", "absent", "not_addressed"]
# Puntuación secundaria fijada antes de ver resultados (hilo 733932: confirmado ~0.95, sospecha ~0.75-0.85,
# no abordado ~0.03-0.08); absent por debajo de not_addressed.
STATUS_P = {"present": 0.95, "possible": 0.75, "not_addressed": 0.06, "absent": 0.02}

SYSTEM = """You read knee MRI radiology reports and score twelve findings. The report may be written in any language.

For each of the twelve findings return:
- status:
  "present" - the report states the finding.
  "possible" - the report raises it as suspected, possible, "cannot be excluded", "rule out", or as one option of a differential diagnosis.
  "absent" - the report states that the structure is normal or that the finding is not there. A general statement such as "ligaments intact", "menisci normal" or "cartilage normal" counts for every finding it covers.
  "not_addressed" - the report says nothing that bears on the finding. Do not turn silence into "absent".
- p: an integer from 0 to 100, your probability that a radiologist looking at the images would mark this finding as present. Base it on what the report states; when the finding is not addressed, use how often it accompanies what the report does describe (for example synovitis is often under-reported when there is a joint effusion) and its usual frequency in knee MRI.

Findings:
- acl, mcl: injury of the anterior cruciate / medial collateral ligament (sprain, partial tear, complete tear, avulsion). Mucoid degeneration, a ganglion cyst or fluid next to an intact ligament is not an injury.
- medial_meniscus, lateral_meniscus: meniscal tear. Degeneration, intrasubstance signal that does not reach the surface, extrusion alone or a discoid shape alone is not a tear.
- medial_oa, lateral_oa, pf_oa: osteoarthritis or degenerative cartilage disease of the medial tibiofemoral, lateral tibiofemoral or patellofemoral compartment (chondropathy, chondromalacia, chondrosis, cartilage thinning, fissures, ulcers or defects, osteophytes, joint space narrowing). "Tricompartmental" applies to all three. An acute traumatic osteochondral lesion alone is not osteoarthritis.
- effusion: joint effusion. "Physiologic" or "no significant" fluid counts as absent.
- synovitis: synovitis or synovial thickening, proliferation or hypertrophy.
- bakers: Baker's (popliteal) cyst.
- contusion: bone contusion, bone bruise or traumatic bone marrow edema. Marrow edema that the report attributes to degeneration, overload or an adjacent fracture line only is not a contusion.
- fracture: any fracture, including subchondral, insufficiency, impaction, avulsion and osteochondral fractures."""

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "object",
                       "properties": {"status": {"type": "string", "enum": STATUS},
                                      "p": {"type": "integer", "minimum": 0, "maximum": 100}},
                       "required": ["status", "p"]} for k in FINDINGS},
    "required": list(FINDINGS),
}


# ------------------------------- lectura -----------------------------------------------------------
def chat(host: str, model: str, report: str, timeout: int, think: bool | None) -> dict:
    body = {"model": model, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 400},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Report:\n\"\"\"\n{' '.join(str(report).split())}\n\"\"\""}]}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(f"http://{host}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))     # servidor local al job: sin proxy
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat_mock(model: str, report: str) -> dict:
    """Respuesta falsa determinista para probar el cableado sin modelo (palabras clave del informe)."""
    txt = str(report).lower()
    claves = {"acl": ["cruzado anterior", "acl", "lca"], "mcl": ["colateral medial", "lli", "mcl"],
              "medial_meniscus": ["menisco interno", "menisco medial", "medial meniscus"],
              "lateral_meniscus": ["menisco externo", "menisco lateral", "lateral meniscus"],
              "medial_oa": ["artrosis femorotibial medial", "condropat"], "lateral_oa": ["femorotibial lateral", "externa"],
              "pf_oa": ["femoropatelar", "rotul"], "effusion": ["derrame", "effusion", "épanchement"],
              "synovitis": ["sinovitis", "synovit"], "bakers": ["baker", "poplíte"], "contusion": ["contusi", "edema óseo"],
              "fracture": ["fractura", "fracture"]}
    h = int(hashlib.md5((model + txt).encode()).hexdigest(), 16)
    out = {}
    for i, (k, ws) in enumerate(claves.items()):
        hit = any(w in txt for w in ws)
        out[k] = {"status": "present" if hit else ("not_addressed" if (h >> i) % 3 else "absent"),
                  "p": 85 + (h >> (i + 3)) % 10 if hit else (h >> (i + 5)) % 15}
    return {"message": {"content": json.dumps(out)}, "eval_count": 120, "prompt_eval_count": 600,
            "eval_duration": 1, "load_duration": 0, "done_reason": "stop"}


def valido(d: dict) -> bool:
    return all(isinstance(d.get(k), dict) and d[k].get("status") in STATUS and isinstance(d[k].get("p"), (int, float))
               and 0 <= d[k]["p"] <= 100 for k in FINDINGS)


def leer(a, reader: str, uids: list[str], informes: dict, cache: Path, deadline: float) -> dict:
    """Lee los informes que faltan en <cache>/<reader>.jsonl. Devuelve {uid: registro} de los válidos."""
    path = cache / f"{reader.replace(':', '_').replace('/', '_')}.jsonl"
    hechos = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("prompt_version") == PROMPT_VERSION and "error" not in r:
                hechos[r["uid"]] = r
    pend = [u for u in uids if u not in hechos]
    print(f"[{reader}] {len(uids)} pedidos, {len(uids) - len(pend)} en caché, {len(pend)} por leer", flush=True)
    if not pend:
        return {u: hechos[u] for u in uids if u in hechos}
    think = False if "qwen3" in reader else None
    lock, t0, n_ok, n_err = threading.Lock(), time.time(), 0, 0
    parar = threading.Event()

    def uno(u):
        if parar.is_set() or time.time() > deadline:
            parar.set()
            return None
        rec = {"uid": u, "lector": reader, "prompt_version": PROMPT_VERSION}
        t = time.time()
        try:
            nonlocal think
            try:
                resp = chat_mock(reader, informes[u]) if a.mock else chat(a.host, reader, informes[u], a.timeout, think)
            except urllib.error.HTTPError as exc:          # modelo sin "think": se reintenta sin el campo
                if think is not None and "think" in exc.read().decode(errors="ignore").lower():
                    think = None
                    resp = chat(a.host, reader, informes[u], a.timeout, think)
                else:
                    raise
            d = json.loads(resp["message"]["content"])
            if not valido(d):
                raise ValueError("respuesta fuera del esquema")
            rec.update({"r": {FINDINGS[k]: [d[k]["status"], int(round(d[k]["p"]))] for k in FINDINGS},
                        "seg": round(time.time() - t, 2)})
            for k in ("eval_count", "prompt_eval_count", "eval_duration", "load_duration", "done_reason"):
                rec[k] = resp.get(k)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return rec

    with open(path, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(uno, u) for u in pend]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            if rec is None:
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if "error" in rec:
                    n_err += 1
                else:
                    n_ok += 1
                    hechos[rec["uid"]] = rec
            if i in (1, 10, 50) or i % 200 == 0:
                print(f"  [{reader}] {i}/{len(pend)} ok={n_ok} err={n_err} {(time.time() - t0) / i:.2f} s/informe "
                      f"(último {rec.get('seg')} s, {rec.get('eval_count')} tokens)", flush=True)
    if parar.is_set():
        print(f"  [{reader}] cortado por el plazo: {n_ok} nuevos", flush=True)
    print(f"[{reader}] fin: ok={n_ok} err={n_err} en {(time.time() - t0) / 60:.1f} min", flush=True)
    return {u: hechos[u] for u in uids if u in hechos}


# ------------------------------- agregación y evaluación -------------------------------------------
def tabla(lecturas: dict[str, dict], uids: list[str]) -> pd.DataFrame:
    """Una fila por estudio: por hallazgo p y status de cada lector, consenso, no abordado y dispersión."""
    filas = []
    for u in uids:
        f = {"StudyInstanceUID": u}
        for c in LAB:
            ps, nas, ss = [], [], []
            for r, rec in lecturas.items():
                x = rec.get(u)
                if x is None:
                    f[f"p_{r}|{c}"], f[f"status_{r}|{c}"] = np.nan, None
                    continue
                st, p = x["r"][c]
                f[f"p_{r}|{c}"], f[f"status_{r}|{c}"] = p / 100, st
                ps.append(p / 100); nas.append(st == "not_addressed"); ss.append(STATUS_P[st])
            f[f"n|{c}"] = len(ps)
            f[f"p|{c}"] = float(np.mean(ps)) if ps else np.nan                  # consenso (decide)
            f[f"p_status|{c}"] = float(np.mean(ss)) if ss else np.nan           # puntuación secundaria
            f[f"na_frac|{c}"] = float(np.mean(nas)) if nas else np.nan
            f[f"no_abordado|{c}"] = bool(np.mean(nas) >= 0.5) if nas else None  # mayoría de lectores
            f[f"disp|{c}"] = float(max(ps) - min(ps)) if len(ps) > 1 else np.nan
            f[f"sd|{c}"] = float(np.std(ps)) if len(ps) > 1 else np.nan
        filas.append(f)
    return pd.DataFrame(filas)


def macro(y, s, m):
    aucs = []
    for j in range(y.shape[1]):
        k = m[:, j]
        yy = y[k, j]
        aucs.append(roc_auc_score(yy, s[k, j]) if 0 < yy.sum() < len(yy) else np.nan)
    return float(np.nanmean(aucs)), aucs


def evaluar_gold(tab: pd.DataFrame, lectores: list[str], n_boot: int = 1000, seed: int = 20261014) -> dict:
    v1 = pd.read_parquet(V1).set_index("StudyInstanceUID")
    g = v1[v1.oficial]
    t = tab.set_index("StudyInstanceUID").loc[g.index]
    y = g[[f"y_{c}" for c in LAB]].to_numpy().astype(int)
    v1t = g[[f"t_{c}" for c in LAB]].to_numpy(float)
    v1m = (g[[f"w_{c}" for c in LAB]].to_numpy(float) > 0) & np.isfinite(v1t)
    todo = np.ones_like(y, bool)
    scores = {"consenso": t[[f"p|{c}" for c in LAB]].to_numpy(float),
              "consenso_status": t[[f"p_status|{c}" for c in LAB]].to_numpy(float)}
    for r in lectores:
        scores[r] = t[[f"p_{r}|{c}" for c in LAB]].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]

    def ic(s, m):
        v = np.array([macro(y[i], s[i], m[i])[0] for i in idx])
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    out = {"n_gold": int(len(y)), "positivos": dict(zip(LAB, y.sum(0).tolist())), "bootstrap": {"replicas": n_boot, "semilla": seed},
           "nota_v1": "v1 se calibró con estos 58 Gold: su AUC aquí está sesgado al alza"}
    for nombre, s in scores.items():
        cub = np.isfinite(s)
        if not cub.all():
            out[nombre] = {"error": f"{int((~cub).sum())} celdas sin lectura"}
            continue
        m, per = macro(y, s, todo)
        out[nombre] = {"macro_58": m, "ic95_58": ic(s, todo), "por_hallazgo_58": dict(zip(LAB, per)),
                       "macro_celdas_v1": macro(y, s, v1m)[0]}
    mv1, per_v1 = macro(y, np.nan_to_num(v1t), v1m)
    out["v1"] = {"macro_celdas_v1": mv1, "ic95_celdas_v1": ic(np.nan_to_num(v1t), v1m),
                 "por_hallazgo_celdas_v1": dict(zip(LAB, per_v1)), "celdas_por_hallazgo": dict(zip(LAB, v1m.sum(0).tolist()))}
    if "macro_58" in out.get("consenso", {}):
        s = scores["consenso"]
        d = np.array([macro(y[i], s[i], v1m[i])[0] - macro(y[i], np.nan_to_num(v1t)[i], v1m[i])[0] for i in idx])
        delta = out["consenso"]["macro_celdas_v1"] - mv1
        out["consenso_vs_v1"] = {"delta_celdas_v1": delta, "ic95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                                 "p_delta_pos": float(np.mean(d[np.isfinite(d)] > 0))}
        ok = out["consenso"]["macro_58"] >= 0.88 and delta > 0
        out["criterio"] = {"macro_58_consenso": out["consenso"]["macro_58"], "umbral": 0.88, "delta_vs_v1": delta,
                           "veredicto": "PASA" if ok else "NO PASA"}
    return out


# ------------------------------- principal ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
    ap.add_argument("--lectores", default="qwen3.5:27b,qwen2.5:72b,llama3.3:70b",
                    help="orden = orden de la fase resto (el más rápido primero)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="JSONL por lector, reanudable entre jobs")
    ap.add_argument("--workers", type=int, default=4, help="= OLLAMA_NUM_PARALLEL")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--horas", type=float, default=9.3, help="plazo desde el arranque para la fase resto")
    ap.add_argument("--fases", default="gold,resto")
    ap.add_argument("--parar-si-falla", action="store_true", help="no leer el resto si el consenso no pasa en Gold")
    ap.add_argument("--limite-resto", type=int, default=0, help="pruebas: máximo de no Gold")
    ap.add_argument("--solo-agregar", action="store_true")
    ap.add_argument("--mock", action="store_true", help="sin servidor: respuestas falsas deterministas (cableado)")
    a = ap.parse_args()
    t0 = time.time()
    deadline = t0 + a.horas * 3600
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    lectores = [x for x in a.lectores.split(",") if x]
    df = pd.read_csv(DATA / "train.csv", usecols=["StudyInstanceUID", "Report"], dtype=str)
    informes = dict(zip(df.StudyInstanceUID, df.Report.fillna("")))
    v1 = pd.read_parquet(V1)
    gold = v1.StudyInstanceUID[v1.oficial].tolist()
    resto = sorted(set(informes) - set(gold))
    random.Random(20261014).shuffle(resto)                       # orden fijo: un corte por plazo es una muestra al azar
    if a.limite_resto:
        resto = resto[:a.limite_resto]
    (a.out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "prompt_version": PROMPT_VERSION,
                                                   "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                                                   "status_p": STATUS_P, "n_gold": len(gold), "n_resto": len(resto),
                                                   "job": os.environ.get("SLURM_JOB_ID", "")}, indent=1, ensure_ascii=False))
    fases = a.fases.split(",")
    lect = {}
    veredicto = None
    if "gold" in fases or a.solo_agregar:
        for r in lectores:
            lect[r] = leer(a, r, gold, informes, a.cache, float("inf")) if not a.solo_agregar else leer_cache(a.cache, r, gold)
        ev = evaluar_gold(tabla(lect, gold), lectores)
        (a.out / "gold_eval.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
        veredicto = ev.get("criterio", {}).get("veredicto")
        print("GOLD:", json.dumps({k: (v.get("macro_58") if isinstance(v, dict) and "macro_58" in v else v)
                                   for k, v in ev.items() if k in ["consenso", "consenso_status", "criterio", *lectores]},
                                  ensure_ascii=False), flush=True)
    if "resto" in fases and not a.solo_agregar:
        if a.parar_si_falla and veredicto != "PASA":
            print(f"criterio en Gold: {veredicto}; no se lee el resto (--parar-si-falla)", flush=True)
        else:
            for r in lectores:
                if time.time() > deadline:
                    print(f"plazo agotado antes de {r}", flush=True)
                    break
                leer(a, r, resto, informes, a.cache, deadline)
    todos = gold + resto if "resto" in fases else gold
    lect = {r: leer_cache(a.cache, r, todos) for r in lectores}
    tab = tabla(lect, todos)
    tab.insert(1, "oficial", tab.StudyInstanceUID.isin(set(gold)))
    tab.to_parquet(a.out / "etiquetas_llm.parquet", index=False)
    cob = {r: len(v) for r, v in lect.items()}
    print(f"etiquetas_llm.parquet: {len(tab)} estudios; lecturas por lector {cob}; "
          f"estudios con los 3 lectores {int((tab[f'n|{LAB[0]}'] == len(lectores)).sum())}; {(time.time() - t0) / 3600:.2f} h", flush=True)


def leer_cache(cache: Path, reader: str, uids: list[str]) -> dict:
    path = cache / f"{reader.replace(':', '_').replace('/', '_')}.jsonl"
    hechos = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("prompt_version") == PROMPT_VERSION and "error" not in r:
                hechos[r["uid"]] = r
    s = set(uids)
    return {u: r for u, r in hechos.items() if u in s}


if __name__ == "__main__":
    main()
