#!/usr/bin/env python
"""Prueba sin ajuste de MedGemma 1.5 4B sobre las resonancias de los estudios con etiqueta oficial.

Sigue el preprocesamiento del informe técnico de MedGemma 1.5 (sec. 2.3.1 y 3.2.1): cortes axiales
como imágenes 2D de 896x896, normalización min-max por volumen, como máximo 85 cortes por consulta
muestreados de forma equidistante, y marcas de texto "SLICE n" intercaladas con las imágenes.

Las imágenes nunca salen del nodo: se envían a un llama-server local (127.0.0.1) arrancado en el job.
Solo se imprime la tabla agregada de aciertos.

Uso (dentro de un job, con llama-server escuchando):
  python imagenes/prueba_medgemma.py --host 127.0.0.1:8080 --out DIR [--limit 4] [--mode por_hallazgo|json]
Salidas en DIR: imagenes_respuestas.jsonl (una línea por estudio) e imagenes_evaluacion.txt
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))

# columna de train.csv -> cómo se le pregunta al modelo
FINDINGS = {
    "ACL": "an anterior cruciate ligament (ACL) injury or tear",
    "MCL": "a medial collateral ligament (MCL) injury or tear",
    "Medial Meniscus": "a medial meniscus tear",
    "Lateral Meniscus": "a lateral meniscus tear",
    "Medial OA": "osteoarthritis of the medial tibiofemoral compartment",
    "Lateral OA": "osteoarthritis of the lateral tibiofemoral compartment",
    "PF OA": "patellofemoral osteoarthritis",
    "Effusion": "a joint effusion",
    "Synovitis": "synovitis",
    "Baker's": "a Baker's (popliteal) cyst",
    "Contusion": "a bone contusion (bone bruise)",
    "Fracture": "a fracture",
}
INTRO = "You are an expert musculoskeletal radiologist. Below are the slices of a knee MRI study."
KEYS = {c: c.lower().replace(" ", "_").replace("'", "") for c in FINDINGS}


def load_series(path: Path) -> list[np.ndarray]:
    """Cortes de una serie ordenados por posición a lo largo de la normal del plano."""
    slices = []
    for name in os.listdir(path):
        if not name.endswith(".dcm"):
            continue
        ds = pydicom.dcmread(path / name)
        iop, pos = ds.get("ImageOrientationPatient"), ds.get("ImagePositionPatient")
        if iop is not None and pos is not None:
            iop = np.asarray(iop, dtype=float)
            key = float(np.dot(np.cross(iop[:3], iop[3:]), np.asarray(pos, dtype=float)))
        else:
            key = float(ds.get("InstanceNumber", 0))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            arr = arr.max() - arr
        slices.append((key, arr))
    slices.sort(key=lambda t: t[0])
    return [a for _, a in slices]


def study_images(study_uid: str, series: pd.DataFrame, planes: list[str], max_slices: int, size: int) -> tuple[list[str], dict]:
    """Devuelve los cortes del estudio como PNG en base64 y un resumen de qué se usó."""
    rows = series[(series.StudyInstanceUID == study_uid) & series.Anatomical_Plane.isin(planes)]
    rows = rows.sort_values(["Anatomical_Plane", "Fluid_Sensitive"], ascending=[True, False])
    frames: list[np.ndarray] = []
    used = []
    for row in rows.itertuples():
        vol = load_series(DATA / "train_series" / study_uid / row.SeriesInstanceUID)
        if len(vol) < 5:
            continue
        lo = min(float(a.min()) for a in vol)
        hi = max(float(a.max()) for a in vol)
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        frames += [((a - lo) * scale).astype(np.uint8) for a in vol]   # min-max por volumen
        used.append(f"{row.Anatomical_Plane}/{'FS' if row.Fluid_Sensitive else 'noFS'}:{len(vol)}")
    total = len(frames)
    if total > max_slices:
        idx = np.linspace(0, total - 1, max_slices).round().astype(int)
        frames = [frames[i] for i in idx]
    images = []
    for arr in frames:
        img = Image.fromarray(arr).convert("RGB").resize((size, size), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images.append(base64.b64encode(buf.getvalue()).decode())
    return images, {"series": used, "cortes_totales": total, "cortes_enviados": len(images)}


class Server:
    """llama-server local que se reinicia cada pocos estudios: su memoria crece con cada imagen recibida."""

    def __init__(self, binary: str, model: str, mmproj: str, host: str, log: Path):
        ip, port = host.split(":")
        self.cmd = [binary, "--model", model, "--mmproj", mmproj, "--host", ip, "--port", port, "--no-webui",
                    "--offline", "-c", "32768", "-np", "1", "-ngl", "999", "--jinja",
                    "--ctx-checkpoints", "0", "--cache-ram", "0"]
        self.host, self.log, self.proc, self.starts = host, log, None, 0

    def start(self) -> None:
        self.stop()
        self.proc = subprocess.Popen(self.cmd, stdout=open(self.log, "a"), stderr=subprocess.STDOUT)
        self.starts += 1
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(150):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server terminó al arrancar (código {self.proc.returncode}); ver {self.log}")
            try:
                with opener.open(f"http://{self.host}/health", timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("llama-server no respondió a tiempo")

    def rss_gb(self) -> float:
        try:
            with open(f"/proc/{self.proc.pid}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1048576
        except Exception:
            pass
        return float("nan")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None


def post(host: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(f"http://{host}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # servidor local: nunca por proxy
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def content(images: list[str], question: str) -> list[dict]:
    parts = [{"type": "text", "text": INTRO}]
    for i, b64 in enumerate(images, 1):
        parts.append({"type": "text", "text": f"SLICE {i}"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    parts.append({"type": "text", "text": question})   # la pregunta va al final para reutilizar el prefijo
    return parts


def p_yes_from(resp: dict) -> tuple[str, float]:
    """Respuesta textual y probabilidad de 'yes' a partir de los logprobs del primer token."""
    choice = resp["choices"][0]
    text = (choice["message"].get("content") or "").strip().lower()
    yes = no = 0.0
    try:
        for cand in choice["logprobs"]["content"][0]["top_logprobs"]:
            tok = cand["token"].strip().lower()
            if tok.startswith("yes"):
                yes += math.exp(cand["logprob"])
            elif tok.startswith("no"):
                no += math.exp(cand["logprob"])
    except (KeyError, IndexError, TypeError):
        pass
    if yes + no > 0:
        return text, yes / (yes + no)
    return text, 1.0 if text.startswith("yes") else 0.0


def ask_per_finding(host: str, images: list[str], timeout: int) -> dict:
    out = {}
    for col, desc in FINDINGS.items():
        q = f"Question: does this knee MRI study show {desc}? Answer with a single word: yes or no."
        resp = post(host, {"messages": [{"role": "user", "content": content(images, q)}], "temperature": 0,
                           "max_tokens": 3, "logprobs": True, "top_logprobs": 10, "cache_prompt": True}, timeout)
        text, p = p_yes_from(resp)
        out[col] = {"answer": text, "p_yes": round(p, 4)}
    return out


def ask_json(host: str, images: list[str], timeout: int) -> dict:
    listing = "\n".join(f"- {KEYS[c]}: {d}" for c, d in FINDINGS.items())
    q = ("For each of the following findings, state whether this knee MRI study shows it. "
         f"Answer in JSON with the value \"yes\" or \"no\" for each key.\n{listing}")
    schema = {"type": "object", "required": list(KEYS.values()),
              "properties": {k: {"type": "string", "enum": ["yes", "no"]} for k in KEYS.values()}}
    resp = post(host, {"messages": [{"role": "user", "content": content(images, q)}], "temperature": 0,
                       "max_tokens": 200, "cache_prompt": True,
                       "response_format": {"type": "json_schema", "json_schema": {"name": "findings", "schema": schema}}}, timeout)
    parsed = json.loads(resp["choices"][0]["message"]["content"])
    return {c: {"answer": parsed[KEYS[c]], "p_yes": 1.0 if parsed[KEYS[c]] == "yes" else 0.0} for c in FINDINGS}


def evaluate(records: list[dict], gold: pd.DataFrame, say) -> None:
    from sklearn.metrics import roc_auc_score

    uids = [r["StudyInstanceUID"] for r in records]
    g = gold.loc[uids]
    say(f"\n-- MedGemma sobre imágenes frente a la etiqueta oficial; n = {len(uids)} estudios")
    say(f"{'hallazgo':<17}{'pos.of':>7}{'dice sí':>8}{'acuerdo%':>9}{'sens%':>7}{'espec%':>8}{'AUC':>7}")
    tot_ok = 0
    aucs = []
    for col in FINDINGS:
        y = g[col].to_numpy().astype(int)
        p = np.array([r["respuestas"][col]["p_yes"] for r in records])
        pred = np.array([r["respuestas"][col]["answer"].startswith("yes") for r in records]).astype(int)
        tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        sens = tp / y.sum() * 100 if y.sum() else float("nan")
        spec = tn / (1 - y).sum() * 100 if (1 - y).sum() else float("nan")
        auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) and len(set(p)) > 1 else float("nan")
        if not math.isnan(auc):
            aucs.append(auc)
        tot_ok += tp + tn
        say(f"{col:<17}{int(y.sum()):>7}{int(pred.sum()):>8}{(tp + tn) / len(y) * 100:>9.0f}{sens:>7.0f}{spec:>8.0f}{auc:>7.2f}")
    say(f"Acuerdo global: {tot_ok / (len(uids) * len(FINDINGS)) * 100:.1f} %; "
        f"AUC media de {len(aucs)} hallazgos: {np.mean(aucs) if aucs else float('nan'):.3f}")
    base = np.maximum(g[list(FINDINGS)].mean(), 1 - g[list(FINDINGS)].mean()).mean() * 100
    say(f"Referencia: responder siempre la clase mayoritaria de cada hallazgo acierta el {base:.1f} % en estos estudios")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="número de estudios; 0 = los 58 etiquetados")
    ap.add_argument("--mode", default="por_hallazgo", choices=["por_hallazgo", "json"])
    ap.add_argument("--negative-for", default="",
                    help="hallazgos separados por comas: elige primero los estudios cuya etiqueta oficial es negativa en ellos")
    ap.add_argument("--exclude-first", type=int, default=0,
                    help="descarta los N primeros estudios por orden de identificador (los de una prueba anterior)")
    ap.add_argument("--planes", default="Axial", help="planos separados por comas (el modelo se entrenó con axial)")
    ap.add_argument("--max-slices", type=int, default=85)
    ap.add_argument("--size", type=int, default=896)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--llama-server", default="", help="ruta del binario: si se da, el script arranca y reinicia el servidor")
    ap.add_argument("--model", default="")
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--restart-every", type=int, default=1, help="reiniciar el servidor cada N estudios")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    cols = list(FINDINGS)
    gold = train.loc[train[cols].notna().all(axis=1), cols].astype(int).sort_index()
    series = pd.read_csv(DATA / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    pool = gold.iloc[args.exclude_first:]
    neg_note = ""
    if args.negative_for:
        neg_cols = [c.strip() for c in args.negative_for.split(",")]
        unknown = [c for c in neg_cols if c not in cols]
        if unknown:
            raise SystemExit(f"Hallazgos desconocidos en --negative-for: {unknown}")
        # Primero los negativos en todos esos hallazgos; a igualdad, los de menos positivos en total.
        order = pd.DataFrame({"neg": pool[neg_cols].sum(axis=1), "tot": pool.sum(axis=1)}).sort_values(["neg", "tot"], kind="stable")
        pool = pool.loc[order.index]
        neg_note = (f" | negativos oficiales en {neg_cols}: {int((order.neg == 0).sum())} disponibles de {len(order)}"
                    f"; positivos en esos hallazgos entre los elegidos: "
                    f"{order.neg.iloc[: args.limit or len(order)].astype(int).tolist()}")
    uids = list(pool.index[: args.limit] if args.limit else pool.index)

    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    say(f"modo={args.mode} planos={args.planes} max_cortes={args.max_slices} tamaño={args.size} estudios={len(uids)}"
        f" excluidos_primeros={args.exclude_first}{neg_note}")
    server = Server(args.llama_server, args.model, args.mmproj, args.host, args.out / "llama_server.log") if args.llama_server else None
    records, n_err, max_rss = [], 0, 0.0
    with open(args.out / "imagenes_respuestas.jsonl", "w") as fh:
        for i, uid in enumerate(uids, 1):
            rec = {"StudyInstanceUID": uid, "mode": args.mode}
            t0 = time.time()
            try:
                if server and (i - 1) % max(1, args.restart_every) == 0:
                    server.start()
                    t0 = time.time()   # el arranque del servidor no cuenta como preparación de imágenes
                images, info = study_images(uid, series, args.planes.split(","), args.max_slices, args.size)
                rec.update(info)
                rec["prep_s"] = round(time.time() - t0, 1)
                if not images:
                    raise ValueError("el estudio no tiene series utilizables en esos planos")
                t1 = time.time()
                rec["respuestas"] = (ask_per_finding if args.mode == "por_hallazgo" else ask_json)(args.host, images, args.timeout)
                rec["modelo_s"] = round(time.time() - t1, 1)
                records.append(rec)
            except Exception as exc:  # un estudio problemático no debe detener la prueba
                rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
                n_err += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            rss = server.rss_gb() if server and server.proc else float("nan")
            max_rss = max(max_rss, rss) if rss == rss else max_rss
            if i <= 3 or i % 10 == 0 or i == len(uids) or "error" in rec:
                say(f"[{i}/{len(uids)}] cortes={rec.get('cortes_enviados')} de {rec.get('cortes_totales')}  "
                    f"prep={rec.get('prep_s')} s  modelo={rec.get('modelo_s')} s  servidor={rss:.1f} GB  "
                    f"{'ERROR ' + rec['error'] if 'error' in rec else 'ok'}")
    if server:
        say(f"Servidor: {server.starts} arranques; memoria máxima observada {max_rss:.1f} GB")
        server.stop()

    say(f"\n== {len(records)} estudios correctos, {n_err} con error")
    if records:
        say(f"Tiempo medio por estudio: preparación {np.mean([r['prep_s'] for r in records]):.1f} s, "
            f"modelo {np.mean([r['modelo_s'] for r in records]):.1f} s; "
            f"cortes enviados por estudio: {np.mean([r['cortes_enviados'] for r in records]):.0f}")
        evaluate(records, gold, say)
    (args.out / "imagenes_evaluacion.txt").write_text("\n".join(lines) + "\n")
    if not records:
        raise SystemExit("Ningún estudio se procesó correctamente.")


if __name__ == "__main__":
    main()
