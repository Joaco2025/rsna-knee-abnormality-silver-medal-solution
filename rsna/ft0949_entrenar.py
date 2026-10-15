#!/usr/bin/env python
"""Fine-tuning corto del checkpoint público 0.949 de Raptor (CoAtNet + RaptorClassifier), all-data.

HEREDA OAI (el checkpoint se entrenó con 2,399 rodillas OAI): el resultado solo vale para la Final A.
Documentación y criterio: experiments/reportes/ft0949.md.

  inicio      raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt (sha256 verificado), strict=True
  datos       corpus native384dense-v10 (rsna/raptor_corpus.py) en el /scratch del nodo
  modos       v1: recorte central del 80 % (307 px) del 384 y reescalado bilineal a 224; backbone img_size=224
              v2: recorte central de 320 px del 384 nativo, sin reescalado; backbone img_size=320
              (v2 es la entrada con la que el 0.957 infiere este checkpoint: INPUT_RES=320, INPUT_MODE="crop")
  receta      la del #53 (rsna/entrenar_coatnet.py) salvo: 5 épocas, lr backbone 1e-5, lr cabeza 2e-4,
              OneCycle con 10 % de calentamiento, all-data (4,349 menos los que no tienen corpus u objetivo)
  selección   ninguna con Gold: se guardan la última época y el promedio uniforme de las 3 últimas,
              en fp16 y sin optimizador
  Gold        AUC solo como diagnóstico (contaminado: los Gold eligieron las épocas del SWA del 0.949)
  clon        Spearman por hallazgo sobre los 58 Gold frente al checkpoint base, media de los 12

Uso:  python rsna/ft0949_entrenar.py --corpus DIR --out DIR --modo v1|v2 [--smoke]
"""
from __future__ import annotations

import os

os.environ.setdefault("TIMM_FUSED_ATTN", "0")      # antes de importar timm: atención sin fusionar (bench v2)

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_coatnet import (LAB, MEAN, STD, RaptorClassifier, Ventanas, macro_auc,  # noqa: E402
                              un_estudio)

P = Path("/lustre/proyectos/p082")
CKPT = P / "code_joako/referencias/rsna_research/pesos/raptor_0949/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
CKPT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
IMG = 384
MODOS = {"v1": {"res": 224, "crop": round(0.8 * IMG)},   # 307 px -> 224
         "v2": {"res": 320, "crop": 320}}               # 320 px, sin reescalado


def preparar(x: torch.Tensor, g: torch.Tensor, device, modo: str) -> torch.Tensor:
    """uint8 (K,3,384,384) -> (1,K,3,res,res) normalizada. Ganancia en [0,1], recorte central, reescalado
    bilineal (solo v1) y normalización ImageNet, en ese orden (el 0.957 reescala antes de normalizar)."""
    m = MODOS[modo]
    x = x.to(device, non_blocking=True).float() / 255.0
    x = (x * g.to(device)).clamp(0, 1)
    lo = (IMG - m["crop"]) // 2
    x = x[:, :, lo:lo + m["crop"], lo:lo + m["crop"]]
    if m["crop"] != m["res"]:
        x = F.interpolate(x, size=(m["res"], m["res"]), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std)[None]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def construir(ck: dict, res: int) -> RaptorClassifier:
    import timm
    bb = timm.create_model(ck["arch"], pretrained=False, num_classes=0, global_pool="avg", in_chans=3, img_size=res)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)   # si no casa, el job muere aquí con la lista de claves
    return model


@torch.no_grad()
def predecir(model, loader, device, chunk: int, modo: str) -> np.ndarray:
    model.eval()
    ps = []
    for x, g, _, _ in loader:
        x = preparar(x, g, device, modo)[0]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats = torch.cat([model.backbone(x[i:i + chunk]) for i in range(0, len(x), chunk)], 0)
            logits = model.head(feats[None])
        ps.append(torch.sigmoid(logits.float())[0].cpu().numpy())
    return np.stack(ps)


def spearman_medio(a: np.ndarray, b: np.ndarray) -> tuple[float, dict]:
    per = {}
    for j, c in enumerate(LAB):
        r = spearmanr(a[:, j], b[:, j]).correlation
        per[c] = float(r) if np.isfinite(r) else float("nan")
    return float(np.nanmean(list(per.values()))), per


def fp16_sd(sd: dict) -> dict:
    return {k: (v.half() if v.is_floating_point() else v.clone()) for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modo", choices=sorted(MODOS), required=True)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--labels", type=Path, default=P / "runs_joako/etiquetas_coatnet_20261014/etiquetas_blendjev.parquet")
    ap.add_argument("--juez-v1", type=Path, default=P / "preproc/etiquetas_v1_2079203.parquet")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--bb-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0588)
    ap.add_argument("--pct-start", type=float, default=0.10)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--promedio", type=int, default=3, help="épocas finales que se promedian")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-horas", type=float, default=0, help="presupuesto; no empieza una época que no quepa")
    ap.add_argument("--abortar-saltos", type=float, default=0.25)
    ap.add_argument("--umbral-clon", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true", help="cableado: pocos estudios, 3 épocas mínimas")
    ap.add_argument("--smoke-pasos", type=int, default=0, help="con --smoke: corta cada época tras N pasos")
    a = ap.parse_args()
    t_inicio = time.time()
    if a.smoke:
        a.epochs, a.k, a.k_eval, a.accum = 3, 4, 6, 1
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "preds").mkdir(exist_ok=True)
    res = MODOS[a.modo]["res"]

    # ---- checkpoint base ----
    sha = sha256(a.ckpt)
    if sha != CKPT_SHA:
        raise SystemExit(f"sha256 del checkpoint {sha} != {CKPT_SHA}")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    model = construir(ck, res).to(device)

    # ---- estudios y etiquetas (all-data; los Gold nunca entrenan) ----
    lab = pd.read_parquet(a.labels).set_index("StudyInstanceUID")
    v1 = pd.read_parquet(a.juez_v1).set_index("StudyInstanceUID")
    lab = lab.loc[v1.index]
    idx = pd.read_csv(a.corpus / "indice.csv", dtype={"uid": str}).set_index("uid")
    ok = set(idx.index[(idx.error.fillna("") == "") & (idx.n_cortes >= 3)])
    gold_all = v1.index[v1.oficial.to_numpy()].tolist()
    rest = v1.index[~v1.oficial.to_numpy()].tolist()
    wcols = [f"w_{c}" for c in LAB]
    train = [u for u in rest if u in ok and lab.loc[u, wcols].sum() > 0]
    gold = [u for u in gold_all if u in ok]
    if a.smoke:
        train, gold = train[:16], gold[:8]
    t_ = {u: np.nan_to_num(lab.loc[u, [f"t_{c}" for c in LAB]].to_numpy(np.float32)) for u in lab.index}
    w_ = {u: lab.loc[u, wcols].to_numpy(np.float32) for u in lab.index}
    excluidos = {"train_sin_corpus": len([u for u in rest if u not in ok]),
                 "train_sin_objetivo": len([u for u in rest if u in ok and lab.loc[u, wcols].sum() == 0]),
                 "gold_sin_corpus": len(gold_all) - len(gold)}

    T = np.stack([t_[u] for u in train]); W = np.stack([w_[u] for u in train])
    prev = np.array([T[W[:, j] > 0, j].mean() if (W[:, j] > 0).any() else np.nan for j in range(12)])
    prev = np.where(np.isfinite(prev), prev, 0.5)    # solo ocurre en --smoke (hallazgo sin celdas con objetivo)
    prev_c = np.clip(prev, 0.03, 0.7)
    pw_np = np.clip((1 - prev_c) / prev_c, 1, 10)
    pw = torch.tensor(pw_np, dtype=torch.float32, device=device)

    steps_ep = len(train) // a.accum
    if a.smoke and a.smoke_pasos:
        steps_ep = min(steps_ep, a.smoke_pasos)
    total_steps = steps_ep * a.epochs
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update({"ckpt_sha256": sha, "ckpt_epocas_swa": ck.get("swa_over"), "ckpt_gold_auc_guardado": ck.get("gold_auc"),
                "hereda_oai": True, "solo_final": "A", "res": res, "crop_px": MODOS[a.modo]["crop"],
                "n_train": len(train), "n_gold": len(gold), "excluidos": excluidos,
                "pasos_por_epoca": steps_ep, "pasos_totales": total_steps,
                "prevalencia": dict(zip(LAB, prev.round(4).tolist())), "pos_weight": dict(zip(LAB, pw_np.round(3).tolist())),
                "commit": os.environ.get("GIT_COMMIT", ""), "job": os.environ.get("SLURM_JOB_ID", ""),
                "torch": torch.__version__, "TIMM_FUSED_ATTN": os.environ.get("TIMM_FUSED_ATTN")})
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    print(json.dumps(cfg, ensure_ascii=False), flush=True)

    def loader(uids, k, train_mode):
        return DataLoader(Ventanas(uids, a.corpus, t_, w_, k, train_mode, a.jitter), batch_size=1,
                          shuffle=train_mode, num_workers=a.workers, collate_fn=un_estudio, pin_memory=amp,
                          persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    dl_train = loader(train, a.k, True)
    dl_gold = loader(gold, a.k_eval, False)
    Y_gold = v1.loc[gold, [f"y_{c}" for c in LAB]].to_numpy().astype(int)

    # ---- referencia: el checkpoint base en el mismo modo y en el modo del 0.957 (v2) ----
    base = {a.modo: predecir(model, dl_gold, device, a.chunk, a.modo)}
    if a.modo != "v2":
        m2 = construir(ck, MODOS["v2"]["res"]).to(device)
        base["v2"] = predecir(m2, dl_gold, device, a.chunk, "v2")
        del m2
    del ck
    for md, pb in base.items():
        np.savez(a.out / "preds" / f"base_{md}.npz", gold=pb, gold_ids=np.array(gold))
    ref = {f"base_gold_{md}": macro_auc(Y_gold, pb)[0] for md, pb in base.items()}
    print(f"base: {ref}", flush=True)

    head_p = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb_p = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb_p, "lr": a.bb_lr}, {"params": head_p, "lr": a.head_lr}],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=total_steps,
                                                pct_start=a.pct_start)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def evaluar(tag: str) -> dict:
        t0 = time.time()
        pg = predecir(model, dl_gold, device, a.chunk, a.modo)
        np.savez(a.out / "preds" / f"{tag}.npz", gold=pg, gold_ids=np.array(gold))
        m, per = macro_auc(Y_gold, pg)
        r = {"gold_diagnostico": m, "gold_por_hallazgo": dict(zip(LAB, per)),
             "segundos_eval": round(time.time() - t0, 1), "no_finitas_pred": int((~np.isfinite(pg)).sum())}
        for md, pb in base.items():
            s, sp = spearman_medio(pg, pb)
            r[f"spearman_vs_base_{md}"] = s
            r[f"spearman_vs_base_{md}_por_hallazgo"] = sp
        return r

    hist, ultimas, abortado = [], [], None
    for ep in range(a.epochs):
        if a.max_horas and hist:
            dur = max(h["segundos_epoca"] + h["metricas"]["segundos_eval"] for h in hist)
            if time.time() - t_inicio + dur * 1.1 + 1200 > a.max_horas * 3600:
                abortado = f"presupuesto: no cabe la época {ep}"
                print(abortado, flush=True)
                break
        model.train()
        t0 = time.time()
        loss_sum, n_seen, n_nonfinite, skipped, gn_list = 0.0, 0, 0, 0, []
        opt.zero_grad(set_to_none=True)
        for i, (x, g, t, w) in enumerate(dl_train):
            if i >= steps_ep * a.accum:
                break
            x = preparar(x, g, device, a.modo)
            t, w = t.to(device), w.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            if not torch.isfinite(logits).all():
                n_nonfinite += 1
            cell = F.binary_cross_entropy_with_logits(logits.float(), t, pos_weight=pw, reduction="none")
            loss = (cell * w).sum() / (12 * x.shape[0])
            scaler.scale(loss / a.accum).backward()
            if math.isfinite(loss.item()):
                loss_sum += loss.item()
            n_seen += 1
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                escala = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < escala:
                    skipped += 1
                else:
                    gn_list.append(float(gn))
                opt.zero_grad(set_to_none=True)
                sched.step()
                paso = (i + 1) // a.accum
                if paso in (1, 10, 50) or paso % 100 == 0:
                    print(f"  ép {ep} paso {paso}/{steps_ep} pérdida {loss_sum / max(n_seen, 1):.4f} "
                          f"lr_bb {sched.get_last_lr()[0]:.2e} lr_cab {sched.get_last_lr()[1]:.2e} "
                          f"escala {scaler.get_scale():.0f} |g| {float(gn):.2f} "
                          f"{n_seen / (time.time() - t0):.2f} estudios/s", flush=True)
        seg = time.time() - t0
        met = evaluar(f"ep{ep}")
        h = {"epoca": ep, "perdida": loss_sum / max(n_seen, 1), "estudios": n_seen, "pasos": steps_ep,
             "pasos_saltados_gradscaler": skipped, "pct_saltados_gradscaler": round(100 * skipped / max(steps_ep, 1), 2),
             "logits_no_finitos": n_nonfinite,
             "gradnorm_mediana": float(np.median(gn_list)) if gn_list else None,
             "gradnorm_p99": float(np.percentile(gn_list, 99)) if gn_list else None,
             "escala_final": scaler.get_scale(), "segundos_epoca": round(seg, 1), "metricas": met}
        hist.append(h)
        with open(a.out / "historial.jsonl", "a") as fh:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(f"época {ep}: pérdida {h['perdida']:.4f} | Gold (diag.) {met['gold_diagnostico']:.4f} | "
              f"Spearman vs base {met[f'spearman_vs_base_{a.modo}']:.4f} | saltados {skipped}/{steps_ep} "
              f"({h['pct_saltados_gradscaler']:.1f} %) | no finitos {n_nonfinite} | "
              f"{seg / 60:.1f}+{met['segundos_eval'] / 60:.1f} min", flush=True)
        # copias de las últimas épocas en la RAM del host (sin mirar Gold)
        ultimas.append({"epoca": ep, "sd": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
        del ultimas[:-a.promedio]
        if ep > 0 and steps_ep and skipped / steps_ep > a.abortar_saltos:
            abortado = f"divergencia: GradScaler saltó {skipped}/{steps_ep} pasos en la época {ep}"
            print(abortado, flush=True)
            break
        if not math.isfinite(h["perdida"]) or met["no_finitas_pred"]:
            abortado = f"NaN: pérdida {h['perdida']} / predicciones no finitas {met['no_finitas_pred']} en la época {ep}"
            print(abortado, flush=True)
            break

    meta = {"arch": arch, "res": res, "modo": a.modo, "crop_px": MODOS[a.modo]["crop"], "lab": LAB,
            "norm": "imagenet", "corpus": "native384dense-v10", "init": "raptor_0949 " + CKPT_SHA[:12],
            "hereda_oai": True, "solo_final": "A"}
    resumen = {"config": cfg, "abortado": abortado, "epocas_hechas": len(hist), **ref}
    if hist:
        torch.save({"model": fp16_sd(model.state_dict()), **meta, "epoch": hist[-1]["epoca"]}, a.out / "last_fp16.pt")
        resumen["ultima"] = {"epoca": hist[-1]["epoca"], **{k: v for k, v in hist[-1]["metricas"].items()
                                                            if not k.endswith("por_hallazgo")}}
    if len(ultimas) == a.promedio and abortado is None:
        sds = [x["sd"] for x in ultimas]
        avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).to(sds[0][k].dtype)
               if sds[0][k].is_floating_point() else sds[-1][k].clone() for k in sds[0]}
        model.load_state_dict(avg)
        met = evaluar(f"prom{a.promedio}")
        torch.save({"model": fp16_sd(avg), **meta, "promedio_de": [x["epoca"] for x in ultimas]},
                   a.out / f"prom{a.promedio}_fp16.pt")
        resumen[f"prom{a.promedio}"] = {"epocas": [x["epoca"] for x in ultimas],
                                        **{k: v for k, v in met.items() if not k.endswith("por_hallazgo")}}
        resumen[f"prom{a.promedio}_por_hallazgo"] = {k: v for k, v in met.items() if k.endswith("por_hallazgo")}

    # ---- criterio fijado antes (ft0949.md) ----
    estable = (any(h["pct_saltados_gradscaler"] < 5.0 for h in hist) and abortado is None
               and all(h["logits_no_finitos"] == 0 for h in hist))
    resumen["pct_saltados_por_epoca"] = [h["pct_saltados_gradscaler"] for h in hist]
    resumen["estable"] = estable
    for cual in (f"prom{a.promedio}", "ultima"):
        if cual not in resumen:
            continue
        s = resumen[cual]["spearman_vs_base_v2"]
        resumen[f"criterio_{cual}"] = {"spearman_vs_base_v2": s, "umbral_clon": a.umbral_clon,
                                       "clon": not (s < a.umbral_clon), "estable": estable,   # NaN cuenta como clon
                                       "veredicto": "PASA" if estable and s < a.umbral_clon else "NO PASA"}
    resumen["horas"] = round((time.time() - t_inicio) / 3600, 2)
    (a.out / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in resumen.items() if k not in ("config",) and not k.endswith("por_hallazgo")},
                     indent=1, ensure_ascii=False), flush=True)
    sys.exit(1 if abortado and not abortado.startswith("presupuesto") else 0)


if __name__ == "__main__":
    main()
