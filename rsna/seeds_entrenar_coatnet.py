#!/usr/bin/env python
"""Semillas ALL-DATA del CoAtNet propio: copia de rsna/entrenar_coatnet.py (piloto, commit 4c63cd5) con
--all-data y pesos guardados en fp16. La receta es idéntica; ver experiments/reportes/seeds_coatnet.md.

Cambios frente al piloto (y nada más):
  --all-data  entrena con los 4,349 no Gold (sin fold fuera); no hay OOF ni criterio frente a effhib
  guardado    last_fp16.pt y swa_fp16.pt: tensores flotantes en fp16, sin optimizador (regla de disco)
  --corpus    puede apuntar a un corpus de lado unificado (rsna/seeds_corpus_lado.py); el código no cambia

Documentación completa de la receta y de cada decisión: experiments/reportes/receta_coatnet.md.

Resumen:
  datos       corpus native384 (rsna/raptor_corpus.py): ≤96 cortes en 5 ranuras, 140 mm, 384 px nativos
  ventanas    tripletes (i-1, i, i+1); 12 al azar por estudio al entrenar, 24 equiespaciadas al evaluar
  aumento     solo la elección de ventanas y una ganancia de intensidad ±10 % por estudio; sin flips
  modelo      timm coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k + RaptorClassifier (atención MIL por hallazgo),
              copiado tal cual del notebook de entrenamiento de dreaddevelopment / celda 22 del 0.940
  pérdida     BCE con logits, pos_weight = (1 − p)/p, p recortado a [0.03, 0.7] y el peso a [1, 10];
              Σ(w · BCE) / (12 · estudios del lote), celdas enmascaradas incluidas en el denominador
  optimizador AdamW wd 0.0588, lr backbone 4.01e-5, lr cabeza 1.95e-3, OneCycle 15 % de calentamiento,
              16 épocas, lote 1 estudio con acumulación 16, recorte de gradiente 3.0, semilla 42
  precisión   fp16 autocast + GradScaler (bf16 cae con SIGSEGV en el backward en la MI210), atención sin fusionar
  selección   top-3 épocas por macro AUC en los 58 Gold y promedio uniforme de pesos (SWA del #53)
  evaluación  por época: Gold (y real) y OOF del fold 0 con tres jueces (v1, blend, blendjev; y = t ≥ 0.5, w > 0)

Uso:  python rsna/entrenar_coatnet.py --corpus DIR --out DIR [--fold 0] [--smoke]
"""
from __future__ import annotations

import os

os.environ.setdefault("TIMM_FUSED_ATTN", "0")      # antes de importar timm: atención sin fusionar

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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
P = Path("/lustre/proyectos/p082")
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
RES = 384
MEAN = (0.485, 0.456, 0.406)                        # el #53 usa normalización ImageNet
STD = (0.229, 0.224, 0.225)
EFFHIB_OOF = P / "runs/2084813/oof_effhib.parquet"


# ------------------------------- datos -------------------------------------------------------------
def centros(n: int, k: int, train: bool) -> list[int]:
    """Centros de ventana como `StudyWindows._centers` del notebook, con todos los cortes válidos."""
    lo, hi = 0, n - 1
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, n - 2))]
    if train:
        pool = cs * (k // len(cs) + 1)
        random.shuffle(pool)
        return pool[:k]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


class Ventanas(Dataset):
    """Un estudio -> (ventanas uint8 (K, 3, H, W), ganancia, t (12), w (12)). Normaliza la GPU."""

    def __init__(self, uids, corpus: Path, t: dict, w: dict, k: int, train: bool, jitter: float):
        self.uids, self.corpus, self.t, self.w = list(uids), corpus, t, w
        self.k, self.train, self.jitter = k, train, jitter

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        vol = np.load(self.corpus / "vols" / f"{uid}.npy", mmap_mode="r")
        cs = centros(int(vol.shape[0]), self.k, self.train)
        x = np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs], 0)
        g = 1.0 + (random.random() - 0.5) * 2 * self.jitter if self.train else 1.0
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(g, dtype=torch.float32),
                torch.from_numpy(self.t[uid]), torch.from_numpy(self.w[uid]))


def un_estudio(batch):
    assert len(batch) == 1
    x, g, t, w = batch[0]
    return x, g, t[None], w[None]


def preparar(x: torch.Tensor, g: torch.Tensor, device) -> torch.Tensor:
    """uint8 (K,3,H,W) -> float normalizada, con la ganancia aplicada en [0, 1] como el notebook."""
    x = x.to(device, non_blocking=True).float() / 255.0
    x = (x * g.to(device)).clamp(0, 1)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std)[None]                  # (1, K, 3, H, W)


# ------------------------------- modelo (copiado tal cual) -------------------------------------------
class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def construir_modelo(weights: Path) -> RaptorClassifier:
    import timm
    bb = timm.create_model(ARCH, pretrained=True, num_classes=0, global_pool="avg", in_chans=3,
                           pretrained_cfg_overlay=dict(file=str(weights)))
    return RaptorClassifier(bb, F_dim=bb.num_features)


# ------------------------------- métricas ----------------------------------------------------------
def macro_auc(y, p, w=None):
    """Idéntica a rsna/entrenar.py: AUC por hallazgo en celdas con w > 0; media ignorando NaN."""
    aucs = []
    for j in range(y.shape[1]):
        m = np.ones(len(y), bool) if w is None else w[:, j] > 0
        yy = y[m, j]
        aucs.append(roc_auc_score(yy, p[m, j]) if 0 < yy.sum() < len(yy) else float("nan"))
    return float(np.nanmean(aucs)), aucs


def jueces(uids, tablas: dict) -> dict:
    out = {}
    for nombre, tab in tablas.items():
        t = tab.loc[uids, [f"t_{c}" for c in LAB]].to_numpy(np.float32)
        w = tab.loc[uids, [f"w_{c}" for c in LAB]].to_numpy(np.float32)
        out[nombre] = ((np.nan_to_num(t) >= 0.5).astype(int), w)
    return out


@torch.no_grad()
def predecir(model, loader, device, chunk: int) -> np.ndarray:
    """Como la inferencia del 0.940: backbone por trozos de `chunk` ventanas, una sola llamada a la cabeza."""
    model.eval()
    ps = []
    for x, g, _, _ in loader:
        x = preparar(x, g, device)[0]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            feats = torch.cat([model.backbone(x[i:i + chunk]) for i in range(0, len(x), chunk)], 0)
            logits = model.head(feats[None])
        ps.append(torch.sigmoid(logits.float())[0].cpu().numpy())
    return np.stack(ps)


def bootstrap_delta(y, w, pa, pb, n=1000, seed=20261014):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ds.append(macro_auc(y[i], pa[i], w[i])[0] - macro_auc(y[i], pb[i], w[i])[0])
    ds = np.array(ds)
    ok = np.isfinite(ds)
    return {"ic95": [float(np.nanpercentile(ds, 2.5)), float(np.nanpercentile(ds, 97.5))] if ok.any() else None,
            "p_delta_pos": float((ds[ok] > 0).mean()) if ok.any() else None, "replicas": n,
            "replicas_validas": int(ok.sum()), "semilla": seed}


# ------------------------------- principal ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=P / "runs_joako/etiquetas_coatnet_20261014/etiquetas_blendjev.parquet")
    ap.add_argument("--juez-blend", type=Path, default=P / "runs_joako/etiquetas_coatnet_20261014/etiquetas_blend.parquet")
    ap.add_argument("--juez-v1", type=Path, default=P / "preproc/etiquetas_v1_2079203.parquet")
    ap.add_argument("--weights", type=Path,
                    default=P / "code_joako/referencias/rsna_research/pesos/coatnet_rmlp_2_rw_384/model.safetensors")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--bb-lr", type=float, default=4.01e-5)
    ap.add_argument("--head-lr", type=float, default=1.95e-3)
    ap.add_argument("--wd", type=float, default=0.0588)
    ap.add_argument("--pct-start", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=3.0)
    ap.add_argument("--jitter", type=float, default=0.10)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-horas", type=float, default=0, help="presupuesto; no empieza una época que no quepa")
    ap.add_argument("--abortar-saltos", type=float, default=0.25,
                    help="aborta si en una época (tras la primera) GradScaler salta más de esta fracción de pasos")
    ap.add_argument("--mlflow", default="")
    ap.add_argument("--mlflow-artifacts", default="")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--smoke", action="store_true", help="prueba de cableado: pocos estudios, 2 épocas")
    ap.add_argument("--smoke-pasos", type=int, default=0, help="con --smoke: corta cada época tras N pasos")
    ap.add_argument("--all-data", action="store_true", help="entrena con todos los no Gold; sin OOF")
    ap.add_argument("--ref-resumen", type=Path, default=P / "runs_joako/2085360/resumen.json",
                    help="resumen.json del piloto: su Gold del SWA es la referencia del criterio all-data")
    ap.add_argument("--corpus-tipo", default="native384dense-v10", help="solo se anota (p. ej. ...-lado)")
    a = ap.parse_args()
    t_inicio = time.time()
    if a.smoke:
        a.epochs, a.k, a.k_eval, a.accum = 2, 4, 6, 4
        if a.smoke_pasos:                            # smoke de login con 1-2 estudios: un estudio por paso
            a.accum = 1
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # CPU solo para la prueba --smoke
    amp = device.type == "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "preds").mkdir(exist_ok=True)

    # ---- estudios, particiones y etiquetas ----
    lab = pd.read_parquet(a.labels).set_index("StudyInstanceUID")
    v1 = pd.read_parquet(a.juez_v1).set_index("StudyInstanceUID")
    bl = pd.read_parquet(a.juez_blend).set_index("StudyInstanceUID")
    lab, bl = lab.loc[v1.index], bl.loc[v1.index]
    idx = pd.read_csv(a.corpus / "indice.csv", dtype={"uid": str}).set_index("uid")
    ok = set(idx.index[(idx.error.fillna("") == "") & (idx.n_cortes >= 3)])
    gold_all = v1.index[v1.oficial.to_numpy()].tolist()
    rest = v1.index[~v1.oficial.to_numpy()]
    fold = rest.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % a.folds)
    oof_all = [] if a.all_data else rest[fold == a.fold].tolist()
    train_all = rest.tolist() if a.all_data else rest[fold != a.fold].tolist()
    wcols = [f"w_{c}" for c in LAB]
    train = [u for u in train_all if u in ok and lab.loc[u, wcols].sum() > 0]
    oof = [u for u in oof_all if u in ok]
    gold = [u for u in gold_all if u in ok]
    if a.smoke:
        train, oof, gold = train[:32], oof[:24], gold[:24]
    t_ = {u: np.nan_to_num(lab.loc[u, [f"t_{c}" for c in LAB]].to_numpy(np.float32)) for u in lab.index}
    w_ = {u: lab.loc[u, wcols].to_numpy(np.float32) for u in lab.index}
    excluidos = {"train_sin_corpus": len([u for u in train_all if u not in ok]),
                 "train_sin_objetivo": len([u for u in train_all if u in ok and lab.loc[u, wcols].sum() == 0]),
                 "oof_sin_corpus": len(oof_all) - len([u for u in oof_all if u in ok]),
                 "gold_sin_corpus": len(gold_all) - len([u for u in gold_all if u in ok])}

    # pos_weight: prevalencia sobre los estudios de entrenamiento, celdas con objetivo (decisión nuestra)
    T = np.stack([t_[u] for u in train]); W = np.stack([w_[u] for u in train])
    prev = np.array([T[W[:, j] > 0, j].mean() if (W[:, j] > 0).any() else np.nan for j in range(12)])
    prev = np.where(np.isfinite(prev), prev, 0.5)    # solo ocurre en --smoke (hallazgo sin celdas con objetivo)
    prev_c = np.clip(prev, 0.03, 0.7)
    pw_np = np.clip((1 - prev_c) / prev_c, 1, 10)
    pw = torch.tensor(pw_np, dtype=torch.float32, device=device)

    steps_ep = len(train) // a.accum                 # como el #53: 4,349 // 16 = 271 → 4,336 pasos en 16 épocas
    if a.smoke and a.smoke_pasos:
        steps_ep = min(steps_ep, a.smoke_pasos)
    total_steps = steps_ep * a.epochs
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update({"n_train": len(train), "n_oof": len(oof), "n_gold": len(gold), "excluidos": excluidos,
                "pasos_por_epoca": steps_ep, "pasos_totales": total_steps,
                "prevalencia": dict(zip(LAB, prev.round(4).tolist())), "pos_weight": dict(zip(LAB, pw_np.round(3).tolist())),
                "commit": os.environ.get("GIT_COMMIT", ""), "job": os.environ.get("SLURM_JOB_ID", ""),
                "torch": torch.__version__, "TIMM_FUSED_ATTN": os.environ.get("TIMM_FUSED_ATTN")})
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    print(json.dumps(cfg, ensure_ascii=False), flush=True)

    def loader(uids, k, train_mode):
        return DataLoader(Ventanas(uids, a.corpus, t_, w_, k, train_mode, a.jitter), batch_size=1,
                          shuffle=train_mode, num_workers=a.workers, collate_fn=un_estudio, pin_memory=True,
                          persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    dl_train = loader(train, a.k, True)
    dl_oof = loader(oof, a.k_eval, False) if oof else None
    dl_gold = loader(gold, a.k_eval, False)
    J_oof = jueces(oof, {"v1": v1, "blend": bl, "blendjev": lab}) if oof else {}
    Y_gold = v1.loc[gold, [f"y_{c}" for c in LAB]].to_numpy().astype(int)

    # ---- modelo y optimizador ----
    model = construir_modelo(a.weights).to(device)
    n_fused = sum(bool(getattr(m, "fused_attn", False)) for m in model.modules())
    print(f"backbone {ARCH} F={model.clsW.shape[1]} | atenciones fusionadas: {n_fused}", flush=True)
    head_p = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    bb_p = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb_p, "lr": a.bb_lr}, {"params": head_p, "lr": a.head_lr}],
                            weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=total_steps,
                                                pct_start=a.pct_start)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ml = None
    if a.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(a.mlflow)
            if mlflow.get_experiment_by_name("rsna-knee") is None:
                mlflow.create_experiment("rsna-knee", artifact_location=a.mlflow_artifacts or None)
            mlflow.set_experiment("rsna-knee")
            mlflow.start_run(run_name=a.run_name or a.out.name)
            mlflow.log_params({k: str(v)[:250] for k, v in cfg.items()})
            ml = mlflow
        except Exception as exc:  # noqa: BLE001  el seguimiento nunca detiene el entrenamiento
            print(f"MLflow desactivado: {type(exc).__name__}: {exc}", flush=True)

    def evaluar(tag: str) -> dict:
        t0 = time.time()
        pg = predecir(model, dl_gold, device, a.chunk)
        po = predecir(model, dl_oof, device, a.chunk) if dl_oof is not None else np.zeros((0, 12), np.float32)
        np.savez(a.out / "preds" / f"{tag}.npz", gold=pg, gold_ids=np.array(gold), oof=po, oof_ids=np.array(oof))
        r = {"gold": macro_auc(Y_gold, pg)[0], "gold_por_hallazgo": dict(zip(LAB, macro_auc(Y_gold, pg)[1])),
             "segundos_eval": round(time.time() - t0, 1),
             "no_finitas_pred": int((~np.isfinite(pg)).sum() + (~np.isfinite(po)).sum())}
        for jn, (y, w) in J_oof.items():
            m, per = macro_auc(y, po, w)
            r[f"oof_{jn}"] = m
            r[f"oof_{jn}_por_hallazgo"] = dict(zip(LAB, per))
        return r

    # ---- entrenamiento ----
    hist, topk = [], []
    abortado = None
    for ep in range(a.epochs):
        if a.max_horas and hist:
            dur = max(h["segundos_epoca"] + h["metricas"]["segundos_eval"] for h in hist)
            if time.time() - t_inicio + dur * 1.1 + 1800 > a.max_horas * 3600:
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
            x = preparar(x, g, device)
            t, w = t.to(device), w.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            if not torch.isfinite(logits).all():
                n_nonfinite += 1
            cell = F.binary_cross_entropy_with_logits(logits.float(), t, pos_weight=pw, reduction="none")
            loss = (cell * w).sum() / (12 * x.shape[0])         # #53: dividir entre las 12 celdas del estudio
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
             "pasos_saltados_gradscaler": skipped,
             "pct_saltados_gradscaler": round(100 * skipped / max(steps_ep, 1), 2),
             "logits_no_finitos": n_nonfinite,
             "gradnorm_mediana": float(np.median(gn_list)) if gn_list else None,
             "gradnorm_p99": float(np.percentile(gn_list, 99)) if gn_list else None,
             "escala_final": scaler.get_scale(), "segundos_epoca": round(seg, 1), "metricas": met}
        hist.append(h)
        with open(a.out / "historial.jsonl", "a") as fh:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        oofs = " ".join(f"{k[4:]} {met[k]:.4f}" for k in ("oof_v1", "oof_blend", "oof_blendjev") if k in met)
        print(f"época {ep}: pérdida {h['perdida']:.4f} | Gold {met['gold']:.4f} | OOF {oofs or '-'} | saltados {skipped}/{steps_ep} "
              f"({h['pct_saltados_gradscaler']:.1f} %) escala {h['escala_final']:.0f} | no finitos {n_nonfinite} | "
              f"{seg / 60:.1f}+{met['segundos_eval'] / 60:.1f} min", flush=True)
        if ml:
            try:
                ml.log_metrics({"loss": h["perdida"], "gold": met["gold"],
                                **{k: met[k] for k in ("oof_v1", "oof_blend", "oof_blendjev") if k in met},
                                "saltados": skipped, "pct_saltados": h["pct_saltados_gradscaler"],
                                "escala": h["escala_final"]}, step=ep)
            except Exception as exc:  # noqa: BLE001
                print(f"MLflow: {exc}", flush=True)
        # top-k por Gold (selección del #53), copias en RAM del host
        if len(topk) < a.topk or met["gold"] > min(x["gold"] for x in topk):
            topk.append({"epoca": ep, "gold": met["gold"],
                         "sd": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
            topk.sort(key=lambda x: -x["gold"])
            del topk[a.topk:]
        if ep > 0 and steps_ep and skipped / steps_ep > a.abortar_saltos:
            abortado = f"divergencia: GradScaler saltó {skipped}/{steps_ep} pasos en la época {ep}"
            print(abortado, flush=True)
            break
        if not math.isfinite(h["perdida"]) or met["no_finitas_pred"]:
            abortado = f"NaN: pérdida {h['perdida']} / predicciones no finitas {met['no_finitas_pred']} en la época {ep}"
            print(abortado, flush=True)
            break

    def fp16_sd(sd: dict) -> dict:
        return {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu().clone()) for k, v in sd.items()}

    meta = {"arch": ARCH, "res": RES, "lab": LAB, "norm": "imagenet", "corpus": str(cfg.get("corpus_tipo", "native384dense-v10")),
            "fold": "all" if a.all_data else a.fold, "folds": a.folds, "seed": a.seed, "dtype_guardado": "fp16"}
    resumen = {"config": cfg, "abortado": abortado, "epocas_hechas": len(hist)}
    if hist:
        torch.save({"model": fp16_sd(model.state_dict()), **meta, "epoch": hist[-1]["epoca"]}, a.out / "last_fp16.pt")
        resumen["ultima"] = {"epoca": hist[-1]["epoca"], **{k: v for k, v in hist[-1]["metricas"].items()
                                                            if not k.endswith("por_hallazgo")}}
    if len(topk) > 1:
        sds = [x["sd"] for x in topk]
        avg = {k: (sum(sd[k].double() for sd in sds) / len(sds)).to(sds[0][k].dtype)
               if sds[0][k].is_floating_point() else sds[0][k].clone() for k in sds[0]}
        model.load_state_dict(avg)
        met = evaluar("swa")
        torch.save({"model": fp16_sd(avg), **meta, "swa_over": [x["epoca"] for x in topk]}, a.out / "swa_fp16.pt")
        resumen["swa"] = {"epocas": [x["epoca"] for x in topk], **{k: v for k, v in met.items()
                                                                  if not k.endswith("por_hallazgo")}}
        resumen["swa_por_hallazgo"] = {k: v for k, v in met.items() if k.endswith("por_hallazgo")}

    # ---- criterio fijado antes: OOF fold 0, juez v1, frente a effhib en los mismos estudios ----
    # fp16 sano (jcastro): al menos una época con < 5 % de pasos saltados por GradScaler; si no, NO PASA.
    resumen["pct_saltados_por_epoca"] = [h["pct_saltados_gradscaler"] for h in hist]
    resumen["escala_por_epoca"] = [h["escala_final"] for h in hist]
    fp16_sano = any(h["pct_saltados_gradscaler"] < 5.0 for h in hist)
    resumen["fp16_alguna_epoca_bajo_5pct_saltados"] = fp16_sano
    if a.all_data:
        # Sin OOF. Criterio de seeds_coatnet.md: entrenamiento sano y Gold del SWA no más de 0.02 por debajo del
        # Gold del SWA del piloto (los dos con el mismo sesgo de selección top-3 por Gold).
        sano = bool(abortado is None and fp16_sano and all(h["logits_no_finitos"] == 0 for h in hist))
        ref_gold = None
        if a.ref_resumen and a.ref_resumen.exists():
            ref_gold = json.loads(a.ref_resumen.read_text()).get("swa", {}).get("gold")
        g = resumen.get("swa", {}).get("gold")
        ok_gold = g is not None and ref_gold is not None and g >= ref_gold - 0.02
        resumen["criterio_all_data"] = {"sano": sano, "gold_swa": g, "gold_swa_piloto": ref_gold, "tolerancia": 0.02,
                                        "veredicto": "PASA" if sano and ok_gold else "NO PASA"}
    try:
        if a.all_data:
            raise RuntimeError("all-data: sin OOF, no aplica el criterio frente a effhib")
        eff = pd.read_parquet(EFFHIB_OOF).set_index("StudyInstanceUID").loc[oof, LAB].to_numpy()
        y, w = J_oof["v1"]
        ref = macro_auc(y, eff, w)[0]
        resumen["effhib_fold"] = ref
        for cual in ("swa", "ultima"):
            if cual not in resumen:
                continue
            tag = "swa" if cual == "swa" else f"ep{resumen['ultima']['epoca']}"
            po = np.load(a.out / "preds" / f"{tag}.npz")["oof"]
            d = macro_auc(y, po, w)[0] - ref
            resumen[f"criterio_{cual}"] = {"delta_vs_effhib": d, "umbral": 0.005,
                                          **bootstrap_delta(y, w, po, eff),
                                          "pasa": bool(d >= 0.005 and abortado is None and fp16_sano and
                                                       all(h["logits_no_finitos"] == 0 for h in hist))}
            resumen[f"criterio_{cual}"]["veredicto"] = "PASA" if resumen[f"criterio_{cual}"]["pasa"] else "NO PASA"
            if d >= 0.005 and not fp16_sano:
                resumen[f"criterio_{cual}"]["motivo"] = "AUC sobre el umbral, pero ninguna época con < 5 % de pasos saltados"
    except Exception as exc:  # noqa: BLE001
        resumen["criterio_error"] = f"{type(exc).__name__}: {exc}"
    resumen["horas"] = round((time.time() - t_inicio) / 3600, 2)
    (a.out / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in resumen.items() if k not in ("config", "swa_por_hallazgo")},
                     indent=1, ensure_ascii=False), flush=True)
    if ml:
        try:
            ml.end_run()
        except Exception:  # noqa: BLE001
            pass
    sys.exit(1 if abortado and not abortado.startswith("presupuesto") else 0)


if __name__ == "__main__":
    main()
