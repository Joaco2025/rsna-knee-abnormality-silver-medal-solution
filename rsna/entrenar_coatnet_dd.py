#!/usr/bin/env python
"""CONTROL del diagnóstico CoAtNet: el entrenamiento de dreaddevelopment portado lo más literal posible.

Fuente: referencias/rsna_research/notebooks/dreaddevelopment/knee-mri-training-the-twelve-finding-model/
knee-mri-training-the-twelve-finding-model.py (en adelante "dd", con números de línea de ese archivo).
Se copia tal cual: StudyWindows sin ROI (dd 172-268: reescalado bilineal 336->384 en CPU, ganancia ±10 %,
normalización en el Dataset), collate (dd 271), build_backbone (dd 278-286), RaptorClassifier (dd 305-332),
grupos AdamW y OneCycle por paso con total = len(loader) × épocas y pct_start 0.15 (dd 484-490), bucle
zero_grad → forward → BCE media con pos_weight → backward → clip 3.0 → step → sched.step (dd 509-519),
lote 8 estudios sin acumulación, drop_last, lr 3e-5 / 1e-3, wd 0.02 (defaults de dd 349-352).

Desviaciones, todas forzadas:
  1. fp16 + GradScaler en vez de bf16 (bf16 cae con SIGSEGV en la MI210, bench_coatnet_v2.md).
  2. Nuestras etiquetas tienen celdas sin objetivo: la BCE media lleva weight = w (dd no tiene máscara). Con
     w = 1 es exactamente la BCEWithLogitsLoss de dd. La prevalencia de pos_weight usa las celdas con w > 0.
  3. Datos: corpus dd44 por estudio (rsna/dd44_corpus.py) en vez de all_vols.npy; partición fold 0 (md5 % 5).
  4. Pesos ImageNet desde el safetensors local (sin red), en vez de descargarlos.
  5. NORM = "imagenet": es lo que usa su notebook de inferencia ("must match training exactly", línea 100);
     el default de entrenamiento de dd es "none" (dd 360), contradicción que no podemos resolver.
  6. Se evalúan Gold y OOF del fold 0 EN CADA ÉPOCA (dd solo hace OOF al final); no cambia el entrenamiento.
  7. Épocas = 6 (diagnóstico). Se guarda solo la última época en fp16; sin SWA.

Uso:  python rsna/entrenar_coatnet_dd.py --corpus DIR_dd44 --out DIR [--epochs 6]
"""
from __future__ import annotations

import os

os.environ.setdefault("TIMM_FUSED_ATTN", "0")

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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_coatnet import LAB, P, RaptorClassifier, jueces, macro_auc  # noqa: E402  (RaptorClassifier = dd 305-332)


class StudyWindows(Dataset):
    """dd 172-268 sin la rama ROI. Devuelve además w (máscara de celdas, desviación 2)."""
    def __init__(self, corpus, ids, labels, weights, res, k, train, aug=True, norm="imagenet"):
        self.corpus, self.ids, self.labels, self.weights = corpus, ids, labels, weights
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self): return len(self.ids)

    def _centers(self, valid, count):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:count]
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    def _resize(self, tri):
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear", align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        uid = self.ids[i]
        vol = np.load(self.corpus / "vols" / f"{uid}.npy", mmap_mode="r")
        m = np.load(self.corpus / "masks" / f"{uid}.npy")
        self._D = vol.shape[0]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        cs = self._centers(valid, self.k)
        tiles = []
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
            tiles.append(self._resize(tri))
        x = torch.from_numpy(np.stack(tiles, 0))
        if self.train and self.aug:
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":
            x = (x - self._mean) / self._std
        return x, torch.from_numpy(self.labels[uid]), torch.from_numpy(self.weights[uid])


def collate(batch):
    return tuple(torch.stack([b[j] for b in batch]) for j in range(3))


def build_backbone(arch, weights: Path):
    import timm
    return timm.create_model(arch, pretrained=True, num_classes=0, in_chans=3, global_pool="avg",
                             pretrained_cfg_overlay=dict(file=str(weights)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=P / "runs_joako/etiquetas_coatnet_20261014/etiquetas_blendjev.parquet")
    ap.add_argument("--juez-blend", type=Path, default=P / "runs_joako/etiquetas_coatnet_20261014/etiquetas_blend.parquet")
    ap.add_argument("--juez-v1", type=Path, default=P / "preproc/etiquetas_v1_2079203.parquet")
    ap.add_argument("--weights", type=Path,
                    default=P / "code_joako/referencias/rsna_research/pesos/coatnet_rmlp_2_rw_384/model.safetensors")
    ap.add_argument("--arch", default="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--norm", default="imagenet", choices=["none", "imagenet"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.workers = 2, 2, 3, 3, 0
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- partición y etiquetas: idénticas al piloto (entrenar_coatnet.py) ----
    lab = pd.read_parquet(a.labels).set_index("StudyInstanceUID")
    v1 = pd.read_parquet(a.juez_v1).set_index("StudyInstanceUID")
    bl = pd.read_parquet(a.juez_blend).set_index("StudyInstanceUID")
    lab, bl = lab.loc[v1.index], bl.loc[v1.index]
    idx = pd.read_csv(a.corpus / "indice.csv", dtype={"uid": str}).set_index("uid")
    ok = set(idx.index[(idx.error.fillna("") == "") & (idx.n_cortes >= 3)])
    gold_ids = [u for u in v1.index[v1.oficial.to_numpy()] if u in ok]
    rest = v1.index[~v1.oficial.to_numpy()]
    fold = rest.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % a.folds)
    wcols = [f"w_{c}" for c in LAB]
    train_ids = [u for u in rest[fold != a.fold] if u in ok and lab.loc[u, wcols].sum() > 0]
    oof_ids = [u for u in rest[fold == a.fold] if u in ok]
    if a.smoke:
        train_ids, oof_ids, gold_ids = train_ids[:6], oof_ids[:4], gold_ids[:4]
    labels = {u: np.nan_to_num(lab.loc[u, [f"t_{c}" for c in LAB]].to_numpy(np.float32)) for u in lab.index}
    weights = {u: lab.loc[u, wcols].to_numpy(np.float32) for u in lab.index}
    T = np.stack([labels[u] for u in train_ids]); W = np.stack([weights[u] for u in train_ids])
    prev = np.clip(np.array([T[W[:, j] > 0, j].mean() if (W[:, j] > 0).any() else 0.5 for j in range(12)]), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    Y_gold = v1.loc[gold_ids, [f"y_{c}" for c in LAB]].to_numpy().astype(int)
    J_oof = jueces(oof_ids, {"v1": v1, "blend": bl, "blendjev": lab})

    tds = StudyWindows(a.corpus, train_ids, labels, weights, a.res, a.k, train=True, norm=a.norm)
    vds = StudyWindows(a.corpus, gold_ids, labels, weights, a.res, a.k_eval, train=False, norm=a.norm)
    ods = StudyWindows(a.corpus, oof_ids, labels, weights, a.res, a.k_eval, train=False, norm=a.norm)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)
    ol = DataLoader(ods, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)

    bb = build_backbone(a.arch, a.weights)
    if a.grad_ckpt:
        bb.set_grad_checkpointing(True)
    model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(len(tl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")          # desviación 1

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update({"n_train": len(train_ids), "n_oof": len(oof_ids), "n_gold": len(gold_ids), "pasos_por_epoca": len(tl),
                "pasos_totales": steps, "pos_weight": dict(zip(LAB, pw.cpu().numpy().round(3).tolist())),
                "job": os.environ.get("SLURM_JOB_ID", ""), "torch": torch.__version__})
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    print(json.dumps(cfg, ensure_ascii=False), flush=True)

    @torch.no_grad()
    def predict(loader):
        model.eval(); out = []
        for x, _, _ in loader:
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                out.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
        return np.concatenate(out)

    (a.out / "preds").mkdir(exist_ok=True)
    for ep in range(a.epochs):
        model.train(); tot = 0.0; t0 = time.time(); skipped = 0
        for x, y, w in tl:
            x, y, w = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), w.to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y, weight=w, pos_weight=pw)   # desviación 2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            escala = scaler.get_scale()
            scaler.step(opt); scaler.update()
            skipped += int(scaler.get_scale() < escala)
            sched.step(); tot += loss.item()
        seg = time.time() - t0
        t1 = time.time()
        pg, po = predict(vl), predict(ol)
        np.savez(a.out / "preds" / f"ep{ep}.npz", gold=pg, gold_ids=np.array(gold_ids), oof=po, oof_ids=np.array(oof_ids))
        met = {"gold": macro_auc(Y_gold, pg)[0], "gold_por_hallazgo": dict(zip(LAB, macro_auc(Y_gold, pg)[1])),
               "segundos_eval": round(time.time() - t1, 1)}
        for jn, (yj, wj) in J_oof.items():
            met[f"oof_{jn}"] = macro_auc(yj, po, wj)[0]
        h = {"epoca": ep, "perdida": tot / len(tl), "pasos": len(tl), "pasos_saltados_gradscaler": skipped,
             "escala_final": scaler.get_scale(), "segundos_epoca": round(seg, 1), "metricas": met}
        with open(a.out / "historial.jsonl", "a") as fh:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
        print(f"ep{ep} loss {h['perdida']:.3f} | GOLD {met['gold']:.4f} | OOF v1 {met['oof_v1']:.4f} "
              f"blendjev {met['oof_blendjev']:.4f} | saltados {skipped}/{len(tl)} | {seg / 60:.1f}+"
              f"{met['segundos_eval'] / 60:.1f} min", flush=True)
        if not math.isfinite(h["perdida"]):
            print("NaN en la pérdida: aborto", flush=True)
            break
    sd = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu())
          for k, v in model.state_dict().items()}
    torch.save({"model": sd, "arch": a.arch, "res": a.res, "lab": LAB, "norm": a.norm, "corpus": "dd44"},
               a.out / "last_fp16.pt")


if __name__ == "__main__":
    main()
