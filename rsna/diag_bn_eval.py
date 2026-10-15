#!/usr/bin/env python
"""Diagnóstico sin GPU: ¿el desajuste de BatchNorm explica el Gold bajo del piloto CoAtNet?

El backbone tiene 25 BatchNorm (stem y etapas 0-1). El piloto entrena con micro-lotes de 1 estudio
(12 ventanas de la misma rodilla), así que en train cada BN normaliza con la media y la varianza DE ESE
ESTUDIO, y sus running_* son una EMA (momentum 0.1) de los ~10 últimos estudios. En eval se usan esas
running_*. Variantes sobre el mismo checkpoint, mismas 24 ventanas Gold:

  A  eval normal (running_* del checkpoint) — debe reproducir el Gold del piloto (fp32 en CPU)
  B  BN con estadísticas del propio estudio (las 24 ventanas juntas), como en el entrenamiento
  C  running_* recalibradas con media ACUMULADA de N estudios de train, uno por lote (sin ruido de EMA)
  D  running_* recalibradas con lotes de 8 estudios (estadística de población, como un lote de dreaddevelopment)

Lectura: B o C >> A  -> el problema es el desajuste train/eval de BN (arreglo barato).
         B ≈ A        -> si BN pesa, es en la dinámica de aprendizaje; solo lo dice entrenar con lotes de varios estudios.

Uso:  python rsna/diag_bn_eval.py --ckpt runs_joako/2085360/swa.pt --corpus DIR --out DIR [--n-recal 128]
"""
from __future__ import annotations

import os

os.environ.setdefault("TIMM_FUSED_ATTN", "0")

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar_coatnet import ARCH, LAB, MEAN, STD, RaptorClassifier, centros, macro_auc  # noqa: E402


def cargar(ckpt: Path) -> RaptorClassifier:
    import timm
    bb = timm.create_model(ARCH, pretrained=False, num_classes=0, global_pool="avg", in_chans=3)
    m = RaptorClassifier(bb, F_dim=bb.num_features)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
    return m.eval()


def bns(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def ventanas(corpus: Path, uid: str, k: int, train: bool) -> torch.Tensor:
    vol = np.load(corpus / "vols" / f"{uid}.npy", mmap_mode="r")
    cs = centros(int(vol.shape[0]), k, train)
    x = torch.from_numpy(np.stack([np.stack([vol[c - 1], vol[c], vol[c + 1]], 0) for c in cs], 0)).float() / 255.0
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


@torch.no_grad()
def predecir(model, corpus, uids, k, por_estudio: bool) -> np.ndarray:
    """por_estudio=True: BN en modo train con momentum 0 (usa la estadística del lote y no toca running_*)."""
    model.eval()
    if por_estudio:
        for b in bns(model):
            b.train(); b.momentum = 0.0
    ps = []
    for u in uids:
        x = ventanas(corpus, u, k, False)
        feats = model.backbone(x)                       # las 24 ventanas en un solo lote
        ps.append(torch.sigmoid(model.head(feats[None]).float())[0].numpy())
    model.eval()
    return np.stack(ps)


@torch.no_grad()
def recalibrar(model, corpus, uids, k, estudios_por_lote: int):
    for b in bns(model):
        b.reset_running_stats(); b.momentum = None      # media acumulada exacta
    model.eval()
    for b in bns(model):
        b.train()
    for i in range(0, len(uids), estudios_por_lote):
        x = torch.cat([ventanas(corpus, u, k, True) for u in uids[i:i + estudios_por_lote]], 0)
        model.backbone(x)
    model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--juez-v1", type=Path, default=Path("/lustre/proyectos/p082/preproc/etiquetas_v1_2079203.parquet"))
    ap.add_argument("--n-recal", type=int, default=128)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k-eval", type=int, default=24)
    ap.add_argument("--listar-uids", type=Path, default=None,
                    help="solo escribe los UID necesarios (Gold + muestra de train) y sale")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    v1 = pd.read_parquet(a.juez_v1).set_index("StudyInstanceUID")
    gold = v1.index[v1.oficial.to_numpy()].tolist()
    rest = v1.index[~v1.oficial.to_numpy()]
    fold = rest.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 5)
    train = sorted(rest[fold != 0].tolist())
    recal = random.Random(20261015).sample(train, a.n_recal)
    if a.listar_uids:
        a.listar_uids.write_text("\n".join(gold + recal))
        return
    idx = pd.read_csv(a.corpus / "indice.csv", dtype={"uid": str}).set_index("uid")
    ok = set(idx.index[(idx.error.fillna("") == "") & (idx.n_cortes >= 3)])
    gold = [u for u in gold if u in ok]
    recal = [u for u in recal if u in ok]
    if a.smoke:
        gold, recal = gold[:3], recal[:4]
    Y = v1.loc[gold, [f"y_{c}" for c in LAB]].to_numpy().astype(int)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    base = cargar(a.ckpt)
    a.out.mkdir(parents=True, exist_ok=True)
    res, preds = {"ckpt": str(a.ckpt), "n_gold": len(gold), "n_recal": len(recal)}, {}
    for nombre, fn in [
        ("A_eval_normal", lambda m: predecir(m, a.corpus, gold, a.k_eval, False)),
        ("B_bn_por_estudio", lambda m: predecir(m, a.corpus, gold, a.k_eval, True)),
        ("C_recal_1_estudio", lambda m: (recalibrar(m, a.corpus, recal, a.k, 1),
                                         predecir(m, a.corpus, gold, a.k_eval, False))[1]),
        ("D_recal_8_estudios", lambda m: (recalibrar(m, a.corpus, recal, a.k, 8),
                                          predecir(m, a.corpus, gold, a.k_eval, False))[1]),
    ]:
        t0 = time.time()
        p = fn(copy.deepcopy(base))
        preds[nombre] = p
        auc, per = macro_auc(Y, p)
        res[nombre] = {"gold": auc, "por_hallazgo": dict(zip(LAB, per)), "segundos": round(time.time() - t0, 1)}
        print(f"{nombre}: Gold {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        (a.out / "resumen.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    np.savez(a.out / "preds_gold.npz", ids=np.array(gold), **preds)


if __name__ == "__main__":
    main()
