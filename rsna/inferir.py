#!/usr/bin/env python
"""Inferencia: de los DICOM de un split a submission.csv. Pensado para correr sin red en Kaggle.

Para cada estudio de <split>.csv: lee sus series, aplica el mismo preprocesamiento que en
entrenamiento (preproc.py), pasa el tensor por uno o varios modelos y promedia sus probabilidades.
Un estudio que no se pueda leer recibe 0.5 en todo, para que el envío nunca falle por un caso.

Uso:
  python rsna/inferir.py --data DIR_COMPETENCIA --split test --checkpoints a.pt b.pt --out submission.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo import LABELS, KneeNet, prepare  # noqa: E402
from preproc import DEPTH, SIZE, build_study, slots_for  # noqa: E402


class TestStudies(Dataset):
    def __init__(self, uids: list[str], series: pd.DataFrame, root: Path, crop: int, n_slots: int = 4):
        self.uids, self.root, self.crop, self.slots = uids, root, crop, slots_for(n_slots)
        cols = ["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
        self.by_study = {u: g[cols].to_dict("records") for u, g in series.groupby("StudyInstanceUID")}

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        uid = self.uids[i]
        ok = 1.0
        try:
            arr, _ = build_study(self.root / uid, self.by_study.get(uid, []), slots=self.slots)
        except Exception:   # estudio ilegible: tensor vacío, se marcará para devolver 0.5
            arr, ok = np.zeros((len(self.slots), DEPTH, SIZE, SIZE), np.uint8), 0.0
        off = (SIZE - self.crop) // 2
        x = torch.from_numpy(np.ascontiguousarray(arr[:, :, off:off + self.crop, off:off + self.crop]))
        present = (x.flatten(1).amax(dim=1) > 0).float()
        if present.sum() == 0:
            ok = 0.0
        return x, present, torch.tensor(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="carpeta con <split>.csv, <split>_series.csv y <split>_series/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    studies = pd.read_csv(args.data / f"{args.split}.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(args.data / f"{args.split}_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = studies.StudyInstanceUID.tolist()
    if args.limit:
        uids = uids[: args.limit]

    models = []
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu")
        model = KneeNet(state["backbone"], None, n_slots=state.get("n_slots", 4), per_label=state.get("per_label", False),
                        context=state.get("context", False), pool=state.get("pool", "media"),
                        view_gate=state.get("view_gate", False))
        model.load_state_dict(state["model"])
        models.append((model.to(device).eval().to(memory_format=torch.channels_last), state["slices"], state["crop"], state.get("n_slots", 4)))
        print(f"modelo {path.name}: red={state['backbone']} cortes={state['slices']} recorte={state['crop']} "
              f"época={state.get('epoch')} AUC oficial={state.get('auc_oficial', float('nan')):.4f}", flush=True)
    crop = models[0][2]
    assert all(m[2] == crop for m in models), "todos los modelos deben usar el mismo recorte"

    n_slots = max(m[3] for m in models)     # el tensor de 6 ranuras contiene al de 4 en sus primeras posiciones
    loader = DataLoader(TestStudies(uids, series, args.data / f"{args.split}_series", crop, n_slots),
                        batch_size=args.batch, num_workers=args.workers, pin_memory=amp)
    probs, oks = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, present, ok) in enumerate(loader, 1):
            x, present = x.to(device, non_blocking=True), present.to(device)
            acc = torch.zeros(x.shape[0], len(LABELS), device=device)
            for model, n_slices, _, k in models:
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    acc += model(prepare(x[:, :k], n_slices, train=False), present[:, :k]).float().sigmoid()
            probs.append((acc / len(models)).cpu().numpy())
            oks.append(ok.numpy())
            if i % 50 == 0:
                done = i * args.batch
                print(f"{done}/{len(uids)} estudios, {(time.time() - t0) / done:.2f} s por estudio", flush=True)
    p = np.concatenate(probs)
    ok = np.concatenate(oks) > 0
    p[~ok] = 0.5
    p = np.clip(np.nan_to_num(p, nan=0.5), 0.0, 1.0)

    sub = pd.DataFrame(p, columns=LABELS)
    sub.insert(0, "StudyInstanceUID", uids)
    sub.to_csv(args.out, index=False)
    total = time.time() - t0
    print(f"{len(sub)} estudios en {total:.0f} s ({total / max(len(sub), 1):.2f} s por estudio); "
          f"ilegibles con 0.5: {int((~ok).sum())}; escrito {args.out}", flush=True)
    print(f"a este ritmo, 1,300 estudios tardarían {total / max(len(sub), 1) * 1300 / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
