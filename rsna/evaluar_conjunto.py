#!/usr/bin/env python
"""Mide varios modelos y su promedio sobre las dos validaciones (58 oficiales y 10 % interno).

Sirve para decidir qué combinación enviar a Kaggle sin gastar envíos.

Uso:  python rsna/evaluar_conjunto.py --data DIR_NPY --labels TABLA.parquet --out DIR a.pt b.pt [...]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrenar import Studies, macro_auc, predict  # noqa: E402
from modelo import LABELS, KneeNet  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="+")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    table = pd.read_parquet(args.labels)
    official = table[table.oficial].reset_index(drop=True)
    rest = table[~table.oficial]
    fold = rest.StudyInstanceUID.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 10)
    val_int = rest[fold == 0].reset_index(drop=True)   # el mismo 10 % que apartó entrenar.py

    names, p_off, p_int = [], [], []
    y_off = t_int = w_int = None
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu")
        model = KneeNet(state["backbone"], None, n_slots=state.get("n_slots", 4), per_label=state.get("per_label", False),
                        context=state.get("context", False), pool=state.get("pool", "media"),
                        view_gate=state.get("view_gate", False))
        model.load_state_dict(state["model"])
        model = model.to(device).to(memory_format=torch.channels_last)

        def run(tab):
            dl = DataLoader(Studies(tab, args.data, state["crop"], False), batch_size=8, num_workers=args.workers, pin_memory=True)
            return predict(model, dl, device, state["slices"], True)

        po, _, _, y_off = run(official)
        pi, t_int, w_int, _ = run(val_int)
        names.append(f"{state['backbone']}@{path.parent.name}/ep{state['epoch']}")
        p_off.append(po); p_int.append(pi)
        del model
        torch.cuda.empty_cache()

    y_off = y_off.astype(int)
    y_int = (t_int >= 0.5).astype(int)

    def rank(p):   # promedio de rangos: no depende de la escala de cada modelo
        return np.stack([rankdata(p[:, j]) / len(p) for j in range(p.shape[1])], axis=1)

    rows, per_label = [], {}
    for k in range(1, len(names) + 1):
        for combo in combinations(range(len(names)), k):
            label = " + ".join(names[i] for i in combo)
            for kind, agg in (("media", lambda ps: np.mean(ps, axis=0)), ("rangos", lambda ps: np.mean([rank(p) for p in ps], axis=0))):
                if k == 1 and kind == "rangos":
                    continue
                a_off, per_off = macro_auc(y_off, agg([p_off[i] for i in combo]))
                a_int, _ = macro_auc(y_int, agg([p_int[i] for i in combo]), w_int)
                rows.append({"combinación": label, "promedio": kind if k > 1 else "-", "AUC oficial": a_off, "AUC interna": a_int})
                per_label[(label, kind if k > 1 else "-")] = per_off

    res = pd.DataFrame(rows)
    lines = [f"validación oficial: {len(official)} estudios; interna: {len(val_int)} estudios", "",
             res.to_string(index=False, float_format=lambda v: f"{v:.4f}")]
    best = res.sort_values("AUC interna", ascending=False).iloc[0]
    lines += ["", f"-- AUC oficial por hallazgo: modelos sueltos y la combinación con mejor AUC interna ({best['combinación']}, {best['promedio']})"]
    cols = [(n, "-") for n in names] + ([(best["combinación"], best["promedio"])] if best["promedio"] != "-" else [])
    lines.append(f"{'hallazgo':<17}" + "".join(f"{('M' + str(i + 1)) if i < len(names) else 'conjunto':>10}" for i in range(len(cols))))
    for j, c in enumerate(LABELS):
        lines.append(f"{c:<17}" + "".join(f"{per_label[key][j]:>10.3f}" for key in cols))
    lines.append(f"{'MEDIA':<17}" + "".join(f"{np.nanmean(per_label[key]):>10.3f}" for key in cols))
    lines += [""] + [f"M{i + 1} = {n}" for i, n in enumerate(names)]
    print("\n".join(lines), flush=True)
    (args.out / "conjunto.txt").write_text("\n".join(lines) + "\n")
    res.to_csv(args.out / "conjunto_resumen.csv", index=False)


if __name__ == "__main__":
    main()
