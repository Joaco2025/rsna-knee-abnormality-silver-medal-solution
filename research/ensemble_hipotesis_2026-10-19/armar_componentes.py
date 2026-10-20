"""Une las salidas de una ejecución del notebook sobre Gold-58 (y/o de s6_gold_infer.py) en un .npz para evaluar_gold.py.

Entradas (todas opcionales; se usa lo que exista):
  --work DIR    gold_dump/ descargado de gold_dump.ipynb (o un /kaggle/working equivalente): raptor_raw.csv (R384 con espejo),
                raptor224_sub.csv (R224), ours_sub.csv (R6 flip), _own.csv (lector), s6_sub.csv (S6 original),
                meta_synovitis.csv (puntuación de metadatos) y, para la paridad, submission_0958.csv / esust_dsj_final.csv
                (se guardan como nb_base0958 / nb_esust_dsj).  Exige 0 fallos de Raptor y de R6 (blend_receipt.json).
  --s6 NPZ      s6_gold_stride1.npz de s6_gold_infer.py (claves s6, s6_d42, s6_d7).  Si --work ya trae s6_sub.csv, el S6
                original de Yuca se guarda como s6_yuca (el del notebook manda: es el que reproduce la mezcla)
Si --work trae R384/R224 CON espejo, sustituyen a los de runs_joako/2089458 (sin espejo); se anota en 'origen'.
Uso: python -I armar_componentes.py --work W --s6 S --out componentes_gold.npz
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CSV = {"r384": "raptor_raw.csv", "r224": "raptor224_sub.csv", "r6": "ours_sub.csv", "lector": "_own.csv", "s6": "s6_sub.csv",
       "nb_base0958": "submission_0958.csv", "nb_esust_dsj": "esust_dsj_final.csv"}
gold = json.loads(Path("/lustre/proyectos/p082/runs_joako/exp_s6_semillas/S6F0/data/gold58.json").read_text())

ap = argparse.ArgumentParser()
ap.add_argument("--work")
ap.add_argument("--s6")
ap.add_argument("--out", required=True)
a = ap.parse_args()
C, origen = {}, {}
if a.work:
    br = Path(a.work) / "blend_receipt.json"
    if br.exists():
        r = json.loads(br.read_text())
        assert r["raptor_failed"] == 0 and r["ours_failed"] == 0, f"fallos en la ejecución: {r['raptor_failed']}/{r['ours_failed']}"
    pm = Path(a.work) / "meta_synovitis.csv"
    if pm.exists():
        dm = pd.read_csv(pm, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        assert set(dm.index) == set(gold)
        C["meta"] = dm.loc[gold, "meta_synovitis"].to_numpy(np.float64)
        origen["meta"] = str(pm)
    for k, f in CSV.items():
        p = Path(a.work) / f
        if p.exists():
            d = pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
            assert set(d.index) == set(gold), f"{f}: filas distintas de los 58 Gold"
            C[k] = d.loc[gold, LAB].to_numpy(np.float64)
            origen[k] = str(p)
if a.s6:
    z = np.load(a.s6, allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    assert set(ids) == set(gold)
    o = [ids.index(u) for u in gold]
    for k in ("s6", "s6_d42", "s6_d7"):
        if k in z.files:
            kk = "s6_yuca" if (k == "s6" and "s6" in C) else k
            C[kk] = np.asarray(z[k], np.float64)[o]
            origen[kk] = a.s6
for k, v in C.items():
    assert v.shape in ((58, 12), (58,)) and np.isfinite(v).all(), k
    if v.ndim == 2 and k in ("r384", "r6", "lector") and (v == 0.5).all(axis=1).any():
        print(f"AVISO {k}: hay estudios con 0.5 en todo (fallo del componente)")
np.savez(a.out, ids=np.array(gold), **C)
Path(a.out).with_suffix(".origen.json").write_text(json.dumps(origen, indent=1))
print("componentes:", sorted(C), "->", a.out)
