"""A3: préstamo de señal entre hallazgos para los tres más débiles en Gold (Synovitis, Lateral OA, PF OA).

Mecanismo: en los informes la sinovitis casi no se menciona (JEV 30 % frente a 47 % en Gold), así que los modelos la aprenden
mal; Effusion la predicen muy bien (Gold 0.98) y en las etiquetas de experto Effusion predice Synovitis con AUC 0.70.
Parejas fijadas a priori por mecanismo clínico, sin buscar entre las 11 posibles:
  Synovitis  <- Effusion           (derrame-sinovitis)
  Lateral OA <- Lateral Meniscus   (degeneración del compartimento lateral)
  PF OA      <- Medial OA          (artrosis generalizada)
r'(objetivo) = (1-lam)*r(objetivo) + lam*r(pareja), sobre rangos 0-1.  Se reporta la rejilla entera.
Datos: runs_joako/2089458/preds_gold.npz (Raptor; Gold contaminado por la selección SWA: interesa el delta).
Salida: a3_prestamo.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

D = Path("/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection")
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
PAREJAS = {"Synovitis": "Effusion", "Lateral OA": "Lateral Meniscus", "PF OA": "Medial OA"}
t = pd.read_csv(D / "train.csv").dropna(subset=L).set_index("StudyInstanceUID")
z = np.load("/lustre/proyectos/p082/runs_joako/2089458/preds_gold.npz", allow_pickle=True)
Y = t.loc[[str(i) for i in z["ids"]], L].values.astype(int)


def rk(v):
    return (rankdata(v) - 1) / (len(v) - 1)


def auc(y, s):
    r = rankdata(s)
    p = y.sum()
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))


rng = np.random.default_rng(0)
B = [rng.integers(0, len(Y), len(Y)) for _ in range(2000)]
out = {}
for nombre in ("r384", "r224", "mezcla_65_35"):
    P = z[nombre]
    res = {}
    for obj, par in PAREJAS.items():
        y = Y[:, L.index(obj)]
        a0 = rk(P[:, L.index(obj)])
        b = rk(P[:, L.index(par)])
        fila = {"base": round(auc(y, a0), 4)}
        for lam in (0.1, 0.2, 0.3, 0.4):
            s = (1 - lam) * a0 + lam * b
            d = [auc(y[ix], s[ix]) - auc(y[ix], a0[ix]) for ix in B if 0 < y[ix].sum() < len(ix)]
            fila[str(lam)] = {"auc": round(auc(y, s), 4), "delta": round(auc(y, s) - auc(y, a0), 4),
                              "ic95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                              "p_mejora": round(float(np.mean(np.array(d) > 0)), 3),
                              "delta_macro": round((auc(y, s) - auc(y, a0)) / 12, 4)}
        res[obj + " <- " + par] = fila
    out[nombre] = res
Path(__file__).with_name("a3_prestamo.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
for n, r in out.items():
    print("==", n)
    for k, f in r.items():
        print("  %-30s base %.3f | " % (k, f["base"]) + " | ".join(
            "l%s %+.3f [%+.3f,%+.3f] P%.2f" % (l, v["delta"], v["ic95"][0], v["ic95"][1], v["p_mejora"])
            for l, v in f.items() if l != "base"))
