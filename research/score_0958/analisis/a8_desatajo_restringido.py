"""A8: desatajo SOLO en los hallazgos donde los metadatos predicen el informe pero no al experto (criterio de A6:
AUC_meta(informe) - AUC_meta(Gold) > 0.15 -> Synovitis 0.755/0.465 y Fracture 0.609/0.429), con un modelo LINEAL
portable a Kaggle (regresión logística L2 sobre rasgos estandarizados; coeficientes en JSON, sin pickle).

ADVERTENCIA de multiplicidad: los dos hallazgos se eligieron mirando Gold (A6) y lambda se lee también en Gold; el IC es
optimista. Sirve para decidir si merece UN envío, no como prueba.
Salidas: a8.json y desatajo_modelo.json (rasgos, medias, desviaciones, coeficientes, vocabulario) junto a este script.
"""
import json, sys
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

A5 = Path(sys.argv[1] if len(sys.argv) > 1 else "/lustre/proyectos/p082/runs_joako/2093187")
src = Path(__file__).with_name("a6_meta_y_duplicados.py").read_text()
g = {"__name__": "a6", "Path": Path, "pd": pd, "np": np, "re": re, "Counter": Counter, "rankdata": rankdata, "A5": A5}
exec(compile(src.split("# ---------------- etiquetas")[0].split('A5 = Path(sys.argv[1])')[1], "a6_features", "exec"), g)
X, L, auc, rk = g["X"], g["L"], g["auc"], g["rk"]
PROJ = Path("/lustre/proyectos/p082")
OBJ = ["Synovitis", "Fracture"]
tr = pd.read_csv(PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection/train.csv", dtype={"StudyInstanceUID": str})
gold = tr.dropna(subset=L).set_index("StudyInstanceUID")
jev = pd.read_parquet(PROJ / "runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet").set_index("StudyInstanceUID")
nogold = [u for u in X.index if u in jev.index and u not in gold.index]
Xf = X.fillna(X.loc[nogold].median())
mu, sd = Xf.loc[nogold].mean(), Xf.loc[nogold].std().replace(0, 1.0)
Z = (Xf - mu) / sd
z = np.load(PROJ / "runs_joako/2089458/preds_gold.npz", allow_pickle=True)
gids = [str(i) for i in z["ids"]]
gok = [u for u in gids if u in X.index]
Yg = gold.loc[gok, L].values.astype(int)
modelo = {"rasgos": list(X.columns), "medias": mu.round(6).to_dict(), "desv": sd.round(6).to_dict(),
          "medianas_relleno": X.loc[nogold].median().round(6).to_dict(), "coef": {}, "intercepto": {}}
mg = {}
for l in OBJ:
    yy = jev.loc[nogold, l].values
    ok = ~np.isnan(yy)
    lr = LogisticRegression(C=0.05, max_iter=2000)
    lr.fit(Z.loc[nogold].values[ok], (yy[ok] >= 0.5).astype(int))
    modelo["coef"][l] = dict(zip(X.columns, np.round(lr.coef_[0], 6)))
    modelo["intercepto"][l] = float(lr.intercept_[0])
    mg[l] = lr.decision_function(Z.loc[gok].values)
out = {"meta_lineal_gold": {l: round(auc(Yg[:, L.index(l)], mg[l]), 3) for l in OBJ}}
rng = np.random.default_rng(0)
B = [rng.integers(0, len(gok), len(gok)) for _ in range(3000)]
for nombre in ("r384", "r224", "mezcla_65_35"):
    Rg = z[nombre][[gids.index(u) for u in gok]]
    R = np.column_stack([rk(Rg[:, k]) for k in range(len(L))])
    out[nombre] = {}
    for lam in (0.1, 0.2, 0.3):
        S = R.copy()
        for l in OBJ:
            S[:, L.index(l)] = R[:, L.index(l)] - lam * rk(mg[l])
        d = []
        for ix in B:
            if (Yg[ix].min(0) == Yg[ix].max(0)).any():
                continue
            d.append(np.mean([auc(Yg[ix, k], S[ix, k]) - auc(Yg[ix, k], R[ix, k]) for k in range(len(L))]))
        d = np.array(d)
        out[nombre][str(lam)] = {"delta_macro": round(float(np.mean([auc(Yg[:, k], S[:, k]) - auc(Yg[:, k], R[:, k]) for k in range(len(L))])), 4),
                                 "ic95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                 "p_mejora": round(float((d > 0).mean()), 3),
                                 "por_hallazgo": {l: round(auc(Yg[:, L.index(l)], S[:, L.index(l)]) - auc(Yg[:, L.index(l)], R[:, L.index(l)]), 4) for l in OBJ}}
Path(__file__).with_name("a8.json").write_text(json.dumps(out, indent=1))
Path(__file__).with_name("desatajo_modelo.json").write_text(json.dumps(modelo, indent=1))
print(json.dumps(out, indent=1))
