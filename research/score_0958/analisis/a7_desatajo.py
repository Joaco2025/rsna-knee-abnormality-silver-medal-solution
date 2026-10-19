"""A7: ¿los modelos aprendieron el atajo «este centro informa X»?  Quitarlo restando la parte explicada por metadatos.

Mecanismo (A6): un modelo de SOLO metadatos predice la etiqueta de INFORME (Synovitis 0.755) pero no la de EXPERTO
(Synovitis 0.465 en Gold): captura hábitos de redacción por centro.  Raptor/S6/R6 se entrenaron con etiquetas de informe y
pueden haber absorbido ese atajo.  Corrección: r' = rango(modelo) - lam · rango(meta), por hallazgo.
Evaluación en Gold-58 (expertos) sobre Raptor (preds_gold.npz; base inflada por la selección SWA: interesa el delta) y,
como control, sobre el ensamble OOF v10 frente a etiquetas de informe (ahí DEBE empeorar si el mecanismo es el descrito).
Uso: python -I a7_desatajo.py <dir del job a5>
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

sys.argv += [] if len(sys.argv) > 1 else ["/lustre/proyectos/p082/runs_joako/2093187"]
exec(compile(Path(__file__).with_name("a6_meta_y_duplicados.py").read_text().split("# ---------------- etiquetas")[0]
             .split('A5 = Path(sys.argv[1])')[1], "a6_features", "exec"),
     g := {"__name__": "a6", "Path": Path, "pd": pd, "np": np, "re": re, "Counter": Counter, "rankdata": rankdata,
           "A5": Path(sys.argv[1])})
X = g["X"]
PROJ = Path("/lustre/proyectos/p082")
L = g["L"]
auc, rk = g["auc"], g["rk"]
tr = pd.read_csv(PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection/train.csv", dtype={"StudyInstanceUID": str})
gold = tr.dropna(subset=L).set_index("StudyInstanceUID")
jev = pd.read_parquet(PROJ / "runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet").set_index("StudyInstanceUID")
z = np.load(PROJ / "runs_joako/2089458/preds_gold.npz", allow_pickle=True)
gids = [str(i) for i in z["ids"]]
gok = [u for u in gids if u in X.index]
Yg = gold.loc[gok, L].values.astype(int)
nogold = [u for u in X.index if u in jev.index and u not in gold.index]
mg = np.zeros((len(gok), len(L)))
for k, l in enumerate(L):
    yy = jev.loc[nogold, l].values
    ok = ~np.isnan(yy)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(X.loc[nogold].values[ok].astype(np.float32), (yy[ok] >= 0.5).astype(int))
    mg[:, k] = clf.predict_proba(X.loc[gok].values.astype(np.float32))[:, 1]
OUT = {}
rng = np.random.default_rng(0)
B = [rng.integers(0, len(gok), len(gok)) for _ in range(2000)]
for nombre in ("r384", "r224", "mezcla_65_35"):
    Rg = z[nombre][[gids.index(u) for u in gok]]
    R = np.column_stack([rk(Rg[:, k]) for k in range(len(L))])
    M = np.column_stack([rk(mg[:, k]) for k in range(len(L))])
    base = np.array([auc(Yg[:, k], R[:, k]) for k in range(len(L))])
    OUT[nombre] = {"macro_base": round(base.mean(), 4)}
    for lam in (0.05, 0.1, 0.2, 0.3):
        S = R - lam * M
        a = np.array([auc(Yg[:, k], S[:, k]) for k in range(len(L))])
        d = []
        for ix in B:
            if (Yg[ix].min(0) == Yg[ix].max(0)).any():
                continue
            d.append(np.mean([auc(Yg[ix, k], S[ix, k]) - auc(Yg[ix, k], R[ix, k]) for k in range(len(L))]))
        OUT[nombre][str(lam)] = {"delta": round(a.mean() - base.mean(), 4), "ic95": [round(float(np.percentile(d, 2.5)), 4),
                                 round(float(np.percentile(d, 97.5)), 4)], "p_mejora": round(float(np.mean(np.array(d) > 0)), 3),
                                 "por_hallazgo": dict(zip(L, np.round(a - base, 3)))}
Path(__file__).with_name("a7.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
for n, r in OUT.items():
    print(n, "base", r["macro_base"])
    for lam in ("0.05", "0.1", "0.2", "0.3"):
        v = r[lam]
        print("   lam %-4s delta %+.4f IC [%+.4f, %+.4f] P %.2f | %s" % (lam, v["delta"], v["ic95"][0], v["ic95"][1], v["p_mejora"],
              {k: x for k, x in v["por_hallazgo"].items() if abs(x) >= 0.005}))
