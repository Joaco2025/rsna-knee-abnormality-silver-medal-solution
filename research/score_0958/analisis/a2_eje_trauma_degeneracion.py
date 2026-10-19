"""A2: ¿Las predicciones de Raptor (componente principal del 0.958) reproducen la estructura trauma frente a degeneración que
tienen las etiquetas de experto? ¿Una corrección determinista entre hallazgos sube el macro AUC en Gold-58?

Entradas (ya existentes, solo lectura): runs_joako/2089458/preds_gold.npz (Raptor 384, 224 y su mezcla 65/35 sobre Gold-58).
OJO: el SWA de Raptor se eligió por AUC en Gold-58, así que el AUC base está inflado; aquí interesa el CAMBIO al aplicar
una transformación que no se ajustó con esos datos (lambda fijado a priori en una rejilla corta y reportado entero, sin elegir).

Transformación (por estudio, sobre rangos normalizados 0-1 de cada hallazgo):
  T = {ACL, MCL, Contusion, Fracture}; Dg = {Medial OA, Lateral OA, PF OA, Baker's}
  hallazgo de T:  r' = r - lam * media(r de Dg)      hallazgo de Dg:  r' = r - lam * media(r de T)      resto sin cambio
Salida: a2_eje.json.  CPU, segundos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

D = Path("/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection")
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
T = ["ACL", "MCL", "Contusion", "Fracture"]
DG = ["Medial OA", "Lateral OA", "PF OA", "Baker's"]
t = pd.read_csv(D / "train.csv").dropna(subset=L).set_index("StudyInstanceUID")
z = np.load("/lustre/proyectos/p082/runs_joako/2089458/preds_gold.npz", allow_pickle=True)
ids = [str(i) for i in z["ids"]]
Y = t.loc[ids, L].values.astype(int)


def rk(P):
    return np.column_stack([(rankdata(P[:, k]) - 1) / (len(P) - 1) for k in range(P.shape[1])])


def fauc(y, s):
    """AUC por columna con rangos (empates promediados), y binaria [n,k], s [n,k]."""
    r = np.column_stack([rankdata(s[:, k]) for k in range(s.shape[1])])
    npos = y.sum(0)
    nneg = len(y) - npos
    return ((r * y).sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def aucs(S):
    return np.array([roc_auc_score(Y[:, k], S[:, k]) for k in range(len(L))])


def transf(R, lam):
    R2 = R.copy()
    it = [L.index(x) for x in T]
    idg = [L.index(x) for x in DG]
    mt, md = R[:, it].mean(1), R[:, idg].mean(1)
    for k in it:
        R2[:, k] = R[:, k] - lam * md
    for k in idg:
        R2[:, k] = R[:, k] - lam * mt
    return R2


out = {}
rng = np.random.default_rng(0)
for nombre in ("r384", "r224", "mezcla_65_35"):
    R = rk(z[nombre])
    base = aucs(R)
    cp = pd.DataFrame(R, columns=L).corr()
    res = {"macro_base": round(base.mean(), 4),
           "corr_pred_Contusion_MedialOA": round(cp.loc["Contusion", "Medial OA"], 3),
           "corr_pred_ACL_MedialOA": round(cp.loc["ACL", "Medial OA"], 3),
           "corr_pred_MCL_MedialMeniscus": round(cp.loc["MCL", "Medial Meniscus"], 3), "lam": {}}
    for lam in (0.1, 0.2, 0.3, 0.5):
        a = aucs(transf(R, lam))
        # bootstrap pareado de la diferencia de macro AUC (AUC por Mann-Whitney, vectorizado)
        R2 = transf(R, lam)
        difs = []
        for _ in range(500):
            ix = rng.integers(0, len(Y), len(Y))
            yb = Y[ix]
            if (yb.min(0) == yb.max(0)).any():
                continue
            difs.append(fauc(yb, R2[ix]).mean() - fauc(yb, R[ix]).mean())
        difs = np.array(difs)
        res["lam"][str(lam)] = {"macro": round(a.mean(), 4), "delta": round(a.mean() - base.mean(), 4),
                                "ic95": [round(np.percentile(difs, 2.5), 4), round(np.percentile(difs, 97.5), 4)],
                                "p_mejora": round(float((difs > 0).mean()), 3),
                                "por_hallazgo": {L[k]: round(a[k] - base[k], 3) for k in range(len(L)) if abs(a[k] - base[k]) > 1e-9}}
    out[nombre] = res
cg = pd.DataFrame(Y, columns=L).corr()
out["corr_gold_Contusion_MedialOA"] = round(cg.loc["Contusion", "Medial OA"], 3)
out["corr_gold_ACL_MedialOA"] = round(cg.loc["ACL", "Medial OA"], 3)
out["corr_gold_MCL_MedialMeniscus"] = round(cg.loc["MCL", "Medial Meniscus"], 3)
Path(__file__).with_name("a2_eje.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])
