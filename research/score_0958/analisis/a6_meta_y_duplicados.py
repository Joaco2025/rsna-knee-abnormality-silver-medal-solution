"""A6: dos fuentes de información que ningún modelo de imagen usa.

(1) METADATOS DICOM (sin texto clínico: viene anonimizado): sexo, fabricante/modelo/campo, bobina, nº y tipo de series,
    contraste, grosor, tokens de SeriesDescription.  HistGradientBoosting por hallazgo, 5 particiones md5 (las de Owen).
    a) AUC de los metadatos solos (etiquetas JEV, y >= 0.5).
    b) ¿Suman a la imagen? Ensamble OOF v10 (rangos) + w · rango(metadatos); w elegido en las particiones de ajuste y
       evaluado en la que queda fuera (anidado); bootstrap pareado del delta.
    c) Gold-58: metadatos entrenados con los 4,349 no-gold, sumados a Raptor (preds_gold.npz; Gold contaminado para
       Raptor: solo interesa el delta).
(2) CASI-DUPLICADOS: correlación de miniaturas 64x64 (corte sagital central) entre estudios distintos de train; si hay
    pares casi idénticos, ¿coinciden sus etiquetas más que al azar?
Uso: python -I a6_meta_y_duplicados.py <dir del job a5>   (CPU; salida a6.json)
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier

PROJ = Path("/lustre/proyectos/p082")
A5 = Path(sys.argv[1])
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
OUT = {}


def auc(y, s):
    r = rankdata(s)
    p = y.sum()
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))


def rk(v):
    return (rankdata(v) - 1) / (len(v) - 1)


# ---------------- features por estudio ----------------
m = pd.read_parquet(A5 / "meta_series.parquet")
for c in m.columns:
    if m[c].dtype == object:
        m[c] = m[c].replace({"": None, "None": None})
tok = lambda s: re.findall(r"[a-z]+", str(s).lower()) if s else []
vocab = Counter(t for s in m.SeriesDescription.dropna() for t in set(tok(s)))
vocab = [t for t, n in vocab.most_common(80) if n >= 30]
coil = Counter(m.ReceiveCoilName.dropna().str.lower()).most_common(25)
modelos = Counter(m.ManufacturerModelName.dropna().str.lower()).most_common(30)
filas = []
for u, g in m.groupby("StudyInstanceUID"):
    f = {"StudyInstanceUID": u, "n_series": len(g), "n_files_tot": g.n_files.sum()}
    f["sexo_F"] = float((g.PatientSex == "F").any()); f["sexo_M"] = float((g.PatientSex == "M").any())
    f["contraste"] = float(g.ContrastBolusAgent.notna().any())
    f["lat_R"] = float((g.Laterality == "R").any()); f["lat_L"] = float((g.Laterality == "L").any())
    f["campo"] = pd.to_numeric(g.MagneticFieldStrength, errors="coerce").max()
    f["grosor_med"] = pd.to_numeric(g.SliceThickness, errors="coerce").median()
    f["tr_med"] = pd.to_numeric(g.RepetitionTime, errors="coerce").median()
    f["te_max"] = pd.to_numeric(g.EchoTime, errors="coerce").max()
    f["filas_max"] = pd.to_numeric(g.Rows, errors="coerce").max()
    f["n_3d"] = float((g.MRAcquisitionType == "3D").sum())
    for p in ("Sagittal", "Coronal", "Axial"):
        f["n_" + p] = float((g.plane == p).sum())
    toks = Counter(t for s in g.SeriesDescription.dropna() for t in set(tok(s)))
    for t in vocab:
        f["tok_" + t] = float(toks.get(t, 0))
    cs = set(g.ReceiveCoilName.dropna().str.lower())
    for k, (c, _) in enumerate(coil):
        f["coil_%d" % k] = float(c in cs)
    ms = set(g.ManufacturerModelName.dropna().str.lower())
    for k, (c, _) in enumerate(modelos):
        f["modelo_%d" % k] = float(c in ms)
    filas.append(f)
X = pd.DataFrame(filas).set_index("StudyInstanceUID")
OUT["n_features"] = X.shape[1]
OUT["contraste_frac_estudios"] = round(float(X.contraste.mean()), 3)

# ---------------- etiquetas, OOF de imagen v10 ----------------
tr = pd.read_csv(PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection/train.csv", dtype={"StudyInstanceUID": str})
gold = tr.dropna(subset=L).set_index("StudyInstanceUID")
jev = pd.read_parquet(PROJ / "runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet").set_index("StudyInstanceUID")
man = (PROJ / "envios/v10_cv25_effA_cnxA_effB_r34A_r34C_140mm/MANIFIESTO.txt").read_text()
fams = {}
for mm in re.finditer(r"^(\w+) job (\d+)\s", man, re.M):
    fams.setdefault(mm.group(1), []).append(mm.group(2))
oof = []
for f_, jobs in fams.items():
    parts = []
    for j in jobs:
        z = np.load(PROJ / "runs" / j / "predicciones.npz", allow_pickle=True)
        parts.append(pd.DataFrame(z["p_interna"][-1], columns=L, index=[str(u) for u in z["uid_interna"]]))
    oof.append(pd.concat(parts))
uids = sorted(set(X.index) & set(jev.index) & set.intersection(*[set(d.index) for d in oof]))
P = np.mean([d.loc[uids, L].values for d in oof], axis=0)
Xu = X.loc[uids].values.astype(np.float32)
Y = jev.loc[uids, L].values
fold = np.array([int(hashlib.md5(u.encode()).hexdigest(), 16) % 5 for u in uids])
OUT["n"] = len(uids)

meta_oof = np.full(P.shape, np.nan)
for k in range(len(L)):
    for f in range(5):
        trm = (fold != f) & ~np.isnan(Y[:, k])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                             l2_regularization=1.0, random_state=0)
        clf.fit(Xu[trm], (Y[trm, k] >= 0.5).astype(int))
        meta_oof[fold == f, k] = clf.predict_proba(Xu[fold == f])[:, 1]
res_a, res_b = {}, {}
Rimg = np.column_stack([rk(P[:, k]) for k in range(len(L))])
Rmet = np.column_stack([rk(meta_oof[:, k]) for k in range(len(L))])
WS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
mix_nest = np.zeros_like(Rimg)
w_elegidos = {}
for k, l in enumerate(L):
    ok = ~np.isnan(Y[:, k])
    y = (Y[:, k] >= 0.5).astype(int)
    res_a[l] = round(auc(y[ok], meta_oof[ok, k]), 4)
    ws = []
    for f in range(5):   # w elegido en las otras 4 particiones, aplicado a la que queda fuera
        trm = ok & (fold != f)
        mejor = max(WS, key=lambda w: auc(y[trm], (1 - w) * Rimg[trm, k] + w * Rmet[trm, k]))
        te = fold == f
        mix_nest[te, k] = (1 - mejor) * Rimg[te, k] + mejor * Rmet[te, k]
        ws.append(mejor)
    w_elegidos[l] = ws
    res_b[l] = round(auc(y[ok], mix_nest[ok, k]) - auc(y[ok], Rimg[ok, k]), 4)
rng = np.random.default_rng(0)
difs = []
for _ in range(300):
    ix = rng.integers(0, len(uids), len(uids))
    d = []
    for k in range(len(L)):
        ok = ~np.isnan(Y[ix, k]); y = (Y[ix, k][ok] >= 0.5).astype(int)
        d.append(auc(y, mix_nest[ix, k][ok]) - auc(y, Rimg[ix, k][ok]))
    difs.append(np.mean(d))
OUT["meta_solo_auc_jev"] = res_a
OUT["meta_solo_macro"] = round(float(np.mean(list(res_a.values()))), 4)
OUT["imagen_mas_meta_delta_por_hallazgo_anidado"] = res_b
OUT["imagen_mas_meta_delta_macro"] = round(float(np.mean(list(res_b.values()))), 4)
OUT["imagen_mas_meta_ic95"] = [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)]
OUT["w_elegidos"] = w_elegidos

# ---------------- Gold: metadatos (todo no-gold) + Raptor ----------------
z = np.load(PROJ / "runs_joako/2089458/preds_gold.npz", allow_pickle=True)
gids = [str(i) for i in z["ids"]]
gok = [u for u in gids if u in X.index]
Yg = gold.loc[gok, L].values.astype(int)
Rg = z["mezcla_65_35"][[gids.index(u) for u in gok]]
nogold = [u for u in X.index if u in jev.index and u not in gold.index]
gres = {}
for k, l in enumerate(L):
    yy = jev.loc[nogold, l].values
    okk = ~np.isnan(yy)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(X.loc[nogold].values[okk].astype(np.float32), (yy[okk] >= 0.5).astype(int))
    mg = clf.predict_proba(X.loc[gok].values.astype(np.float32))[:, 1]
    gres[l] = {"meta_solo": round(auc(Yg[:, k], mg), 3), "raptor": round(auc(Yg[:, k], Rg[:, k]), 3),
               "raptor+meta_0.1": round(auc(Yg[:, k], 0.9 * rk(Rg[:, k]) + 0.1 * rk(mg)), 3)}
OUT["gold"] = gres
OUT["gold_macro"] = {k: round(float(np.mean([v[k] for v in gres.values()])), 4) for k in ("meta_solo", "raptor", "raptor+meta_0.1")}

# ---------------- casi-duplicados ----------------
t = np.load(A5 / "miniaturas.npz")
tu = [str(u) for u in t["uids"]]
T = t["thumbs"].astype(np.float32).reshape(len(tu), -1)
T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-6)
C = (T @ T.T) / T.shape[1]
np.fill_diagonal(C, -1)
nn = C.argmax(1)
top = C[np.arange(len(tu)), nn]
OUT["dup_corr_top1_percentiles"] = {str(p): round(float(np.percentile(top, p)), 3) for p in (50, 90, 99, 99.9)}
OUT["dup_pares_corr_gt_0.95"] = int((top > 0.95).sum())
OUT["dup_pares_corr_gt_0.90"] = int((top > 0.90).sum())
lab = jev.reindex(tu)[L].values
conc = {}
for thr in (0.8, 0.9, 0.95):
    sel = np.flatnonzero(top > thr)
    if len(sel) < 10:
        continue
    a, b = lab[sel], lab[nn[sel]]
    rnd = lab[rng.permutation(len(tu))[:len(sel)]]
    okp = ~np.isnan(a) & ~np.isnan(b)
    okr = ~np.isnan(a) & ~np.isnan(rnd)
    conc[str(thr)] = {"n": int(len(sel)), "acuerdo_vecino": round(float(((a >= .5) == (b >= .5))[okp].mean()), 3),
                      "acuerdo_azar": round(float(((a >= .5) == (rnd >= .5))[okr].mean()), 3)}
OUT["dup_acuerdo_etiquetas"] = conc
(Path(__file__).with_name("a6.json")).write_text(json.dumps(OUT, indent=1, ensure_ascii=False, default=str))
print(json.dumps({k: OUT[k] for k in ("n", "n_features", "contraste_frac_estudios", "meta_solo_macro", "imagen_mas_meta_delta_macro",
                                      "imagen_mas_meta_ic95", "gold_macro", "dup_corr_top1_percentiles", "dup_pares_corr_gt_0.95",
                                      "dup_acuerdo_etiquetas")}, indent=1, ensure_ascii=False))
print("meta solo", OUT["meta_solo_auc_jev"])
print("delta por hallazgo", OUT["imagen_mas_meta_delta_por_hallazgo_anidado"])
