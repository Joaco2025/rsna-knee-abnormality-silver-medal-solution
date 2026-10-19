"""A4: ¿hay sesgo de puntuación por centro (fabricante × campo) que una corrección pueda quitar para subir el AUC global?

OOF: paquete v10 de Owen (5 familias × 5 particiones; última época de p_interna de cada partición = el modelo que no vio
ese estudio), media de familias.  Etiquetas: JEV-blend (y = >= 0.5; NaN = celda sin etiqueta, fuera).  Centro: fabricante
normalizado × MagneticFieldStrength, desde las cabeceras DICOM (runs_joako/<job>/demografia/demografia_cabeceras.csv);
grupos con < 40 estudios -> "otros".
Correcciones evaluadas (todas sobre rangos 0-1 del OOF, por hallazgo):
  (a) SIN etiquetas (aplicable en el test): r' = r - lam * (media_grupo(r) - 0.5)
  (b) supervisada con ajuste cruzado (diagnóstico; no aplicable a los modelos públicos): sesgo_g = media(r_g) - rango medio
      esperado por la prevalencia de g, estimado en las otras 4 particiones; r' = r - lam * sesgo_g
Bootstrap pareado (500) del delta de macro AUC.  CPU, < 1 min.  Uso: python -I a4_sesgo_centro.py <demografia_cabeceras.csv>
"""
import hashlib, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJ = Path("/lustre/proyectos/p082")
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]
CAB = Path(sys.argv[1])
man = (PROJ / "envios/v10_cv25_effA_cnxA_effB_r34A_r34C_140mm/MANIFIESTO.txt").read_text()
fams = {}
for m in re.finditer(r"^(\w+) job (\d+)\s", man, re.M):
    fams.setdefault(m.group(1), []).append(m.group(2))
oof = {}
for f, jobs in fams.items():
    parts = []
    for j in jobs:
        z = np.load(PROJ / "runs" / j / "predicciones.npz", allow_pickle=True)
        parts.append(pd.DataFrame(z["p_interna"][-1], columns=L, index=[str(u) for u in z["uid_interna"]]))
    d = pd.concat(parts)
    assert d.index.is_unique, f
    oof[f] = d
uids = sorted(set.intersection(*[set(d.index) for d in oof.values()]))
P = np.mean([oof[f].loc[uids, L].values for f in oof], axis=0)
fold = np.array([int(hashlib.md5(u.encode()).hexdigest(), 16) % 5 for u in uids])
jev = pd.read_parquet(PROJ / "runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet")
jev = jev.set_index("StudyInstanceUID").reindex(uids)
Yraw = jev[L].values
cab = pd.read_csv(CAB, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


def fab(s):
    s = str(s).lower()
    for k, v in (("siemens", "SIE"), ("ge ", "GE"), ("ge_", "GE"), ("philips", "PHI"), ("toshiba", "CAN"), ("canon", "CAN"),
                 ("hitachi", "HIT"), ("fuji", "HIT")):
        if k in s + " ":
            return v
    return "OTR"


def campo(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "?"
    return "3T" if x > 2.2 else ("1.5T" if x > 1.2 else "<1.2T")


g = np.array([fab(cab.loc[u, "Manufacturer"]) + "_" + campo(cab.loc[u, "MagneticFieldStrength"]) if u in cab.index else "SIN"
              for u in uids])
cnt = pd.Series(g).value_counts()
g = np.array([x if cnt[x] >= 40 else "otros" for x in g])
grupos = sorted(set(g))


def auc(y, s):
    r = rankdata(s)
    p = y.sum()
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))


def macro(S, idx=None):
    out = []
    for k in range(len(L)):
        m = ~np.isnan(Yraw[:, k]) if idx is None else ~np.isnan(Yraw[idx, k])
        y = (Yraw[:, k] if idx is None else Yraw[idx, k])[m] >= 0.5
        s = (S[:, k] if idx is None else S[idx, k])[m]
        out.append(auc(y.astype(int), s))
    return np.array(out)


R = np.column_stack([(rankdata(P[:, k]) - 1) / (len(P) - 1) for k in range(len(L))])
gm = {x: R[g == x].mean(0) for x in grupos}
A = np.array([gm[x] for x in g]) - 0.5                 # desplazamiento de cada grupo (sin etiquetas)
# (b) cruzado: sesgo = media del rango en g - media del rango que tendría g si su puntuación siguiera solo a su prevalencia
Bsup = np.zeros_like(R)
for k_f in range(5):
    tr, te = fold != k_f, fold == k_f
    for k in range(len(L)):
        y = Yraw[:, k]
        for x in grupos:
            mt = tr & (g == x) & ~np.isnan(y)
            ma = tr & ~np.isnan(y)
            if mt.sum() < 20:
                continue
            # rango medio observado en g frente al rango medio de los estudios de train con la misma mezcla de clases
            pos, neg = (y >= 0.5), (y < 0.5)
            esp = (R[ma & pos, k].mean() * (y[mt] >= 0.5).mean() + R[ma & neg, k].mean() * (y[mt] < 0.5).mean())
            Bsup[te & (g == x), k] = R[mt, k].mean() - esp
base = macro(R)
rng = np.random.default_rng(0)
B = [rng.integers(0, len(R), len(R)) for _ in range(500)]
res = {"n": len(uids), "familias": list(fams), "grupos": cnt.to_dict(), "macro_base": round(base.mean(), 4),
       "por_hallazgo_base": dict(zip(L, np.round(base, 4))),
       "desplazamiento_grupo_rango": {x: dict(zip(L, np.round(gm[x] - 0.5, 3))) for x in grupos}}
for nombre, D in (("a_sin_etiquetas", A), ("b_supervisada_cruzada", Bsup)):
    res[nombre] = {}
    for lam in (0.25, 0.5, 1.0):
        S = R - lam * D
        a = macro(S)
        difs = np.array([macro(S, ix).mean() - macro(R, ix).mean() for ix in B])
        res[nombre][str(lam)] = {"macro": round(a.mean(), 4), "delta": round(a.mean() - base.mean(), 4),
                                 "ic95": [round(float(np.percentile(difs, 2.5)), 4), round(float(np.percentile(difs, 97.5)), 4)],
                                 "p_mejora": round(float((difs > 0).mean()), 3),
                                 "por_hallazgo": dict(zip(L, np.round(a - base, 4)))}
Path(__file__).with_name("a4_centro.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
print("n", res["n"], "grupos", res["grupos"], "macro_base", res["macro_base"])
for nombre in ("a_sin_etiquetas", "b_supervisada_cruzada"):
    for lam, v in res[nombre].items():
        print("%-22s lam %-4s delta %+.4f IC [%+.4f, %+.4f] P %.2f" % (nombre, lam, v["delta"], v["ic95"][0], v["ic95"][1], v["p_mejora"]))
