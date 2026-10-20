"""Ideas no convencionales sobre la receta ganadora, medidas en Gold-58 con las salidas REALES de gold_dump (salida_v1).

Referencia: E + dsj (Gold 0.9398) y E-sust-dsj (0.959 público; Gold 0.9394).  Una variable por prueba.  CRITERIO FIJADO ANTES
DE EJECUTAR: una idea pasa a «candidata a envío» solo si sube el macro frente a su referencia con P(mejora) >= 0.9 y ningún
hallazgo fuera de los que la idea toca baja más de 0.005.  Todo lo demás se informa, pero no se envía.

  T1 desatajo también en Fracture: criterio de A8 (los metadatos predicen el informe, 0.609, pero no al experto, 0.429),
     λ 0.1 fijo.  Modelo lineal de metadatos para Fracture con meta_rasgos (los mismos rasgos que el desplegado), ajustado
     sobre los 4,349 no-Gold con etiquetas JEV, igual que exportar_modelo_meta.py para Synovitis.
  T2 quitar el lector (Part B, 1/10): inversión; si no aporta, libera ~0.5 h de Kaggle.
  T3 mezclar en puntuaciones normales (Φ⁻¹ del rango) en lugar de rangos uniformes: da más peso a los extremos (los
     estudios en los que un modelo está muy seguro); es un cambio de espacio, no de pesos.
  T4 desatajo con modelo NO lineal (el HGB de A6/A7) en lugar del lineal desplegado: ¿el límite es el modelo de metadatos?
     Solo informativo (no portable sin más trabajo).
Uso: python -I -B ideas_gold.py  -> ideas_gold.json
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import mezcla_lib as M
import evaluar_gold as EG

PROJ = Path("/lustre/proyectos/p082")
sys.path.insert(0, str(PROJ / "runs_joako/exp_Esust_dsj"))
import meta_rasgos as MR

z = np.load(PROJ / "runs_joako/exp_gold_dump/componentes_gold.npz", allow_pickle=True)
ids = [str(i) for i in z["ids"]]
C = {k: np.asarray(z[k], np.float64) for k in z.files if k != "ids"}
tr = pd.read_csv(PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection/train.csv", dtype={"StudyInstanceUID": str})
gold = tr.dropna(subset=M.LAB).set_index("StudyInstanceUID")
Y = gold.loc[ids, M.LAB].to_numpy().astype(int)
n = len(ids)
rng = np.random.default_rng(0)
boot = [ix for ix in (rng.integers(0, n, n) for _ in range(1000)) if (Y[ix].min(0) == 0).all() and (Y[ix].max(0) == 1).all()]
CNT = np.stack([np.bincount(ix, minlength=n) for ix in boot]).astype(np.float64)


def comparar(P, ref, toca):
    a, b = EG.aucs(Y, P), EG.aucs(Y, ref)
    bd = (EG.aucs_boot(Y, P, CNT) - EG.aucs_boot(Y, ref, CNT)).mean(1)
    d = a - b
    fuera = [k for k, L in enumerate(M.LAB) if L not in toca]
    peor_fuera = float(d[fuera].min()) if fuera else 0.0
    pasa = bool(d.mean() > 0 and (bd > 0).mean() >= 0.9 and peor_fuera >= -0.005)
    return {"macro": round(float(a.mean()), 4), "delta": round(float(d.mean()), 5),
            "ic95": [round(float(np.percentile(bd, 2.5)), 5), round(float(np.percentile(bd, 97.5)), 5)],
            "p_mejora": round(float((bd > 0).mean()), 3), "por_hallazgo": {L: round(float(x), 4) for L, x in zip(M.LAB, d) if abs(x) > 1e-9},
            "peor_fuera_de_lo_tocado": round(peor_fuera, 4), "CANDIDATA": pasa}


E = M.receta("E", C)
E_dsj = M.desatajo(E, C["meta"])
Es_dsj = M.receta("Esust_dsj", C)
OUT = {"referencias": {"E": round(float(EG.aucs(Y, E).mean()), 4), "E_dsj": round(float(EG.aucs(Y, E_dsj).mean()), 4),
                       "Esust_dsj": round(float(EG.aucs(Y, Es_dsj).mean()), 4)}}

# ---- T1: modelo de metadatos para Fracture (mismo procedimiento que exportar_modelo_meta.py) ---------------------------
m = pd.read_parquet(PROJ / "runs_joako/2093187/meta_series.parquet")
for c in MR.CAMPOS:
    m[c] = m[c].where(m[c].notna() & (m[c].astype(str) != "") & (m[c].astype(str) != "None"), None)
mod = json.loads((PROJ / "runs_joako/exp_Esust_dsj/desatajo_synovitis.json").read_text())
X = MR.rasgos(m, mod["vocab"], mod["bobinas"], mod["modelos"])
jev = pd.read_parquet(PROJ / "runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet").set_index("StudyInstanceUID")
nog = [u for u in X.index if u in jev.index and u not in gold.index]
med = X.loc[nog].median()
Xf = X.fillna(med)
mu, sd = Xf.loc[nog].mean(), Xf.loc[nog].std().replace(0, 1.0).fillna(1.0)
Zs = lambda U: ((Xf.loc[U] - mu) / sd).values
scores = {}
for L in ("Fracture", "Synovitis"):
    y = jev.loc[nog, L].values
    ok = ~np.isnan(y)
    lr = LogisticRegression(C=0.05, max_iter=3000).fit(Zs(nog)[ok], (y[ok] >= 0.5).astype(int))
    scores[L] = lr.decision_function(Zs(ids))
assert np.allclose(M.pct(scores["Synovitis"]), M.pct(C["meta"]), atol=1e-9), "el modelo de Synovitis no reproduce el desplegado"
jf = M.LAB.index("Fracture")
from sklearn.metrics import roc_auc_score
OUT["T1_meta_fracture_auc_gold"] = round(float(roc_auc_score(Y[:, jf], scores["Fracture"])), 3)


def dsj_col(P, col, s, lam=0.1):
    P = np.array(P, dtype=np.float64)
    j = M.LAB.index(col)
    P[:, j] = M.pct(P[:, j] - lam * M.pct(s))
    return P


OUT["T1_dsj_fracture_sobre_E_dsj"] = comparar(dsj_col(E_dsj, "Fracture", scores["Fracture"]), E_dsj, {"Fracture"})
OUT["T1_dsj_fracture_sobre_Esust_dsj"] = comparar(dsj_col(Es_dsj, "Fracture", scores["Fracture"]), Es_dsj, {"Fracture"})

# ---- T2: sin lector -----------------------------------------------------------------------------------------------------
_own = dict(M.OWN_W_PER_TARGET)
M.OWN_W_PER_TARGET.update({k: 0.0 for k in M.OWN_W_PER_TARGET})
E0 = M.desatajo(M.receta("E", C), C["meta"]); Es0 = M.receta("Esust_dsj", C)
M.OWN_W_PER_TARGET.update(_own)
OUT["T2_sin_lector_E_dsj"] = comparar(E0, E_dsj, set(M.LAB))
OUT["T2_sin_lector_Esust_dsj"] = comparar(Es0, Es_dsj, set(M.LAB))

# ---- T3: espacio normal (Φ⁻¹ del rango) en todas las mezclas --------------------------------------------------------------
_pct = M.pct
M.pct = lambda v: norm.ppf((pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average").values - 0.5) / len(v))
N_E = M.receta("E", C); N_Es = M.receta("Esust_dsj", C)
M.pct = _pct
N_E = M.desatajo(np.column_stack([_pct(N_E[:, k]) for k in range(12)]), C["meta"])
N_Es = np.column_stack([_pct(N_Es[:, k]) for k in range(12)])
OUT["T3_normal_E_dsj"] = comparar(N_E, E_dsj, set(M.LAB))
OUT["T3_normal_Esust_dsj"] = comparar(N_Es, Es_dsj, set(M.LAB))

# ---- T4: desatajo de Synovitis con el meta HGB de A7 (informativo) --------------------------------------------------------
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = jev.loc[nog, "Synovitis"].values; ok = ~np.isnan(y)
    h = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, random_state=0).fit(X.loc[nog].values[ok].astype(np.float32), (y[ok] >= 0.5).astype(int))
    sh = h.predict_proba(X.loc[ids].values.astype(np.float32))[:, 1]
    OUT["T4_dsj_hgb_vs_lineal_sobre_E"] = comparar(M.desatajo(E, sh), E_dsj, {"Synovitis"})
except Exception as e:
    OUT["T4_error"] = repr(e)

(AQUI / "ideas_gold.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(json.dumps(OUT, indent=1, ensure_ascii=False))
