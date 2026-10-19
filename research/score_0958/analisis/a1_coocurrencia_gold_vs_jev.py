"""A1: ¿la estructura entre hallazgos de las etiquetas de experto (Gold-58, puntuadas como el leaderboard) difiere de la
de las etiquetas de informe con que se entrenaron S6/Raptor (JEV-blend, 4,349 estudios)?

Si en Gold la co-ocurrencia es mucho más fuerte (p. ej. Synovitis~Effusion) que en las etiquetas de entrenamiento, los
modelos la habrán aprendido de menos y una mezcla entre hallazgos podría recuperar AUC frente a etiquetas de experto
aunque el meta-apilado sobre OOF (etiquetas de informe) saliera negativo.  Solo lee CSV/parquet; CPU, segundos.
Salida: a1_coocurrencia.json junto a este script.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = Path("/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection")
JEV = Path("/lustre/proyectos/p082/runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet")
t = pd.read_csv(D / "train.csv")
L = list(t.columns[2:])
g = t.dropna(subset=L)
j = pd.read_parquet(JEV)
j = j[~j.StudyInstanceUID.isin(g.StudyInstanceUID)]

out = {"n_gold": len(g), "n_jev": len(j), "prev_gold": g[L].mean().round(3).to_dict(),
       "prev_jev": j[L].mean().round(3).to_dict(), "nan_jev": j[L].isna().mean().round(3).to_dict()}
cg = g[L].astype(float).corr()
cj = j[L].astype(float).corr()
pares = []
for a in range(len(L)):
    for b in range(a + 1, len(L)):
        pares.append((L[a], L[b], round(cg.iloc[a, b], 3), round(cj.iloc[a, b], 3), round(cg.iloc[a, b] - cj.iloc[a, b], 3)))
pares.sort(key=lambda x: -abs(x[4]))
out["pares_mayor_diferencia"] = pares[:15]
# información cruzada en Gold: AUC de la etiqueta de experto de B como predictor de A (techo de lo que B aporta a A)
cruz = {}
for a in L:
    ya = g[a].values
    fila = {}
    for b in L:
        if a != b and 0 < ya.sum() < len(ya):
            fila[b] = round(roc_auc_score(ya, g[b].values), 3)
    cruz[a] = dict(sorted(fila.items(), key=lambda kv: -kv[1])[:3])
out["auc_cruzado_gold_top3"] = cruz
Path(__file__).with_name("a1_coocurrencia.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: out[k] for k in ("prev_gold", "prev_jev")}, ensure_ascii=False))
for p in pares[:12]:
    print("%-17s %-17s gold %+.2f  jev %+.2f  dif %+.2f" % p)
for a, f in cruz.items():
    print("%-17s %s" % (a, f))
