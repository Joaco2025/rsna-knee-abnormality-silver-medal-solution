"""¿Pueden los 58 Gold ELEGIR los pesos por hallazgo sin sobreajustar?  Validación cruzada repetida DENTRO de Gold.

Gold son etiquetas de experto de train.csv (las mismas que puntúan el test).  Ajustar pocos parámetros con ellas es usar los
datos de entrenamiento; el riesgo es el sobreajuste, que se mide aquí fuera de muestra.

Mezcla (forma general de E-sust): por hallazgo, A3 = a·rr0 + s·pct(S6) + r·pct(R6), con a + s + r = 1 (rejilla de 0.1),
rr0 = 0.65·pct(R384) + 0.35·pct(R224); luego Part B (lector, pesos fijos) y desatajo de Synovitis, como en el notebook.
Equivalencia con el notebook: wr = a + s, ws = s / (a + s).

Referencias (sin ajuste): heliosli (0.959 = E-sust-dsj) y OAI-FOV (corriendo en Kaggle).
Aprendido: en cada pliegue, por hallazgo, el (a, s, r) de la rejilla con mejor AUC en los estudios de entrenamiento del
pliegue (empate -> el más cercano a OAI-FOV); se puntúan los estudios del pliegue de validación.  AUC fuera de muestra con
las puntuaciones de todos los pliegues juntas.  5 pliegues estratificados por la suma de etiquetas, 20 repeticiones.
Salida: cv_pesos_gold.json (incluye los pesos medios elegidos, «bagging», por si pasan).
"""
import itertools, json, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import mezcla_lib as M
from ideas_gold import C, Y
from sklearn.metrics import roc_auc_score

n = Y.shape[0]
RR0 = np.column_stack([M.bloque_raptor(C["r384"], C["r224"], j) for j in range(12)])
S = np.column_stack([M.pct(C["s6"][:, j]) for j in range(12)])
R = np.column_stack([M.pct(C["r6"][:, j]) for j in range(12)])
LEC = np.column_stack([M.pct(C["lector"][:, j]) for j in range(12)])
MET = M.pct(C["meta"])
JS = M.LAB.index("Synovitis")
GRID = [(a / 10, s / 10, round(1 - a / 10 - s / 10, 1)) for a in range(11) for s in range(11 - a)]


def pesos_receta(wrc, ws=0.25):
    out = []
    for c in M.LAB:
        wr = wrc.get(c, 0.5)
        out.append((wr * (1 - ws), wr * ws, 1 - wr))
    return out


HELIO = pesos_receta(M.W_RAPTOR_C)
OAIFOV = pesos_receta({**M.W_RAPTOR_C, "Lateral OA": 0.1, "PF OA": 0.3, "Synovitis": 0.4, "Effusion": 0.75, "Fracture": 0.75})


def columna(j, w, idx=None):
    """Puntuación final de la columna j con pesos w=(a,s,r) sobre los estudios idx (rangos de componentes sobre los 58,
    como en el notebook; Part B y desatajo re-rankean dentro de idx, que no cambia el orden relativo de forma relevante)."""
    a, s, r = w
    b = a * RR0[:, j] + s * S[:, j] + r * R[:, j]
    ow = M.OWN_W_PER_TARGET[M.LAB[j]]
    f = (1 - ow) * M.pct(b) + ow * LEC[:, j]
    if j == JS:
        f = M.pct(f) - M.LAMBDA_DSJ * MET
    return f if idx is None else f[idx]


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


ref_auc = {nom: np.array([auc(Y[:, j], columna(j, W[j])) for j in range(12)]) for nom, W in (("heliosli", HELIO), ("oai_fov", OAIFOV))}

rng = np.random.default_rng(0)
orden_estr = np.argsort(Y.sum(1) + rng.random(n) * 0.1)
oof_rep, elegidos = [], []
for rep in range(20):
    perm = rng.permutation(n)
    fold = np.empty(n, int)
    # estratificación simple: reparto circular sobre el orden por número de hallazgos (desempate aleatorio por repetición)
    for i, u in enumerate(sorted(range(n), key=lambda u: (Y[u].sum(), perm[u]))):
        fold[u] = i % 5
    P = np.zeros((n, 12))
    for k in range(5):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        for j in range(12):
            mejor, mw = -1, None
            for w in GRID:
                v = auc(Y[tr, j], columna(j, w, tr))
                d = sum(abs(x - y) for x, y in zip(w, OAIFOV[j]))
                clave = (round(v, 6), -d)
                if mejor == -1 or clave > mejor:
                    mejor, mw = clave, w
            elegidos.append((j, mw))
            P[va, j] = columna(j, mw, va)
    oof_rep.append(np.array([auc(Y[:, j], P[:, j]) for j in range(12)]))
oof = np.array(oof_rep)
bag = {}
for j in range(12):
    ws = np.array([w for jj, w in elegidos if jj == j])
    bag[M.LAB[j]] = [round(float(x), 3) for x in ws.mean(0)]
BAG = [tuple(bag[c]) for c in M.LAB]
bag_auc_insample = np.array([auc(Y[:, j], columna(j, BAG[j])) for j in range(12)])
OUT = {
    "macro_heliosli": round(float(ref_auc["heliosli"].mean()), 4),
    "macro_oai_fov": round(float(ref_auc["oai_fov"].mean()), 4),
    "macro_aprendido_fuera_de_muestra": {"media": round(float(oof.mean(1).mean()), 4), "sd": round(float(oof.mean(1).std()), 4),
                                          "frac_reps_mejor_que_oai_fov": round(float((oof.mean(1) > ref_auc["oai_fov"].mean()).mean()), 2)},
    "por_hallazgo_fuera_de_muestra_menos_oai_fov": {c: round(float(oof[:, j].mean() - ref_auc["oai_fov"][j]), 4) for j, c in enumerate(M.LAB)},
    "pesos_bagging_a_s_r": bag,
    "pesos_oai_fov_a_s_r": {c: [round(x, 3) for x in OAIFOV[j]] for j, c in enumerate(M.LAB)},
    "macro_bagging_en_muestra_OPTIMISTA": round(float(bag_auc_insample.mean()), 4),
}
(AQUI / "cv_pesos_gold.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(json.dumps(OUT, indent=1, ensure_ascii=False))
