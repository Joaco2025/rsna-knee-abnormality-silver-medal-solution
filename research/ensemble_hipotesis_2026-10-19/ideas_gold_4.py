"""4.ª ronda: ¿los modelos compactos (R6, S6) fallan más en los estudios cuyo anclaje articular SKM-TEA salió mal?

Mecanismo (extensión por estudio del de campo de visión): R6/S6 recortan alrededor de la articulación localizada por la red
SKM-TEA.  Si no la encuentra (método «slab_prior», posición por defecto) o alguna serie queda fuera de campo, el recorte
puede no contener la rodilla bien centrada -> el compacto pierde; Raptor (corte completo) no depende del anclaje.
Fuente del anclaje: runs_joako/2092983/anchors (precompute_anchors.py de nartaa, el mismo de S6).
Medida por estudio: para cada hallazgo, «error de rango» = rango del estudio entre los negativos si es positivo (debería
ser alto) o entre los positivos si es negativo (debería ser bajo) -> pérdida = fracción de pares mal ordenados en los que
participa el estudio (es su contribución al 1 − AUC).  Se compara compacto frente a Raptor en estudios degradados y normales.
Variante fijada: en los estudios degradados, wr = 0.75 en TODOS los hallazgos (más Raptor), sobre E-sust-dsj-OAI-FOV.
n degradados = 7: solo orientativo.
"""
import json, glob, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import mezcla_lib as M
import evaluar_gold as EG
from ideas_gold import C, Y, ids, comparar

anc = {}
for f in glob.glob("/lustre/proyectos/p082/runs_joako/2092983/anchors/*.json"):
    d = json.loads(Path(f).read_text())
    if "study" in d:
        anc[d["study"]] = d
deg = np.array([anc[u]["method"] == "slab_prior" or not all(s["in_field"] for s in anc[u]["series"].values()) for u in ids])


def perdida(P):
    """(n,) pérdida media por estudio: fracción de pares (pos, neg) mal ordenados en los que participa, media sobre hallazgos."""
    L = np.zeros(len(ids))
    for k in range(12):
        p, y = P[:, k], Y[:, k].astype(bool)
        W = (p[:, None] < p[None, :]) + 0.5 * (p[:, None] == p[None, :])   # W[i, j]: i por debajo de j
        lp = (W[np.ix_(y, ~y)]).mean(1)          # positivo i por debajo de negativos
        ln = (W.T[np.ix_(~y, y)]).mean(1)        # negativo j por encima de positivos
        L[y] += lp; L[~y] += ln
    return L / 12


rap = np.column_stack([M.bloque_raptor(C["r384"], C["r224"], j) for j in range(12)])
OUT = {"n_degradados": int(deg.sum()), "n_normales": int((~deg).sum())}
for nombre, P in (("raptor", rap), ("r6", C["r6"]), ("s6", C["s6"])):
    l = perdida(P)
    OUT[f"perdida_{nombre}"] = {"degradados": round(float(l[deg].mean()), 4), "normales": round(float(l[~deg].mean()), 4)}
base = dict(M.W_RAPTOR_C)
W = {"Lateral OA": 0.1, "PF OA": 0.3, "Synovitis": 0.4, "Effusion": 0.75, "Fracture": 0.75}


def oai_fov(peso_deg=None):
    M.W_RAPTOR_C.clear(); M.W_RAPTOR_C.update(base); M.W_RAPTOR_C.update(W)
    if peso_deg is None:
        P = M.receta("Esust_dsj", C)
    else:
        # wr por estudio: se reescribe la mezcla A3 con wr_i = peso_deg en los degradados (mismo resto de la receta)
        fin = np.zeros((len(ids), 12))
        for j, c in enumerate(M.LAB):
            rr = M.bloque_raptor(C["r384"], C["r224"], j, C["s6"], M.W_S6)
            wr = np.where(deg, peso_deg, M.W_RAPTOR_C.get(c, M.W_RAPTOR))
            fin[:, j] = wr * rr + (1 - wr) * M.pct(C["r6"][:, j])
        P = M.desatajo(M.parte_b(fin, C["lector"]), C["meta"])
    M.W_RAPTOR_C.clear(); M.W_RAPTOR_C.update(base)
    return P


ref = oai_fov()
OUT["anclaje_degradado_wr075_sobre_OAI_FOV"] = comparar(oai_fov(0.75), ref, set(M.LAB))
Path(AQUI / "ideas_gold_4.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(json.dumps(OUT, indent=1, ensure_ascii=False))
