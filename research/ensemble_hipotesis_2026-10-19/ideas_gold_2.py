"""Ideas no convencionales, 2.ª ronda (Gold-58, salidas reales de gold_dump).  Criterio igual que ideas_gold.py, fijado antes.

I1  DESAJUSTE DE DEFINICIÓN DE ETIQUETA (OAI).  Mecanismo previo a mirar Gold: Raptor es el único componente supervisado con
    OAI, y solo en Lateral OA, PF OA y Synovitis (training_code/coatnet/RUNS.md).  OAI define OA y sinovitis con MOAKS/KL,
    no con el criterio de los expertos de la competición.  Predicción: (a) esos 3 son los peores de Raptor en Gold;
    (b) en esos 3, la mezcla sin Raptor (su hueco lo ocupa S6, mismas etiquetas JEV sin OAI) es mejor.
    Variantes fijadas: I1a = en los 3 hallazgos OAI, la mitad Raptor de la mezcla = S6 (wr y todo lo demás igual);
                       I1b = en los 3 hallazgos OAI, wr a la mitad.
    Se mide sobre E + dsj y sobre E-sust-dsj.
I2  LADO / ORIENTACIÓN.  Si un componente confunde medial con lateral en algunos estudios, sus Medial y Lateral (OA, menisco)
    quedan intercambiados.  Diagnóstico: por estudio, d = rango(Medial) − rango(Lateral) de cada componente; estudios donde
    un componente tiene d de signo opuesto y grande frente al consenso.  Prueba: para esos estudios, ¿intercambiar Med/Lat
    de ese componente acerca su predicción a la etiqueta?  Solo diagnóstico (no hay corrección que enviar sin más trabajo).
Uso: python -I -B ideas_gold_2.py -> ideas_gold_2.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import mezcla_lib as M
import evaluar_gold as EG
from ideas_gold import comparar, C, Y, ids  # mismos datos, mismo bootstrap (semilla 0), misma regla de candidata

OAI = ["Lateral OA", "PF OA", "Synovitis"]
OUT = {}
# (a) ranking de los hallazgos de Raptor en Gold
ra = EG.aucs(Y, np.column_stack([M.bloque_raptor(C["r384"], C["r224"], j) for j in range(12)]))
orden = [M.LAB[k] for k in np.argsort(ra)]
OUT["I1a_peores3_raptor"] = orden[:3]
OUT["I1a_coincide_con_OAI"] = sorted(orden[:3]) == sorted(OAI)
for nombre, src in (("r6", C["r6"]), ("s6", C["s6"]), ("lector", C["lector"])):
    a = EG.aucs(Y, src)
    OUT[f"peores3_{nombre}"] = [M.LAB[k] for k in np.argsort(a)[:3]]
OUT["auc_por_componente_en_OAI"] = {nombre: {L: round(float(x), 3) for L, x in zip(M.LAB, EG.aucs(Y, src)) if L in OAI}
                                     for nombre, src in (("raptor", np.column_stack([M.bloque_raptor(C["r384"], C["r224"], j) for j in range(12)])),
                                                         ("r6", C["r6"]), ("s6", C["s6"]), ("lector", C["lector"]))}

# (b) variantes: en los hallazgos OAI, Raptor -> S6 (I1a) o wr a la mitad (I1b)
_bloque = M.bloque_raptor
_wrc = dict(M.W_RAPTOR_C)


def bloque_sin_oai(r384, r224, j, s6=None, ws=0.0, h2=None):
    if M.LAB[j] in OAI:
        return M.pct(np.asarray(C["s6"], dtype=np.float64)[:, j])
    return _bloque(r384, r224, j, s6, ws, h2)


E_dsj = M.desatajo(M.receta("E", C), C["meta"]); Es_dsj = M.receta("Esust_dsj", C)
M.bloque_raptor = bloque_sin_oai
V1 = M.desatajo(M.receta("E", C), C["meta"]); V2 = M.receta("Esust_dsj", C)
M.bloque_raptor = _bloque
OUT["I1a_OAI_raptor_por_S6_sobre_E_dsj"] = comparar(V1, E_dsj, set(OAI))
OUT["I1a_OAI_raptor_por_S6_sobre_Esust_dsj"] = comparar(V2, Es_dsj, set(OAI))
for L in OAI:
    M.W_RAPTOR_C[L] = M.W_RAPTOR_C.get(L, M.W_RAPTOR) / 2
V1 = M.desatajo(M.receta("E", C), C["meta"]); V2 = M.receta("Esust_dsj", C)
M.W_RAPTOR_C.clear(); M.W_RAPTOR_C.update(_wrc)
OUT["I1b_OAI_wr_mitad_sobre_E_dsj"] = comparar(V1, E_dsj, set(OAI))
OUT["I1b_OAI_wr_mitad_sobre_Esust_dsj"] = comparar(V2, Es_dsj, set(OAI))

# I2: diagnóstico de lado
pares = [("Medial OA", "Lateral OA"), ("Medial Meniscus", "Lateral Meniscus")]
comp = {"r384": C["r384"], "r224": C["r224"], "r6": C["r6"], "s6": C["s6"], "lector": C["lector"]}
diag = {}
for med, lat in pares:
    jm, jl = M.LAB.index(med), M.LAB.index(lat)
    D = {k: M.pct(v[:, jm]) - M.pct(v[:, jl]) for k, v in comp.items()}
    cons = np.median(np.column_stack(list(D.values())), axis=1)
    ylab = Y[:, jm] - Y[:, jl]                       # +1 solo medial, −1 solo lateral, 0 ambos/ninguno
    for k, d in D.items():
        sosp = np.where((np.sign(d) != np.sign(cons)) & (np.abs(d - cons) > 0.5))[0]
        # ¿la etiqueta da la razón al consenso (lado correcto) o al componente?
        a_favor_cons = int(np.sum(np.sign(ylab[sosp]) == np.sign(cons[sosp]))) if len(sosp) else 0
        a_favor_comp = int(np.sum(np.sign(ylab[sosp]) == np.sign(d[sosp]))) if len(sosp) else 0
        diag[f"{k} {med[:6]}/{lat[:7]}"] = {"sospechosos": int(len(sosp)), "etiqueta_con_consenso": a_favor_cons,
                                            "etiqueta_con_componente": a_favor_comp}
OUT["I2_lado"] = diag
(AQUI / "ideas_gold_2.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(json.dumps(OUT, indent=1, ensure_ascii=False))
