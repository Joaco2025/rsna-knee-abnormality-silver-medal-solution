"""3.ª ronda (Gold-58, salidas reales).  Criterio igual que ideas_gold.py (P >= 0.9, nada fuera de lo tocado < −0.005).

I3  CAMPO DE VISIÓN.  R6 y S6 (familia compacta de heliosli) puntúan recortes de 288 px centrados en anclas SKM-TEA de la
    articulación; Raptor puntúa el corte completo (384, con espejo).  Predicción mecanística: los hallazgos que pueden caer
    FUERA del recorte articular, el derrame (receso suprarrotuliano, craneal a la rótula) y las fracturas alejadas de la
    interlínea, son donde Raptor debe ganar a la familia compacta; los hallazgos articulares (meniscos, ligamentos,
    contusión subcondral), no.
    Variante fijada antes de ejecutar: wr = 0.75 en Effusion y Fracture (hoy 0.5); simétrica a I1b (OAI) en tamaño.
    Se mide (a) sola sobre E-sust-dsj (el 0.959) y (b) combinada con I1b (E-sust-dsj-OAI), para saber si se suman.
    Aviso de honestidad: la tabla por componente de Gold ya se había visto antes de formular I3 (§0); el mecanismo es
    arquitectónico, pero la elección de hallazgos no es ciega.  Por eso se comprueba además la predicción en los
    hallazgos articulares (deben NO favorecer a Raptor).
Uso: python -I -B ideas_gold_3.py -> ideas_gold_3.json
"""
import json, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import mezcla_lib as M
import evaluar_gold as EG
from ideas_gold import comparar, C, Y

FOV = ["Effusion", "Fracture"]
ART = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Contusion"]
OAI = {"Lateral OA": 0.1, "PF OA": 0.3, "Synovitis": 0.4}
rap = np.column_stack([M.bloque_raptor(C["r384"], C["r224"], j) for j in range(12)])
au = {k: EG.aucs(Y, v) for k, v in (("raptor", rap), ("r6", C["r6"]), ("s6", C["s6"]))}
OUT = {"raptor_menos_compacta": {L: round(float(au["raptor"][j] - max(au["r6"][j], au["s6"][j])), 3) for j, L in enumerate(M.LAB)}}
OUT["prediccion_FOV_raptor_gana"] = {L: OUT["raptor_menos_compacta"][L] > 0 for L in FOV}
OUT["prediccion_articulares_raptor_no_gana"] = {L: OUT["raptor_menos_compacta"][L] <= 0 for L in ART}

base = dict(M.W_RAPTOR_C)


def con(pesos):
    M.W_RAPTOR_C.clear(); M.W_RAPTOR_C.update(base); M.W_RAPTOR_C.update(pesos)
    P = M.receta("Esust_dsj", C)
    M.W_RAPTOR_C.clear(); M.W_RAPTOR_C.update(base)
    return P


ref = con({})
oai = con(OAI)
fov = con({L: 0.75 for L in FOV})
ambos = con({**OAI, **{L: 0.75 for L in FOV}})
OUT["I3_FOV_sobre_Esust_dsj"] = comparar(fov, ref, set(FOV))
OUT["I3_FOV_sobre_Esust_dsj_OAI"] = comparar(ambos, oai, set(FOV))
OUT["OAI_mas_FOV_sobre_Esust_dsj"] = comparar(ambos, ref, set(FOV) | set(OAI))
(AQUI / "ideas_gold_3.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(json.dumps(OUT, indent=1, ensure_ascii=False))
