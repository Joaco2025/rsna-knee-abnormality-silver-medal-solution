"""Evalúa las salidas de s6_gold_infer.py (job s6_gold_eval) dentro de la receta OAI-FOV, en Gold-58.

CRITERIO FIJADO ANTES DEL JOB (19-oct): una variante de S6 entra en un envío solo si, sustituyendo al S6 original dentro de
OAI-FOV (resto de componentes = salidas reales de Kaggle, exp_gold_dump/salida_v1), sube el macro con P(mejora) >= 0.9, NO baja
Lateral OA, PF OA ni Synovitis, y ningún otro hallazgo baja más de 0.005.
Referencia justa: el S6 ORIGINAL corrido en el MISMO job (stride 1, crop 288, misma GPU y precisión), no el de Kaggle.
Variantes: d42, d7 (destilados), super_logit (media de logits original + d42 + d7, lo que hace compact_infer con 3 ficheros),
orig_s3 (stride 3, H5), orig_c320 (imagen completa, campo de visión), super_logit_s3.
Uso: python -I -B evaluar_s6_variantes.py runs_joako/<job>/gold
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
with contextlib.redirect_stdout(io.StringIO()):
    import mezcla_lib as M
    import evaluar_gold as EG
    from ideas_gold import C, Y, ids, comparar

D = Path(sys.argv[1])


def cargar(nombre):
    p = D / f"s6_gold_{nombre}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    u = [str(x) for x in z["ids"]]
    o = [u.index(i) for i in ids]
    return {k: np.asarray(z[k], np.float64)[o] for k in ("s6", "s6_d42", "s6_d7")}


base = cargar("stride1")
assert base is not None, "falta s6_gold_stride1.npz"
s3, c320 = cargar("stride3"), cargar("stride1_crop320")
cands = {"orig_yuca": base["s6"], "d42": base["s6_d42"], "d7": base["s6_d7"],
         "super_logit": M.combinar_s6([base["s6"], base["s6_d42"], base["s6_d7"]], "logit"),
         "destilados_logit": M.combinar_s6([base["s6_d42"], base["s6_d7"]], "logit")}
if s3:
    cands["orig_s3"] = s3["s6"]
    cands["super_logit_s3"] = M.combinar_s6([s3["s6"], s3["s6_d42"], s3["s6_d7"]], "logit")
if c320:
    cands["orig_c320"] = c320["s6"]
    cands["super_logit_c320"] = M.combinar_s6([c320["s6"], c320["s6_d42"], c320["s6_d7"]], "logit")

W = {"Lateral OA": 0.1, "PF OA": 0.3, "Synovitis": 0.4, "Effusion": 0.75, "Fracture": 0.75}
b0 = dict(M.W_RAPTOR_C)


def oai_fov(s6):
    M.W_RAPTOR_C.update(W)
    P = M.receta("Esust_dsj", dict(C, s6=s6))
    M.W_RAPTOR_C.clear(); M.W_RAPTOR_C.update(b0)
    return P


OAI = ["Lateral OA", "PF OA", "Synovitis"]
ref = oai_fov(cands["orig_yuca"])
OUT = {"control_s6_kaggle_vs_yuca_spearman": {L: round(float(np.corrcoef(M.pct(C["s6"][:, j]), M.pct(cands["orig_yuca"][:, j]))[0, 1]), 4)
                                               for j, L in enumerate(M.LAB)},
       "oai_fov_con_s6_kaggle": round(float(EG.aucs(Y, oai_fov(C["s6"])).mean()), 4),
       "oai_fov_con_s6_yuca": round(float(EG.aucs(Y, ref).mean()), 4),
       "solo": {k: round(float(EG.aucs(Y, v).mean()), 4) for k, v in cands.items()},
       "en_oai_fov": {}}
for k, v in cands.items():
    if k == "orig_yuca":
        continue
    r = comparar(oai_fov(v), ref, set(M.LAB))
    baja_oai = any(r["por_hallazgo"].get(L, 0) < 0 for L in OAI)
    peor = min(r["por_hallazgo"].values()) if r["por_hallazgo"] else 0.0
    r["PASA_CRITERIO"] = bool(r["delta"] > 0 and r["p_mejora"] >= 0.9 and not baja_oai and peor >= -0.005)
    OUT["en_oai_fov"][k] = r
(D / "evaluacion_s6_variantes.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
print(json.dumps({k: OUT[k] for k in ("oai_fov_con_s6_kaggle", "oai_fov_con_s6_yuca", "solo")}, indent=1))
for k, r in OUT["en_oai_fov"].items():
    print("%-18s Δ %+.4f IC [%+.4f, %+.4f] P %.2f  pasa=%s  %s" % (k, r["delta"], *r["ic95"], r["p_mejora"], r["PASA_CRITERIO"], r["por_hallazgo"]))
