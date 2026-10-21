"""Criterio PREREGISTRADO (journal, 20-oct 21:00) para los S6 entrenados con etiquetas de Omer (jobs 2108250 puro, 2108251 mix).

Base = Dual+C2 (S6d = d42 + d7 dentro de la mitad Raptor al 0.25; s42 encima al 0.25; desatajo λ 0.1; C2 γ 0.1).
Candidata = brazo de encima = media de logits (s42 + S6-Omer).  Decide STRIDE 3: ENTRA si Δ > 0, P ≥ 0.85, ningún hallazgo
< −0.005.  Si pasan las dos variantes, la de mayor Δ.  Stride 1 y el S6-Omer solo, descriptivos.
Uso: python -I -B evaluar_omer.py runs_joako/<job>/gold
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
with contextlib.redirect_stdout(io.StringIO()):
    import mezcla_lib as M, evaluar_gold as EG
    from ideas_gold import C, Y, ids, comparar
D = Path(sys.argv[1])
J = {l: i for i, l in enumerate(M.LAB)}


def carga(n):
    z = np.load(D / f"s6_gold_{n}.npz", allow_pickle=True)
    u = [str(x) for x in z["ids"]]
    o = [u.index(i) for i in ids]
    return {k: np.asarray(z[k], np.float64)[o] for k in z.files if k != "ids"}


def receta(S, top):
    a3 = M.mezcla_a3(C["r384"], C["r224"], C["r6"], M.combinar_s6([S["s6_d42"], S["s6_d7"]], "logit"), ws=0.25)
    b = M.parte_b(a3, C["lector"])
    t = M.combinar_s6([S[k] for k in top], "logit")
    P = M.desatajo(np.column_stack([M.pct(0.75 * M.pct(b[:, j]) + 0.25 * M.pct(t[:, j])) for j in range(12)]), C["meta"])
    m = M.pct(P[:, J["Medial OA"]])
    for f in ("Lateral OA", "PF OA"):
        P[:, J[f]] = M.pct(M.pct(P[:, J[f]]) - 0.1 * m)
    return P


OUT = {}
for st in ("stride3", "stride1"):
    S = carga(st)
    for v in ("s6_omer_puro", "s6_omer_mix"):
        if v not in S:
            continue
        r = comparar(receta(S, ["s6", v]), receta(S, ["s6"]), set(M.LAB))
        peor = min(r["por_hallazgo"].values()) if r["por_hallazgo"] else 0.0
        r["ENTRA"] = bool(r["delta"] > 0 and r["p_mejora"] >= 0.85 and peor >= -0.005) if st == "stride3" else "descriptivo"
        r["solo_macro"] = round(float(EG.aucs(Y, S[v]).mean()), 4)
        OUT[f"{st}/{v}"] = r
        print("%s %s: S6 solo %.4f | base %.4f -> %.4f | Δ %+.5f IC [%+.4f, %+.4f] P %.3f peor %+.4f | ENTRA=%s" % (
            st, v, r["solo_macro"], r["macro"] - r["delta"], r["macro"], r["delta"], *r["ic95"], r["p_mejora"], peor, r["ENTRA"]))
(D / "evaluacion_omer.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
