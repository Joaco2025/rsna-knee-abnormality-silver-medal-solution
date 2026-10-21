"""Criterio PREREGISTRADO (journal, 20-oct 14:00) para los 2 destilados nuevos (job 2099380, semillas 314 y 2718).

Base = Dual+C2: mitad Raptor con S6d = media de logits de d42 + d7 al 0.25; s42 encima al 0.25; desatajo de Synovitis; C2.
Candidata = lo mismo con S6d = media de logits de d42 + d7 + d314 + d2718.
Decide la comparación a STRIDE 3 (lo que correría en Kaggle): ENTRA si Δ > 0, P(mejora) ≥ 0.85 y ningún hallazgo baja más de
0.005.  Stride 1 se informa (descriptivo).  Una sola comparación decisiva; sin variantes de peso.
Uso: python -I -B evaluar_destil4.py runs_joako/<job>/gold
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
with contextlib.redirect_stdout(io.StringIO()):
    import mezcla_lib as M
    from ideas_gold import C, ids, comparar
D = Path(sys.argv[1])
J = {l: i for i, l in enumerate(M.LAB)}


def carga(n):
    z = np.load(D / f"s6_gold_{n}.npz", allow_pickle=True)
    u = [str(x) for x in z["ids"]]
    o = [u.index(i) for i in ids]
    return {k: np.asarray(z[k], np.float64)[o] for k in z.files if k != "ids"}


def receta(S, dest):
    a3 = M.mezcla_a3(C["r384"], C["r224"], C["r6"], M.combinar_s6([S[k] for k in dest], "logit"), ws=0.25)
    b = M.parte_b(a3, C["lector"])
    P = M.desatajo(np.column_stack([M.pct(0.75 * M.pct(b[:, j]) + 0.25 * M.pct(S["s6"][:, j])) for j in range(12)]), C["meta"])
    m = M.pct(P[:, J["Medial OA"]])
    for f in ("Lateral OA", "PF OA"):
        P[:, J[f]] = M.pct(M.pct(P[:, J[f]]) - 0.1 * m)
    return P


OUT = {}
for st in ("stride3", "stride1"):
    S = carga(st)
    r = comparar(receta(S, ["s6_d42", "s6_d7", "s6_d314", "s6_d2718"]), receta(S, ["s6_d42", "s6_d7"]), set(M.LAB))
    peor = min(r["por_hallazgo"].values()) if r["por_hallazgo"] else 0.0
    r["ENTRA"] = bool(r["delta"] > 0 and r["p_mejora"] >= 0.85 and peor >= -0.005) if st == "stride3" else "descriptivo"
    OUT[st] = r
    print("%s: base %.4f -> 4 destilados %.4f | Δ %+.5f IC [%+.4f, %+.4f] P %.3f peor %+.4f | ENTRA=%s\n   %s" % (
        st, r["macro"] - r["delta"], r["macro"], r["delta"], *r["ic95"], r["p_mejora"], peor, r["ENTRA"], r["por_hallazgo"]))
(D / "evaluacion_destil4.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False))
