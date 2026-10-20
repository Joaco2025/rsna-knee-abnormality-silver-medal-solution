"""Paridad de mezcla_lib con las salidas reales de las pruebas sintéticas del notebook (solo lectura de runs_joako).

Para cada prueba que terminó con submission.csv: recalcula submission_a3.csv, submission_0958.csv y submission.csv desde los
CSV intermedios (raptor_raw, raptor224_sub, ours_sub, _own, s6_sub) y compara.  dsj: se comparan las 11 columnas que no son
Synovitis (los metadatos de la prueba no se guardaron) y se comprueba que Synovitis sí cambia.
Uso: python -I paridad.py  ->  paridad.json
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import mezcla_lib as M

R = Path("/lustre/proyectos/p082/runs_joako")
CASOS = {"E": R / "exp_E_0958_s6/pruebas", "Esust": R / "exp_Esust_0958/pruebas", "Esust_dsj": R / "exp_Esust_dsj/pruebas"}


def leer(w, n, ids=None):
    d = pd.read_csv(w / n, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    return d.loc[ids, M.LAB].to_numpy(np.float64) if ids is not None else d


out = {}
for receta, raiz in CASOS.items():
    for w in sorted(raiz.glob("*/work")):
        if not (w / "submission.csv").exists() or not (w / "s6_sub.csv").exists():
            continue
        br = json.loads((w / "blend_receipt.json").read_text())
        if br["raptor_failed"] or br["ours_failed"]:
            continue
        ids = leer(w, "submission.csv").index.tolist()
        C = {"r384": leer(w, "raptor_raw.csv", ids), "r224": leer(w, "raptor224_sub.csv", ids),
             "r6": leer(w, "ours_sub.csv", ids), "lector": leer(w, "_own.csv", ids), "s6": leer(w, "s6_sub.csv", ids)}
        a3 = M.mezcla_a3(C["r384"], C["r224"], C["r6"], C["s6"] if receta != "E" else None,
                         M.W_S6 if receta != "E" else 0.0)
        base = M.parte_b(M.mezcla_a3(C["r384"], C["r224"], C["r6"]), C["lector"])
        fin = M.receta("Esust" if receta == "Esust_dsj" else receta, C)
        real_a3, real_base, real_fin = leer(w, "submission_a3.csv", ids), leer(w, "submission_0958.csv", ids), leer(w, "submission.csv", ids)
        r = {"n": len(ids)}
        if receta == "E":
            r["a3"] = float(np.abs(a3 - real_a3).max())
        else:   # el A3 guardado es el de la receta 0.958 (sin S6); el de E-sust solo vive en memoria
            r["a3_sin_s6"] = float(np.abs(M.mezcla_a3(C["r384"], C["r224"], C["r6"]) - real_a3).max())
        r["base0958"] = float(np.abs(base - real_base).max())
        js = M.LAB.index("Synovitis")
        if receta == "Esust_dsj":
            otras = [j for j in range(12) if j != js]
            r["final_11_columnas"] = float(np.abs(fin[:, otras] - real_fin[:, otras]).max())
            r["synovitis_cambia"] = bool(np.abs(fin[:, js] - real_fin[:, js]).max() > 0) or len(ids) < 3
        else:
            r["final"] = float(np.abs(fin - real_fin).max())
        out[f"{receta}/{w.parent.name}"] = r
Path(__file__).with_name("paridad.json").write_text(json.dumps(out, indent=1))
peor = max(v for r in out.values() for k, v in r.items() if isinstance(v, float))
print(json.dumps(out, indent=1))
print("PARIDAD", "OK" if peor < 1e-6 else "FALLO", "max |dif| = %.2e" % peor, "casos", len(out))
