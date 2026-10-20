"""H3/H4: peso nominal de cada componente en el rango final, por hallazgo (lineal en rangos; ignora el re-rankeo intermedio y la
pérdida de dispersión de rr, que h2_dispersion.py cuantifica en <= 1.8 pp).
  E      : Raptor 0.75·wr·(1-w)  R6 0.75·(1-wr)·(1-w)  S6 0.25            lector 0.75·w
  E-sust : Raptor 0.75·wr·(1-w)  R6 (1-wr)·(1-w)       S6 0.25·wr·(1-w)   lector w
  (Raptor = 0.65·R384 + 0.35·R224; w = OWN_W_PER_TARGET)
Uso: python -I pesos_nominales.py -> pesos_nominales.json
"""
import json, sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import mezcla_lib as M

out = {}
print("%-17s | %-27s | %-27s" % ("", "E: Rap   R6    S6    lect", "E-sust: Rap   R6    S6    lect"))
for c in M.LAB:
    wr, w = M.W_RAPTOR_C.get(c, M.W_RAPTOR), M.OWN_W_PER_TARGET[c]
    e = {"raptor": .75 * wr * (1 - w), "r6": .75 * (1 - wr) * (1 - w), "s6": .25, "lector": .75 * w}
    s = {"raptor": .75 * wr * (1 - w), "r6": (1 - wr) * (1 - w), "s6": .25 * wr * (1 - w), "lector": w}
    out[c] = {"E": {k: round(v, 4) for k, v in e.items()}, "Esust": {k: round(v, 4) for k, v in s.items()}}
    print("%-17s | %.3f %.3f %.3f %.3f     | %.3f %.3f %.3f %.3f" % (c, *e.values(), *s.values()))
Path(__file__).with_name("pesos_nominales.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
