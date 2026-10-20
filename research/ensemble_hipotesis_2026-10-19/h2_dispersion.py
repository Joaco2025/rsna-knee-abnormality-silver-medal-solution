"""H2: ¿cuánto cambia el peso efectivo de la mitad Raptor frente a R6 al no re-rankear rr antes de mezclar?

En la celda A3, rr = 0.65·pct(R384) + 0.35·pct(R224) (y en E-sust, 0.75·rr + 0.25·pct(S6)) NO se re-rankea; pct(R6) sí es
uniforme.  Una suma de uniformes correlacionadas tiene menos dispersión que una uniforme, así que en b = wr·rr + (1-wr)·pct(R6)
la mitad Raptor pesa algo menos que wr.  Peso efectivo (en desviación típica): wr·s / (wr·s + (1 - wr)), s = std(rr)/std(U).

Medido en Gold-58: s de rr sin S6 (R384, R224 reales, sin espejo).
Con S6: no hay S6 en Gold.  Se usa la correlación de rangos Raptor 0.949 ↔ S6 que el propio notebook de S6 declara
(«fold-0 mean 0.876, min 0.731», constantes MIN_RHO_* de s6_original.py) como rango plausible:
  var(rr_sust) = 0.75²·s0² + 0.25² + 2·0.75·0.25·ρ·s0   (en unidades de var(U); ρ = corr(rr0, pct(S6)) ≈ corr(R384, S6))
Uso: python -I h2_dispersion.py -> h2_dispersion.json
"""
import json, sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).parent))
import mezcla_lib as M

z = np.load("/lustre/proyectos/p082/runs_joako/2089458/preds_gold.npz", allow_pickle=True)
n = len(z["ids"])
U = M.pct(np.arange(n))
RHO_S6 = {"media_fold0": 0.876, "minimo_fold0": 0.731}
out = {"n": n, "rho_raptor_s6_supuesta": RHO_S6, "por_hallazgo": {}}
for j, c in enumerate(M.LAB):
    rr0 = M.bloque_raptor(z["r384"], z["r224"], j)
    s0 = float(rr0.std() / U.std())
    wr = M.W_RAPTOR_C.get(c, M.W_RAPTOR)
    eff = lambda s: wr * s / (wr * s + (1 - wr))
    fila = {"wr_nominal": wr, "s_rr_E": round(s0, 4), "wr_efectivo_E": round(eff(s0), 4),
            "rho_384_224": round(float(np.corrcoef(M.pct(z["r384"][:, j]), M.pct(z["r224"][:, j]))[0, 1]), 3)}
    for k, rho in RHO_S6.items():
        s = float(np.sqrt(0.75 ** 2 * s0 ** 2 + 0.25 ** 2 + 2 * 0.75 * 0.25 * rho * s0))
        fila["s_rr_Esust_" + k] = round(s, 4)
        fila["wr_efectivo_Esust_" + k] = round(eff(s), 4)
    out["por_hallazgo"][c] = fila
ph = out["por_hallazgo"].values()
out["resumen"] = {
    "max_desplazamiento_wr_E_pp": round(100 * max(abs(f["wr_nominal"] - f["wr_efectivo_E"]) for f in ph), 2),
    "max_desplazamiento_wr_Esust_media_pp": round(100 * max(abs(f["wr_nominal"] - f["wr_efectivo_Esust_media_fold0"]) for f in ph), 2),
    "max_desplazamiento_wr_Esust_minimo_pp": round(100 * max(abs(f["wr_nominal"] - f["wr_efectivo_Esust_minimo_fold0"]) for f in ph), 2),
    "Esust_vs_E_pp_media": round(100 * max(abs(f["wr_efectivo_E"] - f["wr_efectivo_Esust_media_fold0"]) for f in ph), 2),
}
Path(__file__).with_name("h2_dispersion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("%-17s %5s %6s %7s %9s %9s" % ("hallazgo", "wr", "s_E", "wrE", "wrSust.876", "wrSust.731"))
for c, f in out["por_hallazgo"].items():
    print("%-17s %5.2f %6.4f %7.4f %9.4f %9.4f" % (c, f["wr_nominal"], f["s_rr_E"], f["wr_efectivo_E"],
          f["wr_efectivo_Esust_media_fold0"], f["wr_efectivo_Esust_minimo_fold0"]))
print(out["resumen"])
