"""Evaluador de las hipótesis H1–H4 sobre Gold-58 (etiquetas de experto de train.csv, las 12 columnas completas).

Métrica: la de la competición, ROC-AUC por hallazgo y su media (macro), sklearn.roc_auc_score.  Deltas pareados frente a la
línea base de cada hipótesis, con bootstrap de estudios (B = 1000, semilla 0; se descartan los remuestreos con alguna clase
vacía en algún hallazgo y se informa cuántos quedan).

Componentes (matrices 58 x 12 en el orden de mezcla_lib.LAB, filas en el orden de `ids`):
  r384, r224   SIEMPRE: runs_joako/2089458/preds_gold.npz (fp32 CPU, SIN el espejo de Kaggle; Raptor eligió su SWA con Gold)
  meta         SIEMPRE: puntuación de metadatos del modelo desplegado en E-sust-dsj (desatajo_synovitis.json)
  r6, lector, s6, s6_d42, s6_d7   OPCIONALES: --componentes X.npz (claves con esos nombres + 'ids').  Hoy no existen.
Cada variante declara los componentes que necesita; si falta alguno queda «PENDIENTE» con la lista de lo que falta.

Uso:
  python -I evaluar_gold.py                          -> resultados_gold.json (lo evaluable hoy)
  python -I evaluar_gold.py --componentes C.npz      -> idem con los componentes nuevos
  python -I evaluar_gold.py --autoprueba             -> componentes SINTÉTICOS (ruido sobre Raptor) solo para comprobar que
                                                        todas las variantes corren; su salida NO es un resultado.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.dont_write_bytecode = True
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import mezcla_lib as M

PROJ = Path("/lustre/proyectos/p082")
RJ = PROJ / "runs_joako"
LAB = M.LAB
B, SEMILLA = 1000, 0

# ---------------- variantes fijadas ANTES de ver resultados -------------------------------------------------------------
# (nombre, hipótesis, línea base, componentes necesarios, función)
H3_SYN_PF = {"Synovitis": 0.125, "PF OA": 0.125}
H3_SYN_PF_0 = {"Synovitis": 0.0, "PF OA": 0.0}
H3_OAI3 = {"Synovitis": 0.125, "PF OA": 0.125, "Lateral OA": 0.125}


def _raptor(C):
    return np.column_stack([M.bloque_raptor(C["r384"], C["r224"], j) for j in range(12)])


def _raptor_dsj(C):
    return M.desatajo(_raptor(C), C["meta"])


def _s6_solo(key):
    return lambda C: np.asarray(C[key], dtype=np.float64)


def _s6_comb(modo):
    return lambda C: M.combinar_s6([C["s6"], C["s6_d42"], C["s6_d7"]], modo)


def _s6_dest(modo):
    return lambda C: M.combinar_s6([C["s6_d42"], C["s6_d7"]], modo)


def _con_s6(nombre, s6fun, **kw):
    def f(C):
        C2 = dict(C, s6=s6fun(C))
        return M.receta(nombre, C2, **kw)
    return f


RAPTOR = ["r384", "r224"]
FULL = RAPTOR + ["r6", "lector"]
VARIANTES = [
    # referencia sólo-Raptor (evaluable hoy)
    ("raptor_65_35", "ref", None, RAPTOR, _raptor),
    ("raptor_65_35_dsj", "H4-dsj", "raptor_65_35", RAPTOR + ["meta"], _raptor_dsj),
    # receta completa
    ("base0958", "ref", None, FULL, lambda C: M.receta("base0958", C)),
    ("E", "H4", "base0958", FULL + ["s6"], lambda C: M.receta("E", C)),
    ("Esust", "H4", "E", FULL + ["s6"], lambda C: M.receta("Esust", C)),
    ("Esust_dsj", "H4", "Esust", FULL + ["s6", "meta"], lambda C: M.receta("Esust_dsj", C)),
    # añadidas tras la primera evaluación completa (propuestas en el informe ANTES de ver la mezcla completa):
    ("E_dsj", "H4", "E", FULL + ["s6", "meta"], lambda C: M.desatajo(M.receta("E", C), C["meta"])),
    ("Esust_dsj_vs_E", "H4", "E", FULL + ["s6", "meta"], lambda C: M.receta("Esust_dsj", C)),
    ("solo_r6", "ctx", None, ["r6"], lambda C: np.asarray(C["r6"], float)),
    ("solo_lector", "ctx", None, ["lector"], lambda C: np.asarray(C["lector"], float)),
    ("solo_r384", "ctx", None, ["r384"], lambda C: np.asarray(C["r384"], float)),
    # H2: una sola variable (normalización de rr) frente a su receta
    ("base0958_h2_uniforme", "H2", "base0958", FULL, lambda C: M.receta("base0958", C, h2="uniforme")),
    ("Esust_h2_uniforme", "H2", "Esust", FULL + ["s6"], lambda C: M.receta("Esust", C, h2="uniforme")),
    ("Esust_h2_dispersion", "H2", "Esust", FULL + ["s6"], lambda C: M.receta("Esust", C, h2="dispersion")),
    # H3: peso de S6 por hallazgo dentro de la mitad Raptor (3 variantes, nada más)
    ("Esust_h3_synpf_mitad", "H3", "Esust", FULL + ["s6"], lambda C: M.receta("Esust", C, ws_c=H3_SYN_PF)),
    ("Esust_h3_synpf_cero", "H3", "Esust", FULL + ["s6"], lambda C: M.receta("Esust", C, ws_c=H3_SYN_PF_0)),
    ("Esust_h3_oai3_mitad", "H3", "Esust", FULL + ["s6"], lambda C: M.receta("Esust", C, ws_c=H3_OAI3)),
    # H1: S6 solo y combinaciones de checkpoints
    ("s6_original", "H1", None, ["s6"], _s6_solo("s6")),
    ("s6_d42", "H1", "s6_original", ["s6_d42"], _s6_solo("s6_d42")),
    ("s6_d7", "H1", "s6_original", ["s6_d7"], _s6_solo("s6_d7")),
    ("s6_destilados_logit", "H1", "s6_original", ["s6_d42", "s6_d7"], _s6_dest("logit")),
    ("s6_super_logit", "H1", "s6_original", ["s6", "s6_d42", "s6_d7"], _s6_comb("logit")),
    ("s6_super_rango", "H1", "s6_original", ["s6", "s6_d42", "s6_d7"], _s6_comb("rango")),
    ("Esust_super_logit", "H1", "Esust", FULL + ["s6", "s6_d42", "s6_d7"], _con_s6("Esust", _s6_comb("logit"))),
    ("Esust_destilados_logit", "H1", "Esust", FULL + ["s6_d42", "s6_d7"], _con_s6("Esust", _s6_dest("logit"))),
]


def aucs(Y, P):
    return np.array([roc_auc_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])])


def aucs_boot(Y, P, CNT):
    """AUC de cada remuestreo bootstrap a la vez (B x 12), exacto: Mann-Whitney con pesos = veces que sale cada estudio.
    W[i, j] = 1 si P_i > P_j, 0.5 si empatan; AUC_b = sum_{i pos, j neg} c_i c_j W_ij / (sum c_pos · sum c_neg)
    (= roc_auc_score sobre la muestra con duplicados, empates a 0.5).  CNT: (B, n) recuentos."""
    out = np.empty((CNT.shape[0], Y.shape[1]))
    for k in range(Y.shape[1]):
        p, y = P[:, k], Y[:, k].astype(bool)
        W = (p[:, None] > p[None, :]) + 0.5 * (p[:, None] == p[None, :])
        cp, cn = CNT * y, CNT * ~y
        out[:, k] = np.einsum("bi,ij,bj->b", cp, W, cn) / (cp.sum(1) * cn.sum(1))
    return out


def meta_gold(ids):
    """Puntuación de Synovitis del modelo de metadatos DESPLEGADO (desatajo_synovitis.json) sobre los estudios de Gold."""
    sys.path.insert(0, str(RJ / "exp_Esust_dsj"))
    import meta_rasgos as MR
    modelo = json.loads((RJ / "exp_Esust_dsj/desatajo_synovitis.json").read_text())
    m = pd.read_parquet(RJ / "2093187/meta_series.parquet")
    for c in MR.CAMPOS:
        m[c] = m[c].where(m[c].notna() & (m[c].astype(str) != "") & (m[c].astype(str) != "None"), None)
    X = MR.rasgos(m, modelo["vocab"], modelo["bobinas"], modelo["modelos"])
    falta = [u for u in ids if u not in X.index]
    if falta:
        raise SystemExit("meta: %d estudios de Gold sin cabeceras" % len(falta))
    return np.asarray(MR.puntuar(X.loc[ids], modelo, "Synovitis"), dtype=np.float64)


def cargar(args):
    z = np.load(RJ / "2089458/preds_gold.npz", allow_pickle=True)
    ids = [str(i) for i in z["ids"]]
    C = {"r384": z["r384"].astype(np.float64), "r224": z["r224"].astype(np.float64)}
    # control: la mezcla guardada por el job es la misma que reconstruye mezcla_lib (en orden)
    tr = pd.read_csv(PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection/train.csv", dtype={"StudyInstanceUID": str})
    gold = tr.dropna(subset=LAB).set_index("StudyInstanceUID")
    assert len(gold) == 58 and set(ids) == set(gold.index), "los ids de preds_gold.npz no son los 58 de Gold"
    Y = gold.loc[ids, LAB].to_numpy().astype(int)
    assert set(np.unique(Y)) <= {0, 1} and (Y.min(0) == 0).all() and (Y.max(0) == 1).all()
    C["meta"] = meta_gold(ids)
    origen = {"r384": "runs_joako/2089458/preds_gold.npz (sin espejo)", "r224": "ídem",
              "meta": "desatajo_synovitis.json sobre runs_joako/2093187/meta_series.parquet"}
    if args.componentes:
        zc = np.load(args.componentes, allow_pickle=True)
        idc = [str(i) for i in zc["ids"]]
        assert set(idc) == set(ids), "--componentes: los ids no son los 58 de Gold"
        orden = [idc.index(u) for u in ids]
        for k in zc.files:
            if k == "ids":
                continue
            v = np.asarray(zc[k], dtype=np.float64)[orden]
            assert v.shape in ((58, 12), (58,)) and np.isfinite(v).all(), k
            C[k] = v
            origen[k] = str(args.componentes)
    if args.autoprueba:   # SINTÉTICO: solo para comprobar que todo corre
        rng = np.random.default_rng(1)
        base = C["r384"]
        for k, s in (("r6", .6), ("lector", 1.0), ("s6", .5), ("s6_d42", .5), ("s6_d7", .5)):
            C[k] = 1 / (1 + np.exp(-(np.log(base / (1 - base)) + rng.normal(0, s * 2, base.shape))))
            origen[k] = "SINTÉTICO (autoprueba)"
    return ids, Y, C, origen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--componentes")
    ap.add_argument("--autoprueba", action="store_true")
    ap.add_argument("--salida")
    ap.add_argument("--B", type=int, default=B, help="remuestreos (1000 para resultados; menos solo en la autoprueba)")
    args = ap.parse_args()
    ids, Y, C, origen = cargar(args)
    n = len(ids)
    rng = np.random.default_rng(SEMILLA)
    boot = [ix for ix in (rng.integers(0, n, n) for _ in range(args.B)) if (Y[ix].min(0) == 0).all() and (Y[ix].max(0) == 1).all()]
    CNT = np.stack([np.bincount(ix, minlength=n) for ix in boot]).astype(np.float64)
    ix0 = boot[0]                                  # control: el AUC vectorizado = sklearn en un remuestreo
    P, res, AB = {}, {}, {}
    for nombre, hip, ref, req, fun in VARIANTES:
        falta = [k for k in req if k not in C]
        if falta:
            res[nombre] = {"hipotesis": hip, "estado": "PENDIENTE", "faltan": falta}
            continue
        P[nombre] = fun(C)
        a = aucs(Y, P[nombre])
        AB[nombre] = aucs_boot(Y, P[nombre], CNT)
        assert np.allclose(AB[nombre][0], aucs(Y[ix0], P[nombre][ix0]), atol=1e-12), nombre
        res[nombre] = {"hipotesis": hip, "estado": "OK", "macro": round(float(a.mean()), 4),
                       "por_hallazgo": dict(zip(LAB, np.round(a, 4).tolist()))}
    for nombre, hip, ref, req, fun in VARIANTES:
        if ref is None or nombre not in P or ref not in P:
            if ref is not None and nombre in P:
                res[nombre]["delta_vs"] = {"ref": ref, "estado": "PENDIENTE (falta la línea base)"}
            continue
        d_ph = aucs(Y, P[nombre]) - aucs(Y, P[ref])
        bph = AB[nombre] - AB[ref]
        bd = bph.mean(1)
        res[nombre]["delta_vs"] = {
            "ref": ref, "delta_macro": round(float(d_ph.mean()), 5),
            "ic95_macro": [round(float(np.percentile(bd, 2.5)), 5), round(float(np.percentile(bd, 97.5)), 5)],
            "p_mejora": round(float((bd > 0).mean()), 3), "p_empeora": round(float((bd < 0).mean()), 3),
            "por_hallazgo": dict(zip(LAB, np.round(d_ph, 4).tolist())),
            "hallazgos_que_suben": int((d_ph > 1e-9).sum()), "hallazgos_que_bajan": int((d_ph < -1e-9).sum()),
            "ic95_por_hallazgo": {L: [round(float(np.percentile(bph[:, k], 2.5)), 4), round(float(np.percentile(bph[:, k], 97.5)), 4)]
                                  for k, L in enumerate(LAB) if abs(d_ph[k]) > 1e-9}}
    # paridad con la ejecución real del notebook (solo si --componentes trae sus salidas): mismas fórmulas, mismos rangos
    paridad = {}
    for clave, receta in (("nb_base0958", "base0958"), ("nb_esust_dsj", "Esust_dsj")):
        if clave in C and receta in P:
            nbr = np.column_stack([M.pct(C[clave][:, k]) for k in range(12)])
            paridad[receta] = float(np.abs(nbr - P[receta]).max())
    if paridad:
        print("PARIDAD_NOTEBOOK", json.dumps(paridad), "OK" if max(paridad.values()) < 1e-6 else "DIFERENTE: revisar antes de interpretar")
    salida = {"n_gold": n, "paridad_notebook": paridad, "bootstrap": {"B": args.B, "validos": len(boot), "semilla": SEMILLA}, "origen_componentes": origen,
              "AVISO": ("AUTOPRUEBA CON COMPONENTES SINTÉTICOS: NO ES UN RESULTADO" if args.autoprueba else
                        "Raptor eligió su SWA con Gold: AUC absolutos inflados; interpretar solo deltas"),
              "variantes": res}
    out = Path(args.salida) if args.salida else AQUI / ("autoprueba_gold.json" if args.autoprueba else "resultados_gold.json")
    out.write_text(json.dumps(salida, indent=1, ensure_ascii=False))
    for nombre, r in res.items():
        if r["estado"] != "OK":
            print("%-26s %-6s PENDIENTE: faltan %s" % (nombre, r["hipotesis"], ",".join(r["faltan"])))
            continue
        d = r.get("delta_vs", {})
        txt = ("Δ %+.4f IC [%+.4f, %+.4f] P(+) %.2f  ↑%d ↓%d" % (d["delta_macro"], *d["ic95_macro"], d["p_mejora"],
               d["hallazgos_que_suben"], d["hallazgos_que_bajan"])) if "delta_macro" in d else ""
        print("%-26s %-6s macro %.4f  %s" % (nombre, r["hipotesis"], r["macro"], txt))
    print("->", out)


if __name__ == "__main__":
    main()
