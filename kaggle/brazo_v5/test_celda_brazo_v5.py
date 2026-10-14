#!/usr/bin/env python
"""Prueba en CPU de celda_brazo_v5.py y de su variante diagnóstica. No usa GPU ni el paquete de Owen.

Qué hace: ejecuta la celda con una etapa sintética de N estudios (N=7 y N=1400) que imita el notebook 0.957:
la etapa es submission.csv, la carpeta de datos se llama DATA (no ROOT), no hay TIME_BUDGET ni rsna_frame ni
_RSNA_*. Sustituye la inferencia (_v5_correr), el paquete, las GPU y los decodificadores por simulaciones y
comprueba la compuerta de punta a punta: mezcla, orden de columnas, validación equivalente a la de la celda 6,
guarda de tiempo y variante diagnóstica. La cola real de la celda (TIME_BUDGET = 8 h, ROOT = DATA, T0 = NB_T0)
se ejecuta tal cual en todas las pruebas. Los 12 hallazgos se extraen de la celda 2 del notebook 0.957.

Uso:  python test_celda_brazo_v5.py <directorio_de_salida>
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB = REPO / "referencias" / "0-957-latest-dataset-score-in-one-click.ipynb"
CELDA = HERE / "celda_brazo_v5.py"
DIAG = HERE / "celda_brazo_v5_diagnostico.py"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

RES: list[dict] = []


def check(nombre: str, cond, detalle: str = "") -> None:
    RES.append({"check": nombre, "ok": bool(cond), "detalle": str(detalle)})
    print(("PASS " if cond else "FAIL "), nombre, ("| " + str(detalle)) if detalle else "", flush=True)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- los 12 hallazgos (LAB), tal cual salen de la celda 2 del notebook ------------------------------
def etiquetas_del_notebook():
    nb = json.load(open(NB))
    codigo = "".join(nb["cells"][2]["source"]).split("\n", 1)[1]     # sin la línea mágica %%writefile
    for node in ast.parse(codigo).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LAB" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no se encontró LAB en la celda 2")


LABELS = etiquetas_del_notebook()


def validador_celda6(df, uids, cols):
    """Las aserciones finales de la celda 6 del notebook sobre submission.csv, reescritas aparte."""
    assert list(df.columns) == list(cols), "columnas"
    assert df["StudyInstanceUID"].tolist() == list(uids), "orden de UIDs"
    v = df[list(cols[1:])].to_numpy(float)
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), "valores"


NB_SHA = sha(NB)


@contextlib.contextmanager
def entorno(**kv):
    previo = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: v for k, v in kv.items() if v is not None})
    try:
        yield
    finally:
        for k, v in previo.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


N_MODELOS = 15                  # 3 familias por defecto (eff, effhib, r34att) x 5 particiones
CABECERA_COLA = 'if not globals().get("BRAZO_V5_SOLO_DEFINIR"):'


def correr(nombre, n, fuente: Path, *, restante_s=7 * 3600, con_t0=True, solo_nb_t0=False, gpus=(0, 1), n_ilegibles=0,
           w=None, mod_stage=None, ss_desordenado=False, rerun=False, familias=None, con_ganchos=False):
    """Una ejecución de la celda con una etapa sintética de n estudios. Devuelve lo observable."""
    wd = OUT / nombre
    wd.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(1234 + n)
    uids = [f"1.2.826.0.1.3680043.8.498.{i:020d}" for i in range(n)]
    data = wd / "data"
    data.mkdir()
    pd.DataFrame({"StudyInstanceUID": uids}).to_csv(data / "test.csv", index=False)
    ss_cols = ["StudyInstanceUID", *LABELS]
    if ss_desordenado:
        ss_cols[1], ss_cols[2] = ss_cols[2], ss_cols[1]
    pd.DataFrame(columns=ss_cols).to_csv(data / "sample_submission.csv", index=False)
    stage = pd.DataFrame(rng.normal(size=(n, 12))).rank(pct=True)
    stage.columns = LABELS
    stage.insert(0, "StudyInstanceUID", uids)
    if mod_stage:
        stage = mod_stage(stage)
    stage_path = wd / "submission.csv"
    stage.to_csv(stage_path, index=False)
    original = wd / "_etapa_original.csv"          # copia aparte, para comparar después de ejecutar la celda
    stage.to_csv(original, index=False)
    sha_antes = sha(original)

    # Cabeza (definiciones) y cola (la parte que hace TIME_BUDGET/ROOT/T0 y llama a brazo_v5) por separado,
    # para sustituir la inferencia, el paquete y las GPU entre las dos y ejecutar la cola real.
    texto = fuente.read_text()
    cabeza, cola = texto.rsplit("\n" + CABECERA_COLA, 1)
    ns = {"__name__": "celda_ns", "BRAZO_V5_SOLO_DEFINIR": True}
    if familias is not None:
        ns["BRAZO_V5_FAMILIAS"] = tuple(familias)
    exec(compile(cabeza, str(fuente), "exec"), ns)
    # El notebook 0.957 NO tiene rsna_frame, rsna_phase, rsna_event, rsna_save_predictions, _RSNA_*,
    # TIME_BUDGET ni ROOT. Solo existen T0 (= NB_T0) y DATA.
    eventos: list = []
    if con_ganchos:
        ns["rsna_event"] = lambda kind, **d: eventos.append(("event", kind, d))
        ns["rsna_phase"] = lambda stage_, status="START", **d: eventos.append(("phase", stage_, status, d))
        ns["rsna_save_predictions"] = lambda tag, ids, values, labels=None: eventos.append(("save", tag, len(ids)))
    ns["DATA"] = data
    ns["V5_STAGE"], ns["V5_DIR"] = str(stage_path), str(wd / "brazo_v5")
    presupuesto = 8 * 3600.0            # el que fija la cola de la celda
    if con_t0 or solo_nb_t0:
        t = time.time() - (presupuesto - restante_s)
        ns["NB_T0"] = t
        if con_t0:
            ns["T0"] = t
    if w is not None:
        ns["V5_W"] = dict(w)

    llamada: dict = {}
    ns["_v5_paquete"] = lambda: "/simulado/paquete"
    ns["_v5_checkpoints"] = lambda raiz, k: [f"/simulado/pesos/m{i}.pt" for i in range(len(ns["V5_FAMILIAS"]) * k)]
    ns["_v5_gpus"] = lambda: (list(gpus), f"simulado {list(gpus)}")
    ns["_v5_librerias"] = lambda destino: (True, "")

    def inferencia_simulada(raiz, data_, split, uids_, ckpts, gpus_, limite, destino):
        llamada.update(split=split, n=len(uids_), n_ckpts=len(ckpts), gpus=list(gpus_), a_limite_s=limite - time.time())
        r = np.random.default_rng(99 + n).uniform(0.02, 0.98, size=(n, 12))
        df = pd.DataFrame(r, columns=ns["V5_LABELS"])
        df.insert(0, "StudyInstanceUID", uids_)
        if n_ilegibles:
            df.loc[df.index[:n_ilegibles], ns["V5_LABELS"]] = 0.5
        salida = df.sample(frac=1, random_state=3).reset_index(drop=True)
        llamada["v5"] = salida.copy()
        return salida, []

    ns["_v5_correr"] = inferencia_simulada

    exc, estado = None, None
    with entorno(KAGGLE_IS_COMPETITION_RERUN="1" if rerun else None):
        try:
            ns["BRAZO_V5_SOLO_DEFINIR"] = False
            exec(compile(CABECERA_COLA + cola, str(fuente) + "#cola", "exec"), ns)
            estado = ns["BRAZO_V5_ESTADO"]
        except Exception as e:  # noqa: BLE001
            exc = e
    ej = wd / "brazo_v5" / "estado.json"
    estado_json = json.load(open(ej)) if ej.is_file() else {}
    return dict(nombre=nombre, n=n, ns=ns, uids=uids, estado=estado or estado_json, exc=exc, stage_path=stage_path,
                original=original, stage_antes=stage, sha_antes=sha_antes, llamada=llamada, eventos=eventos, wd=wd,
                restante_s=restante_s)


def etapa_leida(r) -> pd.DataFrame:
    return pd.read_csv(r["stage_path"], dtype={"StudyInstanceUID": str})


def mezcla_esperada(ant: pd.DataFrame, v5: pd.DataFrame, w: dict) -> pd.DataFrame:
    """Reimplementación independiente de la mezcla por rangos: (1-W)·rango_notebook + W·rango_v5, re-rankeada."""
    v5 = v5.set_index("StudyInstanceUID").reindex(ant["StudyInstanceUID"])
    ilegible = (v5[LABELS] == 0.5).all(axis=1).to_numpy()
    out = ant.copy()
    for c in LABELS:
        if w[c] <= 0:
            continue
        r_nb = ant[c].rank(method="average", pct=True).to_numpy()
        x = v5[c].copy()
        x[ilegible] = np.nan
        r_v5 = x.rank(method="average", pct=True).fillna(0.0).to_numpy()
        peso = w[c] * (~ilegible)
        out[c] = pd.Series((1 - peso) * r_nb + peso * r_v5).rank(method="average", pct=True).to_numpy()
    return out


def verificar_aplicado(tag, r, *, n_gpus=2, n_ilegibles=0, w=None, n_modelos=15, con_ganchos=False):
    est, ns, n = r["estado"], r["ns"], r["n"]
    w = w or {c: 0.10 for c in LABELS}
    check(f"{tag}: decisión 'aplicado' sin excepción", est.get("decision") == "aplicado" and r["exc"] is None,
          f"{est.get('decision')} {est.get('motivo', '')[:120]} exc={r['exc']!r}")
    df, ant = etapa_leida(r), pd.read_csv(r["original"], dtype={"StudyInstanceUID": str})
    check(f"{tag}: columnas y orden de filas intactos",
          list(df.columns) == ["StudyInstanceUID", *LABELS] and df.StudyInstanceUID.tolist() == ant.StudyInstanceUID.tolist())
    ss_cols = list(pd.read_csv(r["wd"] / "data" / "sample_submission.csv", nrows=0).columns)
    try:
        validador_celda6(df, r["uids"], ss_cols)
        ok_celda6 = True
    except AssertionError as e:
        ok_celda6 = False
        print("   validador de la celda 6:", repr(e))
    check(f"{tag}: pasa las aserciones finales de la celda 6 (orden de UIDs, columnas, finitos, [0, 1])", ok_celda6)
    check(f"{tag}: la cola de la celda dejó TIME_BUDGET = 8 h y ROOT = DATA",
          ns.get("TIME_BUDGET") == 8 * 3600.0 and str(ns.get("ROOT")) == str(ns["DATA"]),
          f"TIME_BUDGET={ns.get('TIME_BUDGET')} ROOT={ns.get('ROOT')}")
    v = df[LABELS].to_numpy(float)
    check(f"{tag}: valores en (0, 1]", np.isfinite(v).all() and v.min() > 0 and v.max() <= 1.0, f"min={v.min():.5f} max={v.max():.5f}")
    esperado_df = mezcla_esperada(ant, r["llamada"]["v5"], w)
    dif = np.abs(df[LABELS].to_numpy(float) - esperado_df[LABELS].to_numpy(float)).max()
    check(f"{tag}: coincide con la mezcla por rangos recalculada de forma independiente", dif < 1e-9, f"max|Δ|={dif:.2e}")
    if n >= 100:
        cambio = np.abs(v - ant[LABELS].to_numpy(float)).max(axis=0)
        activos = [c for c in LABELS if w[c] > 0]
        check(f"{tag}: cada hallazgo con W>0 cambia respecto de la etapa original",
              all(cambio[LABELS.index(c)] > 0 for c in activos))
        rho = [df[c].corr(ant[c], method="spearman") for c in activos]
        check(f"{tag}: Spearman medio con la etapa original entre 0.8 y 0.999 (W=0.1)",
              0.9 < float(np.mean(rho)) < 0.9995, f"medio={np.mean(rho):.4f}")
    copia = r["wd"] / "brazo_v5" / "submission_antes_v5.csv"
    check(f"{tag}: copia de seguridad submission_antes_v5.csv idéntica a la etapa original", copia.is_file() and sha(copia) == r["sha_antes"])
    check(f"{tag}: sin archivo temporal", not Path(str(r["stage_path"]) + ".tmp_v5").exists())
    ll = r["llamada"]
    check(f"{tag}: inferencia pedida con los {n} UIDs, {n_modelos} modelos y {n_gpus} GPU",
          ll.get("n") == n and ll.get("n_ckpts") == n_modelos and len(ll.get("gpus", [])) == n_gpus, str({k: v for k, v in ll.items() if k != "v5"}))
    esperado = r["restante_s"] - ns["V5_MARGEN_S"]
    check(f"{tag}: plazo duro = restante − margen (±30 s)", abs(ll.get("a_limite_s", -1e9) - esperado) < 30,
          f"{ll.get('a_limite_s', 0):.0f} s vs {esperado:.0f} s")
    if con_ganchos:
        check(f"{tag}: registro 'applied' y fase COMPLETE con ok=True (notebook con ganchos rsna_*)",
              any(e[0] == "event" and e[1] == "yuca_v5_applied" for e in r["eventos"])
              and any(e[0] == "phase" and e[2] == "COMPLETE" and e[3].get("ok") is True for e in r["eventos"]))
    else:
        check(f"{tag}: sin ganchos rsna_* el brazo funciona igual", r["eventos"] == [])
    if n_ilegibles:
        ok_ord = all(
            np.array_equal(df[c].iloc[:n_ilegibles].rank(method="first").to_numpy(), ant[c].iloc[:n_ilegibles].rank(method="first").to_numpy())
            for c in LABELS)
        check(f"{tag}: {n_ilegibles} estudios ilegibles no votan (mantienen el orden relativo del notebook)",
              ok_ord and est.get("ilegibles") == n_ilegibles, f"ilegibles={est.get('ilegibles')}")


def verificar_omitido(tag, r, subcadena=""):
    est = r["estado"]
    check(f"{tag}: decisión 'omitido', sin excepción", est.get("decision") == "omitido" and r["exc"] is None,
          f"{est.get('decision')}: {est.get('motivo', '')[:140]}")
    check(f"{tag}: la etapa queda byte a byte igual", sha(r["stage_path"]) == r["sha_antes"])
    if subcadena:
        check(f"{tag}: el motivo menciona «{subcadena}»", subcadena in est.get("motivo", ""), est.get("motivo", "")[:140])
    check(f"{tag}: sin copia de seguridad ni temporal", not (r["wd"] / "brazo_v5" / "submission_antes_v5.csv").exists()
          and not Path(str(r["stage_path"]) + ".tmp_v5").exists())


def verificar_excepcion(tag, r):
    check(f"{tag}: la variante diagnóstica lanza RuntimeError «V5_EXIGIR»",
          isinstance(r["exc"], RuntimeError) and "V5_EXIGIR" in str(r["exc"]), repr(r["exc"])[:200])
    check(f"{tag}: aun así la etapa queda byte a byte igual", sha(r["stage_path"]) == r["sha_antes"])


def pruebas_notebook_0957():
    """Lo específico del notebook 0.957: valores por defecto, familias, T0/NB_T0, paquete y la cola sin simulaciones."""
    celda_txt = CELDA.read_text()
    cabeza = celda_txt.rsplit("\n" + CABECERA_COLA, 1)[0]
    d: dict = {"BRAZO_V5_SOLO_DEFINIR": True}
    exec(compile(cabeza, str(CELDA), "exec"), d)
    check("por defecto: familias eff, effhib, r34att (sin cnxhib)", d["V5_FAMILIAS"] == ("eff", "effhib", "r34att"), d["V5_FAMILIAS"])
    check("por defecto: V5_W = 0.10 en los 12 hallazgos", set(d["V5_W"]) == set(LABELS) and set(d["V5_W"].values()) == {0.10})
    check('V5_STAGE = "/kaggle/working/submission.csv"', d["V5_STAGE"] == "/kaggle/working/submission.csv", d["V5_STAGE"])
    check("la copia previa se llama submission_antes_v5.csv y ya no existe _pipeline_stage", "submission_antes_v5.csv" in celda_txt
          and "_pipeline_stage" not in celda_txt)
    cola = celda_txt.rsplit("\n" + CABECERA_COLA, 1)[1]
    check("la cola define TIME_BUDGET = 8.0 * 3600 y ROOT = DATA antes de llamar a brazo_v5()",
          "TIME_BUDGET = 8.0 * 3600" in cola and "ROOT = globals().get(\"DATA\"" in cola
          and cola.index("TIME_BUDGET =") < cola.index("ROOT =") < cola.rindex("brazo_v5()"))

    for n in (7, 1400):
        t = f"N={n}"
        # familias elegidas desde fuera
        verificar_aplicado(f"{t} una sola familia (eff)", correr(f"n{n}_fam_eff", n, CELDA, familias=("eff",)), n_modelos=5)
        verificar_aplicado(f"{t} las cuatro familias", correr(f"n{n}_fam_4", n, CELDA, familias=("eff", "cnxhib", "effhib", "r34att")),
                           n_modelos=20)
        r = correr(f"n{n}_fam_mala", n, CELDA, familias=("eff", "foo"))
        check(f"{t} familia desconocida: decisión 'error' y la etapa queda intacta",
              r["estado"].get("decision") == "error" and sha(r["stage_path"]) == r["sha_antes"], r["estado"].get("motivo", "")[:120])
        verificar_excepcion(f"{t} diagnóstica, familia desconocida", correr(f"n{n}_fam_mala_diag", n, DIAG, familias=("eff", "foo")))
        # T0 ausente pero NB_T0 presente (la cola lo copia)
        verificar_aplicado(f"{t} sin T0 pero con NB_T0", correr(f"n{n}_solo_nbt0", n, CELDA, con_t0=False, solo_nb_t0=True))
        # el notebook sí tiene ganchos rsna_* (por ejemplo, un notebook como el 0.940)
        verificar_aplicado(f"{t} con ganchos rsna_*", correr(f"n{n}_ganchos", n, CELDA, con_ganchos=True), con_ganchos=True)

    # Cola real de punta a punta, sin simular la inferencia: /kaggle/working y /kaggle/input apuntan a carpetas temporales.
    # Sin paquete v5 la celda debe omitirse POR FALTA DE PAQUETE, no por falta de T0/TIME_BUDGET: eso prueba que la
    # cola definió TIME_BUDGET y ROOT, y que V5_STAGE y los datos se leen donde dice la celda.
    for fuente, nombre in ((CELDA, "cola_real"), (DIAG, "cola_real_diag")):
        wd = OUT / nombre
        (wd / "working").mkdir(parents=True)
        (wd / "input").mkdir()
        data = wd / "data"
        data.mkdir()
        uids = [f"1.2.826.0.1.3680043.8.498.{i:020d}" for i in range(9)]
        pd.DataFrame({"StudyInstanceUID": uids}).to_csv(data / "test.csv", index=False)
        pd.DataFrame(columns=["StudyInstanceUID", *LABELS]).to_csv(data / "sample_submission.csv", index=False)
        st = pd.DataFrame(np.random.default_rng(5).uniform(0.01, 1, size=(9, 12)), columns=LABELS)
        st.insert(0, "StudyInstanceUID", uids)
        st.to_csv(wd / "working" / "submission.csv", index=False)
        sha0 = sha(wd / "working" / "submission.csv")
        src = fuente.read_text().replace("/kaggle/working", str(wd / "working")).replace("/kaggle/input", str(wd / "input"))
        ns = {"__name__": "celda_ns", "DATA": data, "NB_T0": time.time() - 600}
        ns["T0"] = ns["NB_T0"]
        exc = None
        try:
            exec(compile(src, str(fuente), "exec"), ns)
        except Exception as e:  # noqa: BLE001
            exc = e
        est = ns.get("BRAZO_V5_ESTADO") or json.load(open(wd / "working" / "brazo_v5" / "estado.json"))
        check(f"{nombre}: omitida por falta de paquete (no por T0/TIME_BUDGET) con la cola sin simulaciones",
              est.get("decision") == "omitido" and "dataset" in est.get("motivo", "") and "T0" not in est.get("motivo", ""),
              est.get("motivo", "")[:150])
        check(f"{nombre}: TIME_BUDGET = 8 h y ROOT = DATA tras la cola", ns.get("TIME_BUDGET") == 8 * 3600.0
              and str(ns.get("ROOT")) == str(data), f"{ns.get('TIME_BUDGET')} {ns.get('ROOT')}")
        check(f"{nombre}: submission.csv byte a byte igual", sha(wd / "working" / "submission.csv") == sha0)
        if fuente is DIAG:
            check(f"{nombre}: la diagnóstica lanza RuntimeError «V5_EXIGIR»", isinstance(exc, RuntimeError) and "V5_EXIGIR" in str(exc),
                  repr(exc)[:150])
        else:
            check(f"{nombre}: la normal no lanza", exc is None, repr(exc))

    # Búsqueda del paquete: solo hacen falta los pesos de las familias elegidas.
    wd = OUT / "paquete"
    pkg = wd / "input" / "paquete_v5"
    (pkg / "rsna").mkdir(parents=True)
    (pkg / "pesos").mkdir()
    (pkg / "rsna" / "inferir.py").write_text("# simulado\n")
    base = {}
    exec(compile(cabeza.replace("/kaggle/input", str(wd / "input")), str(CELDA), "exec"), {**base, "BRAZO_V5_SOLO_DEFINIR": True})
    for fam in ("eff", "effhib", "r34att"):
        for nombre_peso in d["V5_MIEMBROS"][fam]:
            (pkg / "pesos" / nombre_peso).write_bytes(b"x")

    def buscar(familias):
        ns2 = {"BRAZO_V5_SOLO_DEFINIR": True, "BRAZO_V5_FAMILIAS": tuple(familias)}
        exec(compile(cabeza.replace("/kaggle/input", str(wd / "input")), str(CELDA), "exec"), ns2)
        try:
            return ns2["_v5_paquete"]()
        except ns2["_V5Omitir"] as e:
            return e

    check("paquete sin pesos de cnxhib: se encuentra con las familias por defecto", buscar(("eff", "effhib", "r34att")) == str(pkg))
    check("paquete sin pesos de cnxhib: no se encuentra si se pide cnxhib", not isinstance(buscar(("eff", "cnxhib")), str))


def main():
    celda_txt, diag_txt = CELDA.read_text(), DIAG.read_text()
    check("la variante diagnóstica = cabecera + celda actual (sincronizada)",
          diag_txt.endswith(celda_txt) and "BRAZO_V5_EXIGIR = True" in diag_txt[: len(diag_txt) - len(celda_txt)])
    check("la celda ya no contiene el 1,322 fijo", "1322" not in celda_txt and "ESPERADOS" not in celda_txt)
    check("los 12 hallazgos de LAB (celda 2 del notebook) son los V5_LABELS de la celda",
          correr("def_labels", 7, CELDA)["ns"]["V5_LABELS"] == LABELS, str(LABELS))
    print(f"notebook sha256 {NB_SHA}", flush=True)

    for n in (7, 1400):
        t = f"N={n}"
        ns0 = correr(f"n{n}_defecto", n, CELDA)["ns"]
        nd = correr(f"n{n}_defecto_diag_ns", n, DIAG)["ns"]
        check(f"{t}: V5_EXIGIR es False en la celda normal y True en la diagnóstica",
              ns0["V5_EXIGIR"] is False and nd["V5_EXIGIR"] is True)

        # 1. la variante normal aplica la mezcla (con la variable de rerun puesta: antes se omitía por el 1,322)
        ilegibles = 10 if n == 1400 else 0
        r = correr(f"n{n}_normal_aplica", n, CELDA, rerun=True, n_ilegibles=ilegibles)
        verificar_aplicado(f"{t} normal (rerun, tiempo de sobra)", r, n_ilegibles=ilegibles)

        # 2. la diagnóstica con tiempo de sobra: aplica y no lanza
        r = correr(f"n{n}_diag_aplica", n, DIAG, rerun=True)
        verificar_aplicado(f"{t} diagnóstica (tiempo de sobra)", r)

        # 3. T0 forzado a «sin tiempo»: la normal omite limpia, la diagnóstica falla
        r = correr(f"n{n}_sin_tiempo_norm", n, CELDA, restante_s=-100.0, rerun=True)
        verificar_omitido(f"{t} normal, T0 forzado a «sin tiempo»", r, "quedan")
        r = correr(f"n{n}_sin_tiempo_diag", n, DIAG, restante_s=-100.0, rerun=True)
        verificar_excepcion(f"{t} diagnóstica, T0 forzado a «sin tiempo»", r)

        # 4. sin T0 en el notebook
        r = correr(f"n{n}_sin_T0_norm", n, CELDA, con_t0=False)
        verificar_omitido(f"{t} normal, sin T0", r, "T0")
        verificar_excepcion(f"{t} diagnóstica, sin T0", correr(f"n{n}_sin_T0_diag", n, DIAG, con_t0=False))

        # 5. borde de la guarda de tiempo
        necesario = ns0["V5_FIJOS_S"] + n * max(ns0["V5_PISO_CPU_S"], N_MODELOS * ns0["V5_GPU_MODELO_S"] / 2) + ns0["V5_MARGEN_S"]
        verificar_omitido(f"{t} borde: faltan 60 s sobre {necesario / 60:.1f} min",
                          correr(f"n{n}_borde_menos", n, CELDA, restante_s=necesario - 60), "quedan")
        verificar_aplicado(f"{t} borde: sobran 60 s sobre {necesario / 60:.1f} min",
                           correr(f"n{n}_borde_mas", n, CELDA, restante_s=necesario + 60))

        # 6. una sola GPU; ninguna GPU
        necesita_1gpu = ns0["V5_FIJOS_S"] + n * max(ns0["V5_PISO_CPU_S"], N_MODELOS * ns0["V5_GPU_MODELO_S"]) + ns0["V5_MARGEN_S"]
        verificar_aplicado(f"{t} con una sola GPU", correr(f"n{n}_una_gpu", n, CELDA, gpus=(0,), restante_s=necesita_1gpu + 600), n_gpus=1)
        verificar_omitido(f"{t} normal, sin GPU", correr(f"n{n}_sin_gpu", n, CELDA, gpus=()), "GPU")
        verificar_excepcion(f"{t} diagnóstica, sin GPU", correr(f"n{n}_sin_gpu_diag", n, DIAG, gpus=()))

        # 7. etapa inválida: columnas desordenadas, fuera de [0,1], UID repetido, sample_submission distinto
        def swap(df):
            c = list(df.columns)
            c[1], c[2] = c[2], c[1]
            return df[c]

        def fuera(df):
            df = df.copy()
            df.loc[df.index[0], "ACL"] = 1.3
            return df

        def repetido(df):
            df = df.copy()
            df.loc[df.index[-1], "StudyInstanceUID"] = df.loc[df.index[0], "StudyInstanceUID"]
            return df

        for etiqueta, slug, kw in (("columnas desordenadas", "cols", dict(mod_stage=swap)),
                                   ("valor fuera de [0,1]", "rango", dict(mod_stage=fuera)),
                                   ("UID repetido", "uid", dict(mod_stage=repetido)),
                                   ("sample_submission con otro orden", "ss", dict(ss_desordenado=True))):
            verificar_omitido(f"{t} etapa inválida ({etiqueta})", correr(f"n{n}_malo_{slug}", n, CELDA, **kw))
            verificar_excepcion(f"{t} diagnóstica, etapa inválida ({etiqueta})", correr(f"n{n}_malo_{slug}_diag", n, DIAG, **kw))

        # 8. demasiados ilegibles (> 5 %): error, no se mezcla
        k = int(np.ceil(0.07 * n))
        r = correr(f"n{n}_ilegibles", n, CELDA, n_ilegibles=k)
        check(f"{t} {k} ilegibles ({k / n:.0%}): decisión 'error', etapa intacta, sin excepción",
              r["estado"].get("decision") == "error" and r["exc"] is None and sha(r["stage_path"]) == r["sha_antes"],
              r["estado"].get("motivo", "")[:120])
        verificar_excepcion(f"{t} diagnóstica, {k} ilegibles", correr(f"n{n}_ilegibles_diag", n, DIAG, n_ilegibles=k))

        # 9. W = 0 en un hallazgo deja su columna exactamente igual
        w = {c: 0.10 for c in LABELS}
        w["Fracture"] = 0.0
        r = correr(f"n{n}_w0", n, CELDA, w=w)
        verificar_aplicado(f"{t} con W=0 en Fracture", r, w=w)
        df, ant = etapa_leida(r), pd.read_csv(r["original"], dtype={"StudyInstanceUID": str})
        check(f"{t} W=0: la columna Fracture queda exactamente igual que en la etapa original",
              np.array_equal(df["Fracture"].to_numpy(), ant["Fracture"].to_numpy()))

    pruebas_notebook_0957()

    fallos = [x for x in RES if not x["ok"]]
    (OUT / "resultados.json").write_text(json.dumps(
        {"notebook_sha256": NB_SHA, "total": len(RES), "fallos": len(fallos), "checks": RES}, indent=1, ensure_ascii=False))
    print(f"\nRESUMEN: {len(RES) - len(fallos)}/{len(RES)} comprobaciones PASS; {len(fallos)} FAIL", flush=True)
    for x in fallos:
        print("  FAIL:", x["check"], "|", x["detalle"][:200])
    return 1 if fallos else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
