#!/usr/bin/env python
"""Prueba en CPU del brazo DINOv2 (celda_para_pegar_0957.py, su variante diagnóstica y script_dinov2.py).

NO PRUEBA DINOv2 REAL: en Yuca falta `transformers` y los pesos (pilkwang/rsna-knee-weights, metaresearch/dinov2) no
están en el clúster. Lo que sí prueba:
  A. Procedencia: el script generado coincide con lo que produce el generador; sus definiciones son idénticas por AST a
     las del notebook 0.940; no toca submission.csv; importa limpio; las funciones numéricas puras (ventanas, _combine,
     write_submission) hacen lo que dicen; su CLI falla de forma ordenada sin transformers.
  B. La celda de punta a punta con etapa sintética de N=7 y N=1400 y la inferencia SIMULADA: mezcla, validación, guarda de
     tiempo, escritura atómica, fail-soft y variante diagnóstica. Más el mecanismo REAL de subproceso (argumentos, código
     de salida, corte por plazo con muerte del grupo de procesos) con un script falso, y la cola real de la celda.
  C. Opcional (--preproceso N): el preproceso REAL extraído del 0.940 (cabeceras, ordenación, decodificación, caché) sobre N
     estudios de train, para medir segundos por estudio de CPU.

Uso:  python test_celda_brazo_dinov2.py <directorio_de_salida> [--preproceso N]
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB_957 = REPO / "referencias" / "0-957-latest-dataset-score-in-one-click.ipynb"
NB_940 = REPO / "referencias" / "bend-the-knee-to-speedy-raptors-the-original.ipynb"
SCRIPT = HERE / "script_dinov2.py"
CELDA = HERE / "celda_para_pegar_0957.py"
DIAG = HERE / "celda_para_pegar_0957_diag.py"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
N_PREPROCESO = int(sys.argv[sys.argv.index("--preproceso") + 1]) if "--preproceso" in sys.argv else 0
COMP = Path("/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection")

RES: list[dict] = []
CABECERA_COLA = 'if not globals().get("BRAZO_DINOV2_SOLO_DEFINIR"):'


def check(nombre: str, cond, detalle: str = "") -> None:
    RES.append({"check": nombre, "ok": bool(cond), "detalle": str(detalle)})
    print(("PASS " if cond else "FAIL "), nombre, ("| " + str(detalle)) if detalle else "", flush=True)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(path) -> str:
    return sha_bytes(Path(path).read_bytes())


def cargar_generador():
    spec = importlib.util.spec_from_file_location("generar_brazo_dinov2", HERE / "generar_brazo_dinov2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def etiquetas_del_notebook():
    nb = json.load(open(NB_957))
    codigo = "".join(nb["cells"][2]["source"]).split("\n", 1)[1]
    for node in ast.parse(codigo).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "LAB" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no se encontró LAB en la celda 2 del 0.957")


LABELS = etiquetas_del_notebook()


def validador_celda6(df, uids, cols):
    """Las aserciones finales de la celda 6 del 0.957 sobre submission.csv, reescritas aparte."""
    assert list(df.columns) == list(cols), "columnas"
    assert df["StudyInstanceUID"].tolist() == list(uids), "orden de UIDs"
    v = df[list(cols[1:])].to_numpy(float)
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), "valores"


def cargar_script_como_modulo(nombre="script_dinov2_test"):
    """El script generado como módulo importable (no se ejecuta main). Devuelve su espacio de nombres."""
    ns = {"__name__": nombre, "__file__": str(SCRIPT)}
    exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), ns)
    return ns


# =====================================================================================================
# A. Procedencia y script
# =====================================================================================================
def pruebas_script():
    gen = cargar_generador()
    salidas = gen.generar()
    salidas.pop("_inventario")
    for ruta, texto in salidas.items():
        check(f"el generador reproduce {Path(ruta).name} byte a byte", Path(ruta).read_text() == texto)

    script = SCRIPT.read_text()
    cabeza_nb = NB_940.read_text()
    check("la cabecera del script cita el sha256 del notebook 0.940 actual", f"sha256 {sha_bytes(cabeza_nb.encode())}" in script)

    # Definiciones idénticas por AST a las del notebook (salvo la sustitución declarada de la ruta de diagnósticos).
    celdas = gen.celdas_940()
    nb_defs = {}
    for i, texto in celdas.items():
        for n in ast.parse(texto).body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                nb_defs.setdefault(n.name, []).append((i, n))
    arbol = ast.parse(script)
    sub = gen.SUSTITUCIONES
    n_ok, n_dif, sin_origen = 0, [], []
    fin_extraido = next(i for i, n in enumerate(arbol.body) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "DIAG_DIR" for t in n.targets))
    for n in arbol.body[:fin_extraido]:
        if not isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            continue
        candidatos = nb_defs.get(n.name, [])
        if not candidatos:
            sin_origen.append(n.name)
            continue
        dump = ast.dump(n)
        coincide = False
        for celda_i, orig in candidatos:
            texto_orig = ast.get_source_segment(celdas[celda_i], orig, padded=False)
            for viejo, nuevo, _veces in sub:
                texto_orig = texto_orig.replace(viejo, nuevo)
            if ast.dump(ast.parse(texto_orig).body[0]) == dump or ast.dump(orig) == dump:
                coincide = True
        if coincide:
            n_ok += 1
        else:
            n_dif.append(n.name)
    check(f"las {n_ok} funciones y clases extraídas son idénticas por AST a las del notebook 0.940",
          n_ok >= 60 and not n_dif and not sin_origen, f"idénticas={n_ok} distintas={n_dif} sin origen={sin_origen}")
    check("no se extrajo rsna_initialize (borra submission.csv) ni rsna_asset_preflight (exige otros datasets)",
          "def rsna_initialize" not in script and "def rsna_asset_preflight" not in script)
    check("el script no abre submission.csv ni borra nada que no sea su caché de píxeles",
          "to_csv('submission.csv'" not in script and "unlink" in script and script.count("unlink(") == 2)
    escribe = [ln.strip() for ln in script.splitlines() if "/kaggle/working" in ln and not ln.strip().startswith("#")]
    check("la única ruta de /kaggle/working que usa el script es la de diagnósticos (sustituida por DIAG_DIR)",
          escribe == ["DIAG_DIR = '/kaggle/working/diagnostics'"], str(escribe))

    ns = cargar_script_como_modulo()
    check("el script se importa sin ejecutar nada (sin GPU, sin datasets) y define main y el pegamento",
          all(callable(ns.get(f)) for f in ("main", "infer_from_package", "_combine", "build_cache", "_glue_main", "find_weights")))
    check("TARGETS del script = los 12 hallazgos de LAB del 0.957 (celda 2) y V5/DV2_LABELS de la celda",
          ns["TARGETS"] == LABELS, str(ns["TARGETS"]))
    check("6 ranuras SLOTS_RECOVERED; EXPECTED_DINOV2_MEMBERS = 20", ns["N_SLOT"] == 6 and ns["EXPECTED_DINOV2_MEMBERS"] == 20)
    check("window_starts(12 cortes, grupo 3) = 10 ventanas (6 ranuras x 10 = 60 imágenes por estudio y miembro)",
          ns["window_starts"](12, 3) == list(range(10)))

    # _combine: media de rangos de los miembros, comprobada con una reimplementación aparte.
    rng = np.random.default_rng(7)
    ids = [f"u{i:03d}" for i in range(40)]
    miembros = [{"id": f"m{k}", "ids": list(ids), "pred": rng.uniform(0.01, 0.99, size=(40, 12))} for k in range(4)]
    todos, acc = ns["_combine"](miembros)
    esperado = np.mean([pd.DataFrame(m["pred"]).rank(pct=True).to_numpy() for m in miembros], axis=0)
    check("_combine = media de rangos percentiles de los miembros", todos == sorted(ids) and np.allclose(acc, esperado))

    # write_submission: orden de test.csv, rangos percentiles, hueco = 0.5.
    wd = OUT / "write_submission"
    wd.mkdir()
    test_df = pd.DataFrame({"StudyInstanceUID": ["b", "a", "c", "z"]})
    pred = rng.uniform(size=(3, 12))
    cwd = os.getcwd()
    try:
        os.chdir(wd)
        out = ns["write_submission"](pred, ["a", "b", "c"], test_df, "etapa.csv")
    finally:
        os.chdir(cwd)
    rel = pd.read_csv(wd / "etapa.csv", dtype={"StudyInstanceUID": str})
    check("write_submission: orden de test.csv, columnas del concurso, rangos percentiles y 0.5 para el UID que no se predijo",
          rel.StudyInstanceUID.tolist() == ["b", "a", "c", "z"] and list(rel.columns) == ["StudyInstanceUID", *LABELS]
          and (rel.loc[3, LABELS] == 0.5).all() and rel.loc[0, LABELS].max() <= 1 and len(out) == 4)

    # Guarda de pesos: con 19 miembros find_weights se niega (no hay salida parcial).
    # CLI: sin transformers/datasets el script debe fallar de forma ordenada, no colgarse ni escribir en submission.csv.
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--comprobar", "--out", str(OUT / "cli_comprobar")],
                       capture_output=True, text=True, timeout=300)
    cola = (r.stderr.strip().splitlines() or [""])[-1]
    check("CLI --comprobar en Yuca falla con código ≠ 0 (faltan transformers y los datasets de Kaggle; era lo esperado)",
          r.returncode != 0 and ("transformers" in cola or "dinov2" in cola.lower() or "FileNotFoundError" in cola), cola[:200])
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--out", str(OUT / "cli_sin_args")], capture_output=True, text=True, timeout=300)
    check("CLI sin --data/--deadline: sale con mensaje y código ≠ 0", r.returncode != 0 and "--data" in (r.stderr + r.stdout))
    check("el CLI no escribió submission.csv ni nada fuera de --out", not (OUT / "cli_comprobar" / "submission.csv").exists())


# =====================================================================================================
# B. La celda con inferencia simulada
# =====================================================================================================
@contextlib.contextmanager
def entorno(**kv):
    previo = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in previo.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def preparar_dir(nombre, n, *, ss_desordenado=False, mod_stage=None):
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
    original = wd / "_etapa_original.csv"
    stage.to_csv(original, index=False)
    return wd, uids, data, stage, stage_path, original


def correr(nombre, n, fuente: Path, *, restante_s=7 * 3600, con_t0=True, solo_nb_t0=False, gpus=(0, 1), vacios=0,
           w=None, mod_stage=None, ss_desordenado=False, informe_extra=None, anticorrelado=False, comprobar_omite=None,
           sha_malo=False, real=None, constantes=None, fake_smi=None):
    """Una ejecución de la celda con una etapa sintética de n estudios. `real` = comportamiento del script falso."""
    wd, uids, data, stage, stage_path, original = preparar_dir(nombre, n, ss_desordenado=ss_desordenado, mod_stage=mod_stage)
    sha_antes = sha(original)
    texto = fuente.read_text()
    cabeza, cola = texto.rsplit("\n" + CABECERA_COLA, 1)
    ns = {"__name__": "celda_ns", "BRAZO_DINOV2_SOLO_DEFINIR": True}
    exec(compile(cabeza, str(fuente), "exec"), ns)
    # El notebook 0.957 NO tiene rsna_frame, rsna_phase, rsna_event, rsna_save_predictions, _RSNA_*, TIME_BUDGET ni ROOT.
    ns["DATA"] = data
    ns["DV2_STAGE"], ns["DV2_DIR"] = str(stage_path), str(wd / "brazo_dinov2")
    presupuesto = 8 * 3600.0
    if con_t0 or solo_nb_t0:
        t = time.time() - (presupuesto - restante_s)
        ns["NB_T0"] = t
        if con_t0:
            ns["T0"] = t
    if w is not None:
        ns["DV2_W"] = dict(w)
    if sha_malo:
        ns["DV2_SCRIPT_SHA256"] = "0" * 64
    for k, v in (constantes or {}).items():
        ns[k] = v

    llamada: dict = {}
    if real is None:
        ns["_dv2_gpus"] = lambda: (list(gpus), f"simulado {list(gpus)}")

        def comprobar_simulado(script, data_):
            llamada["comprobar_script_existe"] = os.path.isfile(script)
            if comprobar_omite:
                raise ns["_DV2Omitir"](comprobar_omite)
            return {"simulado": True}

        ns["_dv2_comprobar"] = comprobar_simulado

        def correr_simulado(script, data_, gpus_, limite, destino):
            llamada.update(n=n, gpus=list(gpus_), a_limite_s=limite - time.time(), data=str(data_), destino=str(destino))
            r = np.random.default_rng(99 + n).uniform(0.02, 0.98, size=(n, 12))
            if anticorrelado:
                r = 1.0 - stage[LABELS].to_numpy() * 0.96 - 0.02 + np.random.default_rng(5).normal(0, 0.001, size=(n, 12))
                r = np.clip(r, 0.01, 0.99)
            df = pd.DataFrame(r, columns=LABELS)
            df.insert(0, "StudyInstanceUID", uids)
            salida = df.sample(frac=1, random_state=3).reset_index(drop=True)
            llamada["dv2"] = salida.copy()
            ids_vacios = list(uids[:vacios])
            informe = {"miembros": 20, "estudios": n, "gpus": len(gpus_), "estudios_sin_series": ids_vacios}
            informe.update(informe_extra or {})
            return salida, informe

        ns["_dv2_correr"] = correr_simulado

    exc, estado = None, None
    ruta_path = os.environ.get("PATH", "")
    if fake_smi is not None:
        carpeta = wd / "fakebin"
        carpeta.mkdir()
        (wd / "smi.txt").write_text("\n".join(fake_smi) + "\n")
        smi = carpeta / "nvidia-smi"
        smi.write_text(f"#!/bin/bash\ncat '{wd / 'smi.txt'}'\n")
        smi.chmod(0o755)
    with entorno(PATH=(str(wd / "fakebin") + ":" + ruta_path) if fake_smi is not None else ruta_path):
        try:
            ns["BRAZO_DINOV2_SOLO_DEFINIR"] = False
            exec(compile(CABECERA_COLA + cola, str(fuente) + "#cola", "exec"), ns)
            estado = ns["BRAZO_DINOV2_ESTADO"]
        except Exception as e:  # noqa: BLE001
            exc = e
    ej = wd / "brazo_dinov2" / "estado.json"
    estado_json = json.load(open(ej)) if ej.is_file() else {}
    return dict(nombre=nombre, n=n, ns=ns, uids=uids, estado=estado or estado_json, exc=exc, stage_path=stage_path,
                original=original, stage_antes=stage, sha_antes=sha_antes, llamada=llamada, wd=wd, restante_s=restante_s)


def etapa_leida(r) -> pd.DataFrame:
    return pd.read_csv(r["stage_path"], dtype={"StudyInstanceUID": str})


def mezcla_esperada(ant: pd.DataFrame, dv2: pd.DataFrame, w: dict, vacios=()) -> pd.DataFrame:
    """Reimplementación independiente de la mezcla por rangos: (1-W)·rango_notebook + W·rango_dinov2, re-rankeada."""
    dv2 = dv2.set_index("StudyInstanceUID").reindex(ant["StudyInstanceUID"])
    vacio = ant["StudyInstanceUID"].isin(set(vacios)).to_numpy()
    out = ant.copy()
    for c in LABELS:
        if w[c] <= 0:
            continue
        r_nb = ant[c].rank(method="average", pct=True).to_numpy()
        x = dv2[c].copy()
        x[vacio] = np.nan
        r_d = x.rank(method="average", pct=True).fillna(0.0).to_numpy()
        peso = w[c] * (~vacio)
        out[c] = pd.Series((1 - peso) * r_nb + peso * r_d).rank(method="average", pct=True).to_numpy()
    return out


def verificar_aplicado(tag, r, *, n_gpus=2, vacios=0, w=None):
    est, ns, n = r["estado"], r["ns"], r["n"]
    w = w or {c: 0.15 for c in LABELS}
    check(f"{tag}: decisión 'aplicado' sin excepción", est.get("decision") == "aplicado" and r["exc"] is None,
          f"{est.get('decision')} {est.get('motivo', '')[:140]} exc={r['exc']!r}")
    df, ant = etapa_leida(r), pd.read_csv(r["original"], dtype={"StudyInstanceUID": str})
    check(f"{tag}: columnas y orden de filas intactos",
          list(df.columns) == ["StudyInstanceUID", *LABELS] and df.StudyInstanceUID.tolist() == ant.StudyInstanceUID.tolist())
    ss_cols = list(pd.read_csv(r["wd"] / "data" / "sample_submission.csv", nrows=0).columns)
    try:
        validador_celda6(df, r["uids"], ss_cols)
        ok6 = True
    except AssertionError as e:
        ok6 = False
        print("   validador de la celda 6:", repr(e))
    check(f"{tag}: pasa las aserciones finales de la celda 6 (orden de UIDs, columnas, finitos, [0, 1])", ok6)
    check(f"{tag}: la cola de la celda dejó TIME_BUDGET = 8 h y ROOT = DATA",
          ns.get("TIME_BUDGET") == 8 * 3600.0 and str(ns.get("ROOT")) == str(ns["DATA"]),
          f"TIME_BUDGET={ns.get('TIME_BUDGET')} ROOT={ns.get('ROOT')}")
    v = df[LABELS].to_numpy(float)
    check(f"{tag}: valores en (0, 1]", np.isfinite(v).all() and v.min() > 0 and v.max() <= 1.0, f"min={v.min():.5f} max={v.max():.5f}")
    if "dv2" in r["llamada"]:
        dv2_previo = r["llamada"]["dv2"]
    else:                                   # modo real: lo que escribió el script falso
        dv2_previo = pd.read_csv(r["wd"] / "brazo_dinov2" / "run" / "dinov2_sub.csv", dtype={"StudyInstanceUID": str})
    esperado = mezcla_esperada(ant, dv2_previo, w, vacios=r["uids"][:vacios])
    dif = np.abs(df[LABELS].to_numpy(float) - esperado[LABELS].to_numpy(float)).max()
    check(f"{tag}: coincide con la mezcla por rangos recalculada de forma independiente", dif < 1e-9, f"max|Δ|={dif:.2e}")
    if n >= 100:
        cambio = np.abs(v - ant[LABELS].to_numpy(float)).max(axis=0)
        activos = [c for c in LABELS if w[c] > 0]
        check(f"{tag}: cada hallazgo con W>0 cambia respecto de la etapa original", all(cambio[LABELS.index(c)] > 0 for c in activos))
        rho = [df[c].corr(ant[c], method="spearman") for c in activos]
        check(f"{tag}: Spearman medio con la etapa original entre 0.9 y 0.9995 (W=0.15)", 0.9 < float(np.mean(rho)) < 0.9995,
              f"medio={np.mean(rho):.4f}")
    copia = r["wd"] / "brazo_dinov2" / "submission_antes_dinov2.csv"
    check(f"{tag}: copia de seguridad submission_antes_dinov2.csv idéntica a la etapa original", copia.is_file() and sha(copia) == r["sha_antes"])
    check(f"{tag}: sin archivo temporal", not Path(str(r["stage_path"]) + ".tmp_dinov2").exists())
    ll = r["llamada"]
    if ll:
        check(f"{tag}: inferencia pedida con {n} estudios y {n_gpus} GPU, y el script incrustado se escribió",
              ll.get("n") == n and len(ll.get("gpus", [])) == n_gpus and (r["wd"] / "brazo_dinov2" / "script_dinov2.py").is_file()
              and ll.get("comprobar_script_existe", True) is not False, str({k: v for k, v in ll.items() if k != "dv2"}))
        esperado_s = r["restante_s"] - ns["DV2_MARGEN_S"]
        check(f"{tag}: plazo duro = restante − margen (±30 s)", abs(ll.get("a_limite_s", -1e9) - esperado_s) < 30,
              f"{ll.get('a_limite_s', 0):.0f} s vs {esperado_s:.0f} s")
        check(f"{tag}: el script escrito es byte a byte el incrustado", sha(r["wd"] / "brazo_dinov2" / "script_dinov2.py") == ns["DV2_SCRIPT_SHA256"])
    if vacios:
        n_vac = vacios
        ok_ord = all(
            np.array_equal(df[c].iloc[:n_vac].rank(method="first").to_numpy(), ant[c].iloc[:n_vac].rank(method="first").to_numpy())
            for c in LABELS)
        check(f"{tag}: {n_vac} estudios sin series no votan (mantienen el orden relativo del notebook)",
              ok_ord and est.get("sin_series") == n_vac, f"sin_series={est.get('sin_series')}")


def verificar_omitido(tag, r, subcadena=""):
    est = r["estado"]
    check(f"{tag}: decisión 'omitido', sin excepción", est.get("decision") == "omitido" and r["exc"] is None,
          f"{est.get('decision')}: {est.get('motivo', '')[:160]}")
    check(f"{tag}: la etapa queda byte a byte igual", sha(r["stage_path"]) == r["sha_antes"])
    if subcadena:
        check(f"{tag}: el motivo menciona «{subcadena}»", subcadena in est.get("motivo", ""), est.get("motivo", "")[:160])
    check(f"{tag}: sin copia de seguridad ni temporal", not (r["wd"] / "brazo_dinov2" / "submission_antes_dinov2.csv").exists()
          and not Path(str(r["stage_path"]) + ".tmp_dinov2").exists())


def verificar_error(tag, r, subcadena=""):
    est = r["estado"]
    check(f"{tag}: decisión 'error', sin excepción, etapa intacta",
          est.get("decision") == "error" and r["exc"] is None and sha(r["stage_path"]) == r["sha_antes"],
          f"{est.get('decision')}: {est.get('motivo', '')[:200]}")
    if subcadena:
        check(f"{tag}: el motivo menciona «{subcadena}»", subcadena in est.get("motivo", ""), est.get("motivo", "")[:200])


def verificar_excepcion(tag, r):
    check(f"{tag}: la variante diagnóstica lanza RuntimeError «DV2_EXIGIR»",
          isinstance(r["exc"], RuntimeError) and "DV2_EXIGIR" in str(r["exc"]), repr(r["exc"])[:200])
    check(f"{tag}: aun así la etapa queda byte a byte igual", sha(r["stage_path"]) == r["sha_antes"])


def pruebas_celda():
    celda_txt = CELDA.read_text()
    diag_txt = DIAG.read_text()
    cabecera_diag = diag_txt[: len(diag_txt) - len(celda_txt)]
    check("la variante diagnóstica = cabecera + celda actual (sincronizada)",
          diag_txt.endswith(celda_txt) and cabecera_diag.startswith("# VARIANTE DIAGNÓSTICA")
          and "BRAZO_DINOV2_EXIGIR = True" in cabecera_diag)
    ns_def = {"BRAZO_DINOV2_SOLO_DEFINIR": True}
    exec(compile(celda_txt.rsplit("\n" + CABECERA_COLA, 1)[0], str(CELDA), "exec"), ns_def)
    check("por defecto: W = 0.15 en los 12 hallazgos", set(ns_def["DV2_W"]) == set(LABELS) and set(ns_def["DV2_W"].values()) == {0.15})
    check("DV2_LABELS = los 12 hallazgos de LAB del 0.957", ns_def["DV2_LABELS"] == LABELS)
    check('DV2_STAGE = "/kaggle/working/submission.csv"', ns_def["DV2_STAGE"] == "/kaggle/working/submission.csv")
    check("el sha256 declarado de DV2_SCRIPT_TEXTO es el de script_dinov2.py", ns_def["DV2_SCRIPT_SHA256"] == sha(SCRIPT)
          and ns_def["DV2_SCRIPT_TEXTO"] == SCRIPT.read_text())
    cola_txt = celda_txt.rsplit("\n" + CABECERA_COLA, 1)[1]
    check("la cola define TIME_BUDGET = 8.0 * 3600 y ROOT = DATA antes de llamar a brazo_dinov2()",
          "TIME_BUDGET = 8.0 * 3600" in cola_txt and 'ROOT = globals().get("DATA"' in cola_txt
          and cola_txt.index("TIME_BUDGET =") < cola_txt.index("ROOT =") < cola_txt.rindex("brazo_dinov2()"))
    nombres_celda = {n.name for n in ast.parse(celda_txt).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    check("la celda no usa rsna_*, _RSNA_* ni nombres del brazo v5 (no choca con él si conviven)",
          not any(s in celda_txt.split("DV2_SCRIPT_TEXTO = r'''")[0] for s in ("rsna_", "_RSNA_", "V5_", "brazo_v5", "_v5_")),
          str(sorted(nombres_celda)))

    for n in (7, 1400):
        t = f"N={n}"
        ns0 = correr(f"n{n}_defecto", n, CELDA)["ns"]
        nd = correr(f"n{n}_defecto_diag_ns", n, DIAG)["ns"]
        check(f"{t}: DV2_EXIGIR es False en la celda normal y True en la diagnóstica", ns0["DV2_EXIGIR"] is False and nd["DV2_EXIGIR"] is True)

        # 1. Tiempo de sobra: la normal aplica la mezcla; la diagnóstica aplica y no lanza.
        vac = 10 if n == 1400 else 0
        verificar_aplicado(f"{t} normal (tiempo de sobra)", correr(f"n{n}_normal_aplica", n, CELDA, vacios=vac), vacios=vac)
        verificar_aplicado(f"{t} diagnóstica (tiempo de sobra)", correr(f"n{n}_diag_aplica", n, DIAG))

        # 2. T0 forzado a «sin tiempo»: la normal omite limpia, la diagnóstica falla.
        verificar_omitido(f"{t} normal, T0 forzado a «sin tiempo»", correr(f"n{n}_sin_tiempo", n, CELDA, restante_s=-100.0), "quedan")
        verificar_excepcion(f"{t} diagnóstica, T0 forzado a «sin tiempo»", correr(f"n{n}_sin_tiempo_d", n, DIAG, restante_s=-100.0))

        # 3. Sin T0 en el notebook / solo NB_T0.
        verificar_omitido(f"{t} normal, sin T0", correr(f"n{n}_sin_T0", n, CELDA, con_t0=False), "T0")
        verificar_excepcion(f"{t} diagnóstica, sin T0", correr(f"n{n}_sin_T0_d", n, DIAG, con_t0=False))
        verificar_aplicado(f"{t} sin T0 pero con NB_T0", correr(f"n{n}_solo_nbt0", n, CELDA, con_t0=False, solo_nb_t0=True))

        # 4. Borde de la guarda de tiempo (con 2 GPU).
        necesario = ns0["DV2_FIJOS_S"] + n * (ns0["DV2_CPU_ESTUDIO_S"] + ns0["DV2_GPU_ESTUDIO_S"] / 2) + ns0["DV2_MARGEN_S"]
        verificar_omitido(f"{t} borde: faltan 60 s sobre {necesario / 60:.1f} min",
                          correr(f"n{n}_borde_menos", n, CELDA, restante_s=necesario - 60), "quedan")
        verificar_aplicado(f"{t} borde: sobran 60 s sobre {necesario / 60:.1f} min",
                           correr(f"n{n}_borde_mas", n, CELDA, restante_s=necesario + 60))

        # 5. Una sola GPU: el término de GPU se duplica; ninguna GPU: se omite.
        necesita_1 = ns0["DV2_FIJOS_S"] + n * (ns0["DV2_CPU_ESTUDIO_S"] + ns0["DV2_GPU_ESTUDIO_S"]) + ns0["DV2_MARGEN_S"]
        verificar_omitido(f"{t} una GPU: faltan 60 s sobre {necesita_1 / 60:.1f} min", correr(f"n{n}_una_gpu_menos", n, CELDA, gpus=(0,),
                          restante_s=necesita_1 - 60), "quedan")
        verificar_aplicado(f"{t} con una sola GPU", correr(f"n{n}_una_gpu", n, CELDA, gpus=(0,), restante_s=necesita_1 + 600), n_gpus=1)
        verificar_omitido(f"{t} normal, sin GPU", correr(f"n{n}_sin_gpu", n, CELDA, gpus=()), "GPU")
        verificar_excepcion(f"{t} diagnóstica, sin GPU", correr(f"n{n}_sin_gpu_d", n, DIAG, gpus=()))

        # 6. Faltan datasets/librerías (la comprobación previa del script los detecta): omitir, o fallar si diagnóstica.
        verificar_omitido(f"{t} normal, falta un requisito (comprobación previa)",
                          correr(f"n{n}_sin_req", n, CELDA, comprobar_omite="falta el dataset X"), "falta el dataset X")
        verificar_excepcion(f"{t} diagnóstica, falta un requisito", correr(f"n{n}_sin_req_d", n, DIAG, comprobar_omite="falta el dataset X"))

        # 7. Etapa inválida: columnas desordenadas, fuera de [0,1], UID repetido, sample_submission en otro orden.
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
            verificar_excepcion(f"{t} diagnóstica, etapa inválida ({etiqueta})", correr(f"n{n}_malo_{slug}_d", n, DIAG, **kw))

        # 8. Demasiados estudios sin series (> 5 %): error, no se mezcla.
        k = int(np.ceil(0.07 * n))
        verificar_error(f"{t} {k} estudios sin series ({k / n:.0%})", correr(f"n{n}_vacios", n, CELDA, vacios=k), "sin series utilizables")
        verificar_excepcion(f"{t} diagnóstica, {k} estudios sin series", correr(f"n{n}_vacios_d", n, DIAG, vacios=k))

        # 9. Informe con un número de miembros distinto de 20: error.
        verificar_error(f"{t} el script informa 19 miembros", correr(f"n{n}_19", n, CELDA, informe_extra={"miembros": 19}), "20")

        # 10. Predicciones anticorrelacionadas con el notebook: error con n >= 100; con n pequeño no se aplica la guarda.
        if n >= 100:
            verificar_error(f"{t} correlación de rangos ≤ 0", correr(f"n{n}_anticorr", n, CELDA, anticorrelado=True), "correlación")
        else:
            verificar_aplicado(f"{t} correlación negativa con pocos estudios: la guarda no se aplica", correr(f"n{n}_anticorr", n, CELDA, anticorrelado=True))

        # 11. El script incrustado no coincide con su sha256 (celda pegada incompleta): error.
        verificar_error(f"{t} sha256 del script incrustado alterado", correr(f"n{n}_sha", n, CELDA, sha_malo=True), "sha256")

        # 12. W = 0 en un hallazgo deja su columna exactamente igual.
        w = {c: 0.15 for c in LABELS}
        w["Fracture"] = 0.0
        r = correr(f"n{n}_w0", n, CELDA, w=w)
        verificar_aplicado(f"{t} con W=0 en Fracture", r, w=w)
        df, ant = etapa_leida(r), pd.read_csv(r["original"], dtype={"StudyInstanceUID": str})
        check(f"{t} W=0: la columna Fracture queda exactamente igual que en la etapa original",
              np.array_equal(df["Fracture"].to_numpy(), ant["Fracture"].to_numpy()))


def script_falso(modo: str) -> str:
    """Un script_dinov2.py de mentira que atiende --comprobar y la ejecución, para probar el mecanismo de subproceso."""
    return f'''import argparse, json, os, subprocess, sys, time
import pandas as pd, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--deadline", type=float); ap.add_argument("--comprobar", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
if a.comprobar:
    if "{modo}" == "sin_requisitos":
        sys.stderr.write("ModuleNotFoundError: No module named 'transformers'\\n"); sys.exit(1)
    json.dump({{"falso": True}}, open(os.path.join(a.out, "comprobacion.json"), "w")); sys.exit(0)
json.dump({{"argv": sys.argv[1:], "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), "orden": os.environ.get("CUDA_DEVICE_ORDER"),
           "cwd": os.getcwd(), "pid": os.getpid()}}, open(os.path.join(a.out, "llamada.json"), "w"))
if "{modo}" == "falla":
    sys.stderr.write("RuntimeError: boom en el script falso\\n"); sys.exit(3)
if "{modo}" == "cuelga":
    nieto = subprocess.Popen(["sleep", "300"])
    open(os.path.join(a.out, "nieto.pid"), "w").write(str(nieto.pid))
    time.sleep(300)
uids = pd.read_csv(os.path.join(a.data, "test.csv"), dtype={{"StudyInstanceUID": str}}).StudyInstanceUID.tolist()
r = np.random.default_rng(1).uniform(0.02, 0.98, size=(len(uids), 12))
labels = {LABELS!r}
df = pd.DataFrame(r, columns=labels); df.insert(0, "StudyInstanceUID", uids)
df.to_csv(os.path.join(a.out, "dinov2_sub.csv"), index=False)
json.dump({{"miembros": 20, "estudios": len(uids), "estudios_sin_series": []}}, open(os.path.join(a.out, "informe.json"), "w"))
'''


def pruebas_subproceso_real():
    """El mecanismo REAL de la celda (argumentos, GPU, código de salida, corte por plazo) con un script falso."""
    n = 7
    smi = ["0, 15109, 15360", "1, 15109, 15360"]
    rapido = {"DV2_FIJOS_S": 0, "DV2_CPU_ESTUDIO_S": 0.0, "DV2_GPU_ESTUDIO_S": 0.0, "DV2_MARGEN_S": 0}

    def con_falso(nombre, modo, fuente=CELDA, restante_s=7 * 3600, constantes=None, fake=smi):
        # Mismo flujo que `correr`, pero sin simular _dv2_comprobar/_dv2_correr: se sustituye el texto del script.
        texto = fuente.read_text()
        falso = script_falso(modo)
        viejo_sha = f"DV2_SCRIPT_SHA256 = {sha(SCRIPT)!r}"
        assert texto.count(viejo_sha) == 1
        cabeza, cola = texto.rsplit("\n" + CABECERA_COLA, 1)
        # El literal incrustado se sustituye por el script falso (y su sha256), sin tocar el resto de la celda.
        ini = cabeza.index("DV2_SCRIPT_SHA256 = ")
        cabeza = cabeza[:ini] + f"DV2_SCRIPT_SHA256 = {sha_bytes(falso.encode())!r}\nDV2_SCRIPT_TEXTO = {falso!r}\n"
        tmpcelda = OUT / f"_celda_falsa_{nombre}.py"
        tmpcelda.write_text(cabeza + "\n" + CABECERA_COLA + cola)
        return correr(nombre, n, tmpcelda, real=True, restante_s=restante_s, constantes=constantes, fake_smi=fake)

    r = con_falso("real_ok", "ok")
    ll = json.load(open(r["wd"] / "brazo_dinov2" / "run" / "llamada.json")) if (r["wd"] / "brazo_dinov2" / "run" / "llamada.json").is_file() else {}
    verificar_aplicado("subproceso real (script falso, éxito)", r)
    argv = ll.get("argv", [])
    check("subproceso real: recibe --data DATA, --out y --deadline = límite − 30 s, y CUDA_VISIBLE_DEVICES=0,1 en orden PCI",
          "--data" in argv and argv[argv.index("--data") + 1] == str(r["ns"]["DATA"]) and "--out" in argv
          and abs(float(argv[argv.index("--deadline") + 1]) - (time.time() + r["restante_s"] - r["ns"]["DV2_MARGEN_S"] - 30)) < 120
          and ll.get("cuda") == "0,1" and ll.get("orden") == "PCI_BUS_ID", str(ll)[:240])

    r = con_falso("real_poca_vram", "ok", fake=["0, 3000, 15360", "1, 15109, 15360"])
    ll = json.load(open(r["wd"] / "brazo_dinov2" / "run" / "llamada.json"))
    check("nvidia-smi real: una GPU con menos de 6000 MB libres no se usa (queda solo la 1)", ll.get("cuda") == "1", str(ll.get("cuda")))
    verificar_omitido("nvidia-smi sin ninguna GPU libre", con_falso("real_sin_vram", "ok", fake=["0, 100, 15360"]), "GPU")

    verificar_error("subproceso real: el script sale con código 3", con_falso("real_falla", "falla"), "código 3")
    verificar_omitido("subproceso real: la comprobación previa falla (falta transformers)", con_falso("real_sin_req", "sin_requisitos"), "transformers")
    verificar_excepcion("subproceso real, diagnóstica: la comprobación previa falla", con_falso("real_sin_req_d", "sin_requisitos", fuente=DIAG))

    # Corte por plazo: con 5 s de plazo, un script que se cuelga debe morir, junto con su nieto, y dejar la etapa intacta.
    t0 = time.time()
    r = con_falso("real_cuelga", "cuelga", restante_s=5.0, constantes=rapido)
    dur = time.time() - t0
    nieto_pid = (r["wd"] / "brazo_dinov2" / "run" / "nieto.pid")
    vivo = None
    if nieto_pid.is_file():
        pid = int(nieto_pid.read_text())
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            vivo = True
        except ProcessLookupError:
            vivo = False
        if vivo:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    verificar_error("subproceso real: un script que se cuelga se corta en el plazo", r, "plazo")
    check("subproceso real: el corte llegó a tiempo y mató también al proceso nieto", dur < 40 and vivo is False,
          f"duración={dur:.1f} s nieto_vivo={vivo}")


def pruebas_cola_real():
    """La cola de la celda sin simular nada salvo `nvidia-smi` (un ejecutable falso): el script REAL, sin transformers, debe
    llevar a «omitido» por falta de requisitos y no por T0/TIME_BUDGET."""
    for fuente, nombre in ((CELDA, "cola_real"), (DIAG, "cola_real_diag")):
        wd, uids, data, stage, stage_path, original = preparar_dir(nombre, 9)
        texto = fuente.read_text()
        ini = texto.index("DV2_SCRIPT_SHA256 = ")
        fin = texto.rindex("\n" + CABECERA_COLA)             # el literal del script va entre los dos; no se toca
        def rutas(s):
            return s.replace("/kaggle/working", str(wd)).replace("/kaggle/input", str(wd / "input"))
        src = rutas(texto[:ini]) + texto[ini:fin] + rutas(texto[fin:])
        carpeta = wd / "fakebin"
        carpeta.mkdir()
        (carpeta / "nvidia-smi").write_text("#!/bin/bash\necho '0, 15109, 15360'\necho '1, 15109, 15360'\n")
        (carpeta / "nvidia-smi").chmod(0o755)
        ns = {"__name__": "celda_ns", "DATA": data, "NB_T0": time.time() - 600}
        ns["T0"] = ns["NB_T0"]
        exc = None
        with entorno(PATH=str(carpeta) + ":" + os.environ.get("PATH", "")):
            try:
                exec(compile(src, str(fuente), "exec"), ns)
            except Exception as e:  # noqa: BLE001
                exc = e
        est = ns.get("BRAZO_DINOV2_ESTADO") or json.load(open(wd / "brazo_dinov2" / "estado.json"))
        check(f"{nombre}: el script REAL se omite por falta de requisitos (no por T0/TIME_BUDGET) con la cola sin simulaciones",
              est.get("decision") == "omitido" and "requisito" in est.get("motivo", "") and "T0" not in est.get("motivo", ""),
              est.get("motivo", "")[:220])
        check(f"{nombre}: TIME_BUDGET = 8 h y ROOT = DATA tras la cola", ns.get("TIME_BUDGET") == 8 * 3600.0 and str(ns.get("ROOT")) == str(data))
        check(f"{nombre}: submission.csv byte a byte igual", sha(stage_path) == sha(original))
        if fuente is DIAG:
            check(f"{nombre}: la diagnóstica lanza RuntimeError «DV2_EXIGIR»", isinstance(exc, RuntimeError) and "DV2_EXIGIR" in str(exc), repr(exc)[:150])
        else:
            check(f"{nombre}: la normal no lanza", exc is None, repr(exc))


# =====================================================================================================
# C. Preproceso real del 0.940 sobre N estudios de train
# =====================================================================================================
def preproceso_real(n):
    """cabeceras → ranuras → lateralidad → caché con el código extraído, sobre n estudios de train. Mide segundos."""
    wd = OUT / "preproceso"
    raiz = wd / "root"
    (raiz / "test_series").mkdir(parents=True)
    series = pd.read_csv(COMP / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = sorted(series.StudyInstanceUID.unique())
    muestra = list(pd.Series(uids).sample(n, random_state=2026))
    for u in muestra:
        os.symlink(COMP / "train_series" / u, raiz / "test_series" / u)
    pd.DataFrame({"StudyInstanceUID": muestra}).to_csv(raiz / "test.csv", index=False)
    series[series.StudyInstanceUID.isin(muestra)].to_csv(raiz / "test_series.csv", index=False)

    ns = cargar_script_como_modulo("script_dinov2_preproceso")
    ns["ROOT"] = raiz
    ns["DIAG_DIR"] = str(wd / "diagnostics")
    ns["_RSNA_TEST_IDS"] = list(muestra)
    nb_cfg = {"img": 336, "group": 3, "slices": 12, "crop_mm": 130.0, "band": [0.2, 0.8],
              "slots": [s[0] for s in ns["SLOTS"]], "rules": dict(ns["RULES_NATIVE"])}     # lo que muestra el log del 0.940
    ns["adopt_config_globals"](nb_cfg)
    t = {}
    t0 = time.time()
    test_series = pd.read_csv(raiz / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))
    hte = ns["annotate"](ns["walk"]("test_series"))
    t["cabeceras"] = time.time() - t0
    t0 = time.time()
    slots = ns["pick_slots"](hte, plane_map)
    lat = ns["lat_of"](hte, "test ")
    t["ranuras_y_lateralidad"] = time.time() - t0
    t0 = time.time()
    estudios, cache, mask = ns["build_cache"](slots, plane_map, lat, "preproceso")
    t["cache"] = time.time() - t0
    total = sum(t.values())
    llenas = int(mask.sum())
    kinds = {}
    for e in ns["_RSNA_AUDIT"]["events"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    series_por_estudio = len(hte) / n
    resumen = {"estudios": n, "series": int(len(hte)), "series_por_estudio": round(series_por_estudio, 2),
               "ranuras_llenas": llenas, "ranuras_posibles": int(mask.size), "segundos": {k: round(v, 1) for k, v in t.items()},
               "s_por_estudio": {k: round(v / n, 3) for k, v in t.items()}, "s_por_estudio_total": round(total / n, 3),
               "forma_cache": list(cache.shape), "eventos": kinds, "cpus": os.cpu_count(),
               "afinidad_cpus": len(os.sched_getaffinity(0))}
    (OUT / "preproceso.json").write_text(json.dumps(resumen, indent=1, sort_keys=True))
    print("[preproceso]", json.dumps(resumen, sort_keys=True), flush=True)
    check("preproceso real: la caché tiene la forma (n, 6 ranuras, 12 cortes, 336, 336) uint8",
          tuple(cache.shape) == (n, 6, 12, 336, 336) and cache.dtype == np.uint8, str(cache.shape))
    # Cobertura de ranuras: no es una propiedad que el código garantice (muchos estudios no tienen T1 ni sagital sin
    # supresión de grasa). El ejemplo del 0.940 llenó 12 de 18 (67 %); aquí se exige solo que ningún estudio quede vacío
    # y que haya al menos la mitad, y se informa el valor medido.
    check("preproceso real: ningún estudio queda sin ninguna ranura y se llenó al menos la mitad de las ranuras",
          (mask.sum(axis=1) > 0).all() and kinds.get("empty_study_rows", 0) == 0 and llenas >= 0.5 * mask.size,
          f"{llenas}/{mask.size} ranuras ({llenas / mask.size:.0%}); {kinds}")
    check("preproceso real: los píxeles usan el rango 0-255 (percentiles 1-99 por serie) y no son todo ceros",
          int(cache.max()) == 255 and float(cache[mask.astype(bool)].mean()) > 5, f"max={int(cache.max())}")
    check("preproceso real: ninguna serie falló al decodificar (si no, faltan decodificadores de DICOM)",
          len(ns["DECODE_FAILED"]) == 0, f"{len(ns['DECODE_FAILED'])} series con corte(s) ilegible(s)")


def main():
    pruebas_script()
    pruebas_celda()
    pruebas_subproceso_real()
    pruebas_cola_real()
    if N_PREPROCESO:
        preproceso_real(N_PREPROCESO)
    else:
        print("preproceso real: omitido (sin --preproceso N)", flush=True)

    fallos = [x for x in RES if not x["ok"]]
    sha_archivos = {p.name: sha(p) for p in (CELDA, DIAG, SCRIPT, HERE / "generar_brazo_dinov2.py", HERE / "celda_brazo_dinov2.py", Path(__file__))}
    (OUT / "resultados.json").write_text(json.dumps(
        {"sha256": sha_archivos, "total": len(RES), "fallos": len(fallos), "checks": RES}, indent=1, ensure_ascii=False))
    for nombre, h in sha_archivos.items():
        print(f"sha256 {h}  {nombre}", flush=True)
    print(f"\nRESUMEN: {len(RES) - len(fallos)}/{len(RES)} comprobaciones PASS; {len(fallos)} FAIL", flush=True)
    for x in fallos:
        print("  FAIL:", x["check"], "|", x["detalle"][:240])
    return 1 if fallos else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
