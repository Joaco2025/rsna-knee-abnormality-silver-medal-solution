# %% [Brazo v5] ---------------------------------------------------------------------------------
# Añade el ensamble v5 de Yuca como un brazo más del notebook 0.957, mezclado por rangos sobre
# /kaggle/working/submission.csv. Por defecto usa las familias eff, effhib y r34att (5 particiones
# cada una: 15 CNN ajustadas desde ImageNet) con un peso de 0.10; cnxhib queda fuera porque solapa con
# el ConvNeXt-small (R7) del propio notebook. Ver experiments/reportes/inventario_base_0957.md, §6 y §7.
#
# Dónde va, en referencias/0-957-latest-dataset-score-in-one-click.ipynb: UNA celda nueva al final,
# después de la celda 6 (la que escribe submission.csv). No se toca ninguna otra celda.
# Entradas extra del notebook: el dataset de Owen con el paquete v5 (rsna/, pesos/, wheels/).
#
# Compuerta fail-soft: si falta tiempo, GPU o el paquete, o si algo falla, submission.csv queda
# intacto y el notebook sigue. Coste estimado y cómo medirlo: experiments/reportes/brazo_v5_plan.md.
#
# Lo que usa del notebook: T0 (celda 4: T0 = NB_T0, un time.time()) y DATA (celda 4, carpeta de la
# competencia). Este notebook no tiene TIME_BUDGET ni ROOT: la celda los define justo antes de llamar a
# brazo_v5() (TIME_BUDGET = 8 h en segundos, ROOT = DATA). No necesita _RSNA_TEST_IDS, _RSNA_LABELS,
# rsna_frame, rsna_phase, rsna_event ni rsna_save_predictions (si existen, los usa; si no, sus
# equivalentes propios: test.csv, V5_LABELS y _v5_validar).
#
# Ajustes sin editar la celda, con una variable en una celda anterior:
#   BRAZO_V5_FAMILIAS = ("eff", "effhib", "r34att")   # familias a usar (subconjunto de V5_MIEMBROS)
#   BRAZO_V5_EXIGIR = True                             # variante diagnóstica (celda_brazo_v5_diagnostico.py)
#   BRAZO_V5_SOLO_DEFINIR = True                       # solo define las funciones (ensayo_tiempo_kaggle.py)

import gc, glob, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
import numpy as np
import pandas as pd

# ---- Parámetros ------------------------------------------------------------------------------
V5_LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
             "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Peso del brazo en cada hallazgo: rango final = (1 - W) * rango de submission.csv + W * rango de v5.
# W = 0 deja la columna del notebook tal cual.
V5_W = {c: 0.10 for c in V5_LABELS}
V5_FOLDS_POR_FAMILIA = 5        # 1..5. Recorte fijo: particiones 0..k-1 (no elegirlas por el AUC de los 58)
# Familias de v5 que se mezclan. Por defecto sin cnxhib (solapa con el ConvNeXt-small del notebook 0.957).
V5_FAMILIAS = tuple(globals().get("BRAZO_V5_FAMILIAS", ("eff", "effhib", "r34att")))

# Modelo de tiempo: s/estudio = max(V5_PISO_CPU_S, n_modelos * V5_GPU_MODELO_S / n_gpus).
# Valores pesimistas del plan, §1; sustituirlos por los que mida ensayo_tiempo_kaggle.py.
V5_PISO_CPU_S = 2.2             # lectura de DICOM + preproceso con los 4 vCPU de Kaggle
V5_GPU_MODELO_S = 0.26          # un modelo, un estudio, una T4
V5_FIJOS_S = 300                # pip, carga de pesos en cada proceso, arranque de CUDA y de workers
# TIME_BUDGET (8 h) deja 1 h por debajo del límite de 9 h de Kaggle; esto es holgura extra dentro de él.
V5_MARGEN_S = 600               # reserva dentro de TIME_BUDGET para el cierre del notebook y holgura
# Solo para un envío de diagnóstico: True convierte "omitido" o "error" en excepción, para que el
# rerun falle y se sepa que el brazo no corrió (el log del rerun oculto no se ve). Nunca en un envío normal.
# Se activa poniendo BRAZO_V5_EXIGIR = True en una celda anterior (celda_brazo_v5_diagnostico.py lo hace).
V5_EXIGIR = bool(globals().get("BRAZO_V5_EXIGIR", False))
V5_LOTE = 4
V5_VRAM_MIN_MB = 6000           # una GPU con menos memoria libre no se usa
V5_MAX_ILEGIBLES = 0.05         # más estudios con 0.5 en todo que esto = fallo sistemático: no se mezcla
# Archivo que escribe la celda 6 del notebook 0.957 y que esta celda mezcla en el sitio.
V5_STAGE = "/kaggle/working/submission.csv"
V5_DIR = "/kaggle/working/brazo_v5"

# Copia revisada del código del paquete: code_joako/kaggle/brazo_v5/rsna/ (idéntica a la de Owen).
V5_SHA256 = {
    "inferir.py": "f4c0f4fa5c7baac54cbd73b297fe93f4250d140355d5587cc5143db319c86537",
    "modelo.py": "522c2155248818549a1110fbef50982ba70800661717b05f957cdd5ba1604f6e",
    "preproc.py": "16a776edc43dadbe6cb797b4a9115396e7899641a28cb461a4d70c4b68039a23",
}
# Miembros por familia en orden de partición 0 -> 4 (MANIFIESTO.txt del paquete) y tamaño de sus pesos.
V5_MIEMBROS = {
    "eff": ["eff_job2084365.pt", "eff_job2084585.pt", "eff_job2084586.pt", "eff_job2084626.pt", "eff_job2084584.pt"],
    "cnxhib": ["cnxhib_job2084671.pt", "cnxhib_job2084673.pt", "cnxhib_job2084676.pt", "cnxhib_job2084684.pt",
               "cnxhib_job2084633.pt"],
    "effhib": ["effhib_job2084674.pt", "effhib_job2084683.pt", "effhib_job2084689.pt", "effhib_job2084694.pt",
               "effhib_job2084634.pt"],
    "r34att": ["r34att_job2084756.pt", "r34att_job2084769.pt", "r34att_job2084771.pt", "r34att_job2084783.pt",
               "r34att_job2084722.pt"],
}
V5_BYTES = {"eff": 92755652, "cnxhib": 118054918, "effhib": 92755652, "r34att": 89768328}


class _V5Omitir(Exception):
    """Motivo esperado para no correr el brazo (tiempo, GPU, paquete): no es un error."""


def _v5_buscar(nombre, niveles=6):
    # Solo los primeros niveles: recorrer todo /kaggle/input tardaría mucho con el test completo.
    for n in range(1, niveles + 1):
        hits = glob.glob("/kaggle/input/" + "*/" * n + nombre)
        if hits:
            return hits
    return []


def _v5_restante():
    """Segundos que quedan del presupuesto del notebook (T0 = time.time() de la celda 4; TIME_BUDGET en s)."""
    t0, presupuesto = globals().get("T0"), globals().get("TIME_BUDGET")
    if t0 is None or presupuesto is None:
        return None
    return float(presupuesto) - (time.time() - float(t0))


def _v5_nb(nombre, *args, **kwargs):
    """Llama a una función de registro del notebook (rsna_phase, rsna_event, ...) si existe; nunca falla."""
    f = globals().get(nombre)
    if callable(f):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"[yuca-v5] {nombre} falló: {type(e).__name__}: {e}", flush=True)
    return None


def _v5_paquete():
    """Raíz del dataset que tiene rsna/inferir.py y los pesos de las familias elegidas (se busca por contenido)."""
    for hit in _v5_buscar("rsna/inferir.py"):
        raiz = os.path.dirname(os.path.dirname(hit))
        if all(os.path.isfile(f"{raiz}/pesos/{n}") for fam in V5_FAMILIAS for n in V5_MIEMBROS[fam][:V5_FOLDS_POR_FAMILIA]):
            return raiz
    raise _V5Omitir("no hay ningún dataset con rsna/inferir.py y los pesos de v5 de las familias "
                    f"{list(V5_FAMILIAS)} (¿adjuntado y compartido?)")


def _v5_checkpoints(raiz, k):
    """Los k primeros miembros de cada familia, tras comprobar código y tamaños frente a la copia revisada."""
    for nombre, esperado in V5_SHA256.items():
        with open(f"{raiz}/rsna/{nombre}", "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h != esperado:
            raise RuntimeError(f"rsna/{nombre} del dataset no es la versión revisada ({h[:12]} != {esperado[:12]})")
    ckpts = []
    for fam in V5_FAMILIAS:
        for n in V5_MIEMBROS[fam][:k]:
            ruta = f"{raiz}/pesos/{n}"
            if os.path.getsize(ruta) != V5_BYTES[fam]:
                raise RuntimeError(f"{n}: {os.path.getsize(ruta)} bytes, se esperaban {V5_BYTES[fam]}")
            ckpts.append(ruta)
    return ckpts


def _v5_gpus():
    """GPUs con VRAM libre suficiente, según nvidia-smi (no inicializa CUDA en este kernel)."""
    if "torch" in sys.modules:   # suelta la caché del kernel; los tensores vivos del notebook no se tocan
        gc.collect()
        try:
            sys.modules["torch"].cuda.empty_cache()
        except Exception:
            pass
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return [], f"nvidia-smi falló: {r.stderr.strip()[:200]}"
    gpus, info = [], []
    for linea in r.stdout.strip().splitlines():
        i, libre, total = (int(x) for x in linea.split(","))
        info.append(f"gpu{i} {libre}/{total} MB libres")
        if libre >= V5_VRAM_MIN_MB:
            gpus.append(i)
    return gpus, "; ".join(info)


def _v5_librerias(destino):
    """Decodificadores de DICOM del paquete, instalados en una carpeta propia sin tocar el entorno del notebook."""
    ruedas = sorted({d for n in range(1, 6) for d in glob.glob("/kaggle/input/" + "*/" * n + "wheels")})
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps", "--target", destino]
    for d in ruedas:
        cmd += ["--find-links", d]
    r = subprocess.run(cmd + ["pydicom==3.0.2", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
                       capture_output=True, text=True, timeout=600)
    return r.returncode == 0, r.stderr.strip()[-300:]


def _v5_cola(ruta, n=15):
    try:
        with open(ruta, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(sin log)"


def _v5_fragmentos(data, split, uids, k, destino):
    """Una carpeta por GPU con su parte de <split>.csv y <split>_series.csv y un enlace a las imágenes.

    Se reparte por estudios, intercalados, para que el preproceso (CPU) se haga una sola vez por estudio.
    """
    estudios = pd.read_csv(f"{data}/{split}.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(f"{data}/{split}_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    dirs = []
    for g in range(k):
        parte = set(uids[g::k])
        d = f"{destino}/fragmento{g}"
        os.makedirs(d, exist_ok=True)
        estudios[estudios.StudyInstanceUID.isin(parte)].to_csv(f"{d}/{split}.csv", index=False)
        series[series.StudyInstanceUID.isin(parte)].to_csv(f"{d}/{split}_series.csv", index=False)
        enlace = f"{d}/{split}_series"
        if os.path.lexists(enlace):
            os.remove(enlace)
        os.symlink(f"{data}/{split}_series", enlace)
        dirs.append(d)
    return dirs


def _v5_correr(raiz, data, split, uids, ckpts, gpus, limite, destino):
    """inferir.py del paquete, un proceso por GPU con su parte de los estudios, cortado en `limite` (time.time()).

    Devuelve las predicciones de todos los estudios y, por fragmento, el tiempo que informa inferir.py.
    """
    libs = f"{V5_DIR}/pylibs"
    frags = _v5_fragmentos(data, split, uids, len(gpus), destino)
    workers = max(1, (os.cpu_count() or 2) // len(gpus))
    rutas_py = [p for p in (libs, os.environ.get("PYTHONPATH", "")) if p]
    procs = []
    try:
        for g, d in zip(gpus, frags):
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g),
                       OMP_NUM_THREADS="1", PYTHONPATH=os.pathsep.join(rutas_py))
            cmd = [sys.executable, f"{raiz}/rsna/inferir.py", "--data", d, "--split", split, "--checkpoints", *ckpts,
                   "--out", f"{d}/sub.csv", "--batch", str(V5_LOTE), "--workers", str(workers)]
            log = open(f"{d}/inferir.log", "w")
            # Sesión propia: al cortar se mata el grupo entero, workers del DataLoader incluidos.
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                           start_new_session=True), log, d))
        while any(p.poll() is None for p, _, _ in procs):
            caidos = [d for p, _, d in procs if p.poll() not in (None, 0)]
            if caidos:
                raise RuntimeError(f"inferir.py falló en {caidos[0]}:\n{_v5_cola(caidos[0] + '/inferir.log')}")
            if time.time() > limite:
                raise TimeoutError("se agotó el plazo del brazo; se corta para no pasar del presupuesto del notebook")
            time.sleep(5)
        for p, _, d in procs:
            if p.returncode != 0:
                raise RuntimeError(f"inferir.py falló en {d}:\n{_v5_cola(d + '/inferir.log')}")
    finally:
        for p, log, _ in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
            log.close()
    partes = [pd.read_csv(f"{d}/sub.csv", dtype={"StudyInstanceUID": str}) for _, _, d in procs]
    return pd.concat(partes, ignore_index=True), frags


def _v5_validar(df, uids, columnas, nombre, rango01=False):
    """Los UIDs de test.csv (el número real, sea cual sea), sin duplicados, columnas en el orden dado y valores finitos."""
    if list(df.columns) != list(columnas):
        raise ValueError(f"{nombre}: columnas {list(df.columns)} en vez de {list(columnas)}")
    ids = df[columnas[0]]
    if len(df) != len(uids) or ids.duplicated().any() or set(ids) != set(uids):
        raise ValueError(f"{nombre}: {len(df)} filas y {ids.nunique()} UIDs únicos; se esperaban los {len(uids)} de test")
    v = df[list(columnas[1:])].to_numpy(dtype=float)
    if not np.isfinite(v).all():
        raise ValueError(f"{nombre}: {int((~np.isfinite(v)).sum())} valores no finitos")
    if rango01 and ((v < 0) | (v > 1)).any():
        raise ValueError(f"{nombre}: valores fuera de [0, 1]")


def _v5_datos():
    """Carpeta de la competencia: ROOT (la celda lo iguala a DATA, de la celda 4 del notebook) o, en una sesión de ensayo, por búsqueda."""
    root = globals().get("ROOT")
    if root is not None and os.path.isfile(os.path.join(str(root), "test.csv")):
        return str(root)
    hits = _v5_buscar("test_series.csv")
    if not hits:
        raise _V5Omitir("no se encuentran los datos de la competencia")
    return os.path.dirname(hits[0])


def brazo_v5():
    estado = {"brazo": "v5", "decision": "omitido", "motivo": "", "inicio": time.strftime("%H:%M:%S")}
    tmp = V5_STAGE + ".tmp_v5"
    t_celda = time.time()
    _v5_nb("rsna_phase", "yuca_v5", "START")
    try:
        os.makedirs(V5_DIR, exist_ok=True)

        # 1. Datos y etapa del notebook. Si la etapa no es válida, no es cosa de este brazo: no se toca.
        data = _v5_datos()
        en_test = pd.read_csv(f"{data}/test.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
        uids = [str(u) for u in (globals().get("_RSNA_TEST_IDS") or en_test)]
        if set(uids) != set(en_test) or len(uids) != len(en_test):
            raise _V5Omitir("_RSNA_TEST_IDS del notebook no coincide con test.csv")
        if not uids:
            raise _V5Omitir("test.csv no tiene estudios")
        labels = list(globals().get("_RSNA_LABELS") or V5_LABELS)
        if set(labels) != set(V5_LABELS) or len(labels) != 12:
            raise _V5Omitir(f"los hallazgos del notebook no son los 12 de v5: {labels}")
        columnas = ["StudyInstanceUID", *labels]
        if not os.path.isfile(V5_STAGE):
            raise _V5Omitir(f"no existe {V5_STAGE}")
        stage = pd.read_csv(V5_STAGE, dtype={"StudyInstanceUID": str})
        ref = f"{data}/sample_submission.csv"
        if os.path.isfile(ref) and list(pd.read_csv(ref, nrows=0).columns) != columnas:
            raise _V5Omitir("el orden de hallazgos del notebook no coincide con sample_submission.csv")
        try:
            _v5_validar(stage, uids, columnas, "etapa del notebook", rango01=True)
            if callable(globals().get("rsna_frame")):
                rsna_frame(stage, uids, labels, "yuca_v5 input stage")
        except Exception as e:
            raise _V5Omitir(f"la etapa del notebook no pasa la validación ({e}); se deja como está")
        if set(V5_W) != set(V5_LABELS) or not all(0.0 <= float(w) <= 1.0 for w in V5_W.values()):
            raise ValueError("V5_W debe tener los 12 hallazgos con pesos en [0, 1]")
        if not any(float(V5_W[c]) > 0 for c in labels):
            raise _V5Omitir("todos los pesos V5_W son 0")
        if not V5_FAMILIAS or len(set(V5_FAMILIAS)) != len(V5_FAMILIAS) or not set(V5_FAMILIAS) <= set(V5_MIEMBROS):
            raise ValueError(f"V5_FAMILIAS debe ser un subconjunto sin repetidos de {list(V5_MIEMBROS)}: {V5_FAMILIAS}")
        if not 1 <= V5_FOLDS_POR_FAMILIA <= 5:
            raise ValueError(f"V5_FOLDS_POR_FAMILIA debe estar entre 1 y 5: {V5_FOLDS_POR_FAMILIA}")

        # 2. Guarda de tiempo.
        restante = _v5_restante()
        if restante is None:
            raise _V5Omitir("no se pueden leer T0/TIME_BUDGET: sin ellos no se puede garantizar el tiempo")
        raiz = _v5_paquete()
        ckpts = _v5_checkpoints(raiz, V5_FOLDS_POR_FAMILIA)
        gpus, info_gpu = _v5_gpus()
        estado.update(restante_min=round(restante / 60, 1), gpus=info_gpu, modelos=len(ckpts), estudios=len(uids))
        if not gpus:
            raise _V5Omitir(f"no hay GPU con {V5_VRAM_MIN_MB} MB libres ({info_gpu}); en CPU no cabe en el tiempo")
        s_estudio = max(V5_PISO_CPU_S, len(ckpts) * V5_GPU_MODELO_S / len(gpus))
        necesario = V5_FIJOS_S + len(uids) * s_estudio
        estado.update(estimado_min=round(necesario / 60, 1))
        if restante < necesario + V5_MARGEN_S:
            raise _V5Omitir(f"quedan {restante / 60:.0f} min y v5 necesita ~{necesario / 60:.0f} "
                            f"+ {V5_MARGEN_S / 60:.0f} de margen")
        limite = t_celda + restante - V5_MARGEN_S

        # 3. Inferencia en las GPU disponibles.
        ok, err = _v5_librerias(f"{V5_DIR}/pylibs")
        if not ok:
            print("[yuca-v5] pip no instaló los decodificadores; se usa lo que tenga el entorno:", err, flush=True)
        t_inf = time.time()
        v5, _ = _v5_correr(raiz, data, "test", uids, ckpts, gpus, limite, f"{V5_DIR}/test")
        estado.update(inferencia_min=round((time.time() - t_inf) / 60, 1))
        v5 = v5.set_index("StudyInstanceUID").reindex(uids).rename_axis("StudyInstanceUID").reset_index()
        _v5_validar(v5, uids, ["StudyInstanceUID"] + V5_LABELS, "predicciones v5", rango01=True)
        v5.to_csv(f"{V5_DIR}/v5_sub.csv", index=False)
        _v5_nb("rsna_save_predictions", "yuca_v5_prob", uids, v5[labels].to_numpy(np.float64), labels)

        # 4. Mezcla por rangos. Un estudio que v5 no pudo leer (0.5 en todo) no vota: queda el rango del notebook.
        v5 = v5.set_index("StudyInstanceUID").reindex(stage["StudyInstanceUID"])
        ilegible = (v5[labels] == 0.5).all(axis=1).to_numpy()
        estado.update(ilegibles=int(ilegible.sum()))
        if ilegible.mean() > V5_MAX_ILEGIBLES:
            raise RuntimeError(f"{int(ilegible.sum())} estudios ilegibles para v5 (> {V5_MAX_ILEGIBLES:.0%}): "
                               "fallo sistemático, ¿decodificadores de DICOM?")
        v5.loc[ilegible, labels] = np.nan
        r_nb = stage[labels].rank(method="average", pct=True).to_numpy()
        r_v5 = v5[labels].rank(method="average", pct=True).fillna(0.0).to_numpy()
        w = np.array([float(V5_W[c]) for c in labels])[None, :] * (~ilegible)[:, None]
        mezcla = (1.0 - w) * r_nb + w * r_v5
        nuevo = stage.copy()
        for j, c in enumerate(labels):
            if float(V5_W[c]) > 0:
                # Se vuelve a rankear, como la mezcla de la celda 6: submission.csv queda en (0, 1].
                nuevo[c] = pd.Series(mezcla[:, j]).rank(method="average", pct=True).to_numpy()
        rho = [pd.Series(r_nb[~ilegible, j]).corr(pd.Series(r_v5[~ilegible, j])) for j in range(len(labels))]
        estado.update(spearman_medio=round(float(np.nanmean(rho)), 3))

        # 5. Escritura atómica: la copia temporal pasa la misma validación que hace la celda 6 del notebook
        #    (UIDs en el orden de test.csv, columnas de sample_submission, finitos y en [0, 1]) y solo
        #    entonces sustituye a submission.csv.
        nuevo.to_csv(tmp, index=False)
        releido = pd.read_csv(tmp, dtype={"StudyInstanceUID": str})
        _v5_validar(releido, uids, columnas, "etapa con v5", rango01=True)
        if callable(globals().get("rsna_frame")):
            rsna_frame(releido, uids, labels, "yuca_v5 output stage")
        shutil.copy2(V5_STAGE, f"{V5_DIR}/submission_antes_v5.csv")
        os.replace(tmp, V5_STAGE)
        estado.update(decision="aplicado", motivo=f"{len(ckpts)} modelos, W={dict(V5_W)}")
    except _V5Omitir as e:
        estado.update(decision="omitido", motivo=str(e))
    except Exception as e:
        estado.update(decision="error", motivo=f"{type(e).__name__}: {e}"[:1500], traza=traceback.format_exc()[-3000:])
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        estado["duracion_min"] = round((time.time() - t_celda) / 60, 1)
        resumen = {k: v for k, v in estado.items() if k not in ("decision", "motivo", "traza")}
        # Registro: evento (si el notebook tiene rsna_event) + línea "(flagged)" + fase COMPLETE con ok.
        if estado["decision"] == "aplicado":
            _v5_nb("rsna_event", "yuca_v5_applied", **resumen)
            print(f"[yuca-v5] applied: {estado['motivo']}", flush=True)
        elif estado["decision"] == "omitido":
            _v5_nb("rsna_event", "yuca_v5_skipped", reason=estado["motivo"], **resumen)
            print(f"[yuca-v5] skipped ({estado['motivo']}); continuing without it (flagged)", flush=True)
        else:
            _v5_nb("rsna_event", "yuca_v5_failed", error=estado["motivo"], **resumen)
            print(f"[yuca-v5] FAILED ({estado['motivo'][:300]}); continuing without it (flagged)", flush=True)
        print("[yuca-v5]", resumen, flush=True)
        _v5_nb("rsna_phase", "yuca_v5", "COMPLETE", ok=estado["decision"] == "aplicado", decision=estado["decision"])
        try:
            os.makedirs(V5_DIR, exist_ok=True)
            with open(f"{V5_DIR}/estado.json", "w") as f:
                json.dump(estado, f, indent=1, ensure_ascii=False)
        except Exception:
            pass
    if V5_EXIGIR and estado["decision"] != "aplicado":
        raise RuntimeError(f"V5_EXIGIR: el brazo v5 no se aplicó ({estado['decision']}: {estado['motivo'][:300]})")
    return estado


if not globals().get("BRAZO_V5_SOLO_DEFINIR"):
    # Notebook 0.957: no define TIME_BUDGET ni ROOT, y sin ellos brazo_v5() se omitiría sin avisar.
    # T0 sí existe (celda 4: T0 = NB_T0); si no, se usa NB_T0 (celda 3). DATA es la carpeta de la competencia (celda 4).
    TIME_BUDGET = 8.0 * 3600            # 1 h por debajo del límite de 9 h de Kaggle (DEADLINE_HOURS del notebook es 8.3)
    ROOT = globals().get("DATA", globals().get("ROOT"))
    if globals().get("T0") is None and globals().get("NB_T0") is not None:
        T0 = NB_T0
    BRAZO_V5_ESTADO = brazo_v5()
