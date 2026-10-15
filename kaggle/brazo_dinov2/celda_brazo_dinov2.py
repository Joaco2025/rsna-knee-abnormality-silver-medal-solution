# %% [Brazo DINOv2] ------------------------------------------------------------------------------
# Añade al notebook 0.957 la familia DINOv2-S x20 del notebook 0.940 (Bend the Knee to Speedy Raptors) como un
# brazo más, mezclado por rangos sobre /kaggle/working/submission.csv con un peso de 0.15 en los 12 hallazgos.
#
# Cómo funciona: la celda lleva incrustado script_dinov2.py (texto de las celdas 3-14 del 0.940 copiado por nodos,
# sin reescribir su lógica; ver kaggle/brazo_dinov2/generar_brazo_dinov2.py). Lo escribe a /kaggle/working/brazo_dinov2
# tras comprobar su sha256 y lo lanza como SUBPROCESO, porque sus globales (T0, TIME_BUDGET, IMG, GROUP, SLOTS, ...)
# chocan con los del 0.957. El subproceso escribe dinov2_sub.csv (rangos percentiles) e informe.json.
#
# Dónde va: UNA celda nueva al final, después de la celda que escribe submission.csv. No se toca ninguna otra.
# Datasets que hay que adjuntar además de los del 0.957 (nombres exactos salen del código del 0.940, celdas 4 y 13):
#   - Dataset  pilkwang/rsna-knee-weights   -> /kaggle/input/datasets/pilkwang/rsna-knee-weights (manifest.json y los 20 .pt)
#   - Modelo   metaresearch/dinov2, framework PyTorch, variación small, versión 1
#              -> /kaggle/input/models/metaresearch/dinov2/PyTorch/small/1 (config.json y pytorch_model.bin)
#   Ver experiments/reportes/brazo_dinov2_plan.md, §2.
#
# Compuerta fail-soft: si falta tiempo, GPU, un dataset, o si algo falla, submission.csv queda intacto y el notebook
# sigue. La celda diagnóstica (celda_para_pegar_0957_diag.py) convierte "omitido" o "error" en una excepción.
#
# Lo que usa del notebook: T0 (celda 4: T0 = NB_T0, un time.time()) y DATA (celda 4, carpeta de la competencia).
# Este notebook no tiene TIME_BUDGET ni ROOT: la celda los define justo antes de llamar a brazo_dinov2().
#
# Ajustes sin editar la celda, con una variable en una celda anterior:
#   BRAZO_DINOV2_EXIGIR = True         # variante diagnóstica
#   BRAZO_DINOV2_SOLO_DEFINIR = True   # solo define las funciones (ensayos)

import glob, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
import numpy as np
import pandas as pd

# ---- Parámetros ------------------------------------------------------------------------------
DV2_LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
              "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Peso del brazo en cada hallazgo: rango final = (1 - W) * rango de submission.csv + W * rango de DINOv2.
# W = 0 deja la columna del notebook tal cual.
DV2_W = {c: 0.15 for c in DV2_LABELS}

# Modelo de tiempo (ver brazo_dinov2_plan.md, §3). A diferencia del brazo v5, aquí el preproceso en CPU de TODOS los
# estudios ocurre antes de la inferencia y no se solapa con ella, así que los dos términos SUMAN:
#   segundos = FIJOS + n * (CPU_ESTUDIO + GPU_ESTUDIO / n_gpus)
DV2_FIJOS_S = 300               # importar transformers, 20 cargas de pesos con huella, cabeceras, reserva de la caché
DV2_CPU_ESTUDIO_S = 1.0         # cabeceras + 6 series x 12 cortes DICOM, recorte, remuestreo a 336. Medido 0.25 s en Yuca (4 CPU)
DV2_GPU_ESTUDIO_S = 3.2         # una T4: 20 miembros x 60 ventanas (6 ranuras x 10), con el prefijo de 6 bloques compartido
DV2_MARGEN_S = 600              # reserva dentro de TIME_BUDGET para el cierre del notebook y holgura
DV2_VRAM_MIN_MB = 6000          # una GPU con menos memoria libre no se usa
DV2_MAX_VACIOS = 0.05           # más estudios sin ninguna serie válida que esto = fallo sistemático: no se mezcla
DV2_COMPROBAR_S = 600           # plazo de la comprobación previa de modelos y librerías
# Solo para un envío de diagnóstico: True convierte "omitido" o "error" en excepción, para que el rerun falle y se
# sepa que el brazo no corrió (el log del rerun oculto no se ve). Nunca en un envío normal.
DV2_EXIGIR = bool(globals().get("BRAZO_DINOV2_EXIGIR", False))
# Archivo que escribe la celda 6 del notebook 0.957 y que esta celda mezcla en el sitio.
DV2_STAGE = "/kaggle/working/submission.csv"
DV2_DIR = "/kaggle/working/brazo_dinov2"


class _DV2Omitir(Exception):
    """Motivo esperado para no correr el brazo (tiempo, GPU, datasets): no es un error."""


def _dv2_buscar(nombre, niveles=6):
    # Solo los primeros niveles: recorrer todo /kaggle/input tardaría mucho con el test completo.
    for n in range(1, niveles + 1):
        hits = glob.glob("/kaggle/input/" + "*/" * n + nombre)
        if hits:
            return hits
    return []


def _dv2_restante():
    """Segundos que quedan del presupuesto del notebook (T0 = time.time() de la celda 4; TIME_BUDGET en s)."""
    t0, presupuesto = globals().get("T0"), globals().get("TIME_BUDGET")
    if t0 is None or presupuesto is None:
        return None
    return float(presupuesto) - (time.time() - float(t0))


def _dv2_datos():
    """Carpeta de la competencia: ROOT (la celda lo iguala a DATA, de la celda 4 del notebook) o, en una sesión de ensayo, por búsqueda."""
    root = globals().get("ROOT")
    if root is not None and os.path.isfile(os.path.join(str(root), "test.csv")):
        return str(root)
    hits = _dv2_buscar("test_series.csv")
    if not hits:
        raise _DV2Omitir("no se encuentran los datos de la competencia")
    return os.path.dirname(hits[0])


def _dv2_gpus():
    """GPUs con VRAM libre suficiente, según nvidia-smi (no inicializa CUDA en este kernel). Devuelve (índices, texto)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
    except OSError as e:
        return [], f"nvidia-smi no disponible: {e}"
    if r.returncode != 0:
        return [], f"nvidia-smi falló: {r.stderr.strip()[:200]}"
    gpus, info = [], []
    for linea in r.stdout.strip().splitlines():
        i, libre, total = (int(x) for x in linea.split(","))
        info.append(f"gpu{i} {libre}/{total} MB libres")
        if libre >= DV2_VRAM_MIN_MB:
            gpus.append(i)
    return gpus, "; ".join(info)


def _dv2_cola(ruta, n=15):
    try:
        with open(ruta, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(sin log)"


def _dv2_validar(df, uids, columnas, nombre, rango01=False):
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


def _dv2_script():
    """Escribe el script incrustado, tras comprobar que el texto pegado es el que se generó, y devuelve su ruta."""
    if hashlib.sha256(DV2_SCRIPT_TEXTO.encode("utf-8")).hexdigest() != DV2_SCRIPT_SHA256:
        raise RuntimeError("el script DINOv2 incrustado no coincide con su sha256: la celda se pegó incompleta o alterada")
    os.makedirs(DV2_DIR, exist_ok=True)
    ruta = f"{DV2_DIR}/script_dinov2.py"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(DV2_SCRIPT_TEXTO)
    return ruta


def _dv2_subproceso(cmd, env_extra, limite, log):
    """Un proceso en sesión propia (al cortar se mata el grupo entero, workers incluidos), cortado en `limite` (time.time())."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", **env_extra)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while p.poll() is None:
                if time.time() > limite:
                    raise TimeoutError("se agotó el plazo del brazo; se corta para no pasar del presupuesto del notebook")
                time.sleep(2)
        finally:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
    return p.returncode


def _dv2_comprobar(script, data):
    """Localiza los pesos, el modelo base y las librerías con el propio script (--comprobar). Falta algo = omitir."""
    salida = f"{DV2_DIR}/comprobar"
    rc = _dv2_subproceso([sys.executable, script, "--comprobar", "--out", salida], {}, time.time() + DV2_COMPROBAR_S,
                         f"{DV2_DIR}/comprobar.log")
    if rc != 0:
        raise _DV2Omitir("falta algún requisito del brazo (¿datasets pilkwang/rsna-knee-weights y metaresearch/dinov2 "
                         f"adjuntados, transformers instalado?): {_dv2_cola(DV2_DIR + '/comprobar.log', 3).strip()[-300:]}")
    with open(f"{salida}/comprobacion.json") as f:
        return json.load(f)


def _dv2_correr(script, data, gpus, limite, destino):
    """script_dinov2.py en un subproceso con las GPU elegidas. Devuelve (dinov2_sub.csv leído, informe.json)."""
    for viejo in ("dinov2_sub.csv", "informe.json"):
        if os.path.exists(f"{destino}/{viejo}"):
            os.remove(f"{destino}/{viejo}")
    os.makedirs(destino, exist_ok=True)
    log = f"{DV2_DIR}/script_dinov2.log"
    rc = _dv2_subproceso([sys.executable, script, "--data", str(data), "--out", destino, "--deadline", str(limite - 30)],
                         {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpus), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                         limite, log)
    if rc != 0:
        raise RuntimeError(f"script_dinov2.py terminó con código {rc}:\n{_dv2_cola(log)}")
    sub = pd.read_csv(f"{destino}/dinov2_sub.csv", dtype={"StudyInstanceUID": str})
    with open(f"{destino}/informe.json") as f:
        return sub, json.load(f)


def brazo_dinov2():
    estado = {"brazo": "dinov2", "decision": "omitido", "motivo": "", "inicio": time.strftime("%H:%M:%S")}
    tmp = DV2_STAGE + ".tmp_dinov2"
    t_celda = time.time()
    try:
        os.makedirs(DV2_DIR, exist_ok=True)

        # 1. Datos y etapa del notebook. Si la etapa no es válida, no es cosa de este brazo: no se toca.
        data = _dv2_datos()
        uids = pd.read_csv(f"{data}/test.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
        if not uids:
            raise _DV2Omitir("test.csv no tiene estudios")
        labels = list(DV2_LABELS)
        columnas = ["StudyInstanceUID", *labels]
        if not os.path.isfile(DV2_STAGE):
            raise _DV2Omitir(f"no existe {DV2_STAGE}")
        stage = pd.read_csv(DV2_STAGE, dtype={"StudyInstanceUID": str})
        ref = f"{data}/sample_submission.csv"
        if os.path.isfile(ref) and list(pd.read_csv(ref, nrows=0).columns) != columnas:
            raise _DV2Omitir("el orden de hallazgos del notebook no coincide con sample_submission.csv")
        try:
            _dv2_validar(stage, uids, columnas, "etapa del notebook", rango01=True)
        except Exception as e:
            raise _DV2Omitir(f"la etapa del notebook no pasa la validación ({e}); se deja como está")
        if set(DV2_W) != set(DV2_LABELS) or not all(0.0 <= float(w) <= 1.0 for w in DV2_W.values()):
            raise ValueError("DV2_W debe tener los 12 hallazgos con pesos en [0, 1]")
        if not any(float(DV2_W[c]) > 0 for c in labels):
            raise _DV2Omitir("todos los pesos DV2_W son 0")

        # 2. Guarda de tiempo.
        restante = _dv2_restante()
        if restante is None:
            raise _DV2Omitir("no se pueden leer T0/TIME_BUDGET: sin ellos no se puede garantizar el tiempo")
        script = _dv2_script()
        gpus, info_gpu = _dv2_gpus()
        estado.update(restante_min=round(restante / 60, 1), gpus=info_gpu, estudios=len(uids))
        if not gpus:
            raise _DV2Omitir(f"no hay GPU con {DV2_VRAM_MIN_MB} MB libres ({info_gpu}); en CPU no cabe en el tiempo")
        necesario = DV2_FIJOS_S + len(uids) * (DV2_CPU_ESTUDIO_S + DV2_GPU_ESTUDIO_S / len(gpus))
        estado.update(estimado_min=round(necesario / 60, 1))
        if restante < necesario + DV2_MARGEN_S:
            raise _DV2Omitir(f"quedan {restante / 60:.0f} min y DINOv2 necesita ~{necesario / 60:.0f} "
                             f"+ {DV2_MARGEN_S / 60:.0f} de margen")
        limite = t_celda + restante - DV2_MARGEN_S
        estado.update(requisitos=_dv2_comprobar(script, data))

        # 3. Inferencia en un subproceso con las GPU disponibles.
        t_inf = time.time()
        dv2, informe = _dv2_correr(script, data, gpus, limite, f"{DV2_DIR}/run")
        estado.update(inferencia_min=round((time.time() - t_inf) / 60, 1), miembros=informe.get("miembros"))
        _dv2_validar(dv2, uids, columnas, "predicciones DINOv2", rango01=True)
        if informe.get("miembros") != 20:
            raise RuntimeError(f"el script informa {informe.get('miembros')} miembros en vez de 20")

        # 4. Mezcla por rangos. Un estudio sin ninguna serie utilizable (el script lo registra) no vota: queda el
        #    rango del notebook.
        dv2 = dv2.set_index("StudyInstanceUID").reindex(stage["StudyInstanceUID"])
        vacios = set(map(str, informe.get("estudios_sin_series", [])))
        vacio = stage["StudyInstanceUID"].isin(vacios).to_numpy()
        estado.update(sin_series=int(vacio.sum()))
        if vacio.mean() > DV2_MAX_VACIOS:
            raise RuntimeError(f"{int(vacio.sum())} estudios sin series utilizables para DINOv2 (> {DV2_MAX_VACIOS:.0%}): "
                               "fallo sistemático, ¿lectura de DICOM?")
        dv2.loc[vacio, labels] = np.nan
        r_nb = stage[labels].rank(method="average", pct=True).to_numpy()
        r_dv2 = dv2[labels].rank(method="average", pct=True).fillna(0.0).to_numpy()
        w = np.array([float(DV2_W[c]) for c in labels])[None, :] * (~vacio)[:, None]
        mezcla = (1.0 - w) * r_nb + w * r_dv2
        nuevo = stage.copy()
        for j, c in enumerate(labels):
            if float(DV2_W[c]) > 0:
                # Se vuelve a rankear, como la mezcla de la celda 6: submission.csv queda en (0, 1].
                nuevo[c] = pd.Series(mezcla[:, j]).rank(method="average", pct=True).to_numpy()
        rho = [pd.Series(r_nb[~vacio, j]).corr(pd.Series(r_dv2[~vacio, j])) for j in range(len(labels))]
        rho_medio = float(np.nanmean(rho))
        estado.update(spearman_medio=round(rho_medio, 3))
        if len(uids) >= 100 and not rho_medio > 0:
            raise RuntimeError(f"correlación de rangos con el notebook ≤ 0 ({rho_medio:.3f}): predicciones sin sentido, no se mezclan")

        # 5. Escritura atómica: la copia temporal pasa la misma validación que hace la celda 6 del notebook
        #    (UIDs en el orden de test.csv, columnas de sample_submission, finitos y en [0, 1]) y solo
        #    entonces sustituye a submission.csv.
        nuevo.to_csv(tmp, index=False)
        releido = pd.read_csv(tmp, dtype={"StudyInstanceUID": str})
        _dv2_validar(releido, uids, columnas, "etapa con DINOv2", rango01=True)
        shutil.copy2(DV2_STAGE, f"{DV2_DIR}/submission_antes_dinov2.csv")
        os.replace(tmp, DV2_STAGE)
        estado.update(decision="aplicado", motivo=f"20 miembros, W={dict(DV2_W)}")
    except _DV2Omitir as e:
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
        if estado["decision"] == "aplicado":
            print(f"[dinov2-brazo] applied: {estado['motivo']}", flush=True)
        elif estado["decision"] == "omitido":
            print(f"[dinov2-brazo] skipped ({estado['motivo']}); continuing without it (flagged)", flush=True)
        else:
            print(f"[dinov2-brazo] FAILED ({estado['motivo'][:300]}); continuing without it (flagged)", flush=True)
        print("[dinov2-brazo]", resumen, flush=True)
        try:
            os.makedirs(DV2_DIR, exist_ok=True)
            with open(f"{DV2_DIR}/estado.json", "w") as f:
                json.dump(estado, f, indent=1, ensure_ascii=False)
        except Exception:
            pass
    if DV2_EXIGIR and estado["decision"] != "aplicado":
        raise RuntimeError(f"DV2_EXIGIR: el brazo DINOv2 no se aplicó ({estado['decision']}: {estado['motivo'][:300]})")
    return estado


# ---- script_dinov2.py incrustado (generado; no editar a mano) --------------------------------------
# @@SCRIPT_DINOV2@@


if not globals().get("BRAZO_DINOV2_SOLO_DEFINIR"):
    # Notebook 0.957: no define TIME_BUDGET ni ROOT, y sin ellos brazo_dinov2() se omitiría sin avisar.
    # T0 sí existe (celda 4: T0 = NB_T0); si no, se usa NB_T0 (celda 3). DATA es la carpeta de la competencia (celda 4).
    TIME_BUDGET = 8.0 * 3600            # 1 h por debajo del límite de 9 h de Kaggle (DEADLINE_HOURS del notebook es 8.3)
    ROOT = globals().get("DATA", globals().get("ROOT"))
    if globals().get("T0") is None and globals().get("NB_T0") is not None:
        T0 = NB_T0
    BRAZO_DINOV2_ESTADO = brazo_dinov2()
