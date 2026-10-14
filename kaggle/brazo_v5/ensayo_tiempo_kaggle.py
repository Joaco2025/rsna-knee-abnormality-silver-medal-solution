# %% [Ensayo de tiempo del brazo v5] ------------------------------------------------------------
# Para una sesión interactiva de Kaggle (GPU T4 x2, sin internet), no para el envío.
# Mide en estudios de TRAIN, con el mismo hardware del rerun, los dos números del modelo de tiempo
# de celda_brazo_v5.py, y no gasta envíos:
#   pasada A: 1 modelo (r34att, partición 0)  -> piso de CPU (lectura DICOM + preproceso), s/estudio
#   pasada B: los 20 modelos                   -> s/estudio completo; de ahí, coste de GPU por modelo
# Cada pasada usa estudios distintos, para que la caché de disco de A no abarate B.
#
# Orden de celdas en la sesión:
#   1) BRAZO_V5_SOLO_DEFINIR = True
#   2) el contenido de celda_brazo_v5.py (solo define funciones)
#   3) esta celda
# Coste: unos 15-25 min de la cuota semanal de GPU (ver brazo_v5_plan.md, §3).

import re

N_PISO, N_COMPLETO, SEMILLA = 120, 200, 2026


def _v5_tiempo_log(frags):
    """Mayor 'N estudios en X s' de los fragmentos: es el tiempo de pared de la pasada sin la carga de pesos."""
    peor = 0.0
    for d in frags:
        m = re.search(r"(\d+) estudios en (\d+) s", _v5_cola(f"{d}/inferir.log", 5))
        if not m:
            raise RuntimeError(f"sin línea de tiempo en {d}/inferir.log")
        peor = max(peor, float(m.group(2)))
    return peor


data = os.path.dirname(_v5_buscar("train_series.csv")[0])
raiz = _v5_paquete()
todos = _v5_checkpoints(raiz, 5)
gpus, info_gpu = _v5_gpus()
print("paquete:", raiz, "| datos:", data, "| GPU:", info_gpu, "| vCPU:", os.cpu_count())
assert gpus, "sin GPU utilizable: el brazo no cabría"
ok, err = _v5_librerias(f"{V5_DIR}/pylibs")
print("decodificadores instalados:", ok, err)

muestra = (pd.read_csv(f"{data}/train.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID
           .sample(N_PISO + N_COMPLETO, random_state=SEMILLA).tolist())
pasadas = {"A_piso": (muestra[:N_PISO], [f"{raiz}/pesos/{V5_MIEMBROS['r34att'][0]}"]),
           "B_20": (muestra[N_PISO:], todos)}
res = {}
for nombre, (uids_p, ckpts_p) in pasadas.items():
    t = time.time()
    pred, frags = _v5_correr(raiz, data, "train", uids_p, ckpts_p, gpus, t + 3600, f"{V5_DIR}/ensayo_{nombre}")
    total = time.time() - t
    pred = pred.set_index("StudyInstanceUID").reindex(uids_p).rename_axis("StudyInstanceUID").reset_index()
    _v5_validar(pred, uids_p, ["StudyInstanceUID"] + V5_LABELS, nombre, rango01=True)
    bucle = _v5_tiempo_log(frags)
    res[nombre] = {"s_estudio": bucle / len(uids_p), "fijos_s": total - bucle,
                   "ilegibles": int((pred[V5_LABELS] == 0.5).all(axis=1).sum())}
    print(nombre, {k: round(v, 2) for k, v in res[nombre].items()})

piso = res["A_piso"]["s_estudio"]
completo = res["B_20"]["s_estudio"]
fijos = res["B_20"]["fijos_s"]
gpu_bound = completo > 1.15 * piso
# Si B está limitada por GPU, su tiempo por estudio es 20 modelos repartidos entre las GPU.
g = completo * len(gpus) / len(todos)
print(f"\npiso de CPU {piso:.2f} s/estudio | 20 modelos {completo:.2f} s/estudio | fijos {fijos:.0f} s | "
      f"{'limitado por GPU' if gpu_bound else 'limitado por CPU: el coste por modelo es una cota superior'}")
print(f"GPU por modelo y estudio en una T4: {'≈' if gpu_bound else '≤'}{g:.3f} s")
print(f"\nproyección para {V5_ESPERADOS_RERUN} estudios con {len(gpus)} GPU:")
for k in (5, 4, 3, 2, 1):
    s = max(piso, 4 * k * g / len(gpus))
    print(f"  {k} particiones por familia ({4 * k:2d} modelos): {(fijos + V5_ESPERADOS_RERUN * s) / 60:5.1f} min")
print(f"\nPara celda_brazo_v5.py, con un 20 % de holgura: V5_PISO_CPU_S = {1.2 * piso:.2f}, "
      f"V5_GPU_MODELO_S = {1.2 * g:.3f}, V5_FIJOS_S = {int(1.2 * fijos) + 60}")
