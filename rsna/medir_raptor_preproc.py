#!/usr/bin/env python
"""Mide el preprocesado de Raptor (rsna/raptor_preproc.py) sobre N estudios de train. Solo CPU, solo lectura.

Qué mide, por brazo (v5@336, v10@384, v8@384 con 44 cortes):
  - segundos por estudio: leer DICOM, ordenar, normalizar, recortar 140 mm y reescalar (volumen uint8);
  - bytes por estudio del volumen uint8 (lo que iría en un caché) y de las ventanas float32 (no se cachean);
  - cuántos cortes, cuántas series y cuántos errores;
  - proyección al caché completo de los 4,407 estudios: GB y horas de CPU (un núcleo y con W procesos).
Dos pasadas por estudio, en este orden:
  1. "compartida": los tres brazos con un lector que reutiliza el orden y los píxeles (como el notebook). Es la
     pasada en frío, la que paga la lectura de Lustre.
  2. "por brazo": cada brazo con un lector nuevo y sin caché. Los archivos ya están en la caché del cliente de
     Lustre, así que mide sobre todo cómputo y la lectura en caliente.
Paridad (primeros --paridad estudios): se ejecutan las funciones del propio notebook (extraídas por AST de las
celdas 2 y 22) y se comparan bit a bit las ventanas con las del port.
No escribe volúmenes ni ventanas: solo un JSON y un CSV pequeños en --out.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raptor_preproc as rp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NB = REPO / "referencias" / "bend-the-knee-to-speedy-raptors-the-original.ipynb"
DATA = Path("/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection")
N_TOTAL = 4407


def cuota(args_lfs: list[str]) -> dict:
    """Parsea `lfs quota` (kB) -> usado/cuota/límite en bytes. Solo lectura."""
    try:
        txt = subprocess.run(["lfs", "quota", *args_lfs], capture_output=True, text=True, timeout=60).stdout
        for linea in txt.splitlines():
            t = linea.split()
            if t and t[0] == "/lustre" and len(t) >= 4:
                kb = [int(x.rstrip("*")) for x in t[1:4]]
                return {"usado_B": kb[0] * 1024, "cuota_B": kb[1] * 1024, "limite_B": kb[2] * 1024, "crudo": linea.strip()}
        return {"error": "sin línea /lustre", "crudo": txt[-300:]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def cargar_referencia(out: Path):
    """Las funciones del notebook, tal cual (AST de las celdas 2 y 22), listas para llamarlas."""
    nb = json.load(open(NB))
    t2 = ast.parse("".join(nb["cells"][2]["source"]))
    f2 = [n for n in t2.body if isinstance(n, ast.FunctionDef) and n.name in ("_dense_allocate", "_dense_unique_linspace")]
    t22 = ast.parse("".join(nb["cells"][22]["source"]))
    ke_src, prep = None, None
    for n in t22.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_KE_SRC" for t in n.targets):
            ke_src = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name == "_ke_prepare_windows":
            prep = n
    assert ke_src and prep and len(f2) == 2, "no se encontraron las definiciones del notebook"
    ke_src = ke_src.replace("import timm\n", "")        # solo lo usa build_backbone, que no se llama aquí
    ke_ns = {"__name__": "_ke_raptor", "_asset_find_asset": lambda *a, **k: None, "rsna_rank01": lambda x: x,
             "rsna_strict_load": lambda *a, **k: None, "rsna_event": lambda *a, **k: None, "rsna_sha": lambda p: ""}
    exec(compile(ke_src, "<_KE_SRC>", "exec"), ke_ns)
    g = {"np": np, "_ke_np": np, "_KE_NS": ke_ns, "_ke_input_lock": threading.Lock(), "_ke_input_ids": set(),
         "_ke_input_audit_path": out / "ref_audit.jsonl", "rsna_event": lambda *a, **k: None}
    exec(compile(ast.Module(body=f2 + [prep], type_ignores=[]), "<celdas 2 y 22>", "exec"), g)
    return g, ke_ns


class Cronometro:
    """Envuelve un lector para repartir el tiempo entre ordenar, leer píxeles y el resto."""

    def __init__(self, reader):
        self.t = {"orden": 0.0, "pixeles": 0.0}
        o, p, c = reader

        def orden(s):
            t0 = time.perf_counter()
            try:
                return o(s)
            finally:
                self.t["orden"] += time.perf_counter() - t0

        def px(f):
            t0 = time.perf_counter()
            try:
                return p(f)
            finally:
                self.t["pixeles"] += time.perf_counter() - t0

        orden.avisos = getattr(o, "avisos", [])
        self.reader = (orden, px, c)


def resumen(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "media": float(x.mean()), "mediana": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "min": float(x.min()), "max": float(x.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--paridad", type=int, default=6, help="estudios en los que se compara con el código del notebook")
    ap.add_argument("--procesos", type=int, default=16, help="procesos supuestos para la proyección de reloj")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    q_ini = {"grupo": cuota(["-g", "p082", "/lustre"]), "usuario": cuota(["-u", "jcastro", "/lustre"])}
    print("cuota al inicio:", json.dumps({k: v.get("crudo") for k, v in q_ini.items()}), flush=True)

    ser = pd.read_csv(DATA / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    todos = sorted(pd.read_csv(DATA / "train.csv", usecols=["StudyInstanceUID"], dtype=str)["StudyInstanceUID"])
    estudios = sorted(np.random.default_rng(a.seed).choice(todos, size=a.n, replace=False).tolist())
    series = {u: g.to_dict("records") for u, g in ser[ser.StudyInstanceUID.isin(estudios)].groupby("StudyInstanceUID")}
    root = str(DATA / "train_series")
    print(f"{len(estudios)} estudios (semilla {a.seed}); series por estudio: media {np.mean([len(series.get(u, [])) for u in estudios]):.1f}", flush=True)

    try:
        ref_g, ref_ns = cargar_referencia(a.out)
        print("referencia del notebook cargada (AST celdas 2 y 22)", flush=True)
    except Exception as e:  # noqa: BLE001
        ref_g, ref_ns = None, None
        print(f"AVISO: no se pudo cargar la referencia del notebook, se omite la paridad: {type(e).__name__}: {e}", flush=True)

    filas, paridad, errores = [], [], []
    comp = {arm["name"]: [] for arm in rp.ARMS}
    for k, uid in enumerate(estudios):
        fila = {"uid": uid, "n_series_csv": len(series.get(uid, []))}
        vols = {}
        # Pasada 1: compartida, en frío
        rd = rp.reader_con_cache(rp.make_reader())
        t0 = time.perf_counter()
        for arm in rp.ARMS:
            try:
                vols[arm["name"]], _ = rp.prepare_volume(arm, uid, series, root, rd)
            except Exception as e:  # noqa: BLE001
                errores.append({"uid": uid, "brazo": arm["name"], "pasada": 1, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        fila["t_compartida_s"] = time.perf_counter() - t0
        rd[0].limpiar()
        # Pasada 2: brazo a brazo, sin caché
        for arm in rp.ARMS:
            nm = arm["name"]
            cr = Cronometro(rp.make_reader())
            t0 = time.perf_counter()
            try:
                vol, info = rp.prepare_volume(arm, uid, series, root, cr.reader)
            except Exception as e:  # noqa: BLE001
                errores.append({"uid": uid, "brazo": nm, "pasada": 2, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                continue
            dt = time.perf_counter() - t0
            fila.update({f"{nm}:t_s": dt, f"{nm}:t_orden_s": cr.t["orden"], f"{nm}:t_pixeles_s": cr.t["pixeles"],
                         f"{nm}:bytes_u8": int(vol.nbytes), f"{nm}:cortes": info["n_cortes"], f"{nm}:series": info["series_usadas"],
                         f"{nm}:avisos": len(info["avisos"]),
                         f"{nm}:igual_a_pasada1": bool(nm in vols and np.array_equal(vols[nm], vol))})
            t0 = time.perf_counter()
            w = rp.ventanas(vol, arm)
            fila[f"{nm}:t_ventanas_s"] = time.perf_counter() - t0
            fila[f"{nm}:bytes_ventanas_f32"] = int(w.numel() * 4)
            fila[f"{nm}:ventanas_ok"] = bool(len(w) == info["ventanas_esperadas"])
            if k < 5:
                b = vol.tobytes()
                comp[nm].append(len(zlib.compress(b, 1)) / len(b))
            if ref_g is not None and k < a.paridad:
                try:
                    ref_g["_ke_input_ids"].clear()
                    w_ref = ref_g["_ke_prepare_windows"](arm, uid, series, root, ref_ns["_make_reader"]())
                    import torch
                    igual = bool(torch.equal(w_ref, w))
                    paridad.append({"uid": uid, "brazo": nm, "ventanas": int(len(w)), "igual_bit_a_bit": igual,
                                    "max_abs_dif": float((w_ref - w).abs().max()) if w_ref.shape == w.shape else None,
                                    "forma_ref": list(w_ref.shape), "forma_port": list(w.shape)})
                    print(f"  paridad {uid[-8:]} {nm}: {'IGUAL' if igual else 'DIFIERE'} {paridad[-1]}", flush=True)
                except Exception as e:  # noqa: BLE001
                    paridad.append({"uid": uid, "brazo": nm, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    print(f"  paridad {uid[-8:]} {nm}: ERROR {e!r}", flush=True)
            del w
        filas.append(fila)
        print(f"[{k + 1}/{len(estudios)}] {uid[-8:]} compartida={fila['t_compartida_s']:.1f}s " +
              " ".join(f"{arm['name'].split('-')[-1]}={fila.get(arm['name'] + ':t_s', float('nan')):.1f}s/{fila.get(arm['name'] + ':cortes', 0)}c" for arm in rp.ARMS), flush=True)

    df = pd.DataFrame(filas)
    df.to_csv(a.out / "medicion_por_estudio.csv", index=False)

    brazos = {}
    for arm in rp.ARMS:
        nm = arm["name"]
        t = resumen(df.get(f"{nm}:t_s", []))
        b = resumen(df.get(f"{nm}:bytes_u8", []))
        brazos[nm] = {
            "config": {"img": arm["img"], "res": arm["res"], "span": arm["span"], "cortes_max": rp.OBJETIVO_CORTES},
            "segundos_por_estudio_en_caliente": t, "bytes_u8_por_estudio": b, "cortes": resumen(df.get(f"{nm}:cortes", [])),
            "segundos_ordenar": resumen(df.get(f"{nm}:t_orden_s", [])), "segundos_leer_pixeles": resumen(df.get(f"{nm}:t_pixeles_s", [])),
            "segundos_ventanas": resumen(df.get(f"{nm}:t_ventanas_s", [])), "bytes_ventanas_f32": resumen(df.get(f"{nm}:bytes_ventanas_f32", [])),
            "razon_compresion_zlib1": float(np.mean(comp[nm])) if comp[nm] else None,
            "errores": sum(1 for e in errores if e["brazo"] == nm and e["pasada"] == 2),
            "pasada2_igual_a_pasada1": bool(df.get(f"{nm}:igual_a_pasada1", pd.Series([True])).all()),
            "ventanas_esperadas_ok": bool(df.get(f"{nm}:ventanas_ok", pd.Series([True])).all()),
        }
        if t.get("n") and b.get("n"):
            gb = b["media"] * N_TOTAL / 1e9
            brazos[nm]["proyeccion_4407"] = {
                "GB": gb, "GiB": b["media"] * N_TOTAL / 2 ** 30,
                "GB_si_zlib1": gb * brazos[nm]["razon_compresion_zlib1"] if brazos[nm]["razon_compresion_zlib1"] else None,
                "horas_CPU_1_nucleo_en_caliente": t["media"] * N_TOTAL / 3600,
                "horas_reloj_con_procesos": t["media"] * N_TOTAL / 3600 / a.procesos, "procesos": a.procesos,
            }
    tc = resumen(df["t_compartida_s"])
    proy_total = None
    if tc.get("n"):
        gb_tot = sum(v["proyeccion_4407"]["GB"] for v in brazos.values() if "proyeccion_4407" in v)
        proy_total = {"GB_3_brazos_sin_comprimir": gb_tot, "horas_CPU_1_nucleo_pasada_compartida_en_frio": tc["media"] * N_TOTAL / 3600,
                      "horas_reloj_con_procesos": tc["media"] * N_TOTAL / 3600 / a.procesos, "procesos": a.procesos,
                      "nota": "La pasada compartida decodifica una vez y sirve a los tres brazos; la lectura en frío de Lustre puede "
                              "escalar peor que lineal con muchos procesos (no medido)."}

    q_fin = {"grupo": cuota(["-g", "p082", "/lustre"]), "usuario": cuota(["-u", "jcastro", "/lustre"])}
    libre = {}
    for k, q in q_fin.items():
        if "cuota_B" in q:
            libre[k] = {"libre_bajo_cuota_GB": (q["cuota_B"] - q["usado_B"]) / 1e9, "libre_bajo_limite_GB": (q["limite_B"] - q["usado_B"]) / 1e9}
    print("\n== Resultado por brazo ==")
    for nm, v in brazos.items():
        print(f"{nm}: t={v['segundos_por_estudio_en_caliente']} bytes={v['bytes_u8_por_estudio']} errores={v['errores']}")
        print("   proyección:", v.get("proyeccion_4407"))
    print("compartida en frío:", tc, "\nproyección total:", proy_total, "\ncuota libre:", libre, flush=True)

    ok_par = [p for p in paridad if "igual_bit_a_bit" in p]
    resumen_par = {"comparaciones": len(paridad), "con_resultado": len(ok_par), "iguales_bit_a_bit": sum(p["igual_bit_a_bit"] for p in ok_par),
                   "errores": [p for p in paridad if "error" in p]}
    print("paridad:", resumen_par, flush=True)
    (a.out / "medicion_raptor_preproc.json").write_text(json.dumps({
        "args": {"n": a.n, "seed": a.seed, "paridad": a.paridad}, "estudios": estudios, "brazos": brazos,
        "pasada_compartida_en_frio_s_por_estudio": tc, "proyeccion_total": proy_total, "cuota_inicio": q_ini, "cuota_fin": q_fin,
        "cuota_libre": libre, "paridad": paridad, "paridad_resumen": resumen_par, "errores": errores, "N_total": N_TOTAL,
        "versiones": {"numpy": np.__version__, "pandas": pd.__version__}}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
