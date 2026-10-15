#!/usr/bin/env python
"""Corpus native384 con LADO UNIFICADO (variante opcional de las semillas): todas las rodillas como derechas.

Mismo volumen que rsna/raptor_corpus.py (brazo native384dense-v10, sin cambios), y después, por ranura:

  lado      Laterality del DICOM (R/L) de las series usadas si es unánime; si falta o es mixta, la posición
            izquierda-derecha de la serie sagital: x = mediana de ImagePositionPatient[0] (LPS: +x es la
            izquierda del paciente) en sus cortes primero, central y último; x > 0 -> L. Es la regla de Owen
            (code/eda/09_lateralidad.py: 97.7 % de acierto con x > 0 frente a la etiqueta en 2,178 estudios),
            pero con la coordenada x directa en vez de la posición sobre la normal.
  sagital   el corpus ordena los cortes por posición sobre n = fila × columna (IOP). Orientación estándar:
            n_x < 0. Se invierte el orden de la ranura si (n_x > 0) XOR (lado == L).
  cor./ax.  el corpus no reorienta los píxeles. Orientación estándar: las columnas crecen hacia +x (fila de
            IOP con r_x > 0). Se voltea el ancho si (r_x < 0) XOR (lado == L).

Así una rodilla izquierda queda en espejo anatómico de una derecha y, de paso, las series con orientación no
estándar se llevan a la estándar. Las etiquetas NO se tocan (lo medial sigue siendo medial). No es el espejo
con intercambio de etiquetas que Owen probó sin éxito: aquí no hay aumento, cada estudio tiene una sola vista.

Salida en --out:  vols/<uid>.npy  e  indice.csv (uid, n_cortes, segundos, error, lado, lado_dicom, x_sag,
cuotas, volteos). Pensado para el /scratch del nodo, dentro de un job.

Uso:  python rsna/seeds_corpus_lado.py --out /scratch/.../corpus --procesos 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raptor_preproc as rp  # noqa: E402

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))
ARM = next(a for a in rp.ARMS if a["name"] == "native384dense-v10")
_SERIES: dict = {}
_OUT: Path = Path(".")


def _init(series, out):
    global _SERIES, _OUT
    _SERIES, _OUT = series, out


def series_por_ranura(rows: list[dict]) -> list[dict | None]:
    """La misma elección de series que rp.prepare_volume (pick_series_for_slot con el conjunto `used`)."""
    used, out = set(), []
    for plane, fluid, _ in ARM["slots"]:
        r = rp.pick_series_for_slot(rows, plane, fluid, used)
        if r is not None:
            used.add(r["SeriesInstanceUID"])
        out.append(r)
    return out


def geometria(sdir: str, order_and_meta) -> dict:
    """IOP, x de los cortes primero/central/último (en el orden del corpus) y Laterality de una serie."""
    import pydicom
    files, _ = order_and_meta(sdir)
    if not files:
        return {}
    pick = sorted({0, len(files) // 2, len(files) - 1})
    hs = [pydicom.dcmread(files[i][0], stop_before_pixels=True) for i in pick]
    iop = np.array(hs[0].ImageOrientationPatient, float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    xs = [float(h.ImagePositionPatient[0]) for h in hs if getattr(h, "ImagePositionPatient", None) is not None]
    lat = {"R": "R", "RIGHT": "R", "L": "L", "LEFT": "L"}.get(str(getattr(hs[0], "Laterality", "") or "").strip().upper(), "")
    return {"r_x": float(r[0]), "n_x": float(n[0]), "x": float(np.median(xs)) if xs else None, "lat": lat}


def unificar(vol: np.ndarray, cuotas: list[int], geos: list[dict], lado: str) -> tuple[np.ndarray, list]:
    out, volteos, i0 = vol.copy(), [], 0
    izq = lado == "L"
    for (plane, _, _), q, g in zip(ARM["slots"], cuotas, geos):
        if not q:
            volteos.append(None)
            continue
        seg = slice(i0, i0 + q)
        if plane == "Sagittal":
            inv = (g["n_x"] > 0) != izq
            if inv:
                out[seg] = vol[seg][::-1]
            volteos.append({"plano": plane, "invertir_orden": bool(inv), "n_x": round(g["n_x"], 3)})
        else:
            flip = (g["r_x"] < 0) != izq
            if flip:
                out[seg] = vol[seg][:, :, ::-1]
            volteos.append({"plano": plane, "voltear_ancho": bool(flip), "r_x": round(g["r_x"], 3)})
        i0 += q
    assert i0 == len(vol), (i0, len(vol))
    return np.ascontiguousarray(out), volteos


def work(uid: str) -> dict:
    t0 = time.time()
    rec = {"uid": uid, "n_cortes": 0, "error": ""}
    dest = _OUT / "vols" / f"{uid}.npy"
    try:
        reader = rp.reader_con_cache(rp.make_reader())     # comparte el orden de las series con prepare_volume
        vol, info = rp.prepare_volume(ARM, uid, _SERIES, str(DATA / "train_series"), reader)
        rows = _SERIES.get(uid, [])
        geos = []
        for r, q in zip(series_por_ranura(rows), info["cuotas"]):
            geos.append(geometria(f"{DATA / 'train_series'}/{uid}/{r['SeriesInstanceUID']}", reader[0]) if (r and q) else {})
        lats = {g["lat"] for g in geos if g.get("lat")}
        sag = [g for (p, _, _), g in zip(ARM["slots"], geos) if p == "Sagittal" and g.get("x") is not None]
        x_sag = sag[0]["x"] if sag else None
        lado_dicom = lats.pop() if len(lats) == 1 else ("mixto" if lats else "")
        lado = lado_dicom if lado_dicom in ("R", "L") else (("L" if x_sag > 0 else "R") if x_sag is not None else "")
        if not lado:
            raise RuntimeError("sin lado: ni Laterality ni serie sagital con posición")
        vol, volteos = unificar(vol, info["cuotas"], geos, lado)
        tmp = dest.with_name(f".{uid}.tmp.npy")
        np.save(tmp, vol)
        os.replace(tmp, dest)
        reader[0].limpiar()
        rec.update({"n_cortes": int(len(vol)), "lado": lado, "lado_dicom": lado_dicom, "x_sag": x_sag,
                    "cuotas": json.dumps(info["cuotas"]), "volteos": json.dumps(volteos)})
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["segundos"] = round(time.time() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procesos", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="solo para pruebas")
    ap.add_argument("--uids", type=Path, default=None, help="solo para pruebas: archivo con un UID por línea")
    ap.add_argument("--lados-owen", type=Path, default=Path("/lustre/proyectos/p082/preproc/lados_train_2084968.parquet"),
                    help="solo lectura: tabla de Owen para medir el acuerdo del lado")
    a = ap.parse_args()
    (a.out / "vols").mkdir(parents=True, exist_ok=True)
    ser = pd.read_csv(DATA / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = sorted(pd.read_csv(DATA / "train.csv", usecols=["StudyInstanceUID"], dtype=str)["StudyInstanceUID"])
    if a.uids:
        pedidos = set(a.uids.read_text().split())
        uids = [u for u in uids if u in pedidos]
    if a.limit:
        uids = uids[:a.limit]
    series = {u: g.to_dict("records") for u, g in ser[ser.StudyInstanceUID.isin(uids)].groupby("StudyInstanceUID")}
    print(f"corpus native384 lado unificado: {len(uids)} estudios, {a.procesos} procesos -> {a.out}", flush=True)
    t0, recs = time.time(), []
    with Pool(a.procesos, initializer=_init, initargs=(series, a.out)) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, uids, chunksize=4), 1):
            recs.append(rec)
            if i % 250 == 0 or i == len(uids):
                print(f"  {i}/{len(uids)} en {time.time() - t0:.0f}s", flush=True)
    idx = pd.DataFrame(recs).sort_values("uid")
    idx.to_csv(a.out / "indice.csv", index=False)
    bad = idx[idx.error != ""]
    ok = idx[idx.error == ""]
    print(f"listo en {time.time() - t0:.0f}s | errores {len(bad)} | lado {ok.lado.value_counts().to_dict()} | "
          f"Laterality {ok.lado_dicom.replace('', 'falta').value_counts().to_dict()}", flush=True)
    vt = [v for s in ok.volteos for v in json.loads(s) if v]
    for plane in ("Sagittal", "Coronal", "Axial"):
        vs = [v for v in vt if v["plano"] == plane]
        k = "invertir_orden" if plane == "Sagittal" else "voltear_ancho"
        eje = "n_x" if plane == "Sagittal" else "r_x"
        if vs:
            print(f"  {plane}: {sum(v[k] for v in vs)}/{len(vs)} ranuras volteadas; orientación no estándar "
                  f"{sum((v[eje] > 0) if plane == 'Sagittal' else (v[eje] < 0) for v in vs)}; "
                  f"|{eje}| < 0.5 (oblicua/ambigua) {sum(abs(v[eje]) < 0.5 for v in vs)}", flush=True)
    if a.lados_owen and a.lados_owen.exists():
        ow = pd.read_parquet(a.lados_owen).set_index("StudyInstanceUID").lado_final
        m = ok.set_index("uid").lado.reindex(ow.index).dropna()
        print(f"  acuerdo con la tabla de Owen ({a.lados_owen.name}): {(m == ow.loc[m.index]).mean() * 100:.2f} % "
              f"en {len(m)} estudios", flush=True)
    for _, r in bad.head(20).iterrows():
        print(f"  ERROR {r.uid}: {r.error}", flush=True)


if __name__ == "__main__":
    main()
