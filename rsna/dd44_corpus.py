#!/usr/bin/env python
"""Corpus "dd44": el formato con el que entrena dreaddevelopment (44 x 336 x 336 uint8 + máscara por estudio).

Port de `_make_reader` y `build_study` del notebook de inferencia de dreaddevelopment
(referencias/rsna_research/notebooks/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model/, líneas 255-342)
con las ranuras y la banda de su corpus de 44 cortes (knee_corpus_v2, hilo 737696: "44 over 6 to 94"):

  SLOTS44 = Sagital fluido 12, Sagital no fluido 10, Coronal fluido 8, Coronal no fluido 6, Axial 8
  banda 0.06-0.94 de cada serie, cortes equiespaciados, percentiles 2-98 de los cortes elegidos,
  recorte de 140 mm centrado y INTER_AREA a 336 px; una ranura sin serie queda a cero (máscara 0).

A diferencia de native384 (rsna/raptor_corpus.py): 44 cortes fijos en vez de hasta 96 densos, 336 px que
el entrenamiento reescala a 384, banda 0.06-0.94 en vez de 0.02-0.98, y ranuras vacías con máscara.

Salida en --out:  vols/<uid>.npy (44, 336, 336) uint8,  masks/<uid>.npy (44,) uint8  e
indice.csv (uid, n_cortes = cortes con máscara, segundos, error). Escribir en el /scratch del job (~22 GB).

Uso:  python rsna/dd44_corpus.py --out /scratch/.../corpus_dd44 --procesos 16 [--uids archivo]
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))
IMG = 336
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = sum(s[2] for s in SLOTS)                     # 44
SPAN = (0.06, 0.94)
_SERIES: dict = {}
_OUT: Path = Path(".")


# ---- verbatim del notebook (salvo el import de apply_modality_lut, que cambia de sitio en pydicom 3) ----
def _make_reader():
    import pydicom, cv2  # noqa: E401
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask
# ---- fin del código del notebook -----------------------------------------------------------------------


def _init(series, out):
    global _SERIES, _OUT
    _SERIES, _OUT = series, out


def work(uid: str) -> dict:
    t0 = time.time()
    rec = {"uid": uid, "n_cortes": 0, "error": ""}
    try:
        vol, mask = build_study(uid, _SERIES, str(DATA / "train_series"), _make_reader())
        for sub, arr in (("vols", vol), ("masks", mask)):
            dest = _OUT / sub / f"{uid}.npy"
            tmp = dest.with_name(f".{uid}.tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, dest)
        rec["n_cortes"] = int(mask.sum())
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["segundos"] = round(time.time() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procesos", type=int, default=16)
    ap.add_argument("--uids", type=Path, default=None, help="solo para pruebas: archivo con un UID por línea")
    a = ap.parse_args()
    for sub in ("vols", "masks"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    ser = pd.read_csv(DATA / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = sorted(pd.read_csv(DATA / "train.csv", usecols=["StudyInstanceUID"], dtype=str)["StudyInstanceUID"])
    if a.uids:
        pedidos = set(a.uids.read_text().split())
        uids = [u for u in uids if u in pedidos]
    series = {u: g.to_dict("records") for u, g in ser[ser.StudyInstanceUID.isin(uids)].groupby("StudyInstanceUID")}
    print(f"corpus dd44: {len(uids)} estudios, {a.procesos} procesos -> {a.out}", flush=True)
    t0, recs = time.time(), []
    with Pool(a.procesos, initializer=_init, initargs=(series, a.out)) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, uids, chunksize=4), 1):
            recs.append(rec)
            if i % 250 == 0 or i == len(uids):
                print(f"  {i}/{len(uids)} en {time.time() - t0:.0f}s", flush=True)
    idx = pd.DataFrame(recs).sort_values("uid")
    idx.to_csv(a.out / "indice.csv", index=False)
    print(f"listo en {time.time() - t0:.0f}s | errores {(idx.error != '').sum()} | cortes con máscara: "
          f"media {idx.n_cortes.mean():.1f}, min {idx.n_cortes.min()}, <44: {int((idx.n_cortes < 44).sum())}", flush=True)


if __name__ == "__main__":
    main()
