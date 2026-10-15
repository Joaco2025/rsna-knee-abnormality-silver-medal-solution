#!/usr/bin/env python
"""Corpus Raptor "native384" para el piloto CoAtNet: un .npy uint8 (n_cortes, 384, 384) por estudio.

Geometría = brazo `native384dense-v10` de rsna/raptor_preproc.py, sin cambios: 5 ranuras (SLOTS64), banda
0.02–0.98, hasta 96 cortes repartidos por capacidad, percentiles 2–98, recorte de 140 mm centrado y 384 px
(paridad bit a bit con el notebook comprobada en el job 2085083). n_cortes ≤ 96; la ventana i usa i-1, i, i+1.

Pensado para escribir en el /scratch local del nodo DENTRO de un job (nunca en Lustre: ~62 GB).
Salida en --out:  vols/<StudyInstanceUID>.npy  e  indice.csv (uid, n_cortes, segundos, error).
Un estudio que falla queda en indice.csv con su error; el entrenamiento decide qué hacer con él.

Uso:  python rsna/raptor_corpus.py --out /scratch/.../corpus --procesos 16
"""
from __future__ import annotations

import argparse
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


def work(uid: str) -> dict:
    t0 = time.time()
    rec = {"uid": uid, "n_cortes": 0, "error": ""}
    dest = _OUT / "vols" / f"{uid}.npy"
    try:
        if dest.exists():
            rec["n_cortes"] = int(np.load(dest, mmap_mode="r").shape[0])
        else:
            vol, info = rp.prepare_volume(ARM, uid, _SERIES, str(DATA / "train_series"), rp.make_reader())
            assert vol.dtype == np.uint8 and vol.shape[1:] == (ARM["img"], ARM["img"]), vol.shape
            tmp = dest.with_name(f".{uid}.tmp.npy")
            np.save(tmp, vol)
            os.replace(tmp, dest)                 # nunca queda un .npy a medias
            rec["n_cortes"] = int(info["n_cortes"])
    except Exception as exc:  # noqa: BLE001  un estudio ilegible no detiene el corpus
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["segundos"] = round(time.time() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procesos", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="solo para pruebas")
    ap.add_argument("--uids", type=Path, default=None, help="solo para pruebas: archivo con un UID por línea")
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
    print(f"corpus native384: {len(uids)} estudios, {a.procesos} procesos -> {a.out}", flush=True)
    t0, recs = time.time(), []
    with Pool(a.procesos, initializer=_init, initargs=(series, a.out)) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, uids, chunksize=4), 1):
            recs.append(rec)
            if i % 250 == 0 or i == len(uids):
                print(f"  {i}/{len(uids)} en {time.time() - t0:.0f}s", flush=True)
    idx = pd.DataFrame(recs).sort_values("uid")
    idx.to_csv(a.out / "indice.csv", index=False)
    bad = idx[idx.error != ""]
    print(f"listo en {time.time() - t0:.0f}s | errores {len(bad)} | cortes: media {idx.n_cortes.mean():.1f}, "
          f"min {idx.n_cortes.min()}, <96: {int((idx.n_cortes < 96).sum())}", flush=True)
    for _, r in bad.head(20).iterrows():
        print(f"  ERROR {r.uid}: {r.error}", flush=True)


if __name__ == "__main__":
    main()
