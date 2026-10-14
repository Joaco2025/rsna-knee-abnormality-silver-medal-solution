#!/usr/bin/env python
"""Preprocesa todos los estudios de un split: un .npy (4, 32, lado, lado) uint8 por estudio.

Uso:  python rsna/preprocesar.py --out DIR [--split train] [--size 320] [--workers 48] [--limit N] [--qa 12]
Salidas en DIR:
  <StudyInstanceUID>.npy   tensor del estudio
  meta.parquet             qué serie se usó en cada ranura, tamaños originales, orientación, errores
  resumen.txt              estadísticas agregadas
  qa/                      (opcional) hojas de contacto PNG para revisión visual por una persona
Es reanudable: los estudios cuyo .npy ya existe se saltan.
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
from preproc import DEPTH, SIZE, build_study, slot_names, slots_for  # noqa: E402

DATA = Path(os.environ.get(
    "KNEE_DATA", "/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection"))
OUT: Path = Path(".")
SPLIT = "train"
SIDE = SIZE
N_SLOTS = 4
CROP_MM = 0.0
Z_RANGE = (0.0, 1.0)


def work(task: tuple[str, list[dict]]) -> dict:
    uid, series = task
    rec = {"StudyInstanceUID": uid}
    t0 = time.time()
    try:
        arr, info = build_study(DATA / f"{SPLIT}_series" / uid, series, size=SIDE, slots=slots_for(N_SLOTS), crop_mm=CROP_MM, z_range=Z_RANGE)
        rec.update(info)
        tmp = OUT / f".{uid}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, OUT / f"{uid}.npy")   # escritura atómica: nunca queda un .npy a medias
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["segundos"] = round(time.time() - t0, 2)
    return rec


def contact_sheet(path: Path, dest: Path) -> None:
    """Hoja de contacto: una fila por ranura, 8 cortes equiespaciados."""
    import cv2

    arr = np.load(path)
    idx = np.linspace(0, arr.shape[1] - 1, 8).round().astype(int)
    rows = [np.concatenate([cv2.resize(arr[k, i], (160, 160)) for i in idx], axis=1) for k in range(arr.shape[0])]
    cv2.imwrite(str(dest), np.concatenate(rows, axis=0))


def main():
    global OUT, SPLIT, SIDE, N_SLOTS, CROP_MM, Z_RANGE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=SIZE, help="lado en píxeles del tensor guardado")
    ap.add_argument("--rehacer", action="store_true", help="volver a generar también los estudios que ya tienen .npy (los sobrescribe)")
    ap.add_argument("--crop-mm", type=float, default=0.0, help="lado en mm de la caja física centrada que se conserva; 0 = campo de visión completo")
    ap.add_argument("--z-min", type=float, default=0.0, help="fracción inicial de los cortes que se cubre")
    ap.add_argument("--z-max", type=float, default=1.0, help="fracción final de los cortes que se cubre")
    ap.add_argument("--series", type=int, default=4, choices=[4, 6], help="ranuras por estudio; 6 añade coronal y axial no sensibles a líquido")
    ap.add_argument("--qa", type=int, default=0, help="número de hojas de contacto PNG a generar")
    args = ap.parse_args()
    OUT, SPLIT, SIDE, N_SLOTS = args.out, args.split, args.size, args.series
    SLOT_NAMES = slot_names(slots_for(N_SLOTS))
    CROP_MM, Z_RANGE = args.crop_mm, (args.z_min, args.z_max)
    import json
    (OUT / "preproc.json").write_text(json.dumps({"size": SIDE, "series": N_SLOTS, "crop_mm": CROP_MM, "z_range": list(Z_RANGE)}))
    OUT.mkdir(parents=True, exist_ok=True)

    series = pd.read_csv(DATA / f"{SPLIT}_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    cols = ["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    tasks = [(uid, grp[cols].to_dict("records")) for uid, grp in series.groupby("StudyInstanceUID", sort=True)]
    if args.limit:
        tasks = tasks[: args.limit]
    todo = tasks if args.rehacer else [t for t in tasks if not (OUT / f"{t[0]}.npy").exists()]
    print(f"split={SPLIT} estudios={len(tasks)} ya_hechos={len(tasks) - len(todo)} por_hacer={len(todo)} "
          f"forma=({N_SLOTS},{DEPTH},{SIDE},{SIDE}) procesos={args.workers}", flush=True)

    t0 = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(work, todo, chunksize=2), 1):
            rows.append(rec)
            if i % 250 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {time.time() - t0:.0f} s  errores={sum('error' in r for r in rows)}", flush=True)

    meta_path = OUT / "meta.parquet"
    meta = pd.DataFrame(rows)
    if meta_path.exists() and not args.rehacer:   # reanudación: conservar lo ya registrado
        old = pd.read_parquet(meta_path)
        meta = pd.concat([old[~old.StudyInstanceUID.isin(meta.StudyInstanceUID)], meta], ignore_index=True) if len(meta) else old
    meta.to_parquet(meta_path, index=False)
    if "lado" in meta:     # lado de cada rodilla, en el formato que espera entrenar.py --lados
        meta[["StudyInstanceUID", "lado_declarado", "x_sagital", "lado"]].assign(lado_final=meta.lado).to_parquet(OUT / "lados.parquet", index=False)
        print(f"lados: {meta.lado.value_counts().to_dict()}; declarados en la cabecera: {int((meta.lado_declarado != '').sum())}", flush=True)

    lines = [f"estudios en meta: {len(meta)}; con error: {int(meta['error'].notna().sum()) if 'error' in meta else 0}; "
             f"tiempo de esta ejecución: {(time.time() - t0) / 60:.1f} min"]
    n_files = sum(1 for n in os.listdir(OUT) if n.endswith(".npy") and not n.startswith("."))
    lines.append(f"archivos .npy: {n_files}; tamaño estimado: {n_files * N_SLOTS * DEPTH * SIDE * SIDE / 1e9:.1f} GB")
    lines.append(f"{'ranura':<10}{'presente%':>10}{'es FS%':>8}{'cortes(med)':>12}{'matriz(med)':>12}{'mm(med)':>9}"
                 f"{'transp%':>9}{'volt.col%':>10}{'volt.fil%':>10}{'media':>7}{'errores':>8}")
    for name in SLOT_NAMES:
        if f"{name}_presente" not in meta:
            continue
        p = meta[f"{name}_presente"].fillna(False).astype(bool)
        sub = meta[p]

        def col(c, f):
            return f(sub[f"{name}_{c}"]) if f"{name}_{c}" in sub and len(sub) else float("nan")

        err = int(meta[f"{name}_error"].notna().sum()) if f"{name}_error" in meta else 0
        lines.append(f"{name:<10}{p.mean() * 100:>10.1f}{col('fs', lambda s: s.mean() * 100):>8.1f}"
                     f"{col('cortes', lambda s: s.median()):>12.0f}{col('filas', lambda s: s.median()):>12.0f}"
                     f"{col('mm', lambda s: s.median()):>9.2f}{col('transpuesta', lambda s: s.mean() * 100):>9.1f}"
                     f"{col('volteo_cols', lambda s: s.mean() * 100):>10.1f}{col('volteo_filas', lambda s: s.mean() * 100):>10.1f}"
                     f"{col('media', lambda s: s.mean()):>7.1f}{err:>8}")
    n_pres = sum(meta[f"{n}_presente"].fillna(False).astype(bool) for n in SLOT_NAMES if f"{n}_presente" in meta)
    lines.append("ranuras presentes por estudio: " + ", ".join(f"{k}: {v}" for k, v in n_pres.value_counts().sort_index().items()))
    print("\n".join(lines), flush=True)
    (OUT / "resumen.txt").write_text("\n".join(lines) + "\n")

    if args.qa:
        qa = OUT / "qa"
        qa.mkdir(exist_ok=True)
        files = sorted(n for n in os.listdir(OUT) if n.endswith(".npy") and not n.startswith("."))
        for n in files[:: max(1, len(files) // args.qa)][: args.qa]:
            contact_sheet(OUT / n, qa / n.replace(".npy", ".png"))
        print(f"hojas de contacto para revisión humana en {qa}", flush=True)


if __name__ == "__main__":
    main()
