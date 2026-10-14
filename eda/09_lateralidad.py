#!/usr/bin/env python
"""¿Se puede saber si cada estudio es de rodilla izquierda o derecha?

El preprocesamiento orienta las imágenes en coordenadas del paciente, así que una rodilla izquierda
y una derecha quedan en espejo (lo medial de una cae donde lo lateral de la otra). Para llevarlas
todas a la misma orientación hace falta el lado. Aquí se mide, con una cabecera por serie:
  - cuántos estudios lo declaran en la etiqueta DICOM Laterality y si sus series coinciden;
  - si la posición del volumen en el eje izquierda-derecha del paciente (centro de las series
    coronales y axiales) lo predice en los estudios donde se conoce.
Solo imprime conteos y estadísticas agregadas.

Uso:  python eda/09_lateralidad.py CABECERAS.parquet [--out LADOS.parquet]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cabeceras", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="tabla StudyInstanceUID, lado declarado, x del centro, lado estimado")
    args = ap.parse_args()
    h = pd.read_parquet(args.cabeceras)
    print(f"series: {len(h):,}; estudios: {h.StudyInstanceUID.nunique():,}; columnas usadas presentes: "
          f"{[c for c in ('Laterality', 'eje_normal', 'pos_normal_mm', 'Anatomical_Plane') if c in h]}")

    lat = h.Laterality.astype(str).str.strip().str.upper().map({"R": "R", "RIGHT": "R", "L": "L", "LEFT": "L"})
    h = h.assign(lado=lat)
    per = h.dropna(subset=["lado"]).groupby("StudyInstanceUID").lado.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixto")
    print(f"\nestudios con Laterality en alguna serie: {len(per):,} ({len(per) / h.StudyInstanceUID.nunique() * 100:.1f} %)")
    print(per.value_counts().to_string())

    # La tabla de cabeceras guarda, de un corte por serie, su posición a lo largo de la normal del corte
    # (pos_normal_mm) y qué eje es esa normal (0 = x, sagital). En las series sagitales esa posición es,
    # salvo el signo de la normal, la coordenada izquierda-derecha del corte.
    study = pd.DataFrame({"lado": per.reindex(h.StudyInstanceUID.unique())})
    sag = h[h.eje_normal == 0]
    # En estos datos la normal sagital apunta a la derecha del paciente (las izquierdas salen negativas),
    # así que se cambia el signo para que +x sea la izquierda, como en el preprocesamiento.
    study["x_sagital"] = -sag.groupby("StudyInstanceUID").pos_normal_mm.median()
    known = study[study.lado.isin(["R", "L"])]
    print(f"\n-- Posición x del centro (mm; +x = izquierda del paciente) en los {len(known):,} estudios con lado conocido")
    for col in [c for c in study if c.startswith("x_")]:
        k = known.dropna(subset=[col])
        y = (k.lado == "L").astype(int)
        q = k.groupby("lado")[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack().round(0)
        best = max(((np.mean((k[col] > th) == (y == 1)), th) for th in np.arange(-60, 61, 5)))
        print(f"\n{col}: AUC de x para 'izquierda' = {roc_auc_score(y, k[col]):.4f}; acierto con x > 0: {np.mean((k[col] > 0) == (y == 1)) * 100:.1f} %; "
              f"mejor umbral {best[1]:.0f} mm: {best[0] * 100:.1f} %; estudios sin x: {int(known[col].isna().sum())}")
        print(q.to_string())
        margin = k[col].abs()
        for m in (10, 20, 30):
            sel = margin > m
            print(f"  con |x| > {m} mm: {sel.mean() * 100:.0f} % de los estudios, acierto {np.mean(((k[col] > 0) == (y == 1))[sel]) * 100:.1f} %")

    col = "x_sagital"
    study["lado_estimado"] = np.where(study[col] > 0, "L", "R")
    study.loc[study[col].isna(), "lado_estimado"] = None
    study["lado_final"] = study.lado.where(study.lado.isin(["R", "L"]), study.lado_estimado)
    print(f"\nlado final (declarado, o estimado por x cuando falta): {study.lado_final.value_counts(dropna=False).to_dict()}")
    unk = study[~study.lado.isin(["R", "L"])]
    print(f"sin lado declarado: {len(unk):,}; de ellos con |x| > 20 mm: {(unk[col].abs() > 20).mean() * 100:.0f} %")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        study.rename_axis("StudyInstanceUID").reset_index().to_parquet(args.out, index=False)
        print(f"tabla escrita en {args.out}")


if __name__ == "__main__":
    main()
