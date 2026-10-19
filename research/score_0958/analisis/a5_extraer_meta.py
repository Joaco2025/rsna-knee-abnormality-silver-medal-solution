"""A5 (extracción, CPU): por estudio de train, cabeceras de texto de UNA imagen por serie (sin píxeles) y una miniatura
64x64 del corte central de la primera serie sagital (para buscar casi-duplicados).  Solo lectura de kaggle_data.
Salida (--out): meta_series.parquet (una fila por serie), miniaturas.npz (uids, thumbs float16 [N,64,64]).
Uso: python -I a5_extraer_meta.py --out DIR --workers N [--limit K]
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("/lustre/proyectos/p082/kaggle_data/competitions/rsna-knee-abnormality-detection")
CAMPOS = ["StudyDescription", "SeriesDescription", "ProtocolName", "BodyPartExamined", "ContrastBolusAgent",
          "ContrastBolusRoute", "PatientSex", "PatientAge", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
          "SliceThickness", "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle",
          "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType", "Laterality", "ImageLaterality",
          "RequestedProcedureDescription", "ReasonForTheRequestedProcedure", "PerformedProcedureStepDescription",
          "ImageComments", "StudyComments", "InstitutionName", "StationName", "SoftwareVersions", "ReceiveCoilName",
          "PixelSpacing", "Rows", "Columns", "AcquisitionDate", "StudyDate", "PatientWeight", "BodyPartThickness"]


def txt(v):
    if v is None:
        return None
    try:
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)):
            return "|".join(str(x) for x in v)
    except Exception:
        pass
    return str(v)


def uno(args):
    import pydicom
    uid, series = args
    filas, thumb = [], None
    for ser, plane in series:
        d = D / "train_series" / uid / ser
        try:
            fs = sorted(os.listdir(d))
        except OSError as e:
            filas.append({"StudyInstanceUID": uid, "SeriesInstanceUID": ser, "error": repr(e)[:100]})
            continue
        fila = {"StudyInstanceUID": uid, "SeriesInstanceUID": ser, "plane": plane, "n_files": len(fs)}
        try:
            ds = pydicom.dcmread(str(d / fs[0]), stop_before_pixels=True)
            for c in CAMPOS:
                fila[c] = txt(getattr(ds, c, None))
        except Exception as e:
            fila["error"] = repr(e)[:100]
        filas.append(fila)
        if thumb is None and plane == "Sagittal" and fs:
            try:
                import cv2
                im = pydicom.dcmread(str(d / fs[len(fs) // 2])).pixel_array.astype(np.float32)
                im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
                im = (im - im.mean()) / (im.std() + 1e-6)
                thumb = im.astype(np.float16)
            except Exception:
                thumb = None
    return filas, uid, thumb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(D / "train_series.csv", dtype=str)
    grupos = [(u, list(zip(g.SeriesInstanceUID, g.Anatomical_Plane))) for u, g in s.groupby("StudyInstanceUID")]
    if a.limit:
        grupos = grupos[:a.limit]
    filas, uids, thumbs = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (f, u, t) in enumerate(ex.map(uno, grupos, chunksize=8), 1):
            filas += f
            if t is not None:
                uids.append(u)
                thumbs.append(t)
            if k % 500 == 0:
                print("estudios", k, "de", len(grupos), flush=True)
    pd.DataFrame(filas).to_parquet(out / "meta_series.parquet", index=False)
    np.savez_compressed(out / "miniaturas.npz", uids=np.array(uids), thumbs=np.stack(thumbs))
    print("HECHO series", len(filas), "miniaturas", len(uids), flush=True)


if __name__ == "__main__":
    main()
