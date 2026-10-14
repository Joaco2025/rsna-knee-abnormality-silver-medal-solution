"""Convierte un estudio DICOM de rodilla en un tensor de tamaño fijo.

La misma lógica debe correr en el cluster (entrenamiento) y en el notebook de Kaggle (inferencia),
así que este módulo solo depende de numpy, pydicom y OpenCV, y no lee nada fuera del estudio.

Salida por estudio: uint8 de forma (4, DEPTH, SIZE, SIZE), una "ranura" por tipo de serie:
    0 sagital sensible a líquido   1 sagital no sensible a líquido
    2 coronal sensible a líquido   3 axial sensible a líquido
Si falta la serie preferida de un plano se usa la otra del mismo plano; si el plano no tiene más
series, la ranura queda a cero y se marca como ausente.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pydicom

DEPTH, SIZE = 32, 320
SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Axial", 1)]
SLOT_NAMES = ["sag_fs", "sag_nofs", "cor_fs", "ax_fs"]
# Variante de 6 ranuras: las cuatro de siempre, en el mismo orden, más coronal y axial no sensibles
# a líquido. Las cuatro primeras se eligen igual, así que un tensor de 6 contiene al de 4.
SLOTS6 = SLOTS + [("Coronal", 0), ("Axial", 0)]
_SHORT = {"Sagittal": "sag", "Coronal": "cor", "Axial": "ax"}


def slots_for(n: int) -> list[tuple[str, int]]:
    assert n in (4, 6), "solo hay variantes de 4 y de 6 ranuras"
    return SLOTS if n == 4 else SLOTS6


def slot_names(slots: list[tuple[str, int]]) -> list[str]:
    return [f"{_SHORT[plane]}_{'fs' if fs else 'nofs'}" for plane, fs in slots]

# Orientación canónica en coordenadas del paciente (LPS: +x izquierda, +y posterior, +z craneal):
# hacia dónde crecen las columnas, hacia dónde crecen las filas y en qué sentido se ordenan los cortes.
CANON = {
    "Sagittal": (np.array([0, 1, 0.0]), np.array([0, 0, -1.0]), np.array([1, 0, 0.0])),
    "Coronal": (np.array([1, 0, 0.0]), np.array([0, 0, -1.0]), np.array([0, 1, 0.0])),
    "Axial": (np.array([1, 0, 0.0]), np.array([0, 1, 0.0]), np.array([0, 0, 1.0])),
}


def select_series(series: list[dict], slots: list[tuple[str, int]] = SLOTS) -> list[dict | None]:
    """Elige una serie por ranura.

    `series`: dicts con SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive y n_slices.
    Dentro de cada grupo se prefiere la serie cuyo número de cortes se acerca más a DEPTH, lo que
    evita las adquisiciones 3D de cientos de cortes cuando hay una 2D equivalente.
    """
    def best(cands: list[dict]) -> dict | None:
        cands = [c for c in cands if c["n_slices"] >= 5]
        return min(cands, key=lambda c: (abs(c["n_slices"] - DEPTH), c["SeriesInstanceUID"])) if cands else None

    chosen: list[dict | None] = []
    used: set[str] = set()
    for plane, fs in slots:
        free = [s for s in series if s["Anatomical_Plane"] == plane and s["SeriesInstanceUID"] not in used]
        pick = best([s for s in free if int(s["Fluid_Sensitive"]) == fs]) or best(free)
        if pick:
            used.add(pick["SeriesInstanceUID"])
        chosen.append(pick)
    return chosen


def _pixels(ds) -> np.ndarray:
    """Píxeles de un corte. Tolera cabeceras que declaran 16 bits con datos de 8 (visto en train)."""
    try:
        return ds.pixel_array.astype(np.float32)
    except ValueError:
        raw = np.frombuffer(ds.PixelData, dtype=np.uint8)
        rows, cols = int(ds.Rows), int(ds.Columns)
        if raw.size < rows * cols:
            raise
        return raw[: rows * cols].reshape(rows, cols).astype(np.float32)


def read_series(path: str | os.PathLike, plane: str) -> tuple[np.ndarray, tuple[float, float], dict]:
    """Lee una serie y la devuelve en orientación canónica: (cortes, filas, columnas) float32."""
    t_col, t_row, t_slice = CANON[plane]
    items = []
    n_bad = 0
    lat = ""
    for name in os.listdir(path):
        if not name.endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(os.path.join(path, name))
            arr = _pixels(ds)
        except Exception:   # un corte ilegible se descarta; la serie sigue con los demás
            n_bad += 1
            continue
        if arr.ndim == 3:  # multicuadro o color: no se espera, se toma el cuadro central
            arr = arr[arr.shape[0] // 2]
        lat = lat or str(ds.get("Laterality", "") or "").strip().upper()
        if ds.get("PhotometricInterpretation") == "MONOCHROME1":
            arr = arr.max() - arr
        iop = np.asarray(ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
        pos = np.asarray(ds.get("ImagePositionPatient", [0, 0, float(ds.get("InstanceNumber", 0))]), dtype=float)
        sp = ds.get("PixelSpacing", [1.0, 1.0])
        items.append((float(pos @ t_slice), arr, iop, (float(sp[0]), float(sp[1]))))
    if not items:
        raise ValueError("serie sin cortes")
    shape = items[len(items) // 2][1].shape
    items = [it for it in items if it[1].shape == shape]
    items.sort(key=lambda it: it[0])
    vol = np.stack([it[1] for it in items])
    u, v = items[0][2][:3], items[0][2][3:]      # u: hacia dónde crecen las columnas; v: las filas
    row_mm, col_mm = items[0][3]
    info = {"transpuesta": False, "volteo_cols": False, "volteo_filas": False, "cortes_ilegibles": n_bad,
            "lateralidad": lat, "pos_media": float(np.mean([it[0] for it in items]))}
    if abs(u @ t_col) < abs(u @ t_row):          # los ejes de la imagen están intercambiados
        vol = vol.transpose(0, 2, 1)
        u, v = v, u
        row_mm, col_mm = col_mm, row_mm
        info["transpuesta"] = True
    if u @ t_col < 0:
        vol = vol[:, :, ::-1]
        info["volteo_cols"] = True
    if v @ t_row < 0:
        vol = vol[:, ::-1, :]
        info["volteo_filas"] = True
    return np.ascontiguousarray(vol), (row_mm, col_mm), info


def _center(vol: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Recorta o rellena con ceros, centrado, hasta (cortes, th, tw)."""
    out = np.zeros((vol.shape[0], th, tw), dtype=vol.dtype)
    h, w = vol.shape[1:]
    sh, sw, dh, dw = max((h - th) // 2, 0), max((w - tw) // 2, 0), max((th - h) // 2, 0), max((tw - w) // 2, 0)
    ch, cw = min(h, th), min(w, tw)
    out[:, dh:dh + ch, dw:dw + cw] = vol[:, sh:sh + ch, sw:sw + cw]
    return out


def to_fixed(vol: np.ndarray, spacing: tuple[float, float], depth: int = DEPTH, size: int = SIZE,
             crop_mm: float = 0.0, z_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Normaliza la intensidad y lleva el volumen a (depth, size, size) uint8.

    Con crop_mm > 0 se toma una caja física de crop_mm x crop_mm centrada en el corte (así todas las
    rodillas quedan a la misma escala, sea cual sea el campo de visión del equipo); si no, el campo de
    visión completo. z_range es la fracción de los cortes que se cubre (p. ej. 0.02 a 0.98)."""
    sample = vol[:, ::4, ::4]
    lo, hi = np.percentile(sample, [0.5, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)

    # Remuestreo en profundidad: interpolación lineal entre cortes vecinos.
    n = vol.shape[0]
    pos = np.linspace(z_range[0] * (n - 1), z_range[1] * (n - 1), depth)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (pos - i0).astype(np.float32)[:, None, None]
    vol = vol[i0] * (1 - w) + vol[i1] * w

    # Relleno a campo de visión cuadrado en milímetros, para no deformar la anatomía.
    if crop_mm > 0:
        vol = _center(vol, max(int(round(crop_mm / spacing[0])), 8), max(int(round(crop_mm / spacing[1])), 8))
    else:
        h_mm, w_mm = vol.shape[1] * spacing[0], vol.shape[2] * spacing[1]
        side = max(h_mm, w_mm)
        pad_h = int(round((side - h_mm) / spacing[0] / 2))
        pad_w = int(round((side - w_mm) / spacing[1] / 2))
        if pad_h or pad_w:
            vol = np.pad(vol, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)))

    interp = cv2.INTER_AREA if vol.shape[1] > size else cv2.INTER_LINEAR
    out = np.stack([cv2.resize(s, (size, size), interpolation=interp) for s in vol])
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def build_study(study_dir: str | os.PathLike, series: list[dict], depth: int = DEPTH, size: int = SIZE,
                slots: list[tuple[str, int]] = SLOTS, crop_mm: float = 0.0,
                z_range: tuple[float, float] = (0.0, 1.0), variants: list[tuple] | None = None):
    """Tensor (ranuras, depth, size, size) uint8 de un estudio y un resumen de lo que se usó.

    El resumen incluye el lado de la rodilla ("lado": L o R), tomado de la etiqueta Laterality o, si
    falta, de la posición sagital del volumen (+x es la izquierda del paciente).
    Con `variants` (lista de (crop_mm, (z0, z1))) cada serie se lee una sola vez y se devuelve un
    diccionario {variante: tensor}; sirve para inferir con modelos de preprocesamientos distintos."""
    study_dir = Path(study_dir)
    for s in series:
        if "n_slices" not in s:
            d = study_dir / s["SeriesInstanceUID"]
            s["n_slices"] = sum(n.endswith(".dcm") for n in os.listdir(d)) if d.is_dir() else 0
    todo = variants or [(crop_mm, tuple(z_range))]
    outs = {v: np.zeros((len(slots), depth, size, size), dtype=np.uint8) for v in todo}
    out = outs[todo[0]]
    info: dict = {}
    side_tag, side_x = "", None
    names = slot_names(slots)
    for k, pick in enumerate(select_series(series, slots)):
        name = names[k]
        info[f"{name}_presente"] = False
        if pick is None:
            continue
        try:
            vol, spacing, o = read_series(study_dir / pick["SeriesInstanceUID"], pick["Anatomical_Plane"])
            for v in todo:
                outs[v][k] = to_fixed(vol, spacing, depth, size, crop_mm=v[0], z_range=v[1])
            if pick["Anatomical_Plane"] == "Sagittal":
                side_tag = side_tag or {"R": "R", "RIGHT": "R", "L": "L", "LEFT": "L"}.get(o["lateralidad"], "")
                side_x = o["pos_media"] if side_x is None else side_x
            info.update({f"{name}_presente": True, f"{name}_serie": pick["SeriesInstanceUID"],
                         f"{name}_fs": int(pick["Fluid_Sensitive"]), f"{name}_cortes": int(vol.shape[0]),
                         f"{name}_filas": int(vol.shape[1]), f"{name}_cols": int(vol.shape[2]),
                         f"{name}_mm": float(spacing[0]), f"{name}_transpuesta": o["transpuesta"],
                         f"{name}_volteo_cols": o["volteo_cols"], f"{name}_volteo_filas": o["volteo_filas"],
                         f"{name}_ilegibles": o["cortes_ilegibles"],
                         f"{name}_media": float(out[k].mean())})
        except Exception as exc:  # una serie ilegible deja su ranura vacía, no pierde el estudio
            info[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:200]
    info["lado_declarado"], info["x_sagital"] = side_tag, side_x
    info["lado"] = side_tag or ("L" if (side_x or 0.0) > 0 else "R")
    return (outs if variants else out), info
