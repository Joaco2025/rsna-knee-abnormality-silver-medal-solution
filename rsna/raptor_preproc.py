"""Preprocesado de Raptor portado del notebook público 0.940 (referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb).

Origen (índices de celda desde 0):
  celda 2   _dense_allocate, _dense_unique_linspace          (reparto de cortes por capacidad, objetivo 96)
  celda 22  _KE_SRC: ARMS, _make_reader, _pick_series_for_slot, _eval_centers, eval_windows, constantes
  celda 22  _ke_prepare_windows                              (selección de series, cortes, normalización, recorte en mm)

Qué hace, por estudio y por "brazo" (configuración de Raptor):
  1. Por cada una de las 5 ranuras (plano, sensibilidad a líquido, nº de cortes) elige una serie sin repetir.
  2. Ordena sus archivos por posición geométrica (IPP · normal) y fija la banda de cortes `span` del brazo.
  3. Reparte hasta 96 cortes únicos entre las ranuras según su capacidad (`_dense_allocate`).
  4. Por ranura: percentiles 2 y 98 de los cortes de la rejilla original, normaliza a [0, 1], recorta 140 mm
     centrados (usa PixelSpacing) y reescala a `img` px (INTER_AREA). Resultado: volumen uint8 (n, img, img).
  5. `eval_windows` forma todos los tripletes (i-1, i, i+1) y los normaliza con ImageNet (float32).

Lo que se guarda en un caché es el paso 4 (uint8); el paso 5 se calcula al vuelo en el entrenamiento.
Diferencias con el notebook, todas deliberadas y sin efecto numérico:
  - sin la auditoría en JSONL ni el contador de identidades duplicadas;
  - sin `rsna_event` (los avisos se devuelven en `info["avisos"]`);
  - `torch` solo se importa dentro de `eval_windows`;
  - el caché de órdenes y de píxeles del notebook es opcional (`reader_con_cache`).
La paridad bit a bit con el código del notebook se comprueba en rsna/medir_raptor_preproc.py.
"""
from __future__ import annotations

import glob

import numpy as np

# ---- Constantes de _KE_SRC (celda 22) ------------------------------------------------------------
IMG = 336
CROP_MM = 140.0
NORM = "imagenet"
OBJETIVO_CORTES = 96            # tercer argumento de _dense_allocate en _ke_prepare_windows
PERCENTILES = (2.0, 98.0)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]

# Los brazos del notebook. "maxspan-v5-reverse" usa el mismo preprocesado que "maxspan-v5" y solo invierte
# el orden de los tripletes al inferir, así que aquí no hace falta. `k_eval` queda sin uso: el notebook lo
# sustituye por 94 y `_ke_prepare_windows` calcula k = n_cortes − 2.
ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
     "res": 384, "img": 336, "slots": SLOTS64, "span": (0.02, 0.98), "k_eval": 62, "reverse": False, "w": 0.60},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
     "res": 384, "img": 384, "slots": SLOTS64, "span": (0.02, 0.98), "k_eval": 62, "reverse": False, "w": 0.10},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
     "res": 384, "img": 384, "slots": SLOTS44, "span": (0.06, 0.94), "k_eval": 42, "reverse": False, "w": 0.20},
]


# ---- Celda 2 -------------------------------------------------------------------------------------
def dense_allocate(capacities, proportions, target):
    capacities = np.asarray(capacities, dtype=np.int64)
    proportions = np.asarray(proportions, dtype=np.float64)
    if capacities.ndim != 1 or capacities.shape != proportions.shape or np.any(capacities < 0) or (not np.isfinite(proportions).all()) or np.any(proportions <= 0) or (int(target) < 1):
        raise ValueError("invalid capacity-aware sampling allocation")
    ideal = int(target) * proportions / proportions.sum()
    initial = np.floor(ideal).astype(np.int64)
    for i in np.argsort(-(ideal - initial), kind="stable")[:int(target) - int(initial.sum())]:
        initial[i] += 1
    quotas = np.minimum(initial, capacities)
    wanted = min(int(target), int(capacities.sum()))
    deficit = np.zeros(len(quotas), dtype=np.float64)
    while int(quotas.sum()) < wanted:
        donors = quotas < capacities
        weights = np.where(donors, proportions, 0.0)
        deficit[donors] += weights[donors] / weights.sum()
        chosen = int(np.argmax(np.where(donors, deficit, -np.inf)))
        quotas[chosen] += 1
        deficit[chosen] -= 1
    assert int(quotas.sum()) == wanted and np.all(quotas <= capacities)
    return quotas


def dense_unique_linspace(capacity, count):
    if not 0 <= count <= capacity:
        raise ValueError("cannot create unique picks beyond physical capacity")
    result = np.linspace(0, capacity - 1, count).round().astype(np.int64)
    if count and (result.min() < 0 or result.max() >= capacity):
        raise AssertionError("out-of-range source")
    assert len(np.unique(result)) == count
    return result


# ---- Celda 22: lector de DICOM -------------------------------------------------------------------
def make_reader():
    """(order_and_meta, read_px, mm_crop_resize), igual que `_make_reader` de _KE_SRC."""
    import pydicom
    import cv2
    try:
        from pydicom.pixels import apply_modality_lut       # pydicom >= 3
    except ImportError:                                      # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    avisos: list[str] = []

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm")
        recs, ps_list = [], []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, "ImageOrientationPatient", None)
                ipp = getattr(h, "ImagePositionPatient", None)
                if iop is not None and ipp is not None and (len(iop) == 6):
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                else:
                    instance = getattr(h, "InstanceNumber", None)
                    if instance is None:
                        avisos.append(f"raptor_order_index_fallback {f}")
                        instance = len(recs)
                    pos = float(instance)
                if not np.isfinite(pos):
                    avisos.append(f"raptor_order_nonfinite_fallback {f}")
                    pos = float(len(recs))
                ps = getattr(h, "PixelSpacing", None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception as exc:  # noqa: BLE001
                avisos.append(f"raptor_order_unreadable_skipped {f} {type(exc).__name__}")
                continue
        recs.sort(key=lambda x: (x[0], x[1]))
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return ([(f, ps) for _, f, ps in recs], med_ps)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(CROP_MM / max(ps, 0.001)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    order_and_meta.avisos = avisos
    return (order_and_meta, read_px, mm_crop_resize)


def reader_con_cache(reader):
    """Comparte entre brazos el orden de cada serie y los píxeles ya leídos (como el caché del notebook).

    El notebook limita el caché de píxeles a 192 MiB con un LRU; aquí el llamador lo vacía con `.limpiar()`
    entre estudios, que es el mismo efecto para este uso.
    """
    order_and_meta, read_px, crop = reader
    ordenes, pixeles = {}, {}

    def orden(sdir):
        if sdir not in ordenes:
            ordenes[sdir] = order_and_meta(sdir)
        return ordenes[sdir]

    def px(f):
        if f not in pixeles:
            v = read_px(f)
            v.setflags(write=False)
            pixeles[f] = v
        return pixeles[f]

    def limpiar():
        ordenes.clear()
        pixeles.clear()

    orden.limpiar = limpiar
    orden.avisos = getattr(order_and_meta, "avisos", [])
    return (orden, px, crop)


def pick_series_for_slot(rows, plane, fluid, used):
    """Una bandera desconocida no es falsa y no debe llegar a int(NaN) (celda 22)."""
    def flag(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "<na>", "unknown"):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in ("true", "yes", "y", "t"):
                return 1
            if text in ("false", "no", "n", "f"):
                return 0
            return None
        if not np.isfinite(number):
            return None
        if number not in (0.0, 1.0):
            return None
        return int(number)

    candidates = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if flag(r.get("Fluid_Sensitive")) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


# ---- Celda 22: _ke_prepare_windows, partido en volumen uint8 + ventanas ---------------------------
def prepare_volume(arm, study_uid, series, series_root, reader):
    """Volumen uint8 (n_cortes, img, img) de un estudio para un brazo, y un resumen.

    `series`: dict {StudyInstanceUID: [filas de *_series.csv como dict]}. `series_root`: carpeta que contiene
    <estudio>/<serie>/*.dcm. Lanza las mismas excepciones que el notebook (<3 cortes únicos, duplicados).
    """
    import cv2
    order_and_meta, read_px, _ = reader
    image_size = int(arm["img"])
    slots = list(arm["slots"])
    span_lo, span_hi = map(float, arm["span"])
    used, pools = set(), []
    rows = series.get(study_uid, [])
    for plane, fluid, count in slots:
        record = pick_series_for_slot(rows, plane, fluid, used)
        if record is None:
            pools.append(None)
            continue
        used.add(record["SeriesInstanceUID"])
        files, median_spacing = order_and_meta(f"{series_root}/{study_uid}/{record['SeriesInstanceUID']}")
        if not files:
            pools.append(None)
            continue
        lo = int(len(files) * span_lo)
        hi = max(int(len(files) * span_hi) - 1, lo)
        original = np.linspace(lo, hi, count).round().astype(int)
        pools.append((files, median_spacing, lo, hi, original))
    capacities = [0 if p is None else p[3] - p[2] + 1 for p in pools]
    quotas = dense_allocate(capacities, [s[2] for s in slots], OBJETIVO_CORTES)
    volume, sources = [], []
    for pool, quota in zip(pools, quotas):
        if pool is None or not quota:
            continue
        files, median_spacing, lo, hi, original = pool
        picks = lo + dense_unique_linspace(hi - lo + 1, int(quota))
        arrays = {}
        for i in sorted(set(original.tolist()) | set(picks.tolist())):
            arrays[i] = read_px(files[i][0])
        all_pixels = np.concatenate([arrays[int(i)].ravel() for i in original])
        low, high = np.percentile(all_pixels, list(PERCENTILES))
        for i in picks:
            path, spacing = files[int(i)]
            a = arrays[int(i)]
            a = np.clip((a - low) / (high - low + 1e-6), 0, 1)
            spacing = spacing if spacing > 0 else median_spacing
            h, w = a.shape
            c = min(int(round(140.0 / max(spacing, .001))), min(h, w))
            y, x = (h - c) // 2, (w - c) // 2
            a = cv2.resize(a[y:y + c, x:x + c], (image_size, image_size), interpolation=cv2.INTER_AREA)
            volume.append((a * 255).astype(np.uint8))
            sources.append(str(path))
    if len(sources) != len(set(sources)):
        raise RuntimeError("capacity-aware sampling selected duplicate source files")
    if len(volume) < 3:
        raise RuntimeError(f"{study_uid}: fewer than three unique source slices")
    volume = np.stack(volume)
    info = {"capacidades": capacities, "cuotas": [int(q) for q in quotas], "n_cortes": int(len(volume)),
            "series_usadas": len(used), "ventanas_esperadas": int(min(OBJETIVO_CORTES, sum(capacities)) - 2),
            "avisos": list(getattr(order_and_meta, "avisos", []))}
    return volume, info


def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = (int(valid.min()), int(valid.max()))
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    """Tripletes (i-1, i, i+1) normalizados con ImageNet: tensor float32 (k, 3, res, res). Celda 22."""
    import torch
    import torch.nn.functional as F
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    volume = np.asarray(vol)
    depth = int(volume.shape[0])
    centers = np.asarray(_eval_centers(mask, depth, k), dtype=np.int64)
    centers = np.clip(centers, 1, depth - 2)
    if tuple(volume.shape[-2:]) == (res, res):
        wins = np.empty((len(centers), 3, res, res), np.float32)
        for j, center in enumerate(centers):
            wins[j] = np.stack([volume[center - 1], volume[center], volume[center + 1]], axis=0).astype(np.float32) / 255.0
        x = torch.from_numpy(wins)
        if norm == "imagenet":
            x = (x - mean) / std
        return x
    triplets = centers[:, None] + np.asarray([-1, 0, 1], dtype=np.int64)
    unique, inverse = np.unique(triplets.reshape(-1), return_inverse=True)
    source = volume[unique].astype(np.float32) / 255.0
    planes = torch.from_numpy(source)
    if tuple(planes.shape[-2:]) != (res, res):
        resized = torch.empty((len(planes), res, res), dtype=torch.float32)
        for start in range(0, len(planes), 16):
            resized[start:start + 16] = F.interpolate(planes[start:start + 16, None], size=(res, res), mode="bilinear", align_corners=False)[:, 0]
        planes = resized
    x = planes[torch.from_numpy(inverse.reshape(-1, 3))]
    if norm == "imagenet":
        x = (x - mean) / std
    return x


def ventanas(volume, arm):
    """Todas las ventanas de un volumen, como `_ke_prepare_windows`: k = n_cortes − 2."""
    mask = np.ones(len(volume), dtype=np.uint8)
    w = eval_windows(volume, mask, k=len(volume) - 2, res=int(arm["res"]), norm=NORM)
    return w
