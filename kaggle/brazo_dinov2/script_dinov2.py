"""Brazo DINOv2 x20 del notebook 0.940 (Bend the Knee to Speedy Raptors), como script independiente.

GENERADO por generar_brazo_dinov2.py; no editar a mano. Las celdas 3-14 del notebook se copian por nodos de
primer nivel (ast), sin reescribir su lógica; ver la lista en experiments/reportes/brazo_dinov2_plan.md.
Notebook de origen: bend-the-knee-to-speedy-raptors-the-original.ipynb, sha256 32065fc04a644fe5e1c6517bb93a39e72af9304ecd83ed862f3d3fa71889407f
Corre en su propio proceso para no mezclar sus globales (T0, TIME_BUDGET, IMG, GROUP, SLOTS, ...) con los del 0.957.

Uso: python script_dinov2.py --data DATA --out CARPETA --deadline EPOCH   (o --comprobar --out CARPETA)
Salida: CARPETA/dinov2_sub.csv (StudyInstanceUID + 12 hallazgos en rango percentil) e informe.json."""
from __future__ import annotations
# La línea anterior es la primera del notebook (celda 3), que ya trae `from __future__ import annotations`.
# ---- celda 3 del notebook 0.940 --------------------------------------------------
import time as _startup_time
import os
import gc
import hashlib
import json
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
def _cuda_execution_probe(index):
    dev = torch.device(f'cuda:{index}')
    try:
        major, minor = torch.cuda.get_device_capability(index)
        probe = nn.Conv2d(3, 4, kernel_size=3, padding=1).eval().to(dev)
        with torch.inference_mode():
            out = probe(torch.zeros((1, 3, 16, 16), device=dev))
            if tuple(out.shape) != (1, 4, 16, 16):
                raise RuntimeError(f'unexpected CUDA probe shape {tuple(out.shape)}')
        torch.cuda.synchronize(index)
        print(f'cuda:{index} probe PASS (compute {major}.{minor})')
        del probe, out
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f'cuda:{index} probe FAIL ({type(exc).__name__}: {exc}); rejecting device')
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
T0 = time.time()
SEED = 2026
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
RUNS = [{'name': 'r224', 'img': 224}, {'name': 'r336', 'img': 336}]
EPOCHS = 10
BATCH_STUDIES = 8
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
RULES_NATIVE = {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'}
RULES_LEGACY = {'order': 'dominant_axis', 'lat': 'corner_x', 'slot_fallback': True, 'decode_fill': 'zero'}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0
LR_HEAD = 0.001
LR_BACKBONE = 8e-06
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
SLOTS_RECOVERED = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
SLOTS_PUBLIC = [('SAG_FLUID', 'Sagittal', None, True), ('COR_FLUID', 'Coronal', None, True), ('AX_FLUID', 'Axial', None, True), ('SAG_STRUCT', 'Sagittal', None, False), ('COR_STRUCT', 'Coronal', None, False), ('AX_STRUCT', 'Axial', None, False)]
SLOT_SCHEME = os.environ.get('SLOT_SCHEME', 'recovered')
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == 'public' else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4), 'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55
FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_SEP = re.compile('[_\\-.]')
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b|smart fat|\\bwater\\b')  # 2026-09-17: GE Dixon water images ('SMART FAT', 'Water: SMART FAT', seq FSEfw) are fat-suppressed; 37/4407 training studies had no fat-sat series detected before
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')
import os, json, hashlib, time, threading, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
_RSNA_AUDIT_LOCK = threading.RLock()
_RSNA_AUDIT = {'contract': 'btkd_speedy_v558_input_uniform', 'events': [], 'phases': {}}
_RSNA_ORDER_MEMO = {}
_RSNA_HEADERS_MEMO = {}
_RSNA_CACHE_FILES = []
_RSNA_TEST_IDS = None
_RSNA_LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
def rsna_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()
def rsna_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, default=str)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
def rsna_event(kind, **details):
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['events'].append({'kind':kind, **details})
def rsna_finite(values, tag, probability=False):
    x = np.asarray(values)
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
        raise RuntimeError(f'{tag}: nonfinite/non-numeric values BEFORE ranking')
    if probability and x.size and (x.min() < 0 or x.max() > 1):
        raise RuntimeError(f'{tag}: probability outside [0,1]')
    return x
def rsna_rank01(values):
    """Parent endpoint scale; tied values share their average rank."""
    x = np.asarray(rsna_finite(values, 'rank01'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rank01 requires a nonempty [study,finding] matrix')
    if len(x) == 1:
        return np.full_like(x, .5)
    return (pd.DataFrame(x).rank(method='average').to_numpy() - 1) / (len(x)-1)
def rsna_rankpct(values):
    x = np.asarray(rsna_finite(values, 'rankpct'), dtype=np.float64)
    if x.ndim != 2 or not len(x):
        raise ValueError('rankpct requires a nonempty matrix')
    return pd.DataFrame(x).rank(method='average', pct=True).to_numpy(np.float64)
def rsna_frame(frame, ids, labels, tag):
    ids = [str(u) for u in ids]; labels = list(labels)
    if len(ids) != len(set(ids)):
        raise RuntimeError(f'{tag}: duplicate expected UID')
    if frame.columns.tolist() != ['StudyInstanceUID', *labels]:
        raise RuntimeError(f'{tag}: label order/schema mismatch')
    frame = frame.copy(); frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
    if frame.StudyInstanceUID.duplicated().any() or set(frame.StudyInstanceUID) != set(ids):
        raise RuntimeError(f'{tag}: missing/extra/duplicate UID')
    frame = frame.set_index('StudyInstanceUID').loc[ids].reset_index()
    rsna_finite(frame[labels].to_numpy(), tag, probability=True)
    return frame
def rsna_save_predictions(tag, ids, values, labels=None):
    x = rsna_finite(values, tag)
    target = Path(DIAG_DIR); target.mkdir(parents=True, exist_ok=True)
    path = target/(tag+'.npz')
    np.savez_compressed(path, study_uids=np.asarray(ids,dtype=str), labels=np.asarray(labels or _RSNA_LABELS,dtype=str), values=x)
    with _RSNA_AUDIT_LOCK:
        _RSNA_AUDIT['phases'][tag]={'path':str(path),'sha256':rsna_sha(path),'shape':list(x.shape)}
def rsna_strict_load(model, state, tag):
    """A random frozen encoder is never an acceptable missing-state fallback."""
    expected = model.state_dict()
    missing = sorted(set(expected)-set(state)); extra = sorted(set(state)-set(expected))
    bad_shape = [k for k in set(expected)&set(state) if tuple(expected[k].shape)!=tuple(state[k].shape)]
    if missing or extra or bad_shape:
        raise RuntimeError(f'{tag}: incomplete/mismatched checkpoint; missing={missing[:20]}, extra={extra[:20]}, shapes={bad_shape[:20]}. A partial encoder needs its exact pinned source weights; random initialization is forbidden.')
    for k,t in state.items():
        if t.is_floating_point() and not bool(t.isfinite().all()):
            raise RuntimeError(f'{tag}: nonfinite checkpoint tensor {k}')
    return model.load_state_dict(state, strict=True)
def rsna_deadline(tag):
    if time.time() - T0 > TIME_BUDGET:
        raise TimeoutError(f'{tag}: full-run budget exceeded; no incomplete output may be submitted')
def rsna_array(shape, tag):
    """Large temporary pixels belong in scratch space, not saved notebook outputs."""
    size = int(np.prod(shape))
    if size <= float(os.environ.get('RSNA_CACHE_RAM_GIB', '1')) * (1 << 30):
        return np.zeros(shape, np.uint8)
    directory = Path(os.environ.get('RSNA_PIXEL_SCRATCH', '/kaggle/temp/rsna_pixels_v559'))
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as scratch_error:  # /kaggle/temp is not guaranteed; /tmp is non-persisted scratch with ~1 TiB free
        rsna_event('scratch_fallback', tag=tag, requested=str(directory), error=f'{type(scratch_error).__name__}: {scratch_error}')
        directory = Path('/tmp/rsna_pixels_v559')
        directory.mkdir(parents=True, exist_ok=True)
    import shutil
    free = shutil.disk_usage(directory).free
    rsna_event('scratch_allocation', tag=tag, bytes=size, directory=str(directory), free_bytes=int(free))
    if free < size + (1 << 30) and str(directory) != '/tmp/rsna_pixels_v559':
        rsna_event('scratch_fallback', tag=tag, requested=str(directory), error=f'insufficient free space ({free} bytes)')
        directory = Path('/tmp/rsna_pixels_v559'); directory.mkdir(parents=True, exist_ok=True); free = shutil.disk_usage(directory).free
    if free < size + (1 << 30):
        raise RuntimeError(f'{tag}: not enough scratch disk for complete cache ({size} bytes; {free} free)')
    fd, path = tempfile.mkstemp(prefix='pixels_', suffix='.npy', dir=directory)
    os.close(fd)
    try:
        a = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=tuple(shape))
        a[:] = 0
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    _RSNA_CACHE_FILES.append(path)
    return a
def rsna_release_pixels(array):
    if isinstance(array,np.memmap):
        path=str(array.filename)
        array.flush()
        # POSIX unlink releases the file after the last view is closed.
        Path(path).unlink(missing_ok=True)
        if path in _RSNA_CACHE_FILES: _RSNA_CACHE_FILES.remove(path)
def rsna_phase(stage, status='START', **extra):
    """Local phase evidence; this does not expose Kaggle's hidden rerun logs."""
    import json, os, resource, time
    from pathlib import Path
    work = Path(DIAG_DIR)
    work.mkdir(parents=True, exist_ok=True)
    current = {'stage': stage, 'status': status,
               'elapsed_seconds': time.time()-T0,
               'max_rss_kib': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), **extra}
    for path in ['/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max']:
        try: current[path.rsplit('/', 1)[-1]] = Path(path).read_text().strip()
        except OSError: pass
    if torch.cuda.is_initialized():
        current['gpus'] = [{'id': i, 'allocated_bytes': torch.cuda.memory_allocated(i),
                            'reserved_bytes': torch.cuda.memory_reserved(i)}
                           for i in range(torch.cuda.device_count())]
    print('[phase] '+json.dumps(current, sort_keys=True), flush=True)
    rsna_json(work/'current_phase.json', current)
    with (work/'phase_events.jsonl').open('a') as f:
        f.write(json.dumps(current, sort_keys=True)+'\n')
    return current
# ---- celda 4 del notebook 0.940 --------------------------------------------------
def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)
def _one_direct(tag, candidates, required):
    matches = []
    for candidate in map(Path, candidates):
        if candidate.is_dir() and all((candidate / item).exists() for item in required):
            matches.append(candidate)
    unique = []
    for match in matches:
        if str(match.resolve()) not in {str(item.resolve()) for item in unique}:
            unique.append(match)
    if len(unique) != 1:
        raise FileNotFoundError(
            f'{tag}: expected exactly one direct mounted artifact, found '
            f'{[str(item) for item in unique]}; checked {[str(Path(item)) for item in candidates]}'
        )
    return unique[0]
def find_dinov2(variant='small'):
    if variant != 'small':
        raise ValueError(f'unpinned DINOv2 variant: {variant}')
    return _one_direct('DINOv2-S model', [
        '/kaggle/input/models/metaresearch/dinov2/PyTorch/small/1',
        '/kaggle/input/models/metaresearch/dinov2/pytorch/small/1',
        '/kaggle/input/dinov2/PyTorch/small/1',
        '/kaggle/input/dinov2/pytorch/small/1',
    ], ['config.json', 'pytorch_model.bin'])
IMG = CACHE_IMG
# ---- celda 5 del notebook 0.940 --------------------------------------------------
HDR_TAGS = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime', 'Laterality', 'PixelSpacing', 'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'ImagePositionPatient', 'ImageOrientationPatient']
def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split('|')]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None
def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
        iop = _hdr_vec(getattr(r, 'ImageOrientationPatient', None), 6)
        ps = _hdr_vec(getattr(r, 'PixelSpacing', None), 2)
        rows, cols = (getattr(r, 'Rows', None), getattr(r, 'Columns', None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else 'R' if m < 0 else 'L'
    return out
def side_from_corner_x(h):
    out = {}
    for st, g in h.groupby('StudyInstanceUID'):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else 'R' if x < 0 else 'L'
    return out
def lat_of(h, tag=''):
    geo = side_from_corner_x(h) if RULES['lat'] == 'corner_x' else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby('StudyInstanceUID'):
        v = [str(x).strip().upper() for x in g['Laterality'].dropna()]
        if RULES['lat'] == 'corner_x' and 'ImageLaterality' in g.columns:
            v += [str(x).strip().upper() for x in g['ImageLaterality'].dropna()]
        v = [x[0] for x in v if x and x[0] in ('L', 'R')]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f'{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)')
    return d
def probe(item):
    split, study, series, path = item
    row = {'split': split, 'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'dir': path}
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith('.dcm')))
        row['files'] = files
        row['n_slices'] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]), stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == 'MultiValue':
                row[t] = '|'.join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row['err'] = str(exc)[:120]
    return row
def walk(split):
    key=(str(ROOT.resolve()),str(split))
    if key in _RSNA_HEADERS_MEMO:
        return _RSNA_HEADERS_MEMO[key].copy(deep=True)
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=['split', 'StudyInstanceUID', 'SeriesInstanceUID', 'dir', 'files', 'n_slices'] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    result=pd.DataFrame(rows)
    _RSNA_HEADERS_MEMO[key]=result.copy(deep=True)
    return result
def annotate(df):
    desc = df['SeriesDescription'].fillna('') + ' ' + df['SequenceName'].fillna('')
    desc = desc.str.lower().str.replace(_SEP, ' ', regex=True)
    opts = df['ScanOptions'].fillna('').str.upper().str.split('|')
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df['fatsat'] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df['RepetitionTime'], errors='coerce')
    te = pd.to_numeric(df['EchoTime'], errors='coerce')
    gre = df['ScanningSequence'].fillna('').str.upper().str.contains('GR')
    t1, t2, pdw = (desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX))
    df['weight'] = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    df['fluid'] = np.isin(df['weight'], ['PD', 'T2'])
    df['px'] = pd.to_numeric(df['PixelSpacing'].fillna('').str.split('|').str[0].replace('', np.nan), errors='coerce')
    return df
# ---- celda 6 del notebook 0.940 --------------------------------------------------
def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df['plane'] = series_df['SeriesInstanceUID'].map(plane_map)
    out = {}
    for study, g in series_df.groupby('StudyInstanceUID'):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g['plane'] == plane) & (g['fatsat'] == fs)
            if fluid is not None:
                sel &= g['fluid'] == fluid
            cand = g[sel]
            if len(cand) == 0 and RULES['slot_fallback'] and (fluid is False):
                cand = g[(g['plane'] == plane) & ~g['fatsat']]
            if len(cand):
                chosen[name] = cand.sort_values('n_slices', ascending=False).iloc[0]
        out[study] = chosen
    return out
# ---- celda 7 del notebook 0.940 --------------------------------------------------
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []
RAPTOR_HEADER_CACHE = {}
def _natural_key(name):
    return tuple((int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(name))))
def _order_dominant_axis(rec):
    files, d = (rec['files'], rec['dir'])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            raw = getattr(ds, 'ImagePositionPatient', None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, 'InstanceNumber', None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare, r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)
def order_slices(rec):
    if RULES['order'] == 'dominant_axis':
        return _order_dominant_axis(rec)
    files, d = (rec['files'], rec['dir'])
    RAPTOR_HEADER_CACHE.pop(str(d), None)
    keyed = []
    raptor_records = []
    raptor_safe = bool(files)
    for f in files:
        k = None
        ds = None
        file_raptor_safe = False
        try:
            try:
                ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
                file_raptor_safe = True
            except Exception:
                ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            file_raptor_safe = False
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        if file_raptor_safe:
            try:
                file_raptor_safe = (
                    iop.shape == (6,) and ipp.shape == (3,)
                    and np.isfinite(iop).all() and np.isfinite(ipp).all()
                    and np.isfinite(k)
                )
                spacing = getattr(ds, 'PixelSpacing', None)
                spacing = float(spacing[0]) if spacing is not None else 0.5
                file_raptor_safe = file_raptor_safe and np.isfinite(spacing) and spacing > 0
                if file_raptor_safe:
                    raptor_records.append((k, os.path.join(d, f), spacing))
            except Exception:
                file_raptor_safe = False
        raptor_safe = raptor_safe and file_raptor_safe
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (sorted(files,key=_natural_key), False)
    if raptor_safe and len(raptor_records) == len(files) and len({item[0] for item in raptor_records}) == len(files):
        raptor_records.sort(key=lambda item: (item[0], item[1]))
        RAPTOR_HEADER_CACHE[str(d)] = (
            [(item[1], item[2]) for item in raptor_records],
            float(np.median([item[2] for item in raptor_records])),
        )
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)
def read_slot(rec, n_slice=None, out_size=None):
    """Same healthy pixels as BTKD; failed slices cannot erase good native planes."""
    n_slice = GROUP if n_slice is None else int(n_slice)
    out_size = IMG if out_size is None else int(out_size)
    files, d, px = rec.get('ordered') or rec['files'], rec['dir'], rec['px']
    if not files:
        rsna_event('slot_no_files', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d))
        return None
    lo, hi = int(SLICE_BAND[0]*(len(files)-1)), int(SLICE_BAND[1]*(len(files)-1))
    idx = np.unique(np.linspace(lo,hi,n_slice).astype(int)) if hi>lo else np.array([len(files)//2])
    while len(idx)<n_slice: idx=np.append(idx,idx[-1])
    decoded, errors = {}, []
    for i in sorted(set(int(v) for v in idx[:n_slice])):
        path=os.path.join(d,files[i])
        try:
            ds=pydicom.dcmread(path,force=True)
            a=ds.pixel_array.astype(np.float32)
            a=a*float(getattr(ds,'RescaleSlope',1) or 1)+float(getattr(ds,'RescaleIntercept',0) or 0)
            if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError('expected finite 2-D pixels')
            decoded[i]=a
        except Exception as exc:
            decoded[i]=None; errors.append({'path':path,'error':f'{type(exc).__name__}: {exc}'})
    planes=[decoded[int(i)] for i in idx[:n_slice]]
    got=[i for i,p in enumerate(planes) if p is not None]
    if errors:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID',d))
        rsna_event('partial_slice_decode',series=str(rec.get('SeriesInstanceUID',d)),errors=errors)
    if not got:
        rsna_event('slot_all_decode_failed', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d), errors=errors[:5])
        return None
    from collections import Counter
    shape=Counter(planes[i].shape for i in got).most_common(1)[0][0]
    odd=[i for i in got if planes[i].shape!=shape]
    if odd:
        rsna_event('slot_shape_mismatch', series=str(rec.get('SeriesInstanceUID', d)), dir=str(d), majority=[int(v) for v in shape], dropped=len(odd))
        for i in odd: planes[i]=None
        got=[i for i in got if planes[i] is not None]
    for i,p in enumerate(planes):
        if p is None:
            planes[i]=np.zeros(shape,np.float32) if RULES['decode_fill']=='zero' else planes[min(got,key=lambda j:abs(j-i))]
    vol=np.stack(planes)
    if px and np.isfinite(px) and px>0:
        want=int(round(CROP_MM/px)); h,w=shape
        if 16<want<min(h,w):
            cy,cx=h//2,w//2; half=want//2
            vol=vol[:,max(0,cy-half):cy+half,max(0,cx-half):cx+half]
    lo_v,hi_v=np.percentile(vol,[1,99])
    vol=np.clip((vol-lo_v)/max(hi_v-lo_v,1e-6),0,1)
    t=torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t=F.interpolate(t,size=(out_size,out_size),mode='bilinear',align_corners=False)
    return (t.squeeze(0)*255).round().clamp(0,255).to(torch.uint8)
# ---- celda 8 del notebook 0.940 --------------------------------------------------
def normalise_laterality(img, plane, lat):
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
# ---- celda 9 del notebook 0.940 --------------------------------------------------
def build_cache(slot_map, plane_map, lat_map, tag):
    studies = list(_RSNA_TEST_IDS) if _RSNA_TEST_IDS is not None else sorted(slot_map)
    if set(studies)!=set(slot_map):
        rsna_event('coverage_mismatch',tag=tag,missing=[str(s) for s in studies if s not in slot_map][:20],extra=len(set(slot_map)-set(studies)))
        slot_map={s:slot_map.get(s,{}) for s in studies}
    sidx={s:i for i,s in enumerate(studies)}
    cache=rsna_array((len(studies),N_SLOT,CACHE_SLICES,IMG,IMG),tag)
    mask=np.zeros((len(studies),N_SLOT),np.float32)
    jobs=[(st,k,plane,slot_map[st][name]) for st in studies for k,(name,plane,_,_) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: {len(studies)} studies, {len(jobs)} slot-series, {cache.nbytes/2**30:.2f} GiB pixels')
    missing=[]; reused=0
    for job in jobs:
        rec=job[3]
        key=(str(Path(rec['dir']).resolve()),RULES['order'],tuple(sorted(rec['files'])))
        saved=_RSNA_ORDER_MEMO.get(key)
        if saved is not None:
            rec['ordered']=list(saved[0]);reused+=1
        else: missing.append((job,key))
    def ordered(item):
        job,key=item
        try:
            files,good=order_slices(job[3])
            if len(files)!=len(job[3]['files']) or set(files)!=set(job[3]['files']):
                raise ValueError(f'ordering lost/added files ({len(files)} vs {len(job[3]["files"])})')
        except Exception as exc:
            rsna_event('ordering_fallback',tag=tag,series=str(job[3].get('SeriesInstanceUID')),error=f'{type(exc).__name__}: {exc}')
            return key,sorted(job[3]['files']),False
        return key,files,bool(good)
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for start in range(0,len(missing),256):
            rsna_deadline(tag+' ordering')
            block=missing[start:start+256]
            for (job,_),(key,files,good) in zip(block,pool.map(ordered,block)):
                job[3]['ordered']=files;_RSNA_ORDER_MEMO[key]=(tuple(files),good)
                if not good: rsna_event('deterministic_order_fallback',series=str(job[3].get('SeriesInstanceUID')))
    done=0; unfilled=0
    def _tolerant_read_slot(j):
        try: return read_slot(j[3],CACHE_SLICES,IMG)
        except Exception as exc:
            rsna_event('slot_read_error',tag=tag,study=str(j[0]),series=str(j[3].get('SeriesInstanceUID')),error=f'{type(exc).__name__}: {exc}'); return None
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for start in range(0,len(jobs),128):
            rsna_deadline(tag+' decoding')
            block=jobs[start:start+128]
            for (st,k,plane,rec),img in zip(block,pool.map(_tolerant_read_slot,block)):
                done+=1
                if img is None:
                    unfilled+=1; rsna_event('slot_unfilled',tag=tag,study=str(st),series=str(rec.get('SeriesInstanceUID')),slot=int(k)); continue
                cache[sidx[st],k]=normalise_laterality(img,plane,lat_map.get(st)).numpy()
                mask[sidx[st],k]=1
    if unfilled: log(f'{tag}: {unfilled} slot(s) left empty and masked (flagged)')
    _empty=[studies[i] for i in np.flatnonzero(mask.sum(1)==0)]
    if _empty:
        rsna_event('empty_study_rows',tag=tag,count=len(_empty),studies=[str(s) for s in _empty])
        log(f'{tag}: {len(_empty)} study(ies) with no series matching this layout kept as all-zero masked rows (parent policy): {_empty[:5]}')
    if done!=len(jobs):
        raise RuntimeError(f'{tag}: incomplete cache; {done}/{len(jobs)} slots')
    if isinstance(cache,np.memmap): cache.flush()
    rsna_event('cache_complete',tag=tag,studies=len(studies),slot_jobs=len(jobs),filled=done,order_cache_hits=reused,bytes=int(cache.nbytes))
    log(f'{tag}: COMPLETE {done}/{len(jobs)}; reused ordering for {reused} series')
    return studies,cache,mask
# ---- celda 10 del notebook 0.940 --------------------------------------------------
class SlotHead(nn.Module):

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and (n_out == len(TARGETS)):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer('slot_prior', p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
# ---- celda 11 del notebook 0.940 --------------------------------------------------
class Model(nn.Module):

    def __init__(self, backbone, dim, pool='cls_mean', prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == 'cls_mean_focal':
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)
# ---- celda 12 del notebook 0.940 --------------------------------------------------
def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else DINOV2_SOURCE
    if p is None:
        raise FileNotFoundError('DINOv2 weights not attached')
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum((p.numel() for p in bb.parameters() if p.requires_grad))
    log(f'backbone: {n_layer} blocks, last {unfreeze_last} trainable ({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}')
    return Model(bb, dim, pool=pool, prior=prior)
_speed_original_build_model = build_model
import copy as _speed_copy
import threading as _speed_dino_threading
_speed_dino_templates = {}
_speed_dino_lock = _speed_dino_threading.RLock()
def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    key = (int(unfreeze_last), None if source is None else str(source), variant,
           pool, bool(prior), N_SLOT, tuple(TARGETS))
    with _speed_dino_lock:
        if key not in _speed_dino_templates:
            template = _speed_original_build_model(unfreeze_last, source, variant, pool, prior)
            assert all(p.device.type == 'cpu' for p in template.parameters())
            _speed_dino_templates[key] = template
        # No mutable state or GPU tensors shared between independently trained models.
        return _speed_copy.deepcopy(_speed_dino_templates[key])
# ---- celda 13 del notebook 0.940 --------------------------------------------------
SHARED_DINO_PREFIX_LAYERS = 6
SHARED_DINO_PREFIX_SHA256 = '0a55b893bde971c864ea5aeea443f075b17c084d13aa36f30bd1c7863f655cf9'
def _shared_prefix_state_sha256(state):
    digest = hashlib.sha256()
    prefixes = (
        'backbone.embeddings.',
        *(
            f'backbone.encoder.layer.{index}.'
            for index in range(
                SHARED_DINO_PREFIX_LAYERS
            )
        ),
    )
    keys = sorted(
        key
        for key in state
        if key.startswith(prefixes)
    )
    if len(keys) != 113:
        raise WeightsError(
            'shared DINOv2 prefix key-count drift: '
            f'{len(keys)} != 113'
        )
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
def _shared_prefix_eligible(members):
    if not members or not DEVS:
        return False
    return all(
        'state' not in member
        and int(member['config']['unfreeze_last'])
        == SHARED_DINO_PREFIX_LAYERS
        and member['config']['variant'] == 'small'
        and member['config'].get('pool', 'cls_mean')
        in POOL_PARTS
        for member in members
    )
def _shared_dino_prefix(model, images, img_size):
    batch, slots = images.shape[:2]
    values = images.reshape(
        batch * slots,
        *images.shape[2:],
    ).float().div_(255.0)
    if (
        img_size is not None
        and img_size != values.shape[-1]
    ):
        values = F.interpolate(
            values,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False,
        )
    values = (
        values - model.mean
    ) / model.std
    hidden = model.backbone.embeddings(values)
    for layer in model.backbone.encoder.layer[
        :SHARED_DINO_PREFIX_LAYERS
    ]:
        hidden = layer(hidden)
        hidden = hidden[0] if isinstance(hidden, (tuple, list)) else hidden
    return hidden
def _shared_dino_tail(
    model,
    hidden,
    batch,
    slots,
    slot_mask,
):
    value = hidden
    for layer in model.backbone.encoder.layer[
        SHARED_DINO_PREFIX_LAYERS:
    ]:
        value = layer(value)
        value = value[0] if isinstance(value, (tuple, list)) else value
    value = model.backbone.layernorm(value)
    patch = value[:, 1:]
    parts = [
        value[:, 0],
        patch.mean(1),
    ]
    if model.pool == 'cls_mean_focal':
        count = max(
            1,
            patch.shape[1] // 8,
        )
        parts.append(
            patch.topk(
                count,
                dim=1,
            ).values.mean(1)
        )
    feature = torch.cat(
        parts,
        dim=1,
    ).reshape(
        batch,
        slots,
        -1,
    )
    return model.head(feature, slot_mask)
@torch.no_grad()
def _predict_shared_dino_members(
    entries,
    cache,
    mask,
    idx,
    dev,
    img_size,
    group,
    starts,
):
    target_idx = {
        target: index
        for index, target in enumerate(TARGETS)
    }
    states = {}
    for member, model, jitter in entries:
        generator = torch.Generator(device=dev)
        jitter_seed = (
            SEED
            + int(
                hashlib.sha256(
                    str(member['id']).encode()
                ).hexdigest()[:8],
                16,
            )
        )
        generator.manual_seed(
            int(jitter_seed) % (2 ** 63 - 1)
        )
        states[member['id']] = {
            'out': [],
            'public': [],
            'soft': [],
            'generator': generator,
            'jitter': jitter,
        }
        model.eval()

    reference = entries[0][1]
    for batch_start in range(
        0,
        len(idx),
        EVAL_BATCH,
    ):
        selected = idx[
            batch_start:
            batch_start + EVAL_BATCH
        ]
        slot_mask = torch.from_numpy(
            mask[selected]
        ).to(dev)
        batch = len(selected)
        slots = cache.shape[1]
        member_windows = {
            member['id']: (
                [],
                [],
                [],
            )
            for member, _, _ in entries
        }

        for start in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(
                    cache[
                        selected,
                        :,
                        start:start + group,
                    ]
                )
            ).to(dev)
            def _dino_shared_forward(_amp):
                with torch.autocast('cuda', enabled=_amp):
                    _common = _shared_dino_prefix(reference, rows, img_size)
                    return {member['id']: _shared_dino_tail(model, _common, batch, slots, slot_mask).float() for member, model, _ in entries}
            original_logits = _dino_shared_forward(dev.type == 'cuda')
            if dev.type == 'cuda' and not all(bool(torch.isfinite(v).all()) for v in original_logits.values()):
                rsna_event('dino_fp16_nonfinite_retry_fp32', window_start=int(start))
                original_logits = _dino_shared_forward(False)

            for member, model, jitter in entries:
                member_id = member['id']
                view_logits = [
                    original_logits[member_id]
                ]
                if jitter:
                    jittered = augment(
                        rows,
                        generator=states[
                            member_id
                        ]['generator'],
                    )
                    with torch.autocast(
                        'cuda',
                        enabled=dev.type == 'cuda',
                    ):
                        jitter_hidden = (
                            _shared_dino_prefix(
                                reference,
                                jittered,
                                img_size,
                            )
                        )
                        view_logits.append(
                            _shared_dino_tail(
                                model,
                                jitter_hidden,
                                batch,
                                slots,
                                slot_mask,
                            ).float()
                        )
                    if dev.type == 'cuda' and not bool(torch.isfinite(view_logits[-1]).all()):
                        rsna_event('dino_jitter_fp16_nonfinite_dropped', member=str(member_id))
                        view_logits.pop()
                view_probs = [
                    torch.sigmoid(value)
                    for value in view_logits
                ]
                probabilities, logits, originals = (
                    member_windows[member_id]
                )
                logits.append(
                    torch.stack(
                        view_logits
                    ).mean(0)
                )
                probabilities.append(
                    torch.stack(
                        view_probs
                    ).mean(0)
                )
                originals.append(
                    view_probs[0]
                )

        for member, _, _ in entries:
            member_id = member['id']
            win_probs, win_logits, win_originals = (
                member_windows[member_id]
            )
            probabilities = torch.stack(win_probs)
            logits = torch.stack(win_logits)
            originals = torch.stack(win_originals)
            value = (
                torch.sigmoid(logits.mean(0))
                if TTA_POOL == 'logit'
                else probabilities.mean(0)
            )
            value = apply_target_window_pool(
                value,
                probabilities,
                logits,
                originals,
                TTA_TARGET_POOL,
                target_idx,
            )
            states[member_id]['out'].append(
                value.cpu().numpy()
            )
            public_value = apply_target_window_pool(
                originals.mean(0),
                originals,
                logits,
                originals,
                PUBLIC_FRONTIER_TARGET_POOL,
                target_idx,
            )
            states[member_id]['public'].append(
                public_value.cpu().numpy()
            )
            states[member_id]['soft'].append(
                legacy_fold_soft_window_pool(
                    originals,
                    target_idx,
                ).cpu().numpy()
            )

    return {
        member['id']: (
            np.concatenate(states[member['id']]['out']),
            np.concatenate(states[member['id']]['public']),
            np.concatenate(states[member['id']]['soft']),
        )
        for member, _, _ in entries
    }
@torch.inference_mode()
def _check_shared_dino_path(model,dev,img_size,tag):
    model.eval()
    g=torch.Generator().manual_seed(SEED)
    im=torch.randint(0,256,(2,N_SLOT,GROUP,img_size,img_size),generator=g,dtype=torch.uint8).to(dev)
    mask=torch.ones(2,N_SLOT,device=dev);mask[1,-1]=0
    direct=model(im,mask,img_size).float()
    split=_shared_dino_tail(model,_shared_dino_prefix(model,im,img_size),2,N_SLOT,mask).float()
    if not torch.isfinite(split).all() or not torch.allclose(split,direct,rtol=1e-5,atol=2e-5):
        raise RuntimeError(f'{tag}: shared-prefix execution differs from full forward')
    rsna_event('dino_shared_path_parity',member=tag,max_abs=float((split-direct).abs().max()))
def _run_shared_dino_group(
    path,
    members,
    cache,
    mask,
    idx,
    starts,
):
    ordered = sorted(
        members,
        key=lambda member: -(
            member.get('holdout') or 0
        ),
    )
    # run_dinov2 retains only submission_public_0899.csv, whose member
    # predictions are built exclusively from the unaugmented views.
    # Jittering here affected only the intermediate submission.csv that the
    # retained public-frontier file replaces at the end of this stage.
    plans = [
        (
            member,
            False,
        )
        for member in ordered
    ]
    assignments = [
        plans[index::len(DEVS)]
        for index in range(len(DEVS))
    ]
    results = {}
    result_lock = threading.Lock()

    def worker(dev, plans_for_device):
        global _DINOV2_MATCHED_MEMBERS
        loaded = []
        for member, jitter in plans_for_device:
            with BUILD_LOCK:
                checkpoint = torch.load(
                    Path(path) / member['file'],
                    map_location='cpu',
                    weights_only=False,
                )
                state = checkpoint['model']
                prefix_sha256 = (
                    _shared_prefix_state_sha256(
                        state
                    )
                )
                if (
                    prefix_sha256
                    != SHARED_DINO_PREFIX_SHA256
                ):
                    raise WeightsError(
                        f"{member['id']}: frozen prefix "
                        'hash mismatch: '
                        f'{prefix_sha256}'
                    )
                model = build_model(
                    int(
                        member['config'][
                            'unfreeze_last'
                        ]
                    ),
                    variant=member['config'][
                        'variant'
                    ],
                    pool=member['config'].get(
                        'pool',
                        'cls_mean',
                    ),
                    prior=bool(
                        member['config'].get(
                            'prior',
                            False,
                        )
                    ),
                ).to(dev)
                model.load_state_dict(state)
                check_fingerprint(
                    model,
                    dev,
                    IMG,
                    checkpoint.get('fingerprint'),
                    tag=f"{member['id']}: ",
                )
                _check_shared_dino_path(model, dev, IMG, member['id'])
                _DINOV2_MATCHED_MEMBERS += 1
                del checkpoint, state
            loaded.append(
                (member, model, jitter)
            )

        predicted = _predict_shared_dino_members(
            loaded,
            cache,
            mask,
            idx,
            dev,
            IMG,
            GROUP,
            starts,
        )
        with result_lock:
            results.update(predicted)
        del loaded
        gc.collect()
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

    with ThreadPoolExecutor(max_workers=len(DEVS)) as pool:
        futures=[pool.submit(worker,dev,plan) for dev,plan in zip(DEVS,assignments)]
        for future in futures: future.result()
    if len(results) != len(ordered):
        raise WeightsError(
            'shared DINOv2 worker did not return '
            f'all members: {len(results)} / '
            f'{len(ordered)}'
        )
    log(
        'shared first 6 DINOv2 blocks for '
        f'{len(ordered)} members; original views '
        f'computed once per {len(DEVS)} device(s)'
    )
    return [
        (
            member,
            results[member['id']],
            jitter,
        )
        for member, jitter in plans
    ]
FINGERPRINT_TOL = 0.002
EXPECTED_DINOV2_MEMBERS = 20
_DINOV2_MATCHED_MEMBERS = 0
def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out
def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=''):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if not np.isfinite(got).all() or not np.isfinite(exp).all():
        raise WeightsError(f'{tag}: nonfinite fingerprint')
    if got.shape != exp.shape:
        raise WeightsError(f'{tag}fingerprint shape {got.shape} != stored {exp.shape}: the architecture is not the one these weights were fitted to')
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f'{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load but do not compute what they computed when fitted - preprocessing, resolution or architecture has moved between the two runs.')
    log(f'{tag}fingerprint matches within {d:.2g}')
    return d
class WeightsError(RuntimeError):
    pass
def find_weights(name='manifest.json'):
    import json
    roots = [
        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights'),
        Path('/kaggle/input/rsna-knee-weights'),
    ]
    valid = []
    for root in roots:
        path = root / name
        if not path.is_file():
            continue
        try:
            man = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise WeightsError(f'invalid pinned weights manifest at {path}: {exc}') from exc
        if not isinstance(man.get('members'), list) or len(man['members']) != EXPECTED_DINOV2_MEMBERS:
            raise WeightsError(f'{path} must list exactly {EXPECTED_DINOV2_MEMBERS} members')
        if len({m.get('id') for m in man['members']}) != EXPECTED_DINOV2_MEMBERS:
            raise WeightsError(f'{path} has duplicate or missing member ids')
        missing = [m['file'] for m in man['members'] if not (root / m['file']).is_file()]
        if missing:
            raise WeightsError(
                f'{root} lists {len(man["members"])} members but misses {missing[0]!r}'
            )
        valid.append(root)
    if len(valid) != 1:
        raise WeightsError(f'expected one pinned rsna-knee-weights root, found {valid}')
    return valid[0]
TTA_OVERLAP = True
TTA_POOL = 'prob'
PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}
LEGACY_FOLD_SOFTPOOL_BETA = {
    'ACL': 6.0, 'MCL': 6.0,
    'Medial Meniscus': 8.0, 'Lateral Meniscus': 8.0,
    "Baker's": 8.0, 'Contusion': 8.0, 'Fracture': 10.0,
}
LEGACY_FOLD_SOFTPOOL_ALPHA = {
    'ACL': 0.20, 'MCL': 0.20,
    'Medial Meniscus': 0.25, 'Lateral Meniscus': 0.25,
    "Baker's": 0.20, 'Contusion': 0.20, 'Fracture': 0.15,
}
LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus': 15.0, 'Medial OA': 2.5, 'Lateral OA': 15.0, 'Contusion': 5.0}
def window_starts(n_slice, group, overlap=None):
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]
def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == 'max':
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == 'mean':
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == 'logit_mean':
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == 'original_mean':
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ('top2', 'top3'):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f'unknown TTA pooling mode for {target}: {mode}')
    return values
def legacy_fold_soft_window_pool(original_probs, target_idx):
    values = original_probs.mean(0).clone()
    for target, beta in LEGACY_FOLD_SOFTPOOL_BETA.items():
        j = target_idx[target]
        x = original_probs[:, :, j]
        weight = torch.softmax(float(beta) * x, dim=0)
        values[:, j] = (weight * x).sum(0)
    return values
@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None, starts=None, jitter=False, jitter_seed=SEED, return_public_frontier=False):
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError('predict_member was given no windows to average over')
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f'unknown target(s) in TTA_TARGET_POOL: {unknown}')
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out, public_soft_out = ([], [], [])
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            for view in views:
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    z = model(view, m, img_size).float()
                if dev.type == 'cuda' and not bool(torch.isfinite(z).all()):
                    rsna_event('dino_legacy_fp16_nonfinite_retry_fp32')
                    with torch.autocast('cuda', enabled=False):
                        z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == 'logit' else probs.mean(0)
        v = apply_target_window_pool(v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx)
        out.append(v.cpu().numpy())
        if return_public_frontier:
            public_v = apply_target_window_pool(original_probs.mean(0), original_probs, logits, original_probs, PUBLIC_FRONTIER_TARGET_POOL, target_idx)
            public_frontier_out.append(public_v.cpu().numpy())
            public_soft = legacy_fold_soft_window_pool(original_probs, target_idx)
            public_soft_out.append(public_soft.cpu().numpy())
    primary = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not return_public_frontier:
        return primary
    public_frontier = np.concatenate(public_frontier_out) if public_frontier_out else np.zeros((0, len(TARGETS)), np.float32)
    public_soft = np.concatenate(public_soft_out) if public_soft_out else np.zeros((0, len(TARGETS)), np.float32)
    return (primary, public_frontier, public_soft)
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
LEGACY_BUNDLE_FILE = 'rsna_20260807_v1.pt'
LEGACY_WEIGHT = 0.5
def find_legacy_bundle():
    candidates = [
        Path('/kaggle/input/datasets/pilkwang/rsna-knee-weights') / LEGACY_BUNDLE_FILE,
        Path('/kaggle/input/rsna-knee-weights') / LEGACY_BUNDLE_FILE,
    ]
    hits = [path for path in candidates if path.is_file()]
    if len(hits) > 1:
        raise WeightsError(f'ambiguous legacy bundle: {hits}')
    return hits[0] if hits else None
def legacy_group_members():
    return {}
def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            raise WeightsError(f"{m['id']}: inline legacy state is forbidden")
        ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
        state, fp = (ck['model'], ck.get('fingerprint'))
        if fp is None:
            raise WeightsError(f"{m['id']}: stored fingerprint is required")
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        global _DINOV2_MATCHED_MEMBERS
        _DINOV2_MATCHED_MEMBERS += 1
    t_ready = time.time()
    jitter_seed = SEED + int(hashlib.sha256(str(m['id']).encode()).hexdigest()[:8], 16)
    public_member = 'state' not in m
    predicted = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter, jitter_seed=jitter_seed, return_public_frontier=public_member)
    if public_member:
        p, public_p, public_soft = predicted
    else:
        p, public_p, public_soft = (predicted, None, None)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == 'cuda':
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, public_soft, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))
def _combine(per_member):
    all_ids = sorted({s for m in per_member for s in m['ids']})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        target_weight = m.get('target_weight')
        w = np.asarray(target_weight if target_weight is not None else [float(m.get('weight', 1.0))] * len(TARGETS), dtype=np.float64)
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        rsna_finite(m['pred'],str(m['id'])+' retained raw',probability=True)
        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise ValueError(f'at least one target has no ensemble vote: {tot}')
    return (all_ids, acc / tot[None, :])
def combine_public_members_by_fold(per_member, pred_key='pred'):
    # Raw-average the four members within each fold, rank each fold,
    # then give all five folds equal weight.
    all_ids = sorted({study for member in per_member for study in member['ids']})
    position = {study: i for i, study in enumerate(all_ids)}
    groups = {}
    for i, member in enumerate(per_member):
        fold = member.get('fold')
        key = f'fold_{fold}' if fold is not None else f'member_{i}'
        groups.setdefault(key, []).append(member)
    fold_ranks, diagnostics = ([], [])
    for key, members_in_fold in sorted(groups.items()):
        matrices = []
        for member in members_in_fold:
            values = np.full((len(all_ids), len(TARGETS)), np.nan, np.float64)
            values[[position[study] for study in member['ids']]] = np.asarray(member[pred_key], np.float64)
            if np.isnan(values).any():
                raise WeightsError(f"{member.get('id')}: incomplete {pred_key} coverage")
            matrices.append(values)
        raw_fold_mean = np.mean(matrices, axis=0)
        fold_ranks.append(pd.DataFrame(raw_fold_mean).rank(method='average', pct=True).to_numpy(np.float64))
        diagnostics.append({'ensemble_group': key, 'members': len(members_in_fold)})
    if len(fold_ranks) != 5:
        raise WeightsError(f'legacy branch requires five folds, found {len(fold_ranks)}')
    return all_ids, np.mean(fold_ranks, axis=0), pd.DataFrame(diagnostics)
def blend_legacy_frontier_and_soft(frontier_rank, soft_rank):
    output = np.asarray(frontier_rank, np.float64).copy()
    for j, target in enumerate(TARGETS):
        alpha = float(LEGACY_FOLD_SOFTPOOL_ALPHA.get(target, 0.0))
        if alpha:
            output[:, j] = (1.0 - alpha) * frontier_rank[:, j] + alpha * soft_rank[:, j]
    return output
def infer_from_package(path, dev=None):
    man = json.loads((Path(path) / 'manifest.json').read_text())
    members = man['members']
    if len(members) != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'expected {EXPECTED_DINOV2_MEMBERS} DINOv2 members, found {len(members)}')
    log(f'weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)')
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('test_series'))
    log(f'test header pass: {len(hte)} series')
    groups = {}
    for m in members:
        groups.setdefault(m['pixel_group'], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    failures = []
    abort = threading.Event()
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None, public_soft=None):
        if not np.isfinite(pred).all():
            rsna_event('dino_member_nonfinite_neutral', member=str(m['id']), rows=int((~np.isfinite(pred)).reshape(len(pred), -1).any(axis=1).sum()))
            pred = np.where(np.isfinite(pred), pred, 0.5)
        if public_pred is not None and not np.isfinite(public_pred).all():
            rsna_event('dino_member_public_nonfinite_neutral', member=str(m['id']))
            public_pred = np.where(np.isfinite(public_pred), public_pred, 0.5)
        rsna_finite(pred, m['id']+' DINO raw', probability=True)
        if public_pred is not None: rsna_finite(public_pred,m['id']+' DINO retained raw',probability=True)
        rsna_save_predictions('dino_'+str(m['id']),ids,public_pred if public_pred is not None else pred,TARGETS)
        if float(np.std(pred)) < 1e-09:
            raise WeightsError(f"{m['id']}: degenerate predictions")
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'fold': m.get('fold'), 'ids': ids, 'pred': public_pred, 'soft_pred': public_soft})
            elif public_pred is not None:
                raise WeightsError(f"{m['id']}: incomplete public-frontier windows")
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s); {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        if len(groups) == 1 and _shared_prefix_eligible(gm):
            shared_results = _run_shared_dino_group(
                path,
                gm,
                Cte,
                Mte,
                idx,
                starts_full,
            )
            for member, predicted, jitter in shared_results:
                prediction, public, soft = predicted
                bank(
                    member,
                    st_te,
                    prediction,
                    starts_full,
                    jitter,
                    public,
                    soft,
                )
            rsna_release_pixels(Cte)
            del Cte, Mte, shared_results
            gc.collect()
            continue
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                rsna_deadline('DINO complete-member inference')
                starts, jit = starts_full, False
                return (pending.pop(0), starts, jit)

        def worker(dev):
            while not abort.is_set():
                m, starts, jit = pop_next()
                if m is None:
                    return
                try:
                    p, public_p, public_soft, (fs, ws) = _run_member(path, m, dev, Cte, Mte, idx, starts, jit)
                    with STATE_LOCK:
                        est['fixed'], est['win'] = (fs, ws)
                    bank(m, st_te, p, starts, jit, public_p, public_soft)
                except Exception as exc:
                    with STATE_LOCK:
                        failures.append((m['id'], str(dev), type(exc).__name__, str(exc)))
                    abort.set()
                    return
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if failures:
            raise WeightsError(f'DINOv2 member failure; no fallback permitted: {failures[0]}')
        rsna_release_pixels(Cte)
        del Cte, Mte
        gc.collect()
    if _DINOV2_MATCHED_MEMBERS != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'fingerprint gate failed: {_DINOV2_MATCHED_MEMBERS} / {EXPECTED_DINOV2_MEMBERS}')
    if len(public_frontier_members) != EXPECTED_DINOV2_MEMBERS:
        raise WeightsError(f'public-frontier inference incomplete: {len(public_frontier_members)} / {EXPECTED_DINOV2_MEMBERS} members')
    log(f'DINOv2 fail-closed gate PASS: {_DINOV2_MATCHED_MEMBERS}/{EXPECTED_DINOV2_MEMBERS} fingerprints and members')
    frontier_ids, frontier_acc = _combine(public_frontier_members)
    sub = write_submission(frontier_acc, frontier_ids, test_df, '_pipeline_stage.csv')
    log(f'submission.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {sub.shape}')
    return sub
def adopt_config_globals(cfg):
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg['img'])
    GROUP = int(cfg['group'])
    CACHE_SLICES = int(cfg['slices'])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg['crop_mm'])
    SLICE_BAND = tuple((float(x) for x in cfg['band']))
    rules = cfg.get('rules') or RULES_NATIVE
    unknown = {k: v for k, v in rules.items() if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f'the members record pixel rules this pipeline cannot reproduce: {unknown}')
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg['slots']):
        raise WeightsError(f"the members were fitted on slots {cfg['slots']} and this pipeline defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")
# ---- celda 14 del notebook 0.940 --------------------------------------------------
def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    if sub[TARGETS].isna().any().any():
        rsna_event('submission_neutral_fill', studies=int(sub[TARGETS].isna().any(axis=1).sum()))
        sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub
def _v37_validate_submission(path, test_df, tag):
    frame = pd.read_csv(path)
    expected = ['StudyInstanceUID'] + TARGETS
    if list(frame.columns) != expected:
        raise ValueError(f'{tag}: columns differ from the competition contract')
    if len(frame) != len(test_df) or not frame['StudyInstanceUID'].is_unique:
        raise ValueError(f'{tag}: row count or StudyInstanceUID uniqueness failed')
    if set(frame['StudyInstanceUID'].astype(str)) != set(test_df['StudyInstanceUID'].astype(str)):
        raise ValueError(f'{tag}: StudyInstanceUID set differs from test.csv')
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{tag}: non-finite prediction')
    return frame
def main():
    pkg = find_weights()
    if pkg is None:
        raise WeightsError('required public checkpoint manifest not found')
    infer_from_package(pkg, DEVS[0])
    _v37_validate_submission('_pipeline_stage.csv', pd.read_csv(ROOT / 'test.csv'), 'public frontier')


# =====================================================================================================
# PEGAMENTO (no extraído del notebook). Todo lo de arriba es texto del notebook 0.940; lo de abajo es nuevo.
# Hace lo que en el notebook hacían rsna_initialize, la comprobación de 2xT4 y la celda 15 (aquí sin borrar
# submission.csv ni exigir otros datasets), con argumentos de línea de órdenes.
# =====================================================================================================
DIAG_DIR = '/kaggle/working/diagnostics'


def _glue_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Brazo DINOv2 x20 del notebook 0.940, como script independiente')
    ap.add_argument('--data', help='carpeta de la competencia (test.csv, test_series.csv, test_series/)')
    ap.add_argument('--out', help='carpeta de salida; el script cambia a ella y escribe dinov2_sub.csv e informe.json')
    ap.add_argument('--deadline', type=float, help='plazo duro, en segundos desde la época (time.time())')
    ap.add_argument('--comprobar', action='store_true', help='solo localiza pesos y modelo base y escribe un JSON; no infiere')
    return ap.parse_args(argv)


def _glue_comprobar(out):
    import importlib
    info = {'python': __import__('sys').version.split()[0], 'torch': str(torch.__version__)}
    for modulo in ('transformers', 'pydicom', 'numpy', 'pandas'):
        info[modulo] = str(importlib.import_module(modulo).__version__)
    info['dinov2_small'] = str(find_dinov2('small'))
    info['pesos'] = str(find_weights())
    info['miembros'] = len(json.loads((Path(info['pesos']) / 'manifest.json').read_text())['members'])
    info['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    Path(out, 'comprobacion.json').write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print('[dinov2] comprobacion OK', json.dumps(info, sort_keys=True), flush=True)


def _glue_main():
    global ROOT, DEVS, TIME_BUDGET, DIAG_DIR, DINOV2_SOURCE, _RSNA_TEST_IDS
    args = _glue_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    DIAG_DIR = str(out / 'diagnostics')
    Path(DIAG_DIR).mkdir(parents=True, exist_ok=True)
    if args.comprobar:
        _glue_comprobar(out)
        return 0
    if not (args.data and args.deadline):
        raise SystemExit('faltan --data y --deadline')
    ROOT = Path(args.data)
    DINOV2_SOURCE = find_dinov2('small')              # celda 4 del notebook: DINOV2_SOURCE = find_dinov2('small')
    test_ids = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    if not test_ids or len(set(test_ids)) != len(test_ids):
        raise RuntimeError('test.csv vacío o con UIDs repetidos')
    _RSNA_TEST_IDS = [str(u) for u in test_ids]
    DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) if _cuda_execution_probe(i)]
    if not DEVS:
        raise RuntimeError('ninguna GPU CUDA pasó la sonda de ejecución')
    TIME_BUDGET = float(args.deadline) - T0           # rsna_deadline compara contra T0 + TIME_BUDGET
    log(f'dinov2: {len(_RSNA_TEST_IDS)} estudios, {len(DEVS)} GPU, plazo en {args.deadline - time.time():.0f} s')
    # Lo que hace la celda 15 del notebook:
    rsna_phase('dinov2', 'START')
    main()
    log('done')
    _speed_dino_templates.clear()
    gc.collect()
    rsna_phase('dinov2', 'COMPLETE')
    if _DINOV2_MATCHED_MEMBERS != EXPECTED_DINOV2_MEMBERS:       # la compuerta de la celda 16
        raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')
    stage = out / '_pipeline_stage.csv'
    sub = _v37_validate_submission(stage, pd.read_csv(ROOT / 'test.csv'), 'dinov2 salida')
    os.replace(stage, out / 'dinov2_sub.csv')
    vacios = sorted({str(s) for e in _RSNA_AUDIT['events'] if e.get('kind') == 'empty_study_rows' for s in e.get('studies', [])})
    kinds = {}
    for e in _RSNA_AUDIT['events']:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    informe = {'estudios': len(sub), 'miembros': _DINOV2_MATCHED_MEMBERS, 'gpus': len(DEVS),
               'segundos': round(time.time() - T0, 1), 'estudios_sin_series': vacios,
               'series_con_fallo_de_decodificacion': len(set(map(str, DECODE_FAILED))), 'eventos': kinds}
    (out / 'informe.json').write_text(json.dumps(informe, indent=1, sort_keys=True) + '\n')
    print('[dinov2] informe', json.dumps(informe, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(_glue_main())
