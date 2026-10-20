"""Construye gold_dump.ipynb = E-sust-dsj con los 58 estudios de Gold como «test» (para medir H1–H4 sobre la mezcla completa).

NO es un envío.  Se ejecuta como «Save & Run All» privado en Kaggle (T4) y se descargan sus salidas (/kaggle/working/gold_dump/).
Base: runs_joako/exp_Esust_dsj/Esust_dsj_estricto.ipynb (sha256 fijado abajo).  Modelos, pesos, recetas y celdas de cada
componente sin cambios; solo se redirige DÓNDE se lee el «test»:

  G0 (celda nueva, antes de la celda 0 [C]):
     * se niega a correr si el test.csv de la competición tiene > 3 estudios (= ejecución de envío sobre el test oculto);
     * crea /tmp/gold58_comp con test.csv (58 UID), sample_submission.csv (0.5), test_series.csv (filas de train_series.csv)
       y test_series/<estudio>/<serie>/ como CARPETAS REALES con un enlace simbólico por DICOM (os.walk no sigue enlaces a
       carpetas por defecto); comprueba que los 58 UID son las filas con las 12 etiquetas de train.csv;
     * exporta C_GOLD_ROOT.
  Parches de texto (cada uno se exige exactamente 1 vez):
     [c0]   c_competicion() -> solo C_GOLD_ROOT
     [A1]   celda nueva tras %%writefile raptor_run.py: find_test_root() mira primero C_GOLD_ROOT
     [A1b]  ídem raptor224_run.py
     [A2]   celda nueva tras la de activos de R6: DATA = C_GOLD_ROOT
     [B]    lector: _o_find("test_series.csv") -> C_GOLD_ROOT
     [S6]   se añade a PARCHES de la etapa 4b (se aplica DESPUÉS de la comprobación sha256 del script de nartaa):
            I.comp = find_mount(COMP, "test.csv") -> I.comp = Path(C_GOLD_ROOT)
     [dsj]  usa c_competicion() -> ya redirigido por [c0]
  Celda final nueva: copia los intermedios a /kaggle/working/gold_dump/ (raptor_raw, raptor224_sub, ours_sub, _own, s6_sub,
     submission_a3, submission_0958, la salida E-sust-dsj, recibos y la puntuación de metadatos), y BORRA
     /kaggle/working/submission.csv.
Salida: runs_joako/exp_gold_dump/{gold_dump.ipynb, kaggle_push_gold/{gold_dump.ipynb, kernel-metadata.json}}
Uso: python -I construir_notebook_gold.py
"""
import copy, hashlib, json, re
from pathlib import Path

PROJ = Path("/lustre/proyectos/p082")
BASE = PROJ / "runs_joako/exp_Esust_dsj/Esust_dsj_estricto.ipynb"
BASE_META = PROJ / "runs_joako/exp_Esust_dsj/kaggle_push_Esust_dsj/kernel-metadata.json"
GOLD = PROJ / "runs_joako/exp_s6_semillas/S6F0/data/gold58.json"
OUT = PROJ / "runs_joako/exp_gold_dump"

raw = BASE.read_bytes()
SHA_BASE = hashlib.sha256(raw).hexdigest()
nb = json.loads(raw)
cells = nb["cells"]
src = lambda c: "".join(c["source"])
gold = json.loads(GOLD.read_text())
assert len(gold) == 58 == len(set(gold))


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def una_vez(texto, viejo, nuevo, donde):
    n = texto.count(viejo)
    assert n == 1, f"{donde}: '{viejo[:70]}' aparece {n} veces"
    return texto.replace(viejo, nuevo)


def buscar(prefijo):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(c).startswith(prefijo)]
    assert len(hits) == 1, (prefijo, hits)
    return hits[0]


i_c0 = buscar("# [C] Celda 0")
i_rap = buscar("%%writefile /tmp/raptor_run.py")
i_r224 = buscar("%%writefile /tmp/raptor224_run.py")
i_r6 = buscar("ARM = 'E_lrprior'")
i_lect = buscar("import time as _g_time")
i_s6 = buscar("# Stage 4b (estricta)")
i_fin = buscar("# Stage 5 (estricta)")
assert src(cells[buscar("%%writefile /tmp/s6_original.py")]).count('    I.comp = find_mount(COMP, "test.csv")\n') == 1

G0 = '''# [GOLD] Celda G0 (equipo p082): los 58 estudios de Gold como «test».  NO ES UN ENVÍO.
# Crea /tmp/gold58_comp con la forma de la carpeta de test de la competición; las etapas siguientes leen de ahí (C_GOLD_ROOT).
import json as _gj, os as _gos, shutil as _gsh, time as _gt
from pathlib import Path as _GP
import pandas as _gpd
_g_t0 = _gt.time()
_GK = _GP(_gos.environ.get("C_GOLD_KAGGLE", "/kaggle"))          # solo para la prueba local; en Kaggle no se define
_g_real = [p for p in (_GK / "input/competitions/rsna-knee-abnormality-detection", _GK / "input/rsna-knee-abnormality-detection")
           if (p / "test.csv").is_file()]
if len(_g_real) != 1:
    raise RuntimeError("GOLD: no se resuelve la carpeta de la competición: %s" % _g_real)
_g_real = _g_real[0]
_g_nt = len(_gpd.read_csv(_g_real / "test.csv"))
if _g_nt > 3:
    raise RuntimeError("GOLD: test oculto detectado (%d estudios). Este notebook mide Gold-58 y NO se envía." % _g_nt)
GOLD_IDS = __GOLD__
_g_tr = _gpd.read_csv(_g_real / "train.csv", dtype={"StudyInstanceUID": str})
_g_lab = list(_gpd.read_csv(_g_real / "sample_submission.csv", nrows=1).columns)
if set(_g_tr.dropna(subset=_g_lab[1:])["StudyInstanceUID"]) != set(GOLD_IDS):
    raise RuntimeError("GOLD: los 58 UID no son las filas de train.csv con las 12 etiquetas")
_g_ser = _gpd.read_csv(_g_real / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
_g_ser = _g_ser[_g_ser["StudyInstanceUID"].isin(GOLD_IDS)].reset_index(drop=True)
GOLD_ROOT = _GP(_gos.environ.get("C_GOLD_DIR", "/tmp/gold58_comp"))
if GOLD_ROOT.exists():
    _gsh.rmtree(GOLD_ROOT)
(GOLD_ROOT / "test_series").mkdir(parents=True)
_gpd.DataFrame({"StudyInstanceUID": GOLD_IDS}).to_csv(GOLD_ROOT / "test.csv", index=False)
_g_ss = _gpd.DataFrame({"StudyInstanceUID": GOLD_IDS})
for _c in _g_lab[1:]:
    _g_ss[_c] = 0.5
_g_ss[_g_lab].to_csv(GOLD_ROOT / "sample_submission.csv", index=False)
_g_ser.to_csv(GOLD_ROOT / "test_series.csv", index=False)
_g_nf = 0
for _u, _s in zip(_g_ser["StudyInstanceUID"], _g_ser["SeriesInstanceUID"]):
    _src = _g_real / "train_series" / _u / _s
    if not _src.is_dir():
        raise RuntimeError("GOLD: falta la carpeta de la serie %s/%s" % (_u, _s))
    _dst = GOLD_ROOT / "test_series" / _u / _s
    _dst.mkdir(parents=True)
    for _f in sorted(_src.iterdir()):
        _gos.symlink(_f, _dst / _f.name)
        _g_nf += 1
for _u in GOLD_IDS:
    if set(_gos.listdir(_g_real / "train_series" / _u)) != set(_gos.listdir(GOLD_ROOT / "test_series" / _u)):
        raise RuntimeError("GOLD: series distintas en %s" % _u)
_gos.environ["C_GOLD_ROOT"] = str(GOLD_ROOT)
GOLD_INFO = {"estudios": len(GOLD_IDS), "series": len(_g_ser), "dicom_enlazados": _g_nf, "origen": str(_g_real),
             "segundos": round(_gt.time() - _g_t0, 1)}
print("GOLD_ROOT_OK", _gj.dumps(GOLD_INFO), flush=True)
'''.replace("__GOLD__", json.dumps(gold))

PARCHE_RAPTOR = '''# [GOLD] {nombre}: find_test_root() mira primero C_GOLD_ROOT (parche de texto exacto sobre el script recién escrito).
from pathlib import Path as _GP
_p = _GP("{ruta}")
_t = _p.read_text()
_v = '    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",\\n'
_n = '    cands = [os.environ["C_GOLD_ROOT"], "/kaggle/input/competitions/rsna-knee-abnormality-detection",\\n'
if _t.count(_v) != 1 or "import os" not in _t:
    raise RuntimeError("GOLD: parche de {nombre} no aplicable")
_p.write_text(_t.replace(_v, _n))
print("GOLD_PARCHE_OK {nombre}", flush=True)
'''

PARCHE_R6 = '''# [GOLD] R6: el «test» es Gold-58 (la celda anterior localizó la competición real; aquí se redirige).
import os as _gos
from pathlib import Path as _GP
DATA = _GP(_gos.environ["C_GOLD_ROOT"])
assert (DATA / "test.csv").is_file() and (DATA / "test_series").is_dir()
print("GOLD_R6_DATA", DATA, flush=True)
'''

FINAL = '''# [GOLD] Volcado de todos los componentes sobre Gold-58 y retirada de submission.csv (este notebook no se envía).
import json as _gj, os as _gos, shutil as _gsh
from pathlib import Path as _GP
_D = _GP("/kaggle/working/gold_dump")
if _D.exists():
    _gsh.rmtree(_D)
_D.mkdir()
_W = _GP("/kaggle/working")
_copiados = {}
for _n in ("raptor_raw.csv", "raptor_sub.csv", "raptor224_sub.csv", "ours_sub.csv", "_own.csv", "s6_sub.csv", "submission_a3.csv",
           "submission_0958.csv", "blend_receipt.json", "submission_receipt.json", "e_base_receipt.json", "c_receipt.json",
           "r224_log.txt"):
    if (_W / _n).is_file():
        _gsh.copy2(_W / _n, _D / _n)
        _copiados[_n] = c_sha(_D / _n)
if (_W / "submission.csv").is_file():
    _gsh.copy2(_W / "submission.csv", _D / "esust_dsj_final.csv")
    _copiados["esust_dsj_final.csv"] = c_sha(_D / "esust_dsj_final.csv")
    _gos.remove(_W / "submission.csv")
if "DSJ_META" in globals():
    DSJ_META.rename("meta_synovitis").to_csv(_D / "meta_synovitis.csv", index_label="StudyInstanceUID")
for _p in _GP("/tmp/c_s6_root/working/k1").glob("*.json"):
    _gsh.copy2(_p, _D / ("s6_" + _p.name))
(_D / "gold_info.json").write_text(_gj.dumps({"gold": GOLD_INFO, "archivos_sha256": _copiados,
                                              "estado": C_ESTADO}, indent=1, default=str))
assert not (_W / "submission.csv").exists()
print("GOLD_DUMP_OK", sorted(_copiados), flush=True)
'''

new = copy.deepcopy(cells)
# parches dentro de celdas existentes
new[i_c0]["source"] = una_vez(src(new[i_c0]),
    '    cands = [p for p in (Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),\n'
    '                         Path("/kaggle/input/rsna-knee-abnormality-detection")) if (p / "test.csv").is_file()]\n',
    '    cands = [p for p in (Path(os.environ["C_GOLD_ROOT"]),) if (p / "test.csv").is_file()]   # [GOLD]\n',
    "c0").splitlines(True)
new[i_lect]["source"] = una_vez(src(new[i_lect]), '_o_find("test_series.csv")', '_o_os.environ["C_GOLD_ROOT"]', "lector").splitlines(True)
new[i_s6]["source"] = una_vez(src(new[i_s6]), "LIMITE_KAGGLE_H, MARGEN_H = 9.0, 0.25\n",
    'PARCHES.append(["    I.comp = find_mount(COMP, \\"test.csv\\")\\n", "    I.comp = Path(os.environ[\\"C_GOLD_ROOT\\"])\\n"])   # [GOLD]\n'
    "LIMITE_KAGGLE_H, MARGEN_H = 9.0, 0.25\n", "s6").splitlines(True)
# celdas nuevas (de atrás hacia delante para no mover índices)
inserts = [(i_fin + 1, code(FINAL)),
           (i_r6 + 1, code(PARCHE_R6)),
           (i_r224 + 1, code(PARCHE_RAPTOR.format(nombre="raptor224_run.py", ruta="/tmp/raptor224_run.py"))),
           (i_rap + 1, code(PARCHE_RAPTOR.format(nombre="raptor_run.py", ruta="/tmp/raptor_run.py"))),
           (i_c0, code(G0)),
           (i_c0, md("## [GOLD] Gold-58 como test — NO ES UN ENVÍO\n\nCopia de E-sust-dsj (sha256 `%s…`) que corre los cinco "
                     "componentes sobre los 58 estudios de Gold de train y deja sus predicciones en `/kaggle/working/gold_dump/`. "
                     "Se niega a correr sobre el test oculto y borra `submission.csv` al final. Constructor: "
                     "`code_joako/research/ensemble_hipotesis_2026-10-19/construir_notebook_gold.py`.\n" % SHA_BASE[:12]))]
for pos, c in sorted(inserts, key=lambda x: -x[0]):
    new.insert(pos, c)
nb["cells"] = new
# comprobaciones: las celdas sin parche son byte a byte las de la base
orig = {src(c) for c in cells}
tocadas = {i_c0, i_lect, i_s6}
for i, c in enumerate(cells):
    if i not in tocadas:
        assert any(src(n) == src(c) for n in new), f"celda {i} cambió"
nuevas = [c for c in new if src(c) not in orig]
assert len(nuevas) == 6 + 3, len(nuevas)
OUT.mkdir(exist_ok=True)
(OUT / "kaggle_push_gold").mkdir(exist_ok=True)
txt = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
for p in (OUT / "gold_dump.ipynb", OUT / "kaggle_push_gold/gold_dump.ipynb"):
    p.write_text(txt)
meta = json.loads(BASE_META.read_text())
meta.update({"id": meta["id"].split("/")[0] + "/rsna-gold58-dump", "title": "RSNA gold58 dump", "code_file": "gold_dump.ipynb",
             "is_private": True})
(OUT / "kaggle_push_gold/kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
sha = hashlib.sha256(txt.encode()).hexdigest()
print("base", SHA_BASE[:16], "| gold_dump.ipynb", sha[:16], "| celdas", len(cells), "->", len(new), "| tocadas", sorted(tocadas))
