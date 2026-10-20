"""Prueba local (CPU, sin GPU, sin Kaggle) del notebook gold_dump.ipynb: todo lo que se puede comprobar sin correr los modelos.

  1. Todas las celdas de código compilan (sin las líneas mágicas %%/!).
  2. G0 se ejecuta TAL CUAL contra una raíz /kaggle falsa (en el scratchpad) cuyo input/competitions/<slug> enlaza a
     kaggle_data (solo lectura): crea la raíz Gold, 58 estudios, 336 series, todos los DICOM enlazados; el guardarraíl del
     test oculto salta con un test.csv falso de 1,284 filas.
  3. [c0] c_competicion() (extraída por AST, sin ejecutar la limpieza de la celda 0) devuelve la raíz Gold, 58 UID, 13 columnas.
  4. [A1]/[A1b] la celda de parche reescribe una copia del script Raptor y su find_test_root() devuelve la raíz Gold.
  5. [S6] los PARCHES de la etapa 4b (con el añadido [GOLD]) se aplican al script de nartaa tras su comprobación sha256.
  6. [B] el lector recibe la raíz Gold y su preprocess.load_series decodifica una serie Gold a través de los enlaces.
  7. [dsj] meta_rasgos.leer_series sobre la raíz Gold da la MISMA puntuación de metadatos que evaluar_gold.py (parquet).
Uso: python -I -B prueba_gold.py <dir de trabajo NUEVO en el scratchpad>
"""
import ast, hashlib, json, os, sys, types
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
PROJ = Path("/lustre/proyectos/p082")
NB = PROJ / "runs_joako/exp_gold_dump/gold_dump.ipynb"
COMP = PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection"
T = Path(sys.argv[1])
T.mkdir(parents=True, exist_ok=False)
nb = json.loads(NB.read_text())
cs = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
cel = lambda pref: [s for s in cs if s.startswith(pref)][0]
res = {}

# 1. compilación
for i, s in enumerate(cs):
    limpio = (s.split("\n", 1)[1] if s.startswith("%%writefile") else
              "\n".join("" if l.startswith(("%", "!")) else l for l in s.splitlines()))
    compile(limpio, f"celda{i}", "exec")
res["1_compilan"] = len(cs)

# 2. G0 tal cual contra /kaggle falso
K = T / "kaggle"
(K / "input/competitions").mkdir(parents=True)
os.symlink(COMP, K / "input/competitions/rsna-knee-abnormality-detection")
os.environ.update(C_GOLD_KAGGLE=str(K), C_GOLD_DIR=str(T / "gold58_comp"))
g = {}
exec(cel("# [GOLD] Celda G0"), g)
GR = Path(os.environ["C_GOLD_ROOT"])
nlinks = sum(1 for _ in GR.glob("test_series/*/*/*"))
res["2_G0"] = dict(g["GOLD_INFO"], enlaces_en_disco=nlinks, raiz=str(GR))
assert g["GOLD_INFO"]["estudios"] == 58 and g["GOLD_INFO"]["series"] == 336 and nlinks == g["GOLD_INFO"]["dicom_enlazados"] > 10000
#    guardarraíl: test oculto (1,284 filas) -> se niega
K2 = T / "kaggle_oculto/input/competitions/rsna-knee-abnormality-detection"
K2.mkdir(parents=True)
pd.DataFrame({"StudyInstanceUID": [str(i) for i in range(1284)]}).to_csv(K2 / "test.csv", index=False)
os.environ["C_GOLD_KAGGLE"] = str(T / "kaggle_oculto")
try:
    exec(cel("# [GOLD] Celda G0"), {})
    raise AssertionError("G0 no se negó con el test oculto")
except RuntimeError as e:
    res["2_guardarrail_test_oculto"] = str(e)[:90]
os.environ["C_GOLD_KAGGLE"] = str(K)

# 3. c_competicion por AST (sin la limpieza de la celda 0)
c0 = cel("# [C] Celda 0")
mod = ast.parse(c0)
defs = [n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in ("c_borrar", "c_fallo", "c_competicion")]
g0 = {"Path": Path, "os": os, "json": json, "shutil": __import__("shutil"), "C_WORK": T / "work", "C_ESTADO": {}}
exec(compile(ast.Module(body=defs, type_ignores=[]), "c0", "exec"), g0)
comp, ids, cols = g0["c_competicion"]()
assert Path(comp) == GR and sorted(ids) == sorted(g["GOLD_IDS"]) and len(cols) == 13
res["3_c_competicion"] = {"raiz": str(comp), "ids": len(ids), "columnas": len(cols)}

# 4. parches Raptor
for nombre, pref in (("raptor_run.py", "%%writefile /tmp/raptor_run.py"), ("raptor224_run.py", "%%writefile /tmp/raptor224_run.py")):
    script = T / nombre
    script.write_text(cel(pref).split("\n", 1)[1])
    parche = cel(f"# [GOLD] {nombre}").replace(f'"/tmp/{nombre}"', repr(str(script)))
    exec(parche, {})
    txt = script.read_text()
    fn = [n for n in ast.parse(txt).body if isinstance(n, ast.FunctionDef) and n.name == "find_test_root"][0]
    gg = {"os": os}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), nombre, "exec"), gg)
    assert gg["find_test_root"]() == str(GR)
    for f in ("test.csv", "test_series.csv", "sample_submission.csv"):
        assert (GR / f).is_file()
    res["4_" + nombre] = gg["find_test_root"]()

# 5. S6: sha256 del script de nartaa + PARCHES (con el [GOLD])
s6 = cel("%%writefile /tmp/s6_original.py").split("\n", 1)[1]
c4b = cel("# Stage 4b (estricta)")
cab = c4b.split("LIMITE_KAGGLE_H, MARGEN_H")[0]
gs = {}
exec(cab, gs)                                  # solo imports, S6_SHA y PARCHES (+ el append [GOLD])
assert hashlib.sha256(s6.rstrip("\n").encode()).hexdigest() == gs["S6_SHA"]
t = s6
for a, b in gs["PARCHES"]:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
assert '    I.comp = Path(os.environ["C_GOLD_ROOT"])\n' in t and 'find_mount(COMP, "test.csv")' not in t
compile(t, "s6_run.py", "exec")
res["5_s6"] = {"sha_nartaa_ok": True, "parches": len(gs["PARCHES"])}

# 6. lector: raíz y una serie decodificada a través de los enlaces
lect = cel("import time as _g_time")
assert '_o_os.environ["C_GOLD_ROOT"]' in lect and '_o_find("test_series.csv")' not in lect
pre = cel("%%writefile /kaggle/working/own_src/preprocess.py").split("\n", 1)[1]
gp = {"__name__": "preprocess_prueba"}
exec(pre, gp)
ser = pd.read_csv(GR / "test_series.csv", dtype=str).iloc[0]
vol = gp["load_series"](f"{GR}/test_series/{ser.StudyInstanceUID}/{ser.SeriesInstanceUID}", ser.Anatomical_Plane)
v = vol[0] if isinstance(vol, tuple) else vol
res["6_lector_serie"] = {"forma": list(np.asarray(v).shape), "media": round(float(np.asarray(v).mean()), 2)}
assert np.asarray(v).size > 0 and float(np.asarray(v).std()) > 0

# 7. dsj: misma puntuación de metadatos por la raíz Gold que desde el parquet de cabeceras
sys.path.insert(0, str(PROJ / "runs_joako/exp_Esust_dsj"))
import meta_rasgos as MR
mod_dsj = json.loads((PROJ / "runs_joako/exp_Esust_dsj/desatajo_synovitis.json").read_text())
m = MR.leer_series(GR, "test", GR / "test_series.csv", workers=4)
X = MR.rasgos(m, mod_dsj["vocab"], mod_dsj["bobinas"], mod_dsj["modelos"])
s_raiz = pd.Series(MR.puntuar(X, mod_dsj, "Synovitis"), index=X.index).loc[g["GOLD_IDS"]]
sys.path.insert(0, str(Path(__file__).parent))
import evaluar_gold as EG
s_parq = pd.Series(EG.meta_gold(g["GOLD_IDS"]), index=g["GOLD_IDS"])
d = float(np.abs(s_raiz.values - s_parq.values).max())
rho = float(pd.Series(s_raiz.values).corr(pd.Series(s_parq.values), method="spearman"))
res["7_dsj_meta"] = {"max_dif": d, "spearman": round(rho, 6)}

(T / "prueba_gold.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
print("PRUEBA_GOLD", "OK" if d < 1e-6 else "AVISO meta distinta (ver 7)")
