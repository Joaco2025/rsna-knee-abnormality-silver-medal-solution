# Dos celdas para añadir AL FINAL de un notebook público (p. ej. rsna-versia-7) que ya haya escrito
# /kaggle/working/submission.csv. Añaden nuestra rama entrenada en Yuca y la mezclan por rangos.
# Antes: agregar como entrada del notebook el dataset privado con nuestro paquete (rsna/, pesos/, wheels/).

# %% ---------------------------------------------------------------------------------------------
# Celda 1: nuestra rama, en un subproceso aparte para que libere la memoria de la GPU al terminar.
import glob, os, shutil, subprocess, sys, time

# El límite de Kaggle son 9 h para TODO el notebook. Los notebooks públicos pueden gastar hasta ~8.3 h
# (el de R7 + Raptor tiene ese plazo interno), así que nuestra rama solo corre si queda tiempo y con un
# tope: si se pasa, se corta y se envía la mezcla pública sin tocar.
LIMITE_H = 8.7                      # margen bajo las 9 h de Kaggle
INICIO = globals().get("NB_T0")     # el notebook de R7 + Raptor guarda aquí su hora de inicio
if INICIO is None:
    INICIO = time.time() - float(os.environ.get("HORAS_YA_GASTADAS", "0")) * 3600
QUEDA = LIMITE_H * 3600 - (time.time() - INICIO)
print(f"tiempo restante para nuestra rama: {QUEDA / 3600:.2f} h")

def buscar(nombre):
    for nivel in range(1, 6):
        hits = glob.glob("/kaggle/input/" + "*/" * nivel + nombre)
        if hits:
            return hits
    return []

hits = buscar("rsna/inferir.py")
assert hits, "No se encuentra nuestro paquete: añade el dataset con rsna/, pesos/ y wheels/ como entrada"
YUCA = os.path.dirname(os.path.dirname(hits[0]))
hits = buscar("test_series.csv")
assert hits, "No se encuentran los datos de la competencia"
DATA = os.path.dirname(hits[0])

# Se guarda la mezcla pública tal cual, para poder comparar y para no perderla si algo falla después.
shutil.copy("/kaggle/working/submission.csv", "/kaggle/working/submission_publica.csv")

# Lectores de DICOM, sin red. Si el notebook público ya los instaló, esto no cambia nada.
ruedas = sorted({d for n in range(1, 6) for d in glob.glob("/kaggle/input/" + "*/" * n + "wheels")})
cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps"]
for d in ruedas:
    cmd += ["--find-links", d]
subprocess.run(cmd + ["pydicom==3.0.2", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)

CKPTS = sorted(glob.glob(f"{YUCA}/pesos/*.pt"))
print(f"{len(CKPTS)} modelos propios desde {YUCA}; datos en {DATA}")
YUCA_OK = False
if QUEDA < 0.5 * 3600:
    print("no queda tiempo suficiente: se envía la mezcla pública tal cual")
else:
    try:
        subprocess.run([sys.executable, f"{YUCA}/rsna/inferir.py", "--data", DATA, "--split", "test",
                        "--checkpoints", *CKPTS, "--out", "/kaggle/working/yuca_sub.csv",
                        "--batch", "4", "--workers", str(min(4, os.cpu_count() or 2))],
                       check=True, timeout=QUEDA - 600)
        YUCA_OK = True
    except Exception as exc:     # tiempo agotado o fallo: la mezcla pública queda como envío
        print(f"nuestra rama no terminó ({type(exc).__name__}); se envía la mezcla pública tal cual")

# %% ---------------------------------------------------------------------------------------------
# Celda 2: mezcla por rangos, con un peso de nuestra rama distinto para cada hallazgo.
import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Peso de nuestra rama: más donde rinde mejor en los 58 estudios oficiales, cero donde flojea.
# Son valores de partida, no medidos contra los modelos públicos: lo que decide es la tabla de Kaggle.
PESO_YUCA = {"ACL": 0.18, "Baker's": 0.18, "Medial OA": 0.16, "MCL": 0.15, "Medial Meniscus": 0.15,
             "Effusion": 0.15, "Fracture": 0.10, "Contusion": 0.08, "PF OA": 0.05, "Lateral Meniscus": 0.04,
             "Lateral OA": 0.00, "Synovitis": 0.00}

pub = pd.read_csv("/kaggle/working/submission_publica.csv", dtype={"StudyInstanceUID": str})
if not globals().get("YUCA_OK", False) or not os.path.exists("/kaggle/working/yuca_sub.csv"):
    pub.to_csv("/kaggle/working/submission.csv", index=False)
    raise SystemExit("sin rama propia: queda la mezcla pública")
yuca = pd.read_csv("/kaggle/working/yuca_sub.csv", dtype={"StudyInstanceUID": str})
yuca = yuca.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID").reindex(pub["StudyInstanceUID"]).reset_index()

# Estudios que nuestra rama no pudo leer (deja 0.5 en todo) o que le faltan: ahí manda solo la rama pública.
sin_yuca = yuca[LABELS].isna().any(axis=1) | (yuca[LABELS] == 0.5).all(axis=1)
print("estudios:", len(pub), "| sin predicción propia:", int(sin_yuca.sum()))

r_pub = pub[LABELS].rank(method="average", pct=True)
r_yuca = yuca[LABELS].rank(method="average", pct=True)
final = pub.copy()
for col in LABELS:
    w = np.where(sin_yuca, 0.0, PESO_YUCA[col])
    final[col] = (1.0 - w) * r_pub[col] + w * r_yuca[col].fillna(0.0)
final[LABELS] = final[LABELS].rank(method="average", pct=True)

ref = pd.read_csv(f"{DATA}/sample_submission.csv")
assert list(final.columns) == list(ref.columns), "las columnas no coinciden con sample_submission.csv"
assert len(final) == len(pd.read_csv(f"{DATA}/test.csv")), "falta algún estudio"
assert np.isfinite(final[LABELS].to_numpy()).all() and ((final[LABELS] >= 0) & (final[LABELS] <= 1)).all().all()
final.to_csv("/kaggle/working/submission.csv", index=False)
print("mezcla pública + propia escrita:", final.shape)
