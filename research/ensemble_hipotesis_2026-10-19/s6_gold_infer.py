"""BORRADOR, NO EJECUTADO: S6 (original + destilados) sobre Gold-58, con eval_stride 1 y 3, en un solo paso por stride (H1 + H5).

Por qué así: compact_infer.py (Heliosli, sha 34f433fe…, sin cambios) ya carga TODOS los <arm>_s*_f*.pt de --models, los corre
sobre las MISMAS ventanas (una sola decodificación DICOM y un solo anclaje SKM-TEA por estudio) y, con --dump-logits, guarda
el logit de CADA modelo por estudio.  Una ejecución por stride da S6 original, d42, d7 y cualquier combinación.
Los destilados se llaman igual que el original (E_prior_s42_fall.pt): aquí se enlazan con semillas ficticias distintas
(s42 = original; s1042 = destilado semilla 42; s1007 = destilado semilla 7) para que el glob no los confunda.

Geometría = B_GEOMETRY de s6_original.py en el notebook (convnext_small, 320/288, v4, 3 cortes, 3.5 mm, chunk 192, fp16 en
T4; aquí bf16/fp16 según la GPU, como el portátil de nartaa).  Lanzador = la constante LAUNCHER de esa misma celda, extraída
del notebook E-sust-dsj (no se reescribe).

Salida (en --out, carpeta NUEVA):
  stride{1,3}/predictions.csv, logits.json, receipt de compact_infer
  s6_gold_stride{1,3}.npz  -> claves ids, s6, s6_d42, s6_d7 (probabilidades por modelo) para evaluar_gold.py --componentes
  coste.json               -> ventanas y segundos de GPU por estudio y stride (H5)

Requiere GPU (compact_infer usa torch.device('cuda')), los checkpoints de 2093337 y las anclas de 2092983.  Ver el informe.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

PROJ = Path("/lustre/proyectos/p082")
EXP = PROJ / "runs_joako/exp_s6_semillas"
NB = PROJ / "runs_joako/exp_Esust_dsj/Esust_dsj_estricto.ipynb"
SRC = EXP / "entradas/s6_extras_extraido/heliosli"
CODE = EXP / "entradas/orthofoundation_v1/cloud_code"
ANATOMY = EXP / "entradas/orthofoundation_v1/anatomy.pt"
DATA = PROJ / "kaggle_data/competitions/rsna-knee-abnormality-detection"
GOLD = EXP / "S6F0/data/gold58.json"
ORIGINAL = EXP / "entradas/s6fall_v5/E_prior_s42_fall.pt"
SHA_ORIGINAL = "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
GEOM = {"backbone": "convnext_small", "size": 320, "crop": 288, "views": "v4", "slices": 3, "phys_mm": 3.5, "chunk": 192,
        "item_timeout": 900}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def launcher(out):
    nb = json.loads(NB.read_text())
    src = "".join(nb["cells"][22]["source"])
    m = re.search(r"\nLAUNCHER = r'''(.*?)'''\n", src, re.S)
    if not m:
        raise SystemExit("no encuentro LAUNCHER en la celda 22 del notebook")
    p = out / "k1_b_infer.py"
    p.write_text(m.group(1).replace("__SRC__", repr(str(SRC))).replace("__CODE__", repr(str(CODE))))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="carpeta NUEVA (runs_joako/<job>)")
    ap.add_argument("--d42", required=True, help="runs_joako/2093337_0/train/models/E_prior_s42_fall.pt")
    ap.add_argument("--d7", required=True, help="runs_joako/2093337_1/train/models/E_prior_s7_fall.pt")
    ap.add_argument("--s2026", default="", help="runs_joako/2098694_0/train/models/E_prior_s2026_fall.pt (opcional)")
    ap.add_argument("--s1234", default="", help="runs_joako/2098694_1/train/models/E_prior_s1234_fall.pt (opcional)")
    ap.add_argument("--d314", default="", help="runs_joako/2099380_0/train/models/E_prior_s314_fall.pt (destilado, opcional)")
    ap.add_argument("--d2718", default="", help="runs_joako/2099380_1/train/models/E_prior_s2718_fall.pt (destilado, opcional)")
    ap.add_argument("--variantes", default="1:288,3:288,1:320",
                    help="stride:crop separados por comas; 1:288 = notebook; 3:288 = H5; 1:320 = imagen completa (campo de visión)")
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--limite", type=int, default=0, help="> 0: solo los primeros N estudios de Gold (humo)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if sha(ORIGINAL) != SHA_ORIGINAL:
        raise SystemExit("sha del S6 original distinto")
    modelos = out / "modelos"
    modelos.mkdir()
    mapa = {"s6": ("E_prior_s42_fall.pt", ORIGINAL), "s6_d42": ("E_prior_s1042_fall.pt", Path(a.d42)),
            "s6_d7": ("E_prior_s1007_fall.pt", Path(a.d7))}
    if a.s2026:
        mapa["s6_s2026"] = ("E_prior_s2026_fall.pt", Path(a.s2026))
    if a.s1234:
        mapa["s6_s1234"] = ("E_prior_s1234_fall.pt", Path(a.s1234))
    if a.d314:
        mapa["s6_d314"] = ("E_prior_s1314_fall.pt", Path(a.d314))
    if a.d2718:
        mapa["s6_d2718"] = ("E_prior_s12718_fall.pt", Path(a.d2718))
    for k, (nombre, p) in mapa.items():
        os.symlink(p, modelos / nombre)
    orden = sorted(n for n, _ in mapa.values())                 # = sorted(glob) de compact_infer.load_group
    idx = {k: orden.index(n) for k, (n, _) in mapa.items()}
    gold = json.loads(GOLD.read_text())
    if a.limite:
        gold = gold[:a.limite]
    (out / "gold.json").write_text(json.dumps(gold))
    lanz = launcher(out)
    info = {"modelos": {k: {"archivo": n, "origen": str(p), "sha256": sha(p), "indice_en_dump": idx[k]} for k, (n, p) in mapa.items()},
            "geometria": GEOM, "amp": a.amp, "n": len(gold)}
    coste = {}
    for var in a.variantes.split(","):
        st, cr = (int(x) for x in var.split(":"))          # «stride:crop»; 1:288 = la configuración del notebook
        nombre = f"stride{st}" + ("" if cr == GEOM["crop"] else f"_crop{cr}")
        d = out / nombre
        cmd = [sys.executable, "-u", str(lanz), "--data", str(DATA), "--split", "train", "--code", str(CODE),
               "--models", str(modelos), "--arm", "E_prior", "--backbone", GEOM["backbone"], "--size", str(GEOM["size"]),
               "--crop", str(cr), "--views", GEOM["views"], "--slices", str(GEOM["slices"]),
               "--phys-mm", str(GEOM["phys_mm"]), "--eval-stride", str(st), "--anchor-weights", str(ANATOMY),
               "--amp", a.amp, "--chunk", str(GEOM["chunk"]), "--workers", "4", "--item-timeout", str(GEOM["item_timeout"]),
               "--assignment", str(out / "gold.json"), "--output", str(d), "--tmp", f"/tmp/s6gold_{os.getpid()}_{nombre}",
               "--dump-logits", str(d / "logits.json")]
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:                              # una variante que no corre (p. ej. crop 320) no tumba las demás
            coste[nombre] = {"error": f"compact_infer código {r.returncode}"}
            continue
        dump = json.loads((d / "logits.json").read_text())
        falta = [u for u in gold if u not in dump]
        if falta:
            coste[nombre] = {"error": f"{len(falta)} estudios sin logits: {falta[:3]}"}
            continue
        Z = np.array([dump[u] for u in gold], dtype=np.float64)          # (n, modelos, 12)
        P = 1 / (1 + np.exp(-Z))
        np.savez(out / f"s6_gold_{nombre}.npz", ids=np.array(gold), **{k: P[:, i, :] for k, i in idx.items()})
        coste[nombre] = {"segundos_totales": round(time.time() - t0, 1)}
        for f in d.glob("*.json"):
            if f.name != "logits.json":
                coste[nombre]["receipt_" + f.name] = json.loads(f.read_text()).get("stats")
    (out / "info.json").write_text(json.dumps(info, indent=1))
    (out / "coste.json").write_text(json.dumps(coste, indent=1))
    print("S6_GOLD_OK", json.dumps(coste))


if __name__ == "__main__":
    main()
