#!/usr/bin/env python
"""Tablas de objetivos para el piloto CoAtNet (receta del hilo 746792, mensaje #53, sin OAI).

Fuentes, todas por estudio y hallazgo:
  nuestra  = tabla híbrida (t_, w_; NaN con w = 0 donde nadie dice nada)
  v4       = stevenleehans llm_labels_v4_blend.csv; 0.25 exacto es su marca de "no abordado"
  jev      = riadmohamed42 JEV HYBRID_labels_scores.csv (denso: siempre tiene valor)

blend (acordado con jcastro; el #53 promedia sin pesos porque sus etiquetas propias son densas):
  nuestra y v4 con valor   -> t = media de t,  w = media de w      (w_v4 = 2·|p − 0.5|)
  solo nuestra (v4 = 0.25) -> t, w de la nuestra
  solo v4 (nuestra NaN)    -> t = p_v4, w = 2·|p_v4 − 0.5|
  ninguna                  -> NaN, w = 0
blendjev (#53: "mean of blend and JEV HYBRID wherever JEV has a value; otherwise blend"):
  blend y JEV con valor    -> t = (t_blend + jev) / 2, w = w_blend
  blend con valor, JEV NaN -> t, w de blend (JEV tiene 1 estudio sin ACL, MCL ni Fracture)
  blend NaN                -> sigue enmascarada (decisión nuestra: el JEV de esas celdas es el 0.25 de v4
                              reescalado, es decir, silencio; no se convierte en objetivo)
Baker's: toda celda que siga enmascarada en blendjev pasa a t = 0.05, w = 0.5 (negativo débil, pedido
por jcastro; en Gold P(positivo | silencio) = 1/29).

Salidas en --out (formato de rsna/entrenar.py: StudyInstanceUID, oficial, t_*, w_*, y_*):
  etiquetas_blend.parquet       juez "blend" (sin el relleno de Baker's)
  etiquetas_blendjev.parquet    objetivo de entrenamiento (con el relleno de Baker's)
  resumen_etiquetas.json        cobertura y origen de cada celda

Uso:  python rsna/etiquetas_coatnet.py --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path("/lustre/proyectos/p082")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
HIBRIDA = P / "runs_cons/datos/etiquetas_candidatas_v2_hibridas.parquet"
V1 = P / "preproc/etiquetas_v1_2079203.parquet"
V4 = P / "runs_cons/datos/RSNA Knee LLM Report Labels/llm_labels_v4_blend.csv"
JEV = P / "code_joako/referencias/rsna_research/jev/HYBRID_labels_scores.csv"
V4_SILENCIO = 0.25
BAKER_T, BAKER_W = 0.05, 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def externa(path: Path, index: pd.Index, nan_ok: bool = False) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    assert not d.StudyInstanceUID.duplicated().any(), f"UID duplicados en {path.name}"
    d = d.set_index("StudyInstanceUID")
    assert set(index) <= set(d.index), f"faltan estudios en {path.name}"
    d = d.loc[index, LABELS].astype("float64")
    assert nan_ok or d.notna().all().all(), f"{path.name} tiene NaN"
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    hy = pd.read_parquet(HIBRIDA).set_index("StudyInstanceUID")
    v1 = pd.read_parquet(V1).set_index("StudyInstanceUID")
    idx = v1.index
    hy = hy.loc[idx]
    v4, jev = externa(V4, idx), externa(JEV, idx, nan_ok=True)   # JEV: 1 estudio sin ACL, MCL ni Fracture

    base = pd.DataFrame(index=idx)
    base["oficial"] = v1.oficial
    blend, bjev = base.copy(), base.copy()
    resumen = {"fuentes": {p.name: sha(p) for p in (HIBRIDA, V1, V4, JEV)}, "por_hallazgo": {}}
    for c in LABELS:
        th, wh = hy[f"t_{c}"].astype("float64"), hy[f"w_{c}"].fillna(0).astype("float64")
        assert ((wh > 0) == th.notna()).all(), f"híbrida incoherente en {c}"
        p4 = v4[c]
        w4 = (2 * (p4 - 0.5).abs()).clip(0, 1)
        has_h = wh > 0
        has_4 = (p4 != V4_SILENCIO) & (w4 > 0)

        t = pd.Series(np.nan, index=idx)
        w = pd.Series(0.0, index=idx)
        both, only_h, only_4 = has_h & has_4, has_h & ~has_4, ~has_h & has_4
        t[both] = (th[both] + p4[both]) / 2
        w[both] = (wh[both] + w4[both]) / 2
        t[only_h], w[only_h] = th[only_h], wh[only_h]
        t[only_4], w[only_4] = p4[only_4], w4[only_4]
        blend[f"t_{c}"], blend[f"w_{c}"] = t.astype("float32"), w.astype("float32")

        has_b = w > 0
        tj = t.where(~(has_b & jev[c].notna()), (t + jev[c]) / 2)
        wj = w.copy()
        relleno = pd.Series(False, index=idx)
        if c == "Baker's":
            relleno = ~has_b
            tj[relleno], wj[relleno] = BAKER_T, BAKER_W
        bjev[f"t_{c}"], bjev[f"w_{c}"] = tj.astype("float32"), wj.astype("float32")

        for tab in (blend, bjev):
            tab[f"y_{c}"] = v1[f"y_{c}"]
        ng = ~base.oficial
        resumen["por_hallazgo"][c] = {
            "nuestra_y_v4": int((both & ng).sum()), "solo_nuestra": int((only_h & ng).sum()),
            "solo_v4": int((only_4 & ng).sum()), "enmascaradas_blend": int((~has_b & ng).sum()),
            "relleno_baker": int((relleno & ng).sum()),
            "cobertura_blendjev_%": round(100 * float(((wj > 0) & ng).sum()) / int(ng.sum()), 2),
            "t_medio_blendjev": round(float(tj[(wj > 0) & ng].mean()), 4),
        }

    for tab in (blend, bjev):
        for c in LABELS:
            assert ((tab[f"w_{c}"] > 0) == tab[f"t_{c}"].notna()).all(), c
            assert tab[f"t_{c}"].dropna().between(0, 1).all(), c
    args.out.mkdir(parents=True, exist_ok=True)
    blend.reset_index().to_parquet(args.out / "etiquetas_blend.parquet", index=False)
    bjev.reset_index().to_parquet(args.out / "etiquetas_blendjev.parquet", index=False)
    resumen["salidas"] = {n: sha(args.out / n) for n in ("etiquetas_blend.parquet", "etiquetas_blendjev.parquet")}
    (args.out / "resumen_etiquetas.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False))
    print(json.dumps(resumen, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
