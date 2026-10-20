"""Reconstrucción pura (numpy/pandas) de las mezclas E, E-sust, E-sust-dsj y de las variantes H2/H3.

Copia literal de la lógica del notebook (celda A3 «Stage 3», Part B modo «end», celda final de E y `final_esust.py`):
  pct(v)       = rango medio / n                      (pandas rank(method='average', pct=True))
  rr           = (1 - A224) · pct(R384) + A224 · pct(R224)                    sin volver a rankear
  [E-sust]  rr = (1 - ws) · rr + ws · pct(S6)                                 sin volver a rankear
  b            = wr_c · rr + (1 - wr_c) · pct(R6)                             wr_c = W_RAPTOR_C.get(c, 0.5)
  base         = rank( (1 - w_c) · rank(b) + w_c · rank(lector) )             w_c = OWN_W_PER_TARGET[c]
  [E]   final  = rank( 0.75 · rank(base) + 0.25 · rank(S6) )
  [dsj] Syn    = rank( Syn - 0.1 · rank(meta) )
Sin fallos de componentes (rap_fail = our_fail = vacío): el caso de Gold-58 y de las pruebas usadas para la paridad.

Variantes (una sola variable cada una):
  h2='uniforme'   rr -> pct(rr) antes de mezclar con R6
  h2='dispersion' rr de E-sust reescalado (alrededor de su media) a la std de rr sin S6 (punto ajustado por heliosli)
  ws_c            peso de S6 por hallazgo dentro de la mitad Raptor (H3); ws escalar si no se da
  s6 puede ser una matriz ya combinada (H1: media de logits o de rangos de varios checkpoints; ver combinar_s6)
"""
import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
A224 = 0.35
W_RAPTOR = 0.5
W_RAPTOR_C = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
OWN_W_PER_TARGET = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02,
                    "MCL": 0.02, "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008,
                    "Effusion": 0.004, "Synovitis": 0.003}
W_S6 = 0.25
LAMBDA_DSJ = 0.1


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method="average", pct=True).values


def _col(M, j):
    return np.asarray(M, dtype=np.float64)[:, j]


def bloque_raptor(r384, r224, j, s6=None, ws=0.0, h2=None):
    """rr de la columna j.  h2: None (receta), 'uniforme' o 'dispersion'."""
    rr0 = (1.0 - A224) * pct(_col(r384, j)) + A224 * pct(_col(r224, j))
    rr = rr0
    if s6 is not None and ws > 0:
        rr = (1.0 - ws) * rr0 + ws * pct(_col(s6, j))
    if h2 == "uniforme":
        rr = pct(rr)
    elif h2 == "dispersion":
        sd0, sd = rr0.std(), rr.std()
        rr = rr.mean() + (rr - rr.mean()) * (sd0 / sd if sd > 0 else 1.0)
    elif h2 is not None:
        raise ValueError(h2)
    return rr


def mezcla_a3(r384, r224, r6, s6=None, ws=0.0, ws_c=None, h2=None):
    n = len(np.asarray(r384))
    out = np.zeros((n, len(LAB)))
    for j, c in enumerate(LAB):
        w_s6 = ws if ws_c is None else ws_c.get(c, ws)
        rr = bloque_raptor(r384, r224, j, s6, w_s6, h2)
        wr = W_RAPTOR_C.get(c, W_RAPTOR)
        out[:, j] = wr * rr + (1.0 - wr) * pct(_col(r6, j))
    return out


def parte_b(a3, lector):
    a3 = np.asarray(a3, dtype=np.float32)            # el notebook guarda A3 como float32 antes de rankear
    out = np.zeros(a3.shape)
    for j, c in enumerate(LAB):
        w = OWN_W_PER_TARGET.get(c, 0.15)
        out[:, j] = pct((1 - w) * pct(a3[:, j]) + w * pct(_col(lector, j)))
    return out


def receta(nombre, C, ws_c=None, h2=None, s6_key="s6"):
    """C: dict de matrices (n, 12): r384, r224, r6, lector, s6 y meta (vector n de la puntuación de metadatos)."""
    s6 = C.get(s6_key)
    if nombre == "base0958":
        return parte_b(mezcla_a3(C["r384"], C["r224"], C["r6"], h2=h2), C["lector"])
    if nombre == "E":
        base = parte_b(mezcla_a3(C["r384"], C["r224"], C["r6"], h2=h2), C["lector"])
        return np.column_stack([pct((1 - W_S6) * pct(base[:, j]) + W_S6 * pct(_col(s6, j))) for j in range(len(LAB))])
    if nombre in ("Esust", "Esust_dsj"):
        fin = parte_b(mezcla_a3(C["r384"], C["r224"], C["r6"], s6, W_S6, ws_c, h2), C["lector"])
        if nombre == "Esust_dsj":
            fin = desatajo(fin, C["meta"])
        return fin
    raise ValueError(nombre)


def desatajo(fin, meta, lam=LAMBDA_DSJ):
    fin = np.array(fin, dtype=np.float64)
    j = LAB.index("Synovitis")
    fin[:, j] = pct(fin[:, j] - lam * pct(meta))
    return fin


def combinar_s6(mats, modo):
    """H1.  mats: lista de matrices de probabilidad (n, 12).  'logit' = sigmoide de la media de logits; 'rango' = media de pct."""
    mats = [np.clip(np.asarray(m, dtype=np.float64), 1e-6, 1 - 1e-6) for m in mats]
    if modo == "logit":
        z = np.mean([np.log(m / (1 - m)) for m in mats], axis=0)
        return 1 / (1 + np.exp(-z))
    if modo == "rango":
        return np.mean([np.column_stack([pct(m[:, j]) for j in range(m.shape[1])]) for m in mats], axis=0)
    raise ValueError(modo)
