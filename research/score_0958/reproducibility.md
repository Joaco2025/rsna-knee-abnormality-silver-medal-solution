# Reproducibilidad

## Análisis de esta carpeta

Todos son de CPU, duran segundos, solo leen datos y se pueden repetir en el login con el venv del proyecto:

```bash
cd /lustre/proyectos/p082/code_joako/research/score_0958/analisis
P=/lustre/proyectos/p082/venvs/rsna/bin/python
$P -I a1_coocurrencia_gold_vs_jev.py    # -> a1_coocurrencia.json
$P -I a2_eje_trauma_degeneracion.py     # -> a2_eje.json  (bootstrap 500, semilla 0)
$P -I a3_prestamo_hallazgos_debiles.py  # -> a3_prestamo.json (bootstrap 2000, semilla 0)
```

## Entradas que leen los análisis

| Entrada | Qué es |
|---|---|
| `kaggle_data/competitions/rsna-knee-abnormality-detection/train.csv` | etiquetas de Gold-58 (las únicas filas con las 12 columnas) |
| `runs_joako/exp_s6_semillas/entradas/s6fall_v5/training_code/labels/labels_v1_blendjev.parquet` | etiquetas JEV-blend, comprobadas contra el `SHA256SUMS` de nartaa en la verificación de D |
| `runs_joako/2089458/preds_gold.npz` | Raptor 384, 224 y su mezcla 65/35 sobre Gold-58; generado por `diag_r224_gold.py` del job 2089458, fp32 en CPU, **sin el espejo de Kaggle** |

## Inspecciones de código

| Notebook o fuente | sha256 |
|---|---|
| Notebook «0.958» | `9485a6fc…` |
| E | `043ab37c…` |
| `compact_infer` de R6 (v4c), extraído del `SOURCES` comprimido de la celda 11 de E | `8401812cd637…` |
| `compact_infer` de S6 | `34f433fe72b7…` |

Hay una copia de las fuentes de R6 en el scratchpad de la sesión (no es un artefacto del proyecto).

La configuración de entrenamiento de S6 está en `entradas/s6fall_v5/E_prior_s42_fall.run_config.json` (`extent: false`,
`aug: crop`).

## Limitaciones

- Gold-58 es pequeño: los IC del bootstrap son de ±0.01–0.05 por hallazgo.
- Para Raptor, Gold está contaminado (SWA elegido por Gold): su AUC base está inflado y solo se interpretan los deltas.
- No hay predicciones de R6, S6 ni del lector en Gold, así que no se midió la complementariedad entre componentes.
- La búsqueda externa fue parcial (la herramienta de Kaggle no está autenticada para notebooks ni para el leaderboard).
- Los scores del público vienen de recibos comunicados por jcastro o de lo que declaran los autores. Ninguno se reprodujo
  aquí.
