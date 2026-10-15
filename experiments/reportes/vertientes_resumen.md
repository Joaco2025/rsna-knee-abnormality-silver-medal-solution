# Vertientes del 2026-10-14: resumen de lo encolado

Estado al cierre de la sesión (2026-10-14, ~23:30). El piloto **2085360 está RUNNING** en `amdgpu-hpc-05`; todo lo demás está en cola (`QOSGrpGRES`: 4 GPU de cuota compartidas con Owen) o esperando dependencia. Cuota del grupo al último envío: **545.9 GB de 700 GB** (~154 GB libres; umbral de parada 20 GiB). El código de esta sesión **no está commiteado** (regla de la sesión): cada job copia sus scripts y sus sha256 a `runs_joako/<job>/codigo/` e `inputs.txt`.

## Tabla

| Tarea | JOBID | Recursos | Dependencia | Escribe (todo en `/lustre/proyectos/p082/`) | Criterio de éxito |
|---|---|---|---|---|---|
| A · ft0949 V1 (224 desde el 80 %) | **2085552** | gpu, 1 MI210, 16 CPU, 64 GB, 6 h | — | `runs_joako/2085552/`: `last_fp16.pt`, `prom3_fp16.pt` (~146 MB c/u), `historial.jsonl`, `preds/`, `resumen.json`; log `code_joako/logs/ft0949-v1_2085552.out` | Estable (≥ 1 época con < 5 % de pasos saltados, sin NaN ni aborto) **y** Spearman medio de `prom3` frente a la base en modo 320 sobre los 58 Gold **< 0.95** → `criterio_prom3.veredicto`. **Hereda OAI: solo Final A** |
| A · ft0949 V2 (recorte de 320) | **2085553** | ídem | — | `runs_joako/2085553/` (ídem); log `logs/ft0949-v2_2085553.out` | Ídem. Solo Final A |
| B · semilla 7 all-data | **2085556** | gpu, 1 MI210, 16 CPU, 64 GB, 12 h, `--deadline` 19-oct 23:59 | `afterany:2085360` + gate (piloto con `criterio_swa` = PASA) | `runs_joako/2085556/`: `gate.txt`; si abre: `last_fp16.pt`, `swa_fp16.pt`, `historial.jsonl`, `resumen.json`; log `logs/seeds-s7_2085556.out` | Sano (fp16 < 5 % en alguna época, sin NaN) **y** Gold del SWA ≥ Gold del SWA del piloto − 0.02 → `criterio_all_data.veredicto`. Gold sesgado por la selección top-3 |
| B · semilla 2026 all-data | **2085557** | ídem | ídem | `runs_joako/2085557/` (ídem); log `logs/seeds-s2026_2085557.out` | Ídem |
| B · lado unificado, fold 0, semilla 42 (variante) | **2085558** | ídem | ídem | `runs_joako/2085558/` (ídem, + `indice_corpus.csv` con lado y volteos); log `logs/seeds-lado-f0_2085558.out` | Mismo criterio que el piloto (OOF fold 0, juez v1 ≥ effhib + 0.005) → `criterio_swa.veredicto`, **pareado con 2085360** (mismos estudios y semilla; solo cambia la orientación) |
| C · R7 limpio | — **no viable** | — | — | `rsna/r7_sources/` (7 fuentes, sha256 OK) y `experiments/reportes/r7_limpio.md` | — (falta el paquete `training` / `cloud_code.zip`; además el R7 es una CNN, y eso choca con el brazo de Owen) |
| D · LLM, solo fase Gold | **2085625** | gpu, 1 MI210, 8 CPU, 64 GB, 1.5 h, `--deadline` 19-oct 23:59 | — | `runs_joako/2085625/`: `gold_eval.json`, `etiquetas_llm.parquet` (58 Gold), `ollama_serve.log`; caché `runs_joako/llm_reetiquetado_cache/`; log `logs/llm-reetiquetar_2085625.out` | Consenso de 3 lectores: macro AUC en los 58 Gold **≥ 0.88** **y** > v1 en las celdas de v1 (v1 = 0.840, sesgado al alza) → `criterio.veredicto`. La fase resto (4,349) solo si PASA y con un nuevo OK |

Reportes: [ft0949.md](ft0949.md), [seeds_coatnet.md](seeds_coatnet.md), [r7_limpio.md](r7_limpio.md), [llm_reetiquetado.md](llm_reetiquetado.md).

## Comandos para revisar cada uno

Estado de todos (no más de una vez por minuto):

```bash
squeue -u jcastro -o "%.9i %.16j %.8T %.10M %.10l %.22E %R"
sacct -j 2085360,2085552,2085553,2085556,2085557,2085558,2085625 -X --format=JobID,JobName%18,State,Elapsed,Start,End
```

A, ft0949 (V1 = 2085552, V2 = 2085553):

```bash
tail -n 20 /lustre/proyectos/p082/code_joako/logs/ft0949-v1_2085552.out
python3 -c "import json; r=json.load(open('/lustre/proyectos/p082/runs_joako/2085552/resumen.json')); print(r.get('abortado'), r.get('pct_saltados_por_epoca')); print(r.get('criterio_prom3')); print(r.get('criterio_ultima'))"
# V2: igual con 2085553 y logs/ft0949-v2_2085553.out
```

B, gate y semillas (2085556, 2085557; variante 2085558):

```bash
cat /lustre/proyectos/p082/runs_joako/2085556/gate.txt
python3 -c "import json; r=json.load(open('/lustre/proyectos/p082/runs_joako/2085556/resumen.json')); print(r.get('abortado'), r.get('swa',{}).get('epocas'), r.get('swa',{}).get('gold')); print(r.get('criterio_all_data'))"
# variante de lado, pareada con el piloto:
python3 -c "import json; a=json.load(open('/lustre/proyectos/p082/runs_joako/2085360/resumen.json')); b=json.load(open('/lustre/proyectos/p082/runs_joako/2085558/resumen.json')); print('piloto', a.get('criterio_swa')); print('lado  ', b.get('criterio_swa'))"
grep -E "acuerdo con la tabla de Owen|ranuras volteadas|lado \{" /lustre/proyectos/p082/code_joako/logs/seeds-lado-f0_2085558.out
```

D, LLM Gold (2085625):

```bash
tail -n 30 /lustre/proyectos/p082/code_joako/logs/llm-reetiquetar_2085625.out
python3 -c "import json; e=json.load(open('/lustre/proyectos/p082/runs_joako/2085625/gold_eval.json')); print(e.get('criterio')); print({k: round(v['macro_58'],4) for k,v in e.items() if isinstance(v,dict) and 'macro_58' in v}); print('v1', round(e['v1']['macro_celdas_v1'],4), e.get('consenso_vs_v1'))"
```

Piloto (decide el gate de B):

```bash
python3 -c "import json; r=json.load(open('/lustre/proyectos/p082/runs_joako/2085360/resumen.json')); print(r.get('criterio_swa')); print(r.get('criterio_ultima'))"
```

## Avisos

1. **ft0949 (2085552/3) no lleva `--deadline`** (se encolaron antes de añadirlo a los demás). Son jobs de 6 h: si no han arrancado el **19-oct a las 17:59**, terminarían después del límite. En ese caso jcastro puede hacer `scontrol update JobId=2085552 Deadline=2026-10-19T23:59:00` (y lo mismo con 2085553), o cancelarlos a mano. El agente no puede: `scontrol update` es nivel 3.
2. **Orden en la cola:** el piloto ya corre. Cuando termine (~7-11 h) se liberan las tres de B a la vez, y compiten con A, D y los jobs de Owen por 4 GPU. Si el piloto NO PASA, las tres de B ocupan la GPU unos segundos y terminan sin entrenar.
3. **La semilla 2085558 (lado unificado)**, si mejora, exige canonicalizar el lado también en la inferencia de Kaggle (Laterality / ImagePositionPatient / ImageOrientationPatient). Ese código todavía no existe en el notebook.
4. **D:** que `ollama/0.34.0` cargue `qwen3.5:27b` no está comprobado. Si falla, el consenso sale de 2 lectores y `gold_eval.json` lo dice.
