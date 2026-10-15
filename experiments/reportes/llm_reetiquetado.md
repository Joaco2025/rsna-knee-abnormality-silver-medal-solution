# D. Reetiquetado con LLM local (prefijo `llm_`)

Fecha: 2026-10-14. Código: [rsna/llm_reetiquetar.py](../../rsna/llm_reetiquetar.py), [jobs/llm_reetiquetar.slurm](../../jobs/llm_reetiquetar.slurm).

## Inventario (solo lectura)

### `/lustre/proyectos/ollama-models/`

Dueño `root`. Contiene `blobs/` y `manifests/registry.ollama.ai/library/`.

| Modelo | Blob de pesos | Tamaño |
|---|---|---|
| `qwen2.5:72b` | `sha256-6e7fdda508e9…` | 47.4 GB |
| `llama3.3:70b` | `sha256-4824460d29f2…` | 42.5 GB |
| `qwen3.5:27b` | `sha256-d4b8b4f4c350…` | 17.4 GB (parámetros por defecto: temperature 1, top_k 20, presence_penalty 1.5; se sobrescriben) |

Los otros 12 blobs son plantillas, licencias y parámetros (< 10 MB). Total ≈ 107 GB.

### Runtime

- **Hay runtime instalado: módulos del sistema `ollama/0.12.2`, `ollama/0.32.15` y `ollama/0.34.0`** (`module avail`, en `/shared/modules`). No hay `ollama`, `vllm` ni `llama.cpp` en el PATH sin módulo.
- `ollama/0.34.0` con `rocm/7.2.2` **ya funcionó en las MI210**: es el que usó Owen para extraer etiquetas con `qwen2.5:72b` (jobs 2078913/4, `code/jobs/extraer_etiquetas.slurm`), con el servidor dentro del job en 127.0.0.1. Este job copia ese patrón. No se instala nada.
- **No comprobado:** que `ollama/0.34.0` cargue `qwen3.5:27b`. No se puede probar sin GPU: la partición cpu está llena y en el login no se pueden dejar servidores. Si falla, sus 58 lecturas Gold salen con error, el consenso se calcula con los otros dos lectores y `gold_eval.json` lo registra.

## El problema de cómputo

Owen midió con `qwen2.5:72b` **21.8 s por informe** (295 tokens de salida a 14.4 tok/s; log de 2078913). Con eso, un solo lector de 70B sobre los 4,407 informes son ~27 h. **Tres lectores sobre todo train no caben en 1 job de 10 h con 1 GPU.**

Lo que sí cabe, y es lo que decide: **los 58 Gold con los 3 lectores** (~30-45 min con las cargas). De ahí el diseño en dos fases:

1. **Gold** con los 3 lectores → `gold_eval.json` y criterio.
2. **Resto** (4,349 no Gold) **solo si el consenso PASA** (`--parar-si-falla`), lector a lector hasta 9.3 h, en orden aleatorio fijo (semilla 20261014). Si el plazo corta, lo leído es una muestra al azar.

Medidas para abaratar:

- **Salida compacta:** por hallazgo, `status` + `p` entero (~100-150 tokens frente a 295).
- **4 informes en paralelo** (`OLLAMA_NUM_PARALLEL=4`; con 72B caben pesos + caché KV de 4 × 4,096 tokens, ~53 GB de 64 GB).
- **Orden por velocidad:** `qwen3.5:27b` primero.

Estimación sin medir: 27B ~1-2 h para los 4,349; cada 70B ~4-5 h. En un job: 27B completo, `qwen2.5:72b` completo o casi, `llama3.3:70b` parcial. **Completar el consenso de 3 lectores en todo train requiere un segundo job.** El caché `runs_joako/llm_reetiquetado_cache/<lector>.jsonl` hace que retome donde quedó.

## Diseño de la lectura

- **Esquema JSON forzado** con `format` de Ollama. Por hallazgo: `status` ∈ {present, possible, absent, **not_addressed**} y `p` ∈ [0, 100], la probabilidad de que un radiólogo que ve las imágenes lo marque.
- **«No abordado» explícito:** el prompt dice «Do not turn silence into "absent"». Para los no abordados, `p` se apoya en lo que el informe sí describe (p. ej., sinovitis infrarreportada con derrame) y en la frecuencia habitual. Es la lección de los hilos 733932 y 747658 (estos se leyeron como datos).
- **Definiciones de los 12 hallazgos:** las del extractor de Owen (`code/etiquetas/extraer.py`, v2), sin las escalas de gravedad.
- Temperatura 0, semilla 0, `num_ctx` 4,096 (informes de 1,097 caracteres de media, p95 2,452), `num_predict` 400, `think: false` para `qwen3.5`.

## Salida por celda (`etiquetas_llm.parquet`)

Una fila por estudio (`StudyInstanceUID`, `oficial`). Para cada hallazgo `<H>`:

| Columna | Qué es |
|---|---|
| `p\|<H>` | **Probabilidad de consenso:** media de los `p` de los lectores disponibles |
| `no_abordado\|<H>` | Mayoría de lectores con `not_addressed`; `na_frac\|<H>` es la fracción |
| `disp\|<H>`, `sd\|<H>` | Dispersión entre lectores: máx − mín y desviación típica |
| `n\|<H>` | Lectores que respondieron |
| `p_status\|<H>` | Puntuación secundaria: estado → probabilidad fija (present 0.95, possible 0.75, not_addressed 0.06, absent 0.02), fijada antes de ver resultados |
| `p_<lector>\|<H>`, `status_<lector>\|<H>` | La lectura de cada lector |

## Validación y criterio (fijado antes de lanzar)

- **Contra los 58 Gold** (`y_*` de `etiquetas_v1_2079203.parquet`): AUC por hallazgo y macro de cada lector, del consenso y de la puntuación secundaria, con IC 95 % por bootstrap de 1,000 réplicas por estudio (semilla 20261014).
- **Comparación con v1, pareada:** solo en las celdas donde v1 tiene valor (`w > 0`; Synovitis 25, Baker's 29, el resto 46-56). **v1 se calibró con estos mismos Gold**, así que su AUC está sesgado al alza y la comparación favorece a v1. Referencia medida en el smoke: **v1 = 0.840** [0.805, 0.876] en sus celdas.
- **Consenso útil (PASA)** = macro del consenso en los 58 Gold (todas las celdas) **≥ 0.88** **y** macro del consenso > macro de v1 en las celdas de v1 (punto; IC y P(Δ > 0) como información).
- **Contexto:** Owen midió 83 % de acuerdo con qwen2.5:72b y con MedGemma 27B («el límite es el informe, no el lector»). En el foro, Claude 4.5 llega a 0.892 (hilo 747658, mensaje #7), y un radiólogo pudo etiquetar viendo imágenes que contradicen el informe (mensajes #3 y #5). **0.88 es exigente**; un NO PASA es el resultado más probable y no gastaría más GPU.

## Recursos

| Recurso | Valor |
|---|---|
| Partición, GPU | gpu, 1 MI210 |
| CPU, RAM | 8, 64 GB |
| `--time` | 10 h; plazo interno 9.3 h para la fase resto; `--deadline=2026-10-19T23:59` |
| Fase gold | ~30-45 min (3 cargas de 17-47 GB desde Lustre, ~5 min cada una, más 174 lecturas) |
| Fase resto | Solo si PASA; hasta agotar el plazo |
| Lustre | `runs_joako/<job>/` < 50 MB; caché JSONL ≤ ~30 MB por lector completo |
| Modelos | Leídos de `/lustre/proyectos/ollama-models` (solo lectura, `OLLAMA_NOPRUNE=1`) |
| Red | Ninguna: el servidor escucha en 127.0.0.1 y el cliente ignora los proxys |

## Smoke (login, `--mock`, sin modelo ni servidor)

- `--mock` sustituye la llamada HTTP por una respuesta falsa determinista (palabras clave del informe). Prueba lectura y caché, esquema, tabla, evaluación con bootstrap, criterio y parquet. No mide la calidad del modelo ni su velocidad.
- **Corrida 1** (58 Gold + 2 no Gold, 3 lectores): completa. `gold_eval.json` con AUC por lector y consenso, comparación con v1 y veredicto «NO PASA», como debe con un lector falso. Parquet de 60 × 158.
- **Corrida 2** con el mismo caché y `--parar-si-falla`: los 58 × 3 salen del caché (0 por leer) y el resto no se lee.
- Lo único sin probar es la llamada real a Ollama. Esa ruta es la de `extraer.py` de Owen, más el campo `think` y un reintento sin él si el modelo lo rechaza.

## Estado

**Encolada solo la fase Gold** (decisión de jcastro, 2026-10-14): **JOBID 2085625**, `--time=01:30:00`, `FASES=gold`. Cuota antes: 545.9 GB de 700 GB. Escribe `runs_joako/2085625/gold_eval.json` (criterio en `criterio.veredicto`) y `etiquetas_llm.parquet` con los 58 Gold. Las respuestas quedan en `runs_joako/llm_reetiquetado_cache/`.

```bash
sbatch --time=01:30:00 --export=ALL,FASES=gold jobs/llm_reetiquetar.slurm      # lanzado = 2085625
# si PASA y jcastro lo aprueba, la fase resto reutiliza el caché (no relee los Gold):
sbatch jobs/llm_reetiquetar.slurm      # FASES por defecto = gold,resto (no pasarlo en --export: la coma lo parte)
```

## Resultados

(pendiente)
