# Benchmark de CoAtNet v2: escalera de diagnóstico en la MI210

**Fecha:** 2026-10-14. **Job:** 2085126, `COMPLETED` en 10 min, nodo `amdgpu-hpc-03`, 1 GPU.
**Modelo:** `coatnet_rmlp_2_rw_384` de timm 1.0.30, torch 2.8.0+rocm6.4, 384 px, `num_classes=12`.
**Alcance:** solo cifras medidas por este job. Sin red, sin instalar nada, sin tocar envs ni nada de Owen.
**Reporte anterior de AGY:** inválido (ver el final). Este informe lo sustituye.

## Resumen

1. **El segfault es del backward en bf16.** Cae en los cuatro intentos bf16, con atención fusionada o sin fusionar y con cualquier backend de atención. fp32 y fp16 funcionan con el mismo modelo.
2. **El 47 % de VRAM no causó el segfault.** La escalera lo reprodujo con la GPU limpia. Lo más probable es que fuera memoria de un job que acababa de terminar en ese nodo.
3. **Recomendación: entrenar en fp16 AMP con GradScaler y atención sin fusionar.** Es la única combinación que terminó. Con B=32 mide 40.5 img/s y cabe en 42.7 GB.

## Tabla de peldaños

Todos con AdamW y BCE sobre datos y pesos aleatorios, sin dataloader. Cada peldaño es un subproceso con `faulthandler` activo. Cada cifra sale de 10 iteraciones medidas, después de la primera iteración y dos más de calentamiento.

| Peldaño | Configuración | Estado | img/s | VRAM pico, asignada / reservada | Primera iteración |
|---|---|---|---|---|---|
| 1 | fp32, solo forward, B=2 | **OK** | 63.3 | 0.76 / 0.94 GB | 9.5 s |
| 2 | fp32, fwd+bwd, B=8 | **OK** | 26.9 | 11.98 / 14.06 GB | 47.1 s |
| 3 | bf16, fwd+bwd, B=8, sin fusionar | **CRASH** (código -11, SIGSEGV) | n/d | n/d | n/d |
| 4 | bf16, fwd+bwd, B=8, fusionada | **CRASH** (código -11, SIGSEGV) | n/d | n/d | n/d |
| 5 | fp16 + GradScaler, fwd+bwd, B=8, sin fusionar | **OK** | 32.8 | 10.53 / 11.68 GB | 47.2 s |
| 6a | como el 5, B=16 | **OK** | 37.6 | 19.80 / 21.93 GB | 70.9 s |
| 6b | como el 5, B=32 | **OK** | 40.5 | 38.38 / 42.71 GB | 130.4 s |
| 4b | diagnóstico, bf16 fusionada, sin backend flash | **CRASH** (código -11) | n/d | n/d | n/d |
| 4c | diagnóstico, bf16 fusionada, solo backend math | **CRASH** (código -11) | n/d | n/d | n/d |

- **Ningún OOM.** La VRAM total de la GPU es 68.7 GB.
- El peldaño 6 se hizo sobre el 5, el de mayor img/s entre 3, 4 y 5 que terminó OK (era el único).
- Los peldaños 4b y 4c no estaban en el pedido. Los añadí, dentro del mismo job, porque el 4 falló.
- Los peldaños 1 y 2 usan la atención por defecto de timm, que es la fusionada (16 de 16 capas).
- En los peldaños 3 y 5 comprobé que `TIMM_FUSED_ATTN=0` surtió efecto: 0 de 16 capas de atención fusionadas. En los 4, 4b y 4c eran 16 de 16.

## Causa más probable del segfault

**Un fallo nativo en el backward de autograd cuando se usa bf16, en esta pila ROCm 6.4 con torch 2.8.** Qué operación exacta falla no está identificado.

Evidencia, toda medida en este job:

1. **Dónde cae.** En los peldaños 3, 4, 4b y 4c el último estadio impreso es `fwd_ok`: el forward terminó y se sincronizó. El volcado de `faulthandler` muestra el hilo principal dentro de `loss.backward()` y el hilo que falla sin ningún frame de Python, es decir, en código nativo. Los cuatro volcados son iguales.
2. **Es el bf16, no el modelo.** El mismo grafo termina en fp32 (peldaño 2) y en fp16 (peldaños 5, 6a, 6b), con el mismo tamaño de lote.
3. **No es la atención.** Falla con la atención sin fusionar (3), fusionada (4), fusionada sin el backend flash (4b) y fusionada con solo el backend math (4c). En 4b y 4c verifiqué que los indicadores de SDPA cambiaron como se pretendía.
4. **No es la GPU sucia.** Los cuatro peldaños bf16 arrancaron con 0.10 GB ocupados, que es el contexto del propio proceso. Al inicio del job `rocm-smi` marcaba 0.016 % de VRAM y no había procesos KFD.
5. **Coincide con el job original.** El 2085054 murió en la primera iteración bf16 B=8, antes de imprimir nada más.

Lo que sigue sin saberse: qué operación del backward falla. Candidatas [inferencia, sin probar]: las convoluciones bf16 que ejecuta MIOpen, que CoAtNet usa en sus primeras etapas. Para localizarla hace falta otro job de GPU, que no está autorizado. Dos opciones baratas: un backward bf16 por bloques del modelo, y el mismo backward bf16 con MIOpen desactivado (`torch.backends.cudnn.enabled=False`) para ver si el crash desaparece.

## Sobre el 47 % de VRAM ocupada

- **Hecho.** El job 2085054 empezó a las 10:36:39 en `amdgpu-hpc-03`, el mismo segundo en que terminó el job 2085002_4 de Owen en ese nodo (`sacct`). La otra GPU del nodo la tiene `ollama-serve` de otro usuario desde las 00:25.
- **Hecho.** En este job la GPU llegó limpia: 11 MB usados y sin procesos KFD.
- **Hecho.** Una instantánea de `rocm-smi` tomada justo al terminar un peldaño todavía muestra la memoria del proceso que acaba de salir (59 % tras el 6b, que reservó 42.7 GB). Al arrancar el peldaño siguiente, pocos segundos después, solo había 0.10 GB ocupados. La memoria tarda segundos en liberarse.
- **Inferencia, consistente pero no probada.** El 47 % del 2085054 fue la memoria del job de Owen, que terminaba ese mismo segundo. No hay un `--showpids` de aquel momento que lo confirme.
- **Qué hacer si se quiere blindar los jobs de entrenamiento:** al inicio, esperar a que `rocm-smi --showmeminfo vram` marque menos de 1 GB antes de lanzar Python. No se probó.

## Recomendación

1. **Precisión: fp16 AMP con GradScaler.** Es la única que terminó el backward. No uses bf16 en esta pila hasta localizar la operación que cae.
2. **Atención: sin fusionar (`TIMM_FUSED_ATTN=0`).** Es la única variante medida con éxito. La atención fusionada en fp16 no estaba en la escalera y no se probó.
3. **Lote: B=32 si no hay otra razón para uno menor.** Mide 40.5 img/s frente a 37.6 con B=16 y 32.8 con B=8. La ganancia por duplicar el lote cae de 14.5 % a 7.8 %. Con B=32 la reserva es de 42.7 GB, y la GPU tiene 68.7 GB.
4. **Cuenta con un arranque lento.** La primera iteración tarda de 47 a 130 s, según el lote. No separé la causa. Cada peldaño usó una caché de MIOpen nueva dentro del job, en `runs_joako/bench_coatnet_v2/2085126/miopen`. Una caché que sobreviva entre jobs podría recortarlo si el lote y la precisión no cambian; no se probó.
5. **Coherencia con la inferencia.** El notebook público ya hace su inferencia de Raptor con `torch.autocast` en float16 en las T4 (celda 22). Entrenar en fp16 usa la misma precisión.

## Límites de estas cifras

- Datos y pesos aleatorios. Mide velocidad y memoria, no convergencia. No se comprobó que el entrenamiento en fp16 sea numéricamente estable con datos reales.
- Una sola GPU y un solo proceso. No hay carga de datos ni E/S, así que el rendimiento de un entrenamiento real será menor.
- Solo 10 iteraciones medidas por peldaño y una ejecución. No hay repeticiones para estimar la variación.
- No se probó fp16 con atención fusionada, ni bf16 con MIOpen desactivado.

## Archivos

- Job: `bench_coatnet/bench_v2.slurm`, que lanza `bench_ladder.py`, que lanza un `bench_rung.py` por peldaño.
- Salidas en `runs_joako/bench_coatnet_v2/2085126/`: `escalera.json`, un `rung_<id>.log` por peldaño con el volcado de `faulthandler`, y la caché de MIOpen.
- Log del job: `logs/bench_coatnet_v2_2085126.out`. Incluye `rocm-smi --showpids` y `--showmeminfo vram` del inicio.
- El benchmark original (`bench.py`, `bench.slurm`) se conserva sin cambios.

## Reporte anterior de AGY: inválido

`experiments/reportes/agy_bench_coatnet.md` queda marcado como inválido con un aviso al principio. Su tabla de la sección 5 eran estimaciones, no mediciones. El job 2085054 no se quedó en cola: corrió y murió con segfault a los 57 s, sin medir nada. Supone además 3,500 estudios y 62 ventanas por estudio, sin verificar. El texto original se conserva debajo del aviso.
