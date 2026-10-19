# Hallazgos externos (2026-10-19)

**Acceso:** la herramienta de Kaggle (MCP) da `Unauthenticated` para listar notebooks y leaderboard; sí lee discusiones y
metadatos de datasets. Se completó con búsqueda web, que no indexa el listado de código de Kaggle en tiempo real. **No
podemos afirmar que no exista un notebook público por encima de 0.958; solo que no lo encontramos.**

| Fuente | Fecha | Qué dice | Nivel de evidencia |
|---|---|---|---|
| Write-up de goodpjw2008 (notebook «0.958», v27) | 16-oct | Escalera de envíos (ver `current_state.md`). Los saltos: R7 (0.956), R6 + flip (0.957), pesos por hallazgo de heliosli (0.958). R7 añadido a R6: sin cambio. Flip en R7: 0.957 | [autor]; el 0.958 lo confirmó nuestro recibo |
| heliosli, «blend-mine-in-private-lb-luck-to-all-upvoters» | ~15-oct | Fuente de los pesos por hallazgo (Syn 0.8, Lat OA 0.2, PF OA 0.6, Baker 0.35) | [autor] a través del write-up; notebook no leído (comentarios cerrados) |
| Host (Carr, RSNA), discusión 743416 | 14-oct | Criterio de datos externos: «registro + licencia» vale; acuerdo institucional no vale | [hecho] |
| Discusión 743416 | 16 y 18-oct | Abierta la restricción geográfica de OAI (LG22 desde China); pregunta del 18-oct sin respuesta | [hecho] |
| Discusión 748295 («Stuck at .954») | 11–14 oct | Un usuario del top: «Short answer is labels… AI, not by hand. And don't rely on reports alone.» Otro: PF OA, Lateral OA y Synovitis son los débiles | [autor], sin números |
| Discusión 745861 («Tricks») | 11-oct | «Use 58 samples to train a model with very few parameters… prior is the key» | [autor], sin números públicos |
| Discusión 745861 | 15-oct | 3DReasonKnee (HF, rajpurkarlab): cajas y segmentaciones sobre OAI | sin verificar; **OAI**, mismo riesgo de reglas |
| Discusión 748111 | 10-oct | Un «single sub 0.960» (captura de pantalla); su autor dice en Rednote que es «basically a single model» | captura sin método; no reproducible |
| Discusión 735304 («Best single-model score») | 9–16 oct | Modelos únicos 5-fold en el público: ConvNeXt-T 0.959, EfficientNetV2-S 0.956, CoAtNet 0.952; ConvNeXt-T 2.5D 0.953 y ViT-S DINOv2 0.951 (ensamble 0.955). En todos los casos con **pseudoetiquetas de imagen** mezcladas con las de informe (≈50/50, «soft bootstrapping»), con mejora «en Gold y en el LB». Las clases débiles son Synovitis (~0.80 en Gold) y PF OA / Lateral OA | [autor], varios usuarios; coherente con nuestro Gold de Raptor |
| README de `nartaa/rsna-knee-s6fall-weights` (local) | 17-oct | S6: 0.950 solo, sin OAI, etiquetas JEV, gold excluido, nada seleccionado en Gold | [autor] + [archivo] |
| `training_code/coatnet/RUNS.md` (local) | 17-oct | Raptor 0.949: OAI supervisa **solo** PF OA, Lateral OA y Synovitis; SWA top-3 por AUC en Gold | [archivo] |
| Blog de HF de bishnoiyash | 9-ago | Pipeline de agosto (0.903); sin interés actual | [autor] |
| Búsqueda web («0.958/0.959») | 19-oct | Ningún notebook por encima de 0.958; 0.956–0.957 en los listados indexados | limitada |

## Lo que no pudimos verificar

- Las últimas versiones de los notebooks de heliosli y nartaa (¿hay R8 o S7?).
- El listado de código ordenado por score.
- El leaderboard por encima de nuestro puesto.

Para eso jcastro tendría que autorizar la herramienta de Kaggle (`authorize`) o revisarlo a mano en el navegador.
