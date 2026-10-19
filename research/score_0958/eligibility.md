# Elegibilidad de componentes

Esto no es una opinión legal; es la aplicación del criterio del host (743416, 14-oct): «registro + aceptación de
licencia» vale, «acuerdo institucional o afiliación» no vale; además, transparencia y criterio de «reasonableness».

| Recurso | Usado por | Acceso a los datos | Licencia de pesos o código | Clasificación |
|---|---|---|---|---|
| OAI (NDA) | Raptor 384 y 224 (solo PF OA, Lat OA, Synovitis) | registro en Login.gov + certificado de uso de datos de NDA; **bloqueado geográficamente para China** (LG22) | pesos de nartaa: dataset «publication-swa-weights» (licencia no revisada aquí) | **Riesgo relevante** (pregunta abierta en el foro) |
| fastMRI / fastMRI+ | R7 (declarado); R6 sin verificar | formulario individual + acuerdo de uso de NYU | — | Pendiente de verificar para R6; probablemente compatible con el criterio |
| SKM-TEA (red de anatomía de heliosli) | R6, R7, S6 (anclas) | checkpoint de Stanford, Apache-2.0, sin acceso restringido (según el foro) | Apache-2.0 | Compatible (evidencia moderada) |
| KneeXNet (localizador) | paquete de código de heliosli | — | — | Pendiente |
| Etiquetas JEV / blendjev | Raptor, S6 | tablas públicas CC0 en Kaggle | CC0 | Compatible |
| Lector de goodpjw2008 | base «0.958» (1/10) | solo datos de la competición | Apache-2.0 | Compatible |
| Pesos de S6 (nartaa) | E, C | — | **CC BY-NC 4.0** | Compatible para competir; revisar si choca con la licencia exigida a los ganadores (solo importa si hay premio) |
| R6 (heliosli) | base «0.958» | — | «other» | Pendiente |
| OrthoFoundation `best.pt` / pesos OFL-L | candidato H13 | — | no revisada | Pendiente; no usar sin revisarla |

**Para la elección de finales:** todos los candidatos actuales (0.956, C, «0.958», E y las variantes de E) dependen de
Raptor (OAI). Ninguna variante de este plan cambia ese riesgo. El seguro sigue siendo el final B sin OAI (ver
`runs_joako/exp_finalB_sinOAI/LEEME.md`).
