# Datos externos y riesgo de los finales (notebooks 0.957 y 0.940)

**Fecha:** 2026-10-14. **Solo lectura y solo propuesta:** no se lanzó nada, no se tocó ningún notebook ni se envió nada a Kaggle.
**Fuentes leídas:** `referencias/0-957-latest-dataset-score-in-one-click.ipynb` (7 celdas, **sin salidas guardadas**), `referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb` (24 celdas, con el log de 3 estudios), los 7 fuentes incrustados en la celda 4 del 0.957 (`SOURCES`, extraídos y leídos por búsqueda dirigida, no línea a línea), el hilo 743416 completo vía el MCP de Kaggle (solo lectura) y los inventarios previos [inventario_base_0957.md](inventario_base_0957.md) y [agy_inventario_base_0940.md](agy_inventario_base_0940.md).
**Etiquetas:** [hecho] = está en el texto o el código citado; [inferido] = deducido de nombres o contexto; [no verificado] = no pude comprobarlo.

---

## 0. La regla del host, tal como está escrita

Hilo 743416 («Please clarify which external knee MRI datasets are allowed»). Respuesta del host, 2026-10-14 13:18 UTC, firmada por Christopher Carr, RSNA [hecho]:

> «we are not prepared to designate specific datasets as permitted or off limits. Instead we would reiterate the provision of the contest rules that states that external data sets used must be, "either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants." Datasets that require only registration and acknowledgement of licensing terms would clearly meet these criteria. Datasets that require an institutional research agreement or limit access to users with institutional affiliations clearly do not. Beyond that guidance, we would simply advise adhering to the broader "reasonableness" criterion and also providing transparency in disclosing your training methods, including datasets used.»

Tres consecuencias para lo que sigue:

1. **El host no emitió un veredicto por dataset.** Todo lo que clasifico abajo es *mi aplicación de su regla*, no una decisión suya.
2. **La regla tiene dos mitades:** «solo registro y términos de licencia» (válido) frente a «acuerdo institucional o afiliación institucional» (no válido). Añade un criterio de «reasonableness» y pide **transparencia** sobre los datos usados.
3. **Quedan sin responder dos preguntas que afectan directamente a este informe:**
   - ¿Se pueden usar **checkpoints públicos entrenados con OAI**? Se preguntó dos veces (comentarios del 2026-09-27 y del 2026-10-08) y la respuesta del host no la menciona.
   - ¿Un dataset con registro bloqueado desde ciertos países (error «LG22», China) cumple «equally accessible»? Se preguntó el 2026-10-14 13:26, ocho minutos después de la respuesta, y no hay respuesta en lo que leí.

Otros datos del hilo que uso: la selección final es el **2026-10-22** (lo dice un comentario del 2026-10-08 [hecho]); un participante cuenta que obtuvo OAI como individuo, con cuenta Login.gov, sin institución y con «Institutional Approval Required: No» en la página de permisos de NDA [hecho, autorreportado]; otro comentario dice que NDA respondió que OAI está disponible para todos, pero es una **captura de pantalla que no pude ver** [no verificado]; el mismo hilo recoge que KneeCoT sí fue prohibido por el host por exigir un acuerdo con el hospital (enlace a la discusión 734109, que no abrí).

---

## 1. Datos externos por componente, con la cita

### 1.1 Notebook 0.957 («R7 + Raptor», bundle `8c7b666e27dc0e8f`)

| # | Componente | Dato o modelo externo | Dónde aparece |
|---|---|---|---|
| 1 | **Raptor**, checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (SHA-256 `7e5315da…44ef`) | **OAI** (Osteoarthritis Initiative, vía NIMH Data Archive) | Celda 2: nombre del archivo `…blendjev_oai…` (en `_weight_name`, en `ARMS` y en la comprobación del hash) y dataset `rsna-knee-oai-original384-s42-pair`. Celda 1: «Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI)… Dataset identifier(s): 10.15154/0hcg-f676». Celda 1 también dice «These are our retrained checkpoints, rather than those baseline weights». |
| 2 | Raptor, encoder | **ImageNet-12k → 1k** vía timm (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`) | Celda 2, `ARMS`. |
| 3 | Raptor, etiquetas | **Etiquetas débiles de reportes por LLM**: stevenleehans, pilkwang, riadmohamed42 (JEV) y etiquetas propias derivadas de reportes. «Source reports and study-level label tables are not included here». | Celda 1. El docstring de la celda 2 describe el método para los pesos base: reportes de la competición convertidos a 12 probabilidades por un modelo de lenguaje, 4,349 estudios; los 58 con etiqueta real solo se usan para validar. |
| 4 | **R7 ConvNeXt-small**, entrenamiento auxiliar | **fastMRI+** (cajas de lesión sobre volúmenes fastMRI de rodilla) | Celda 0: «fastMRI auxiliary training». `compact_train.py` línea 2: «--fm-aux DIR = the fastMRI+ lesion-box auxiliary task on the shared encoder»; línea 1286: «the fastMRI+ lesion-box auxiliary task (spec card…)»; línea 1897: «the fm384_v1 store DIR of the fastMRI+ lesion-box auxiliary task»; línea 3047: «training only; the fold models carry no aux_* key». El recibo de entrenamiento lo registra (`receipt["fm_aux"]`). |
| 5 | R7, anclas de rodilla | **SKM-TEA** (V-Net de segmentación del joint, checkpoint `anatomy.pt`) | `knee_anchor_bank_v2.py` línea 4: «centroid of the SKM-TEA joint tissues»; línea 44: «the SKM-TEA V-Net through the mainline loader»; líneas 180-182 y 214 y 261: `method = "skm_tea_joint"` / `"skm_tea_joint_avg%d"`; línea 236: `--weights` por defecto `F:/rsnaknee/pretrained/skm_tea/model-raw.ckpt`. `px_cache.py` línea 85: «SKM-TEA joint centre projected through the geometry bank». **Celda 4:** «the anchors are rebuilt from the test DICOM headers + the SKM-TEA checkpoint of the code dataset», con comprobación de hash contra `manifest['anchor_weights_sha256']`. `compact_infer.py` líneas 330 y 551. |
| 6 | R7, encoder | **ImageNet** (pesos de timm en caché local, `HF_HUB_OFFLINE=1`) | `compact_train.py` línea 482: «init == "timm": public ImageNet weights from the local hub cache». |
| 7 | R7, comprobación de lado | Clasificador de orientación ResNet34 propio (`side_common.py`) | `side_common.py` línea 34. **Con qué datos se entrenó no consta** [no verificado]. |
| 8 | R7, etiquetas | Las mismas etiquetas débiles de reportes del punto 3 | Celda 1. |
| 9 | Rama condicional OFL-L | **OrthoFoundation** («OFL-L», pesos SSL públicos) | Celda 4: `if manifest['backbone'] == 'ofl_l'` carga `rsna-knee-ofl-ssl-weights`. La celda 0 dice ConvNeXt-small, así que **la rama no debería ejecutarse**, pero `manifest.json` no está en el notebook [no verificado]. |
| 10 | Rama condicional DINOv3 | `vit_large_patch16_dinov3_qkvb` | `compact_train.py` línea 294 y `probe_infer.py` línea 186. Es código disponible; nada indica que el bundle lo use [no verificado]. |

**Sobre el `skm_tea_joint` que mencionas:** el notebook 0.957 **no trae ninguna salida guardada** (0 outputs en las 7 celdas; tampoco en `kaggle/notebooks/0957_mas_v5_diag.ipynb`, que tiene 8 celdas). La cadena `skm_tea_joint` solo existe en el **código incrustado** (el campo `method` que escribe `knee_anchor_bank_v2.py`), y una búsqueda sobre todo el repo la encuentra únicamente en esos dos `.ipynb`. Si el log que viste es de una ejecución en Kaggle, ese log no está en el repo y no lo he visto.

**Cuándo se usa SKM-TEA de verdad:** solo si `manifest['views']` es `v3k` o `v4` (celda 4 y `compact_infer.py` línea 329). Ese campo está en `manifest.json`, dentro del dataset `rsna-knee-compact-r7cnxfsss` y **no en el notebook** [no verificado]. Dos indicios a favor de que sí se use: el notebook trae el cableado completo y el hash `anchor_weights_sha256`, y el dataset de código se llama `…orthofoundation-fold0-epoch7…` y contiene `anatomy.pt` «si `views` es `v3k` o `v4`». Un indicio en contra: nada en la celda 0 menciona vistas centradas en la rodilla. **Hay que abrir el manifiesto para saberlo.**

**Transparencia en el propio notebook:** la celda 1 («Credits and required acknowledgement») reconoce OAI, pero **no cita fastMRI ni SKM-TEA**, aunque la celda 0 sí dice «fastMRI auxiliary training» y el código usa SKM-TEA. El host pide divulgar los datasets usados; el notebook los divulga a medias.

### 1.2 Notebook 0.940 («Bend the Knee to Speedy Raptors»)

Búsqueda de `OAI`, `osteoarthritis`, `NDA`, `fastMRI`, `SKM`, `stanford`, `mrnet`, `hospital` sobre las 24 celdas: **cero coincidencias** relevantes [hecho]. Los créditos (celda 0) son solo autores de Kaggle. Pero el notebook es de **inferencia** y casi todos sus pesos vienen de terceros sin descripción de con qué se entrenaron.

| # | Componente | Dato o modelo externo | Dónde aparece |
|---|---|---|---|
| 1 | **DINOv2-S**, 20 miembros (pilkwang, 5 folds) | Base: DINOv2 de Meta (preentrenado en imágenes naturales, no médicas). Ajuste fino de pilkwang: datos no descritos | Celda 4 (`/kaggle/input/models/metaresearch/dinov2/PyTorch/small/1`) y dataset `pilkwang/rsna-knee-weights` (celda 1, línea 124). |
| 2 | **A5** (Mattia Angeli, 5 folds) | Backbone de timm leído de `cfg['backbone']` dentro de cada checkpoint, `pretrained=False`; datos de entrenamiento no descritos [no verificado] | Celda 18, líneas 358-369; dataset `mattiaangeli/knee-mri-fold-weights`. |
| 3 | **RadImageNet** ResNet-50 (`ResNet50.pt`, SHA-256 `08629f7e…0734`) + cabezas E10 / E13 / E11 + calibrador Ridge | **RadImageNet** (1.35 M de imágenes radiológicas) como preentrenamiento del encoder; cabezas y calibrador de prvsiyan, sofiaanjenje y antoinegg1 con datos no descritos | Celda 21, líneas 104-119 (rutas `marwanmath/resnet-50-radimagenet-marwan`, `prvsiyan/…heads-20260812`, `sofiaanjenje/rsna-knee-e13-train`, `antoinegg1/…e11-diverse-heads-v20`) y líneas 140-141, 197-248, 460-483. |
| 4 | **Raptor**, 4 brazos (dreaddevelopment) | Checkpoints v5 / v8 / v10 de `raptor-knee-maxspan`, `raptor-knee-native384`, `raptor-knee-native384dense`. Encoder `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (ImageNet-12k). **Datos de entrenamiento no descritos en este notebook** | Celda 22, líneas 42-69. |
| 5 | **4 variantes CoAt** (Mattia Angeli: `resgated`, `global96`, `d4_swa3`, `repairv1`) | Mismo encoder ImageNet-12k; datos de entrenamiento no descritos | Celda 22, líneas 648-995. |

El texto de la celda 2 del 0.957 sí describe el método de **otro** dataset de dreaddevelopment (`raptor-knee-widedense`: reportes de la competición + etiquetas de un LLM, 4,349 estudios). **No puedo trasladar esa descripción a los datasets `maxspan` / `native384` / `native384dense` que usa el 0.940**, porque son otros archivos [no verificado].

---

## 2. Clasificación con la regla del host

Criterio que aplico: **limpio** = solo datos de la competición, ImageNet/modelos de propósito general o recursos con acceso por registro y términos, y sin depender de nada con afiliación institucional; **dudoso** = el acceso parece cumplir la regla pero hay una condición abierta (restricción geográfica, licencia, procedencia sin documentar, pregunta sin responder del host); **problemático** = el acceso exige o parece exigir afiliación o acuerdo institucional.

### 2.1 Notebook 0.957

| # | Componente | Clase | Razón |
|---|---|---|---|
| 1 | Raptor con checkpoint «oai» | **Dudoso (alto)** | OAI en NDA parece «registro + términos»: un participante cuenta aprobación inmediata como individuo, sin institución. Pero (a) el propio notebook lo llama «controlled access»; (b) hay informes de bloqueo por país, lo que choca con «equally accessible» y el host no respondió; (c) lo que se envía son **pesos entrenados con OAI**, y el host no contestó si eso vale; (d) cómo entró OAI en el entrenamiento solo lo [inferido] del nombre del archivo y de la celda 1. Es la pieza más debatida del hilo. |
| 2 | Raptor, encoder ImageNet-12k | **Limpio** | Pesos públicos de timm de propósito general. |
| 3 | Etiquetas débiles de reportes (LLM) | **Limpio, con una nota** | Son reportes de la propia competición y datasets públicos de Kaggle accesibles para todos. Nota: un comentario del hilo cuestiona que los LLM «profesores» tengan preentrenamiento desconocido; el host no se pronunció. |
| 4 | R7 con fastMRI+ auxiliar | **Problemático** | La página de NYU, según una búsqueda web que hice hoy, pide en la solicitud la **afiliación institucional** del investigador y el uso propuesto, y exige aceptar un Data Sharing Agreement; la tabla del hilo dice lo mismo («Application required, including institutional affiliation»). Eso encaja en «limit access to users with institutional affiliations». Contra: un comentario del hilo dice que no hace falta correo académico ni aprobación, y un artículo de 2020 dice que es abierto con registro. **No vi el formulario vigente**, así que esta clasificación es la prudente, no una verificación. Efecto: entró solo en entrenamiento, pero está incorporado en los pesos R7 que no se pueden separar. |
| 5 | R7 con anclas SKM-TEA | **Dudoso (bajo), condicional** | El acceso parece cumplir la regla: la página de AIMI lo ofrece al público «sin coste, para investigación no comercial» (según la búsqueda); un comentario del hilo habla de registro en Redivis; **no encontré exigencia de afiliación**. Condiciones abiertas: la licencia no comercial y su efecto sobre redistribuir un checkpoint derivado en un dataset de Kaggle [no verificado]; si `anatomy.pt` es el modelo original o uno ajustado [no verificado]; y que **el notebook no lo divulga**. Solo cuenta si `views` es `v3k`/`v4`. |
| 6 | R7, encoder ImageNet | **Limpio** | Igual que el 2. |
| 7 | Clasificador de lado ResNet34 | **Dudoso (bajo)** | Datos de entrenamiento no documentados. Muy probablemente datos de la competición, pero no consta. |
| 8 | Rama OFL-L / rama DINOv3 | **Limpio si no se ejecutan; dudoso si sí** | Condicionales. No he podido confirmar que estén desactivadas. |

**Balance del 0.957:** una pieza problemática (fastMRI+ en R7), una dudosa-alta (OAI en Raptor) y una dudosa-baja condicional (SKM-TEA). **Las dos mitades del 0.5 / 0.5 llevan algo.**

### 2.2 Notebook 0.940

| # | Componente | Clase | Razón |
|---|---|---|---|
| 1 | DINOv2-S, 20 miembros | **Limpio** | Base de Meta de propósito general; ajuste fino de un participante, previsiblemente sobre datos de la competición pero **no descrito** [no verificado]. |
| 2 | A5 | **Limpio, sin verificar** | Datos de entrenamiento y backbone no descritos. |
| 3 | RadImageNet (encoder) | **Limpio, con una nota** | Los pesos se descargan sin registro (Hugging Face, licencia MIT según la ficha; el Atlas de RSNA indica CC BY 4.0: **las dos licencias difieren, hay que mirar el archivo de licencia**), así que son «equally accessible». El conjunto de imágenes en sí se pide por solicitud. Nada en el hilo ni en la regla del host lo cubre, y no vi ninguna exigencia institucional. |
| 4 | Cabezas E10 / E13 / E11 y calibrador Ridge | **Limpio, sin verificar** | Datos de entrenamiento no descritos. |
| 5 | Raptor 4 brazos | **Dudoso (por opacidad)** | No hay rastro de OAI en el 0.940, pero **no hay descripción de con qué se entrenaron** los brazos v5 / v8 / v10. El hilo cuenta que «a public model trained on OAI has now spread through the code section via blind fork ensembling». No puedo ni confirmar ni descartar que alguno de estos brazos esté afectado. |
| 6 | 4 variantes CoAt | **Limpio, sin verificar** | Igual que el 2. |

**Balance del 0.940:** ningún componente aparece como problemático **en lo documentado**; el riesgo restante es de procedencia no documentada en piezas de terceros, sobre todo Raptor. Ausencia de evidencia no es evidencia de ausencia.

### 2.3 Componentes propios que podrían entrar en un final

| Componente | Clase | Razón |
|---|---|---|
| **Brazo v5** (paquete de Owen: `eff`, `cnxhib`, `effhib`, `r34att`; 20 CNN) | **Limpio, con una nota** | Todos parten de pesos ImageNet de torchvision (`auditoria_owen_0918.md`, tabla de modelos). La nota está en el mismo reporte: las etiquetas híbridas de `runs_cons/` vienen de «etiquetas públicas de otros participantes» y su construcción **no se verificó** (misma limitación que las etiquetas débiles del punto 2.1.3). |

---

## 3. Dos finales propuestos (solo propuesta)

Referencia de rendimiento: puntuaciones públicas **0.957** y **0.940** según los títulos de los notebooks. No tengo puntuación privada de ninguno. La diferencia pública es 0.017.

### Final A, «máxima»

- **Componentes:** el notebook 0.957 completo (Raptor «oai» con peso 0.5 + R7 ConvNeXt-small con fastMRI+ auxiliar con peso 0.5, mezcla por rangos, `DEADLINE_HOURS = 8.3`). **Opcional:** nuestra celda v5 detrás con `V5_W = 0.20` (`kaggle/notebooks/0957_mas_v5_diag.ipynb`), solo si antes se mide que no empeora; ese peso se diseñó contra la mezcla del 0.940 y en el 0.957 no está medido ([inventario_base_0957.md §6](inventario_base_0957.md)).
- **Qué aporta:** el mejor público conocido (0.957).
- **Riesgos:**
  1. **fastMRI+ en R7 (problemático):** si el host o la auditoría de los premios lo considera afiliación institucional, el modelo incumple la regla y no se puede quitar sin reentrenar.
  2. **OAI en Raptor (dudoso-alto):** puede resolverse a favor o en contra; el host ya evitó pronunciarse una vez y quedan preguntas abiertas (pesos derivados, restricción por país).
  3. **SKM-TEA (dudoso-bajo, condicional):** solo si `views` es `v3k`/`v4`.
  4. **Transparencia:** habría que declarar OAI, fastMRI+ y, si aplica, SKM-TEA. El notebook original solo cita OAI.
  5. **Exposición:** este riesgo se concentra en los equipos que se auditan a fondo (los premios), según lo que dicen varios participantes en el hilo; no es una afirmación del host.
- **Antes de elegirlo:** abrir `manifest.json` de `rsna-knee-compact-r7cnxfsss` (¿`views`?, ¿`backbone`?) y comprobar el formulario vigente de fastMRI.

### Final B, «más limpia»

- **Componentes:** el notebook 0.940 completo (DINOv2-S 20 miembros + A5 + RadImageNet con calibrador + Raptor 4 brazos + 4 variantes CoAt, con los pesos por hallazgo de 0.70 a 1.00). **Opcional:** el brazo v5 con peso bajo (por ejemplo 0.10-0.20, al estilo `celda_brazo_v5.py`), porque encaja con el 0.940 para el que se diseñó, es ImageNet-only y añade familias distintas.
- **Qué aporta:** menor exposición documentada; ningún componente problemático por lo que consta en el notebook.
- **Riesgos:**
  1. **Procedencia opaca de terceros:** Raptor (4 brazos), A5, CoAt y las cabezas Rad no describen sus datos. Si alguno se entrenó con OAI u otro dato dudoso, no se ve desde aquí.
  2. **RadImageNet:** el host no se pronunció; la discrepancia de licencia (MIT / CC BY 4.0) debe cerrarse.
  3. **Rendimiento:** pierde unas 0.017 en público frente al A. Esa diferencia puede no mantenerse en el privado.
  4. **Etiquetas del v5 si se añade:** la nota de `runs_cons/` del apartado 2.3.
  5. **Transparencia:** declarar RadImageNet como preentrenamiento y las fuentes de etiquetas.
- **Antes de elegirlo:** pedir o buscar en los datasets de dreaddevelopment, Mattia Angeli, prvsiyan, sofiaanjenje y antoinegg1 una descripción de datos de entrenamiento, o aceptar la opacidad como riesgo residual.

### Cómo se comparan

| | A «máxima» | B «más limpia» |
|---|---|---|
| Público (referencia) | 0.957 | 0.940 |
| Componente problemático documentado | 1 (fastMRI+ en R7) | 0 |
| Componentes dudosos | OAI en Raptor (alto), SKM-TEA (bajo, condicional) | Raptor por opacidad; RadImageNet por licencia |
| Se puede quitar la pieza dudosa sin reentrenar | No (está en los pesos) | Sí, por miembros (cada familia es un bloque separado) |
| Divulgación necesaria | OAI, fastMRI+, SKM-TEA | RadImageNet, etiquetas débiles |
| Qué lo cambiaría | Un «no» del host sobre OAI o fastMRI | Que se demuestre OAI en Raptor v5 / v8 / v10 |

No propongo una tercera mezcla («0.957 sin la mitad problemática»): R7 no se puede separar de su entrenamiento fastMRI+, y Raptor «oai» tampoco de OAI, así que no existe una versión del 0.957 que conserve una mitad limpia sin sustituirla por piezas del 0.940.

---

## 4. Lo que no pude comprobar

1. **`views` y `backbone` reales de R7**, en `manifest.json` (fuera del notebook). Decide si SKM-TEA se usa y si OFL-L está desactivada.
2. **Cómo entra OAI en el checkpoint de Raptor.** Solo tengo el nombre del archivo, el nombre del dataset y la cita de NDA en la celda 1.
3. **La captura de pantalla de la respuesta de NDA** en el hilo (imagen adjunta).
4. **El formulario vigente de fastMRI** y los términos exactos de SKM-TEA en AIMI/Redivis. Mi resumen sale de resultados de búsqueda web, no de leer las páginas oficiales ni los acuerdos.
5. **Datos de entrenamiento de los componentes de terceros del 0.940** (todos), y del clasificador de lado del 0.957.
6. **El log que dices del 0.957 con `skm_tea_joint`**: no lo encuentro en el repo; la cadena está solo en el código incrustado.
7. **Que el host acepte o rechace nada de esto.** Lo único oficial es el párrafo citado en el apartado 0, que explícitamente no designa datasets.

Fuentes web usadas para los puntos de acceso (consultadas hoy, no leídas íntegramente): [fastMRI, catálogo de datos de NYU](https://datacatalog.med.nyu.edu/dataset/10389), [Radiology: AI 2020 (fastMRI)](https://pubs.rsna.org/doi/10.1148/ryai.2020190007), [SKM-TEA en Stanford AIMI](https://aimi.stanford.edu/datasets/skm-tea-knee-mri), [pesos RadImageNet en Hugging Face](https://huggingface.co/Lab-Rasool/RadImageNet), [RadImageNet en el Atlas de RSNA](https://atlas.rsna.org/cards/61caf4ba-8941-4699-9824-9e341d9661b9).
