# Plan experimental (criterios fijados antes de ver ningún resultado)

**Validación disponible:** no hay ningún conjunto limpio y grande puntuado como el test. Por eso los cambios de mezcla se
deciden en el **leaderboard público**. Esto es legítimo solo con pocos envíos, cada uno con un mecanismo fijado antes y
comparado contra el mejor envío anterior en el **mismo** conjunto público.

**Resolución del público:** se muestran 3 decimales, pero el puesto refleja el score sin redondear del mejor envío del
equipo. «Mejor puesto con el mismo score mostrado» detecta mejoras menores que 0.001.

**Presupuesto:** ~6–7 h de Kaggle por envío y cierre el 22-oct a las 23:59 UTC. Caben unos 6–8 envíos con resultado; el
último debe empezar antes de ~15:00 UTC del 22-oct.

## P0. E (ya corriendo, sin coste nuevo)

**Éxito:** muestra ≥ 0.959, o 0.958 con mejor puesto que 987 contando la deriva (comparar con los equipos vecinos).
**Fracaso:** 0.957 o menos.

## P1. E-sust (prioridad A): S6 dentro de la mitad Raptor

**Qué cambia:** solo la celda final.
```
mitad_raptor = 0.75 · [0.65·rango(384) + 0.35·rango(224)] + 0.25 · rango(S6)
```
Después: la mezcla por hallazgo de heliosli con R6 (sin cambios) y el lector a 1/10 (sin cambios).

**Qué se mantiene:** modelos, checkpoints, TTA, pesos de heliosli, lector, tiempo de ejecución (igual que E) y versión
estricta (falla con `C_FALLO`).

**Por qué podría mejorar:**
- S6 (0.950) y Raptor (0.949) comparten etiquetas y el mismo papel.
- R6 es el componente con supervisión distinta, y fue el salto más grande de la escalera pública.
- E reduce un 25 % el peso de R6 en todos los hallazgos; E-sust no lo toca.

**Prueba mínima de refutación:** el mismo envío.

**Coste:**
- ~2–3 h de construcción. Hay que reescribir la etapa 5 de E para recalcular la mezcla A3 desde `raptor_sub.csv`,
  `raptor224_sub.csv`, la salida de R6 y S6, y reaplicar la Parte B.
- Prueba sintética + prueba de 3 estudios en Kaggle (~40 min).
- 1 envío (~7 h).

**Éxito:** mejor puesto que el mejor de {«0.958», E} con el mismo score mostrado, o un score mostrado mayor.
**Fracaso:** puesto igual o peor; en ese caso no se prueban más variantes de dónde colocar a S6.

**Orden según P0:**
- **E mejora:** E-sust se compara contra E.
- **E no mejora:** E-sust es la última oportunidad de S6 en la mezcla A. Si tampoco mejora, A = «0.958».

## P2. D, semillas de S6 (prioridad B, ya preparada)

- Solo tiene sentido para A si P0 o P1 muestran que S6 suma. Para el final B tiene sentido en cualquier caso.
- **Coste:** job de anclas 2092983 (1 GPU, ≤ 8 h, en cola) + 3 GPU × 12–22 h + paquete de Kaggle.
- **Corte:** si las semillas no están entrenadas el 21-oct a las 12:00 UTC, no entran en ningún envío.
- **Éxito (en A):** con S6 = media de 1 + k semillas en la misma posición que gane P0/P1, mejor puesto que esa versión.
- **Restricción:** en E o E-sust caben 1–2 semillas extra (~0.4 h cada una).

## P3. H13: ViT-L OrthoFoundation (prioridad C, solo si sobra tiempo el 20-oct)

**Mínimo:** inferencia de `best.pt` en Gold-58 y en 3 estudios de test en Kaggle.
**Para seguir hace falta:**
- AUC en Gold ≥ 0.90 sin selección por Gold.
- Correlación de rangos con Raptor ≤ 0.85 por hallazgo débil.
- Tiempo en T4 ≤ 1.5 h para 1,284 estudios.

**Se descarta** si falla cualquiera de las tres, o si la licencia de los pesos OFL-L no está clara.

## Lo que NO se vuelve a probar

- CNN propias en la mezcla A (v9c2 → 0.955).
- Meta-apilado.
- Transformaciones entre hallazgos (A2 y A3 de hoy).
- Flip TTA en S6 (incompatible).
- R7 junto a R6.
- Flip en R7.
- Ajustar pesos por hallazgo sobre Gold-58 sin un mecanismo detrás.
