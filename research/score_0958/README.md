# Investigación para superar 0.958 (2026-10-19)

**En una frase:** con lo disponible, la única palanca defendible a corto plazo es **dónde se coloca S6 en la mezcla**
(E-sust: dentro de la mitad Raptor, sin diluir a R6). Las vías de post-procesado y TTA quedaron refutadas hoy con datos
o por código.

## Estado

- El 0.958 lo produjo la copia de la v25/27 de goodpjw2008: Raptor 384 + 224 con espejo, R6 con flip, pesos por hallazgo
  de heliosli y lector a 1/10. Recibo de jcastro: 0.958, puesto 987, ~6 h.
- E (base + S6 al 0.25) está corriendo.
- No hay OOF de ningún modelo público.
- Gold-58 es el único conjunto puntuado como el test, y solo tenemos predicciones de Raptor en él (contaminadas).

## Conclusiones

1. **El cuello de botella son 4 hallazgos:** Lateral OA (0.78 en Gold), Synovitis (0.82), PF OA (0.86) y Lateral
   Meniscus (0.89). Los otros 8 están en 0.95–0.98.
2. **Refutado hoy** (análisis A1–A3 y código):
   - Corrección trauma ↔ degeneración: Raptor ya la reproduce.
   - Préstamo entre hallazgos débiles: Δ ≤ 0.
   - Flip TTA en S6: el código lo impide y S6 nunca vio espejos.
3. **Lo que sí tiene mecanismo:** S6 comparte etiquetas y papel con Raptor, mientras que R6 es el componente con
   supervisión distinta y el que más movió el público. E le quita un 25 % de peso a R6 en todos los hallazgos.
   **E-sust** pone S6 dentro de la mitad Raptor y deja intactos a R6 y los pesos de heliosli.
4. Las semillas de S6 (D) tienen un techo bajo para la mezcla A (S6 pesa 0.25 y solo caben 1–2 semillas por tiempo);
   sirven sobre todo al final B.
5. El ViT-L de OrthoFoundation (`best.pt`) es la única fuente de diversidad de arquitectura real que no usa nadie, pero
   no tiene ninguna evidencia de score y su coste es desconocido: prioridad C.

## Archivos

| Archivo | Contenido |
|---|---|
| [current_state.md](current_state.md) | Inventario verificado |
| [technical_analysis.md](technical_analysis.md) | Cuello de botella, refutaciones, inferencia y diversidad |
| [nonconventional_hypotheses.md](nonconventional_hypotheses.md) | 18 hipótesis por mecanismo, con estado |
| [opportunity_matrix.csv](opportunity_matrix.csv) | Puntuaciones 0–5 y prioridad |
| [external_findings.md](external_findings.md) | Fuentes públicas y nivel de evidencia |
| [eligibility.md](eligibility.md) | Datos externos y licencias |
| [experiment_plan.md](experiment_plan.md) | P0–P3 con criterios fijados de antemano |
| [reproducibility.md](reproducibility.md) | Comandos, entradas y limitaciones |
| `analisis/` | Scripts A1–A3 y sus JSON |
