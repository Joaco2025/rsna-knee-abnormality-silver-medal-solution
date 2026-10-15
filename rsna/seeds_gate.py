#!/usr/bin/env python
"""Gate de las semillas: sale con 0 solo si el resumen del piloto dice PASA en el checkpoint que decide (SWA).

El piloto (rsna/entrenar_coatnet.py) escribe `criterio_swa.veredicto` = "PASA" / "NO PASA" en resumen.json.
Cualquier otra cosa (archivo ausente, JSON roto, clave ausente, NO PASA) devuelve 3 y una línea con el motivo.

Uso:  python rsna/seeds_gate.py runs_joako/2085360/resumen.json
"""
import json
import sys
from pathlib import Path


def veredicto(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"no existe {path}"
    try:
        r = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"JSON ilegible: {type(exc).__name__}: {exc}"
    c = r.get("criterio_swa")
    if not isinstance(c, dict):
        return False, f"sin criterio_swa (abortado={r.get('abortado')!r}, criterio_error={r.get('criterio_error')!r})"
    v = c.get("veredicto")
    return v == "PASA", f"criterio_swa.veredicto={v!r} delta_vs_effhib={c.get('delta_vs_effhib')}"


if __name__ == "__main__":
    ok, motivo = veredicto(Path(sys.argv[1]))
    print(f"GATE {'ABIERTO' if ok else 'CERRADO'}: {motivo}", flush=True)
    sys.exit(0 if ok else 3)
