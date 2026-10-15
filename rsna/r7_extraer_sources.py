#!/usr/bin/env python
"""Extrae los 7 fuentes incrustados (SOURCES) de la celda 4 del notebook 0.957 sin ejecutar nada.

Lee el .ipynb como JSON, analiza la celda con `ast` y evalúa solo los literales `SOURCES` y `SOURCE_SHA`
(ast.literal_eval). Escribe cada fuente en --out y comprueba su sha256 contra SOURCE_SHA del propio notebook.
El código extraído es material de terceros: se lee y se audita, no se ejecuta desde aquí.

Uso:  python rsna/r7_extraer_sources.py referencias/0-957-latest-dataset-score-in-one-click.ipynb rsna/r7_sources
"""
import ast
import hashlib
import json
import sys
from pathlib import Path


def literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main():
    nb, out = Path(sys.argv[1]), Path(sys.argv[2])
    cells = json.loads(nb.read_text())["cells"]
    src = "".join(cells[4]["source"])
    lines = [l for l in src.splitlines() if not l.lstrip().startswith(("%", "!"))]   # magias de IPython fuera
    tree = ast.parse("\n".join(lines))
    sources, shas = literal(tree, "SOURCES"), literal(tree, "SOURCE_SHA")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, text in sources.items():
        data = text.encode()
        sha = hashlib.sha256(data).hexdigest()
        (out / name).write_bytes(data)
        manifest[name] = {"bytes": len(data), "lineas": text.count("\n") + 1, "sha256": sha,
                          "sha256_notebook": shas.get(name), "coincide": sha == shas.get(name)}
        print(f"{name:26s} {len(data):>8,} B {manifest[name]['lineas']:>6} líneas  sha256 "
              f"{'OK' if manifest[name]['coincide'] else 'DISTINTO'}")
    (out / "MANIFEST.json").write_text(json.dumps({"notebook": str(nb), "celda": 4, "fuentes": manifest}, indent=1))


if __name__ == "__main__":
    main()
