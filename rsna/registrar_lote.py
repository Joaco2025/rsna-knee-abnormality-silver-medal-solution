#!/usr/bin/env python
"""Anota en MLflow, desde un archivo JSON, lo que no se registró en vivo: corridas de otros
miembros del equipo, envíos de conjuntos de modelos, resúmenes de validación cruzada y etiquetas
para agrupar o descartar corridas.

El archivo es una lista de registros. Cada uno busca (o crea) la corrida `nombre` y le añade:
  historial   carpeta con historial.csv: las métricas por época que falten
  params      parámetros (solo al crear la corrida; MLflow no deja cambiarlos)
  tags        etiquetas (se pueden repetir y corregir)
  metricas    métricas sueltas, p. ej. kaggle_publica
  pesos       archivos enviados: se anota su nombre y su SHA-256, no se copian
Se puede repetir sin duplicar nada.

Uso:  python rsna/registrar_lote.py registros/ARCHIVO.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import pandas as pd

PROJ = Path("/lustre/proyectos/p082")
URI = f"sqlite:///{PROJ}/mlflow/mlflow.db"
PER_EPOCH = ["loss", "auc_interna", "auc_oficial", "estudios_por_s"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lote", type=Path)
    args = ap.parse_args()
    mlflow.set_tracking_uri(URI)
    exp = mlflow.set_experiment("rsna-knee")
    client = mlflow.MlflowClient()

    for rec in json.loads(args.lote.read_text()):
        name = rec["nombre"]
        found = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{name}'")
        if found:
            run_id, created = found[0].info.run_id, False
        else:
            run_id, created = client.create_run(exp.experiment_id, run_name=name).info.run_id, True
            for k, v in rec.get("params", {}).items():
                client.log_param(run_id, k, str(v)[:480])
        done = []
        if "historial" in rec:
            hist = pd.read_csv(Path(rec["historial"]) / "historial.csv")
            seen = {m.step for m in client.get_metric_history(run_id, "auc_interna")}
            new = hist[~hist.epoch.isin(seen)]
            for row in new.itertuples():
                for key in PER_EPOCH:
                    if key in hist:
                        client.log_metric(run_id, key, float(getattr(row, key)), step=int(row.epoch))
            client.log_metric(run_id, "mejor_auc_oficial", float(hist.auc_oficial.max()))
            client.log_metric(run_id, "mejor_auc_interna", float(hist.auc_interna.max()))
            done.append(f"{len(new)} épocas nuevas")
        for k, v in rec.get("metricas", {}).items():
            client.log_metric(run_id, k, float(v))
        for k, v in rec.get("tags", {}).items():
            client.set_tag(run_id, k, str(v))
        for i, path in enumerate(rec.get("pesos", []), 1):
            path = Path(path)
            client.set_tag(run_id, f"pesos.{i}.archivo", path.name)
            client.set_tag(run_id, f"pesos.{i}.sha256", sha256(path))
        # Las corridas creadas a mano quedan abiertas si no se cierran; las que siguen entrenando no se tocan.
        if created or rec.get("cerrar"):
            client.set_terminated(run_id)
        done += [f"{len(rec.get(k, {}))} {k}" for k in ("metricas", "tags", "pesos") if rec.get(k)]
        print(f"{name}: {'creada' if created else 'ya existía'}; " + ", ".join(done), flush=True)


if __name__ == "__main__":
    main()
