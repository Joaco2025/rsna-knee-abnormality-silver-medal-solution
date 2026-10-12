#!/usr/bin/env python
"""Carga en MLflow el historial de entrenamientos ya ejecutados (runs/<jobid>/historial.csv).

Sirve para los jobs que no pudieron registrar en vivo. Se puede repetir: en cada entrenamiento
solo añade las épocas que aún no están en MLflow.

Uso:  python rsna/registrar_mlflow.py JOBID [JOBID...]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import mlflow
import pandas as pd

PROJ = Path("/lustre/proyectos/p082")
URI = f"sqlite:///{PROJ}/mlflow/mlflow_joako.db"
METRICS = {"loss": "loss", "auc_interna": "auc_interna", "auc_oficial": "auc_oficial", "estudios_por_s": "estudios_por_s"}


def params_from_log(job: str) -> dict:
    """Parámetros que el entrenamiento imprime en su primera línea de resumen."""
    out = {}
    log_dirs = [PROJ / "code_joako" / "logs", PROJ / "code" / "logs"]
    for log_dir in log_dirs:
        for log in log_dir.glob(f"*_{job}.out"):
            text = log.read_text(errors="ignore")
            m = re.search(r"^entrenamiento=.*$", text, re.M)
            if m:
                out.update(dict(kv.split("=", 1) for kv in m.group(0).split() if "=" in kv))
            m = re.search(r"^commit=(\w+)", text, re.M)
            if m:
                out["commit"] = m.group(1)
            m = re.search(r"opciones=(.*)$", text, re.M)
            if m:
                out["opciones"] = m.group(1).strip()
        if out:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="+")
    args = ap.parse_args()
    mlflow.set_tracking_uri(URI)
    artifacts_dir = PROJ / "mlflow" / "artifacts_joako"
    exp = mlflow.get_experiment_by_name("rsna-knee")
    if exp is None:
        exp_id = mlflow.create_experiment("rsna-knee", artifact_location=str(artifacts_dir))
        exp = mlflow.get_experiment(exp_id)
    else:
        if exp.artifact_location and Path(exp.artifact_location).resolve() != artifacts_dir.resolve():
            raise SystemExit(
                f"ERROR: Discrepancia en MLflow: el experimento 'rsna-knee' ya existe con artifact_location='{exp.artifact_location}', "
                f"pero se requiere '{artifacts_dir}'. Deteniendo ejecución."
            )
    client = mlflow.MlflowClient()
    for job in args.jobs:
        p_joako = PROJ / "runs_joako" / job / "historial.csv"
        p_owen = PROJ / "runs" / job / "historial.csv"
        if p_joako.exists() and p_owen.exists():
            raise SystemExit(f"ERROR: Ambigüedad para job {job}: existe historial.csv en runs_joako y en runs.")
        elif p_joako.exists():
            hist_path = p_joako
        elif p_owen.exists():
            print(f"[INFO] Job {job}: leyendo historial compartido de Owen (solo lectura): {p_owen}")
            hist_path = p_owen
        else:
            print(f"{job}: sin historial.csv en runs_joako ni en runs")
            continue
        hist = pd.read_csv(hist_path)
        name = f"job-{job}"
        found = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = '{name}'")
        if found:
            run_id = found[0].info.run_id
        else:
            run_id = client.create_run(exp.experiment_id, run_name=name).info.run_id
            for k, v in {**params_from_log(job), "job": job}.items():
                client.log_param(run_id, k, str(v)[:480])
        done = {m.step for m in client.get_metric_history(run_id, "auc_oficial")}
        new = hist[~hist.epoch.isin(done)]
        for row in new.itertuples():
            for col, key in METRICS.items():
                client.log_metric(run_id, key, float(getattr(row, col)), step=int(row.epoch))
        client.log_metric(run_id, "mejor_auc_oficial", float(hist.auc_oficial.max()))
        client.log_metric(run_id, "mejor_auc_interna", float(hist.auc_interna.max()))
        print(f"{job}: {len(new)} épocas nuevas ({len(hist)} en total); mejor AUC oficial {hist.auc_oficial.max():.4f}")


if __name__ == "__main__":
    main()
