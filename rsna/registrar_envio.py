#!/usr/bin/env python
"""Anota en MLflow un envío a Kaggle: la puntuación en el entrenamiento que lo produjo y los
pesos exactos enviados como una versión del modelo registrado `rsna-knee`.

Uso:
  python rsna/registrar_envio.py --job 2080214 --pesos RUTA.pt --puntuacion 0.875 \
      --notebook "notebook000579244b v2" --fecha 2026-10-11 [--nota "..."]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mlflow

PROJ = Path("/lustre/proyectos/p082")
URI = f"sqlite:///{PROJ}/mlflow/mlflow_joako.db"
MODEL = "rsna-knee"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--pesos", type=Path, required=True)
    ap.add_argument("--puntuacion", type=float, required=True, help="AUC en la tabla pública de Kaggle")
    ap.add_argument("--notebook", default="")
    ap.add_argument("--fecha", default="")
    ap.add_argument("--nota", default="")
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
    found = client.search_runs([exp.experiment_id], f"tags.mlflow.runName = 'job-{args.job}'")
    if not found:
        raise SystemExit(f"No hay un registro job-{args.job} en MLflow; cárgalo antes con registrar_mlflow.py")
    run = found[0]
    run_id = run.info.run_id
    sha = hashlib.sha256(args.pesos.read_bytes()).hexdigest()

    client.log_metric(run_id, "kaggle_publica", args.puntuacion)
    for k, v in {"kaggle.enviado": "si", "kaggle.puntuacion_publica": str(args.puntuacion),
                 "kaggle.notebook": args.notebook, "kaggle.fecha": args.fecha,
                 "pesos.sha256": sha, "pesos.archivo": args.pesos.name}.items():
        if v:
            client.set_tag(run_id, k, v)
    client.log_artifact(run_id, str(args.pesos), artifact_path="pesos")

    try:
        client.create_registered_model(MODEL, description="Modelos de imagen enviados a Kaggle (RSNA Knee Abnormalities Detection).")
    except mlflow.exceptions.MlflowException:
        pass   # ya existe
    source = f"{run.info.artifact_uri}/pesos/{args.pesos.name}"
    desc = (f"Entrenamiento job {args.job}. Tabla pública de Kaggle: {args.puntuacion}. "
            f"AUC oficial local (58 estudios): {run.data.metrics.get('mejor_auc_oficial', float('nan')):.4f}. {args.nota}").strip()
    mv = client.create_model_version(MODEL, source=source, run_id=run_id, description=desc,
                                     tags={"job": args.job, "kaggle_publica": str(args.puntuacion),
                                           "notebook": args.notebook, "sha256": sha})
    print(f"job-{args.job}: puntuación {args.puntuacion} anotada; modelo {MODEL} versión {mv.version} "
          f"-> {source}  sha256={sha[:12]}")


if __name__ == "__main__":
    main()
