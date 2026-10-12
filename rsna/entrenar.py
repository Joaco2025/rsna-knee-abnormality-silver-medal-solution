#!/usr/bin/env python
"""Entrena el modelo de imagen con los objetivos derivados de los informes.

Entrena con los estudios sin etiqueta oficial y valida de dos formas:
  - oficial: AUC contra las etiquetas reales de los 58 estudios oficiales (la métrica de Kaggle,
    pero con muy pocos casos: cada AUC tiene un margen de unos ±0.12);
  - interna: AUC contra los objetivos derivados (binarizados en 0.5) de un 10 % apartado. Tiene
    muchos más casos, pero mide el acuerdo con Qwen, no con la verdad.

Con --folds K --fold I se aparta la partición I de K en vez de ese 10 %; entrenando las K se
obtiene una predicción fuera de muestra para cada estudio (predicciones.npz de cada corrida).

Uso:  python rsna/entrenar.py --data DIR_NPY --labels TABLA.parquet --out DIR [opciones]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo import LABELS, KneeNet, prepare  # noqa: E402


class Studies(Dataset):
    def __init__(self, table: pd.DataFrame, data: Path, crop: int, train: bool):
        self.uids = table.StudyInstanceUID.tolist()
        self.t = np.nan_to_num(table[[f"t_{c}" for c in LABELS]].to_numpy(np.float32))
        self.w = table[[f"w_{c}" for c in LABELS]].to_numpy(np.float32)
        self.y = table[[f"y_{c}" for c in LABELS]].to_numpy(np.float32)
        self.data, self.crop, self.train = data, crop, train

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        arr = np.load(self.data / f"{self.uids[i]}.npy", mmap_mode="r")
        size = arr.shape[-1]
        if self.train:
            r, c = np.random.randint(0, size - self.crop + 1, 2)
        else:
            r = c = (size - self.crop) // 2
        x = torch.from_numpy(np.ascontiguousarray(arr[:, :, r:r + self.crop, c:c + self.crop]))
        present = (x.flatten(1).amax(dim=1) > 0).float()
        return x, present, torch.tensor(self.t[i]), torch.tensor(self.w[i]), torch.tensor(self.y[i])


def macro_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> tuple[float, list[float]]:
    aucs = []
    for j in range(y.shape[1]):
        m = np.ones(len(y), bool) if w is None else w[:, j] > 0
        yy = y[m, j]
        aucs.append(roc_auc_score(yy, p[m, j]) if 0 < yy.sum() < len(yy) else float("nan"))
    return float(np.nanmean(aucs)), aucs


@torch.no_grad()
def predict(model, loader, device, n_slices: int, amp: bool):
    model.eval()
    ps, ts, ws, ys = [], [], [], []
    for x, present, t, w, y in loader:
        x = prepare(x.to(device, non_blocking=True), n_slices, train=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x, present.to(device))
        ps.append(logits.float().sigmoid().cpu().numpy())
        ts.append(t.numpy()); ws.append(w.numpy()); ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ts), np.concatenate(ws), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--weights", default="", help="pesos preentrenados de torchvision (.pth)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--slices", type=int, default=16, help="cortes por ranura que ve el modelo")
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 2)
    ap.add_argument("--limit", type=int, default=0, help="estudios de entrenamiento, para pruebas")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=10, help="número de particiones de la validación interna")
    ap.add_argument("--fold", type=int, default=0, help="partición que se aparta para validar")
    ap.add_argument("--mlflow", default="", help="URI de seguimiento de MLflow (p. ej. sqlite:////ruta/mlflow.db); vacío = no registrar")
    ap.add_argument("--mlflow-artifacts", default="", help="directorio de artefactos del experimento")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert 0 <= args.fold < args.folds, "--fold debe estar entre 0 y --folds - 1"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    amp = not args.no_amp

    table = pd.read_parquet(args.labels)
    table = table[[(args.data / f"{u}.npy").exists() for u in table.StudyInstanceUID]].reset_index(drop=True)
    official = table[table.oficial]
    rest = table[~table.oficial]
    fold = rest.StudyInstanceUID.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % args.folds)
    val_int, train = rest[fold == args.fold], rest[fold != args.fold]
    has_target = train[[f"w_{c}" for c in LABELS]].sum(axis=1) > 0
    train = train[has_target]
    if args.limit:
        train = train.head(args.limit)
    print(f"entrenamiento={len(train)} validación interna={len(val_int)} validación oficial={len(official)} "
          f"partición={args.fold}/{args.folds} red={args.backbone} cortes={args.slices} recorte={args.crop} lote={args.batch} épocas={args.epochs} amp={amp}", flush=True)

    def loader(tab, train_mode):
        return DataLoader(Studies(tab, args.data, args.crop, train_mode), batch_size=args.batch, shuffle=train_mode,
                          num_workers=args.workers, pin_memory=True, drop_last=train_mode,
                          persistent_workers=args.workers > 0)

    dl_train, dl_int, dl_off = loader(train, True), loader(val_int, False), loader(official, False)
    model = KneeNet(args.backbone, args.weights or None).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl_train)
    warm = max(1, len(dl_train) // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ml = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow)
            if mlflow.get_experiment_by_name("rsna-knee") is None:
                mlflow.create_experiment("rsna-knee", artifact_location=args.mlflow_artifacts or None)
            mlflow.set_experiment("rsna-knee")
            ml = mlflow
            ml.start_run(run_name=args.run_name or args.out.name)
            ml.log_params({k: str(v) for k, v in vars(args).items()})
            ml.log_params({"n_train": len(train), "n_val_int": len(val_int), "n_val_oficial": len(official),
                           "commit": os.environ.get("GIT_COMMIT", ""), "job": os.environ.get("SLURM_JOB_ID", "")})
        except Exception as exc:   # el seguimiento nunca debe detener el entrenamiento
            print(f"MLflow desactivado: {type(exc).__name__}: {exc}", flush=True)
            ml = None

    history, best = [], {"oficial": -1.0, "interna": -1.0}
    preds_int, preds_off = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for step, (x, present, t, w, _) in enumerate(dl_train, 1):
            x = prepare(x.to(device, non_blocking=True), args.slices, train=True)
            t, w = t.to(device), w.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x, present.to(device))
            loss = (F.binary_cross_entropy_with_logits(logits.float(), t, reduction="none") * w).sum() / w.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            seen += x.shape[0]; loss_sum += loss.item() * x.shape[0]
            if step in (5, 20, 50) or step % 100 == 0:
                rate = seen / (time.time() - t0)
                print(f"  época {epoch} paso {step}/{len(dl_train)} pérdida {loss_sum / seen:.4f} "
                      f"{rate:.1f} estudios/s ({rate * 4 * args.slices:.0f} cortes/s)", flush=True)
            if not math.isfinite(loss.item()):
                raise SystemExit("la pérdida dejó de ser finita")
        train_s = time.time() - t0

        p_int, t_int, w_int, _ = predict(model, dl_int, device, args.slices, amp)
        auc_int, per_int = macro_auc((t_int >= 0.5).astype(int), p_int, w_int)
        p_off, _, _, y_off = predict(model, dl_off, device, args.slices, amp)
        preds_int.append(p_int); preds_off.append(p_off)
        auc_off, per_off = macro_auc(y_off.astype(int), p_off)
        row = {"epoch": epoch, "loss": loss_sum / max(seen, 1), "auc_interna": auc_int, "auc_oficial": auc_off,
               "segundos_entrenamiento": round(train_s, 1), "estudios_por_s": round(seen / train_s, 2)}
        history.append(row)
        print(f"época {epoch}: pérdida {row['loss']:.4f} | AUC interna {auc_int:.4f} | AUC oficial {auc_off:.4f} | "
              f"{train_s / 60:.1f} min ({row['estudios_por_s']} estudios/s)", flush=True)
        if ml:
            try:
                ml.log_metrics({"loss": row["loss"], "auc_interna": auc_int, "auc_oficial": auc_off,
                                "estudios_por_s": row["estudios_por_s"]}, step=epoch)
                # MLflow no admite apóstrofos en los nombres (Baker's)
                ml.log_metrics({f"oficial_{c}".replace("'", ""): a for c, a in zip(LABELS, per_off) if a == a}, step=epoch)
                ml.log_metrics({f"interna_{c}".replace("'", ""): a for c, a in zip(LABELS, per_int) if a == a}, step=epoch)
            except Exception as exc:
                print(f"MLflow: {type(exc).__name__}: {exc}", flush=True)

        state = {"model": model.state_dict(), "backbone": args.backbone, "slices": args.slices, "crop": args.crop,
                 "epoch": epoch, "auc_interna": auc_int, "auc_oficial": auc_off, "labels": LABELS}
        torch.save(state, args.out / "last.pt")
        for key, val in (("oficial", auc_off), ("interna", auc_int)):
            if val > best[key]:
                best[key] = val
                torch.save(state, args.out / f"best_{key}.pt")
                if key == "interna":
                    best["por_hallazgo"] = {"oficial": per_off, "interna": per_int, "epoch": epoch}
        pd.DataFrame(history).to_csv(args.out / "historial.csv", index=False)
        # Predicciones de cada época, (épocas, estudios, hallazgos), para evaluar conjuntos sin volver a inferir.
        np.savez(args.out / "predicciones.npz", fold=args.fold, folds=args.folds,
                 uid_interna=val_int.StudyInstanceUID.to_numpy(str), p_interna=np.stack(preds_int),
                 uid_oficial=official.StudyInstanceUID.to_numpy(str), p_oficial=np.stack(preds_off))

    print(f"\n== mejor AUC interna {best['interna']:.4f}, mejor AUC oficial {best['oficial']:.4f}")
    sel = best.get("por_hallazgo")
    if sel:
        print(f"-- AUC por hallazgo en la época {sel['epoch']} (la de mejor AUC interna)")
        print(f"{'hallazgo':<17}{'oficial(58)':>12}{'interna':>9}")
        for c, a, b in zip(LABELS, sel["oficial"], sel["interna"]):
            print(f"{c:<17}{a:>12.3f}{b:>9.3f}")
        print(f"{'MEDIA':<17}{np.nanmean(sel['oficial']):>12.3f}{np.nanmean(sel['interna']):>9.3f}")
    (args.out / "resumen.json").write_text(json.dumps({"historial": history, "mejor": {k: v for k, v in best.items() if k != "por_hallazgo"}}, indent=2))
    if ml:
        try:
            ml.log_metrics({"mejor_auc_interna": best["interna"], "mejor_auc_oficial": best["oficial"]})
            ml.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
