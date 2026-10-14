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
from modelo import LABELS, MCL, SWAP, KneeNet, drop_slot, mirror_batch, prepare  # noqa: E402
from preproc import slots_for  # noqa: E402


def mirror(arr: np.ndarray, planes: list[str]) -> np.ndarray:
    """Espejo izquierda-derecha coherente de un estudio (ranuras, cortes, filas, columnas): convierte una
    rodilla izquierda en la orientación de una derecha. En sagital se invierte el orden de los cortes;
    en coronal y axial, las columnas."""
    out = np.empty_like(arr)
    for k, plane in enumerate(planes):
        out[k] = arr[k, ::-1] if plane == "Sagittal" else arr[k, :, :, ::-1]
    return out


class Studies(Dataset):
    def __init__(self, table: pd.DataFrame, data: Path, crop: int, train: bool, left: set[str] | None = None,
                 mirror_p: float = 0.0):
        self.left = left or set()
        self.mirror_p = mirror_p if train else 0.0     # espejo con intercambio de etiquetas, solo al entrenar
        self.uids = table.StudyInstanceUID.tolist()
        self.t = np.nan_to_num(table[[f"t_{c}" for c in LABELS]].to_numpy(np.float32))
        self.w = table[[f"w_{c}" for c in LABELS]].to_numpy(np.float32)
        self.y = table[[f"y_{c}" for c in LABELS]].to_numpy(np.float32)
        self.data, self.crop, self.train = data, crop, train

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        arr = np.load(self.data / f"{self.uids[i]}.npy", mmap_mode="r")
        if self.uids[i] in self.left:
            arr = mirror(np.asarray(arr), [p for p, _ in slots_for(arr.shape[0])])
        t, w = self.t[i], self.w[i]
        if self.mirror_p > 0 and np.random.rand() < self.mirror_p:
            arr = mirror(np.asarray(arr), [p for p, _ in slots_for(arr.shape[0])])
            t, w = t[SWAP], w[SWAP]
            w[MCL] = 0.0
        size = arr.shape[-1]
        if self.train:
            r, c = np.random.randint(0, size - self.crop + 1, 2)
        else:
            r = c = (size - self.crop) // 2
        x = torch.from_numpy(np.ascontiguousarray(arr[:, :, r:r + self.crop, c:c + self.crop]))
        present = (x.flatten(1).amax(dim=1) > 0).float()
        return x, present, torch.tensor(t), torch.tensor(w), torch.tensor(self.y[i])


def macro_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> tuple[float, list[float]]:
    aucs = []
    for j in range(y.shape[1]):
        m = np.ones(len(y), bool) if w is None else w[:, j] > 0
        yy = y[m, j]
        aucs.append(roc_auc_score(yy, p[m, j]) if 0 < yy.sum() < len(yy) else float("nan"))
    return float(np.nanmean(aucs)), aucs


@torch.no_grad()
def predict(model, loader, device, n_slices: int, amp: bool, tta_planes: list[str] | None = None):
    """Con tta_planes, promedia con la predicción del estudio en espejo, devolviendo cada salida a su sitio."""
    model.eval()
    ps, ts, ws, ys = [], [], [], []
    for x, present, t, w, y in loader:
        raw, present = x.to(device, non_blocking=True), present.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            p = model(prepare(raw, n_slices, train=False), present).float().sigmoid()
            if tta_planes:
                pm = model(prepare(mirror_batch(raw, tta_planes), n_slices, train=False), present).float().sigmoid()[:, SWAP]
                pm[:, MCL] = p[:, MCL]
                p = (p + pm) / 2
        ps.append(p.cpu().numpy())
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
    ap.add_argument("--acumular", type=int, default=1, help="lotes cuyos gradientes se suman antes de cada paso del optimizador")
    ap.add_argument("--slices", type=int, default=16, help="cortes por ranura que ve el modelo")
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-red", type=float, default=0.0, help="tasa de aprendizaje de la red preentrenada; 0 = la misma que el resto")
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 2)
    ap.add_argument("--limit", type=int, default=0, help="estudios de entrenamiento, para pruebas")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atencion", default="comun", choices=["comun", "hallazgo"],
                    help="hallazgo: cada hallazgo tiene su propia atención sobre los cortes")
    ap.add_argument("--lados", type=Path, default=None,
                    help="tabla con el lado de cada estudio (eda/09_lateralidad.py): las rodillas izquierdas se llevan en espejo a la orientación de las derechas")
    ap.add_argument("--espejo", type=float, default=0.0,
                    help="probabilidad de entrenar cada estudio en espejo intercambiando las etiquetas medial y lateral (requiere --lados)")
    ap.add_argument("--tta-espejo", action="store_true", help="al validar, promediar con la predicción en espejo (requiere --lados)")
    ap.add_argument("--sinovitis-derrame", action="store_true", help="sinovitis = salida propia + beta aprendido * salida de derrame")
    ap.add_argument("--vistas", default="comun", choices=["comun", "hallazgo"],
                    help="hallazgo: cada hallazgo aprende cuánto pesa cada serie (requiere --atencion hallazgo)")
    ap.add_argument("--peso-hallazgos", default="", help='multiplicadores del peso en la pérdida, p. ej. "Lateral OA=3,Synovitis=3"')
    ap.add_argument("--contexto", default="no", choices=["no", "gru"],
                    help="gru: posición del corte y red recurrente entre los cortes de cada serie, antes de la atención")
    ap.add_argument("--pool", default="media", choices=["media", "mediamax", "rejilla", "espacial"],
                    help="cómo se resume cada corte: media espacial; mitad media y mitad máximo; rejilla de 2x2; o espacial: "
                         "atención 2D con posición propia de cada hallazgo (requiere --atencion hallazgo)")
    ap.add_argument("--aug", default="suave", choices=["suave", "fuerte"],
                    help="fuerte añade giro, escala, desplazamiento, gamma, ruido y borrado por serie")
    ap.add_argument("--quitar-serie", type=float, default=0.0, help="probabilidad de apagar una serie al azar de cada estudio")
    ap.add_argument("--mixup", type=float, default=0.0, help="alfa de la mezcla de pares de estudios; 0 = sin mezcla")
    ap.add_argument("--consistencia", type=float, default=0.0,
                    help="peso de la penalización por predecir distinto en la versión suave y en la fuerte del mismo estudio")
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
    n_slots = int(np.load(args.data / f"{train.StudyInstanceUID.iloc[0]}.npy", mmap_mode="r").shape[0])   # series por estudio
    print(f"entrenamiento={len(train)} validación interna={len(val_int)} validación oficial={len(official)} "
          f"partición={args.fold}/{args.folds} aug={args.aug} quitar_serie={args.quitar_serie} mixup={args.mixup} "
          f"consistencia={args.consistencia} series={n_slots} atencion={args.atencion} contexto={args.contexto} pool={args.pool} vistas={args.vistas} lados={'si' if args.lados else 'no'} espejo={args.espejo} tta_espejo={args.tta_espejo} sinovitis_derrame={args.sinovitis_derrame} peso_hallazgos={args.peso_hallazgos or '-'} acumular={args.acumular} red={args.backbone} cortes={args.slices} recorte={args.crop} lote={args.batch} épocas={args.epochs} amp={amp}", flush=True)

    left = None
    if args.lados:
        sides = pd.read_parquet(args.lados)
        left = set(sides.StudyInstanceUID[sides.lado_final == "L"])
        print(f"lados: {len(left)} estudios izquierdos se entrenan y validan en espejo", flush=True)

    assert not (args.espejo or args.tta_espejo) or args.lados, "--espejo y --tta-espejo requieren --lados"
    planes = [p for p, _ in slots_for(n_slots)] if n_slots in (4, 6) else None
    tta = planes if args.tta_espejo else None
    info = {}
    if (args.data / "preproc.json").exists():      # cómo se hicieron los tensores (recorte físico, cobertura de cortes)
        info = json.loads((args.data / "preproc.json").read_text())

    def loader(tab, train_mode):
        return DataLoader(Studies(tab, args.data, args.crop, train_mode, left, args.espejo), batch_size=args.batch, shuffle=train_mode,
                          num_workers=args.workers, pin_memory=True, drop_last=train_mode,
                          persistent_workers=args.workers > 0)

    if args.peso_hallazgos:      # más peso en la pérdida a ciertos hallazgos; las medidas de validación no cambian
        train = train.copy()
        for item in args.peso_hallazgos.split(","):
            name, factor = item.rsplit("=", 1)
            assert name in LABELS, f"hallazgo desconocido: {name}"
            train[f"w_{name}"] = train[f"w_{name}"] * float(factor)
    dl_train, dl_int, dl_off = loader(train, True), loader(val_int, False), loader(official, False)
    model = KneeNet(args.backbone, args.weights or None, n_slots=n_slots, per_label=args.atencion == "hallazgo",
                    context=args.contexto == "gru", pool=args.pool, view_gate=args.vistas == "hallazgo",
                    syn_eff=args.sinovitis_derrame).to(device).to(memory_format=torch.channels_last)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_red or args.lr}, {"params": rest, "lr": args.lr}], weight_decay=args.wd)
    steps = args.epochs * len(dl_train) // args.acumular
    warm = max(1, len(dl_train) // 2 // args.acumular)
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
        opt.zero_grad(set_to_none=True)
        t0, seen, loss_sum = time.time(), 0, 0.0
        for step, (x, present, t, w, _) in enumerate(dl_train, 1):
            raw, present = x.to(device, non_blocking=True), present.to(device)
            t, w = t.to(device), w.to(device)
            strong = args.aug == "fuerte" or args.consistencia > 0
            if args.consistencia > 0:   # la versión suave sirve de referencia y no recibe gradiente
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    ref = model(prepare(raw, args.slices, train=True), present).float().sigmoid()
            x = prepare(raw, args.slices, train=True, strong=strong)
            if args.quitar_serie > 0:
                x, present = drop_slot(x, present, args.quitar_serie)
            lam, perm = 1.0, None
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm = torch.randperm(x.shape[0], device=device)
                x = lam * x + (1 - lam) * x[perm]
                present = torch.maximum(present, present[perm])
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x, present)

            def bce(tt, ww):
                return (F.binary_cross_entropy_with_logits(logits.float(), tt, reduction="none") * ww).sum() / ww.sum().clamp(min=1)

            loss = bce(t, w) if perm is None else lam * bce(t, w) + (1 - lam) * bce(t[perm], w[perm])
            if args.consistencia > 0:
                if perm is not None:
                    ref = lam * ref + (1 - lam) * ref[perm]
                ramp = min(1.0, (step + (epoch - 1) * len(dl_train)) / len(dl_train))   # sube durante la primera época
                loss = loss + args.consistencia * ramp * ((logits.float().sigmoid() - ref) ** 2).mean()
            scaler.scale(loss / args.acumular).backward()
            if step % args.acumular == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            seen += x.shape[0]; loss_sum += loss.item() * x.shape[0]
            if step in (5, 20, 50) or step % 100 == 0:
                rate = seen / (time.time() - t0)
                print(f"  época {epoch} paso {step}/{len(dl_train)} pérdida {loss_sum / seen:.4f} "
                      f"{rate:.1f} estudios/s ({rate * 4 * args.slices:.0f} cortes/s)", flush=True)
            if not math.isfinite(loss.item()):
                raise SystemExit("la pérdida dejó de ser finita")
        train_s = time.time() - t0

        p_int, t_int, w_int, _ = predict(model, dl_int, device, args.slices, amp, tta)
        auc_int, per_int = macro_auc((t_int >= 0.5).astype(int), p_int, w_int)
        p_off, _, _, y_off = predict(model, dl_off, device, args.slices, amp, tta)
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

        state = {"model": model.state_dict(), "backbone": args.backbone, "slices": args.slices, "crop": args.crop, "n_slots": n_slots, "per_label": args.atencion == "hallazgo", "context": args.contexto == "gru", "pool": args.pool, "view_gate": args.vistas == "hallazgo", "mirror_left": bool(args.lados),
                 "mirror_tta": args.tta_espejo, "syn_eff": args.sinovitis_derrame,
                 "crop_mm": float(info.get("crop_mm", 0.0)), "z_range": list(info.get("z_range", [0.0, 1.0])),
                 "size": int(np.load(args.data / f"{train.StudyInstanceUID.iloc[0]}.npy", mmap_mode="r").shape[-1]),
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
