#!/usr/bin/env python
"""Continúa el preentrenamiento de la red 2D sobre nuestros cortes, sin etiquetas (MoCo v3).

Parte de los pesos de ImageNet y enseña a la red que dos versiones aumentadas del mismo corte
(con sus dos vecinos como canales, igual que en el modelo de los 12 hallazgos) se parecen entre sí
más que a los demás cortes del lote. La clave la da una copia de la red que se actualiza por media
móvil. No usa informes ni etiquetas, así que puede ver todos los estudios de entrenamiento.

Salida: pesos en el formato de torchvision, utilizables con `entrenar.py --weights`.

Uso:  python rsna/preentrenar_moco.py --data DIR_NPY --weights IMAGENET.pth --out DIR [opciones]
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo import MEAN, STD, augment  # noqa: E402


class Slices(Dataset):
    """De cada estudio, K cortes al azar (con vecinos) de sus series presentes: (K, 3, lado, lado) uint8."""

    def __init__(self, files: list[Path], per_study: int):
        self.files, self.k = files, per_study

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        arr = np.load(self.files[i], mmap_mode="r")
        slots = [s for s in range(arr.shape[0]) if arr[s, arr.shape[1] // 2].any()] or [0]
        out = np.empty((self.k, 3, *arr.shape[2:]), np.uint8)
        for j in range(self.k):
            s, d = slots[np.random.randint(len(slots))], np.random.randint(arr.shape[1])
            out[j] = arr[s, [max(d - 1, 0), d, min(d + 1, arr.shape[1] - 1)]]
        return torch.from_numpy(out)


def view(x: torch.Tensor, size: int) -> torch.Tensor:
    """Una versión aumentada de (N, 3, H, W) uint8: recorte de escala aleatoria llevado a `size`,
    brillo y contraste, y la aumentación fuerte del modelo (giro, gamma, ruido, borrado)."""
    n, _, h, w = x.shape
    x = x.float() / 255.0
    side = torch.sqrt(0.35 + 0.65 * torch.rand(n, device=x.device))          # fracción del lado que se conserva
    shift = (torch.rand(n, 2, device=x.device) * 2 - 1) * (1 - side)[:, None]
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = side
    theta[:, :, 2] = shift
    x = F.grid_sample(x, F.affine_grid(theta, [n, 3, size, size], align_corners=False), mode="bilinear", align_corners=False)
    gain = 1 + 0.6 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)
    bias = 0.3 * (torch.rand(n, 1, 1, 1, device=x.device) - 0.5)
    x = (x * gain + bias).clamp(0, 1)
    x = augment(x[:, None, None])[:, 0, 0]                                   # espera (B, S, T, 3, H, W)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


def mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_out))


def contrast(q: torch.Tensor, k: torch.Tensor, temp: float) -> torch.Tensor:
    q, k = F.normalize(q.float(), dim=1), F.normalize(k.float(), dim=1)
    return F.cross_entropy(q @ k.t() / temp, torch.arange(len(q), device=q.device)) * 2 * temp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="resnet34")
    ap.add_argument("--weights", default="", help="pesos de partida de torchvision (.pth)")
    ap.add_argument("--epochs", type=int, default=30, help="pasadas; cada una toma --por-estudio cortes de cada estudio")
    ap.add_argument("--batch", type=int, default=16, help="estudios por lote")
    ap.add_argument("--por-estudio", type=int, default=16, help="cortes por estudio en cada lote")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--momentum", type=float, default=0.99)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 2)
    ap.add_argument("--limit", type=int, default=0, help="estudios, para pruebas")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda")

    assert args.backbone.startswith("resnet"), "por ahora solo redes resnet de torchvision"
    files = sorted(p for p in args.data.glob("*.npy") if not p.name.startswith("."))
    if args.limit:
        files = files[: args.limit]
    dl = DataLoader(Slices(files, args.por_estudio), batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    net = torchvision.models.get_model(args.backbone, weights=None)
    if args.weights:
        net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    feat = net.fc.in_features
    fc, net.fc = net.fc, nn.Identity()          # la capa de ImageNet se guarda aparte para devolverla al final
    base = nn.Sequential(net, mlp(feat, 1024, 256)).to(device).to(memory_format=torch.channels_last)
    pred = mlp(256, 1024, 256).to(device)
    target = copy.deepcopy(base).requires_grad_(False)

    params = list(base.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(dl)
    warm = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    print(f"estudios={len(files)} cortes por pasada={len(dl) * args.batch * args.por_estudio} lote={args.batch * args.por_estudio} "
          f"red={args.backbone} lado={args.size} pasadas={args.epochs} pesos={args.weights or 'ninguno'}", flush=True)

    step = 0
    for epoch in range(1, args.epochs + 1):
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x in dl:
            x = x.to(device, non_blocking=True).flatten(0, 1)
            v1, v2 = view(x, args.size), view(x, args.size)
            with torch.autocast("cuda", dtype=torch.float16):
                q1, q2 = pred(base(v1)), pred(base(v2))
                with torch.no_grad():
                    k1, k2 = target(v1), target(v2)
            loss = contrast(q1, k2, args.temp) + contrast(q2, k1, args.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            m = 1 - (1 - args.momentum) * 0.5 * (1 + math.cos(math.pi * step / steps))   # sube hacia 1 al final
            with torch.no_grad():
                for pt, pb in zip(target.parameters(), base.parameters()):
                    pt.mul_(m).add_(pb.detach(), alpha=1 - m)
                for bt, bb in zip(target.buffers(), base.buffers()):
                    bt.copy_(bb)
            seen += len(x); loss_sum += loss.item() * len(x)
            if not math.isfinite(loss.item()):
                raise SystemExit("la pérdida dejó de ser finita")
            if step in (5, 20, 50) or step % 200 == 0:
                print(f"  pasada {epoch} paso {step}/{steps} pérdida {loss_sum / seen:.4f} {seen / (time.time() - t0):.0f} cortes/s", flush=True)
        print(f"pasada {epoch}: pérdida {loss_sum / seen:.4f} | {(time.time() - t0) / 60:.1f} min", flush=True)
        net.fc = fc
        torch.save(net.state_dict(), args.out / f"{args.backbone}-moco.pth")   # formato torchvision, para --weights
        net.fc = nn.Identity()


if __name__ == "__main__":
    main()
