"""Modelo de imagen para los 12 hallazgos. Lo usan el entrenamiento y el notebook de inferencia.

Entrada: el tensor de preproc.py, (4 ranuras, 32 cortes, alto, ancho) uint8.
Cada corte se convierte en una imagen de 3 canales con sus dos vecinos (2.5D), pasa por una red
convolucional 2D preentrenada, los cortes de cada ranura se combinan con atención y las cuatro
ranuras se concatenan para dar 12 salidas.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def make_encoder(name: str, weights_path: str | None) -> tuple[nn.Module, int]:
    net = torchvision.models.get_model(name, weights=None)
    if weights_path:
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    if name.startswith("resnet"):
        return nn.Sequential(*list(net.children())[:-2]), net.fc.in_features
    if name.startswith(("convnext", "efficientnet")):
        feat = net.classifier[-1].in_features
        return net.features, feat
    raise ValueError(f"red no contemplada: {name}")


class KneeNet(nn.Module):
    def __init__(self, backbone: str = "resnet34", weights_path: str | None = None,
                 n_slots: int = 4, n_out: int = len(LABELS), dropout: float = 0.3, per_label: bool = False,
                 context: bool = False, pool: str = "media"):
        super().__init__()
        self.backbone, self.per_label, self.context, self.pool = backbone, per_label, context, pool
        self.encoder, feat = make_encoder(backbone, weights_path)
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, feat))
        if context:   # posición del corte dentro de la serie y una red recurrente que comunica los cortes entre sí
            self.slice_emb = nn.Parameter(torch.zeros(1, feat, 32))
            self.rnn = nn.GRU(feat, feat // 2, num_layers=1, batch_first=True, bidirectional=True)
            self.rnn_norm = nn.LayerNorm(feat)
        self.attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, n_out if per_label else 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat * n_slots, 512), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(512, n_out))

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """x: (B, S, T, 3, H, W) normalizado; present: (B, S) con 1 si la ranura tiene serie."""
        b, s, t = x.shape[:3]
        f = self.encoder(x.flatten(0, 2).contiguous(memory_format=torch.channels_last))
        f = f.mean(dim=(2, 3)) if self.pool == "media" else 0.5 * (f.mean(dim=(2, 3)) + f.amax(dim=(2, 3)))
        f = f.float().view(b, s, t, -1) + self.slot_emb[None, :, None, :]
        if self.context:
            with torch.autocast(device_type=x.device.type, enabled=False):
                pos = F.interpolate(self.slice_emb, size=t, mode="linear", align_corners=False)     # (1, F, T)
                f = f + pos.transpose(1, 2)[:, None]
                f = self.rnn_norm(f + self.rnn(f.flatten(0, 1))[0].view(b, s, t, -1))
        a = self.attn(f).softmax(dim=2)
        if not self.per_label:
            z = (a * f).sum(dim=2) * present[..., None].to(f.dtype)
            return self.head(z.flatten(1))
        # Atención propia de cada hallazgo (idea de Joako): cada uno elige sus cortes; la cabeza se comparte.
        z = torch.einsum("bstj,bstf->bjsf", a, f) * present[:, None, :, None].to(f.dtype)    # (B, J, S, F)
        h = self.head[:-1](z.flatten(2))                                                     # (B, J, 512)
        return (h * self.head[-1].weight[None]).sum(-1) + self.head[-1].bias


def augment(x: torch.Tensor) -> torch.Tensor:
    """Aumentación fuerte de (B, S, T, 3, H, W) en [0, 1]. Cada serie recibe una transformación
    propia, la misma en todos sus cortes: giro de ±10°, escala de ±10 %, desplazamiento de ±5 %,
    gamma, ruido y un rectángulo borrado. No hay volteos: invertirían medial y lateral.
    """
    b, s, t, c, h, w = x.shape
    n, dev = b * s, x.device
    ang = (torch.rand(n, device=dev) - 0.5) * 2 * math.radians(10)
    scale = 1 + (torch.rand(n, device=dev) - 0.5) * 0.2
    shift = (torch.rand(n, 2, device=dev) - 0.5) * 0.2          # en unidades de la rejilla [-1, 1]
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1), torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    flat = x.reshape(n, t * c, h, w)
    grid = F.affine_grid(theta, list(flat.shape), align_corners=False)
    flat = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.6)
    flat = flat.clamp(min=1e-4) ** gamma
    flat = flat + torch.randn_like(flat) * (torch.rand(n, 1, 1, 1, device=dev) * 0.03)

    # Rectángulo borrado, de entre el 10 % y el 25 % del lado.
    size = (0.10 + 0.15 * torch.rand(n, 2, device=dev)) * torch.tensor([h, w], device=dev)
    top = torch.rand(n, 2, device=dev) * (torch.tensor([h, w], device=dev) - size)
    rows = torch.arange(h, device=dev)[None, :, None]
    cols = torch.arange(w, device=dev)[None, None, :]
    hole = ((rows >= top[:, 0, None, None]) & (rows < (top[:, 0] + size[:, 0])[:, None, None])
            & (cols >= top[:, 1, None, None]) & (cols < (top[:, 1] + size[:, 1])[:, None, None]))
    flat = flat.masked_fill(hole[:, None], 0.0)
    return flat.clamp(0, 1).reshape(b, s, t, c, h, w)


def drop_slot(x: torch.Tensor, present: torch.Tensor, p: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Con probabilidad p apaga una serie al azar de cada estudio, si le quedan al menos dos."""
    b, s = present.shape
    pick = torch.multinomial(present + 1e-6, 1).squeeze(1)
    go = (torch.rand(b, device=x.device) < p) & (present.sum(dim=1) >= 2)
    keep = torch.ones_like(present)
    keep[go, pick[go]] = 0
    return x * keep.view(b, s, 1, 1, 1, 1).to(x.dtype), present * keep


def prepare(x: torch.Tensor, n_slices: int, train: bool, strong: bool = False) -> torch.Tensor:
    """De (B, S, D, H, W) uint8 en GPU a (B, S, T, 3, H, W) float normalizado.

    Toma T cortes equiespaciados (con desplazamiento aleatorio al entrenar) y a cada uno le añade
    sus dos vecinos como canales. Con strong=True aplica además la aumentación fuerte.
    """
    b, s, d, h, w = x.shape
    stride = max(1, d // n_slices)
    off = int(torch.randint(0, stride, (1,))) if train else stride // 2
    idx = torch.arange(n_slices, device=x.device) * stride + off
    idx = idx.clamp(max=d - 1)
    stack = torch.stack([(idx - 1).clamp(min=0), idx, (idx + 1).clamp(max=d - 1)], dim=1)   # (T, 3)
    x = x[:, :, stack].float() / 255.0                                                      # (B, S, T, 3, H, W)
    if train:   # variación suave de brillo y contraste por estudio
        gain = 1 + 0.2 * (torch.rand(b, 1, 1, 1, 1, 1, device=x.device) - 0.5)
        bias = 0.1 * (torch.rand(b, 1, 1, 1, 1, 1, device=x.device) - 0.5)
        x = (x * gain + bias).clamp(0, 1)
    if strong:
        x = augment(x)
    mean = torch.tensor(MEAN, device=x.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 1, 1, 3, 1, 1)
    return (x - mean) / std
