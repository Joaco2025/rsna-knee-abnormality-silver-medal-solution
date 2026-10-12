"""Modelo de imagen para los 12 hallazgos. Lo usan el entrenamiento y el notebook de inferencia.

Entrada: el tensor de preproc.py, (4 ranuras, 32 cortes, alto, ancho) uint8.
Cada corte se convierte en una imagen de 3 canales con sus dos vecinos (2.5D), pasa por una red
convolucional 2D preentrenada, los cortes de cada ranura se combinan con atención y las cuatro
ranuras se concatenan para dar 12 salidas.
"""
from __future__ import annotations

import torch
import torch.nn as nn
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
                 n_slots: int = 4, n_out: int = len(LABELS), dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.encoder, feat = make_encoder(backbone, weights_path)
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, feat))
        self.attn = nn.Sequential(nn.Linear(feat, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat * n_slots, 512), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(512, n_out))

    def forward(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """x: (B, S, T, 3, H, W) normalizado; present: (B, S) con 1 si la ranura tiene serie."""
        b, s, t = x.shape[:3]
        f = self.encoder(x.flatten(0, 2).contiguous(memory_format=torch.channels_last))
        f = f.mean(dim=(2, 3)).float().view(b, s, t, -1) + self.slot_emb[None, :, None, :]
        a = self.attn(f).softmax(dim=2)
        z = (a * f).sum(dim=2) * present[..., None].to(f.dtype)
        return self.head(z.flatten(1))


def prepare(x: torch.Tensor, n_slices: int, train: bool) -> torch.Tensor:
    """De (B, S, D, H, W) uint8 en GPU a (B, S, T, 3, H, W) float normalizado.

    Toma T cortes equiespaciados (con desplazamiento aleatorio al entrenar) y a cada uno le añade
    sus dos vecinos como canales.
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
    mean = torch.tensor(MEAN, device=x.device).view(1, 1, 1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 1, 1, 3, 1, 1)
    return (x - mean) / std
