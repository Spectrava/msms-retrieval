"""A JEPA encoder for MS2: predict the latent of a whole spectrum from a masked view of it.

Nothing is reconstructed and no molecule enters the loss - the prediction is made in latent space,
which is the JEPA part. The question this answers is whether a *predictive* encoder is a better
substrate for generating structures than the discriminative DreaMS features: an encoder that has
learned to fill in the peaks it was not shown has to carry them.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

BINS = 1000


def mask_windows(x: torch.Tensor, windows: int = 3, fraction: float = 0.2) -> torch.Tensor:
    """Zero out `windows` contiguous m/z windows per spectrum, covering `fraction` of the range."""
    bins, width = x.shape[1], int(x.shape[1] * fraction / windows)
    out, idx = x.clone(), torch.arange(x.shape[1], device=x.device)
    for _ in range(windows):
        start = torch.randint(0, bins - width + 1, (len(x),), device=x.device)[:, None]
        out[(idx >= start) & (idx < start + width)] = 0.0
    return out


def mlp(dim_in: int, hidden: int, dim_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, dim_out))


class JEPA(nn.Module):
    """Encoder over the binned spectrum and a predictor that maps a masked-view latent to the
    whole-spectrum latent. The target comes from an EMA copy of the encoder, so the task never
    degenerates into predicting the encoder's own output."""

    def __init__(self, bins: int = BINS, hidden: int = 512, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.encoder = mlp(bins, hidden, dim)
        self.predictor = mlp(dim, hidden, dim)
        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.9) -> None:
        for teacher, student in zip(self.teacher.parameters(), self.encoder.parameters()):
            teacher.mul_(momentum).add_(student.detach(), alpha=1 - momentum)

    def loss(self, x: torch.Tensor, masked: torch.Tensor, spread_weight: float = 0.1) -> torch.Tensor:
        """1 - cosine between the predicted whole-spectrum latent and the teacher's, plus a term that
        keeps latents apart: without it the loss is minimised by mapping every spectrum to one vector."""
        pred = F.normalize(self.predictor(self.encoder(masked)), dim=-1)
        error = 1 - (pred * F.normalize(self.teacher(x), dim=-1)).sum(-1)
        n = len(pred)
        collapse = 0.0 if n < 2 else ((pred @ pred.T).sum() - n) / (n * (n - 1))
        return error.mean() + spread_weight * collapse


def linear_probe(latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Least-squares map from latent back to the input - a readout, not part of the training loss, so
    it measures how much of the spectrum the latent still carries without the encoder being trained
    to reconstruct it."""
    w = torch.linalg.lstsq(latent.double(), target.double()).solution
    return (latent.double() @ w).float()
