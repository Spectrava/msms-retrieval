"""An encoder that reads the peak list as a sequence of m/z tokens instead of a histogram.

A 1000-bin histogram throws away which peaks are there and how strong they are relative to one
another; a fragment-identification model needs both. Each spectrum becomes the m/z bins of its
strongest peaks, ascending, and a transformer attends over that set. The latent is then handed to
the same `LatentDecoder` used for every other substrate, so the comparison stays about the encoder.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

BINS = 1000
MAX_PEAKS = 128


def peak_tokens(mz, intensity, offsets, bins: int = BINS, max_peaks: int = MAX_PEAKS) -> np.ndarray:
    """The `max_peaks` strongest m/z bins per spectrum, ascending; 0 is padding."""
    out = np.zeros((len(offsets) - 1, max_peaks), dtype=np.int32)
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        m = np.clip(np.floor(np.asarray(mz[a:b])), 1, bins)
        keep = np.argsort(np.asarray(intensity[a:b]))[::-1][:max_peaks]
        out[i, :len(keep)] = np.sort(m[keep])
    return out


class EncoderBlock(nn.Module):
    """Pre-norm self-attention with no causal mask: peak order carries no dependency, so every peak
    should see every other peak."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        return x + y + self.ff(self.norm(x + y))


class PeakEncoder(nn.Module):
    """Peak tokens -> latent, with a JEPA head that predicts the whole-spectrum latent from a view
    with some peaks removed."""

    def __init__(self, bins: int = BINS, hidden: int = 256, layers: int = 4, heads: int = 4,
                 dim: int = 256, max_peaks: int = MAX_PEAKS):
        super().__init__()
        self.dim, self.max_peaks = dim, max_peaks
        self.token = nn.Embedding(bins + 1, hidden)
        self.pos = nn.Embedding(max_peaks, hidden)
        self.blocks = nn.ModuleList([EncoderBlock(hidden, heads) for _ in range(layers)])
        self.out = nn.Linear(hidden, dim)
        self.predictor = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                      nn.Linear(hidden, dim))
        self.teacher = copy.deepcopy(self.out)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def pooled(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        real = tokens != 0
        return (x * real[:, :, None]).sum(1) / real.sum(1, keepdims=True).clamp(min=1)

    def latent(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.out(self.pooled(tokens)), dim=-1)

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.9) -> None:
        for teacher, student in zip(self.teacher.parameters(), self.out.parameters()):
            teacher.mul_(momentum).add_(student.detach(), alpha=1 - momentum)

    def drop_peaks(self, tokens: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
        """Blank a random quarter of the peaks that are actually present."""
        real = tokens != 0
        pick = torch.rand(tokens.shape, device=tokens.device) < fraction
        return torch.where(real & pick, torch.zeros_like(tokens), tokens)

    def jepa_loss(self, tokens: torch.Tensor, fraction: float = 0.25,
                  spread_weight: float = 1.0) -> torch.Tensor:
        """The penalty is the mean cosine between different latents: unlike `1 - spread` it is not
        capped, so collapsing every spectrum onto one vector costs the full weight, not 0.1."""
        pred = F.normalize(self.predictor(self.latent(self.drop_peaks(tokens, fraction))), dim=-1)
        target = F.normalize(self.teacher(self.pooled(tokens)), dim=-1)
        error = 1 - (pred * target).sum(-1)
        n = len(pred)
        collapse = 0.0 if n < 2 else ((pred @ pred.T).sum() - n) / (n * (n - 1))
        return error.mean() + spread_weight * collapse
