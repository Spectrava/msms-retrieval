"""The pretrained DreaMS spectrum encoder (Bushuiev et al., 2025).

Only the released architecture is ported: Fourier m/z features read from the checkpoint, a peak MLP and a
stem, 7 pre-norm Transformer blocks with the parameter-free Graphormer m/z bias, a final LayerNorm.
``load_dreams`` renames the checkpoint's upstream module names to these ones and drops the task heads.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DIM, DEPTH, HEADS, MLP_DIM = 1024, 7, 8, 4096
PEAK_DIM, FOURIER_DIM, FOURIER_HIDDEN = 44, 980, 512
MAX_MZ, MIN_RESOLUTION = 1000.0, 1e-4
MAX_PEAKS = 128  # DreaMS pads every spectrum to the 128-peak cap of its data format; padding is visible to attention
DROPOUT = 0.1
PRECURSOR_INTENSITY = 1.1


class MLP(nn.Module):
    """Linear layers, each followed by dropout (not the last) and a ReLU (also the last)."""

    def __init__(self, dims):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers += [nn.Linear(a, b)] + [nn.Dropout(DROPOUT)] * (i < len(dims) - 2) + [nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FourierFeatures(nn.Module):
    """cos and sin of 2 pi mz / period for periods 2e-4, 4e-4, .., 0.9998 Da and 2, 3, .., 999 Da."""

    def __init__(self):
        super().__init__()
        periods = [MIN_RESOLUTION * i for i in range(2, math.ceil(1 / MIN_RESOLUTION), 2)]
        periods += list(range(2, math.ceil(MAX_MZ)))
        self.register_buffer("frequencies", torch.tensor([1.0 / p for p in periods], dtype=torch.float32))

    def forward(self, mz):
        # float32 in this op order, as trained: phases reach 3e7 rad
        phase = 2.0 * torch.pi * mz.unsqueeze(-1) * self.frequencies
        return torch.cat([phase.cos(), phase.sin()], -1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(4 * DIM, DIM) * (0.4 / DIM) ** 0.5)  # q, k, v, out; no biases

    def forward(self, x, bias, query_pad):
        B, T, _ = x.shape
        q, k, v = F.linear(x, self.weights[:3 * DIM]).view(B, T, 3, HEADS, -1).permute(2, 0, 3, 1, 4)
        # DreaMS masks padded query rows to a uniform softmax: a zero query and a zero bias row do the same.
        q = q.masked_fill(query_pad, 0.0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=DROPOUT if self.training else 0.0)
        return F.linear(out.transpose(1, 2).reshape(B, T, DIM), self.weights[3 * DIM:])


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(DIM, MLP_DIM, bias=False)
        self.output_projection = nn.Linear(MLP_DIM, DIM, bias=False)

    def forward(self, x):
        return self.output_projection(F.dropout(F.relu(self.input_projection(x)), DROPOUT, self.training))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm, self.feedforward_norm = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.attention, self.feedforward = Attention(), FeedForward()

    def forward(self, x, bias, query_pad):
        x = x + F.dropout(self.attention(self.attention_norm(x), bias, query_pad), DROPOUT, self.training)
        return x + F.dropout(self.feedforward(self.feedforward_norm(x)), DROPOUT, self.training)


class DreaMSEncoder(nn.Module):
    """Spectrum batch (sna.spectra) -> (global [B, D], tokens [B, 1+N, D], token_mask [B, 1+N]).

    Token 0 is the precursor (precursor_mz, intensity 1.1); global is its final-norm output.
    shortcut: upstream DreaMS takes the embedding from the last feed-forward output instead of the
    final-norm residual stream, so these global vectors are not directly comparable to published
    DreaMS retrieval scores.
    """

    dim = DIM

    def __init__(self, max_peaks: int):
        super().__init__()
        self.max_peaks = max_peaks
        self.peak_encoder = MLP([2, PEAK_DIM])
        self.fourier_features = FourierFeatures()
        self.fourier_encoder = MLP([2 * len(self.fourier_features.frequencies)] + [FOURIER_HIDDEN] * 4 + [FOURIER_DIM])
        self.blocks = nn.ModuleList(Block() for _ in range(DEPTH))
        self.final_norm = nn.LayerNorm(DIM)

    def forward(self, batch):
        mz, intensity, mask = batch["mz"], batch["intensity"], batch["mask"]
        B, N = mask.shape
        assert N <= self.max_peaks, f"{N} peaks, the encoder takes at most {self.max_peaks}"
        # Released DreaMS masks padded queries but not padded keys, so outputs depend on the slot count: use max_peaks.
        pad = (0, self.max_peaks - N)
        mz = torch.cat([batch["precursor_mz"][:, None], F.pad(mz, pad)], 1)
        intensity = torch.cat([intensity.new_full((B, 1), PRECURSOR_INTENSITY), F.pad(intensity, pad)], 1)
        valid = torch.cat([mask.new_ones(B, 1), F.pad(mask, pad)], 1)

        fourier = self.fourier_encoder(self.fourier_features(mz))
        x = torch.cat([self.peak_encoder(torch.stack([mz / MAX_MZ, intensity], -1)), fourier], -1)
        x = F.dropout(x, DROPOUT, self.training)
        # Graphormer bias S_query - S_key (S = summed Fourier features), shared by all heads and layers;
        # S_query is constant along a softmax row, so in effect it is a per-key bias.
        s = fourier.float().sum(-1)
        bias = (s[:, :, None] - s[:, None, :]).masked_fill(~valid[:, :, None], 0.0)[:, None]
        query_pad = ~valid[:, None, :, None]
        for block in self.blocks:
            x = block(x, bias, query_pad)
        x = self.final_norm(x)
        return x[:, 0], x[:, :1 + N], valid[:, :1 + N]


def _load_checkpoint(path):
    """The released checkpoint pickles a `dreams` class we never use. Stub the module so
    torch.load can unpickle it without installing the upstream DreaMS package.
    shortcut: the stub only covers these checkpoints; install pluskal-lab/DreaMS if one needs real dreams code."""
    import sys, types
    if "dreams" not in sys.modules:
        for dotted in ("dreams", "dreams.models", "dreams.models.dreams", "dreams.models.dreams.dreams"):
            sys.modules[dotted] = types.ModuleType(dotted)
        sys.modules["dreams.models.dreams.dreams"].DreaMS = type("DreaMS", (), {})
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


def _released_state(state: dict) -> dict:
    """Rename the released checkpoint's upstream module names to this port's; the task heads are dropped."""
    out = {}
    for key, value in state.items():
        if not key.startswith("backbone."):
            continue  # head: the contrastive projection head the encoder never reads
        name = key[len("backbone."):]
        if name.split(".")[0] in ("ff_out", "ro_out"):
            continue  # m/z masking and retention-order heads
        if name == "fourier_enc.b":
            out["fourier_features.frequencies"] = value.reshape(-1)
        elif name.startswith("fourier_enc"):
            continue  # "Fourier frequencies" is the same tensor as b
        elif name.startswith(("ff_peak.ff", "ff_fourier.ff")):
            stem = "peak_encoder" if name.startswith("ff_peak") else "fourier_encoder"
            out[f"{stem}.layers" + name.split(".ff", 1)[1]] = value
        elif name.startswith("transformer_encoder.atts"):
            out[name.replace("transformer_encoder.atts", "blocks").replace(".weights", ".attention.weights")] = value
        elif name.startswith("transformer_encoder.ffs"):
            _, i, proj, param = name.split(".")[1:]
            out[f"blocks.{i}.feedforward.{'input' if proj == 'in_proj' else 'output'}_projection.{param}"] = value
        elif name.startswith("transformer_encoder.scales"):
            _, i, param = name.split(".")[1:]
            i = int(i)
            norm = "final_norm" if i == 2 * DEPTH else f"blocks.{i // 2}.{'feedforward' if i % 2 else 'attention'}_norm"
            out[f"{norm}.{param}"] = value
        else:
            raise ValueError(f"unexpected checkpoint key {key}")
    return out


def load_dreams(path) -> DreaMSEncoder:
    """Encoder of a released DreaMS checkpoint (SSL or contrastive stage); every other head is dropped."""
    encoder = DreaMSEncoder(MAX_PEAKS)
    state = _released_state(_load_checkpoint(path)["state_dict"])
    if not torch.allclose(state["fourier_features.frequencies"], encoder.fourier_features.frequencies, atol=1e-2):
        # the released checkpoint's frequencies are a trainable parameter, so they drifted a little
        # (<= 1.5e-3) during training; a different Fourier strategy would differ by far more.
        raise ValueError(f"{path} uses Fourier frequencies other than the ported set")
    encoder.load_state_dict(state)  # strict: any other architecture difference fails here
    return encoder
