"""A generative MS2 embedding: a latent a molecule decoder is trained to invert.

The DreaMS embedding is discriminative - it is trained to tell spectra apart, so nothing can be
read back out of it. Here the spectrum embedding is pushed through a small encoder into a
bottleneck latent that a transformer decoder is trained, jointly, to turn into a SMILES. The
latent is therefore a generative representation of the spectrum: the model is scored on whether
it can write a structure, not on whether it can find a similar spectrum.

Generation from a held-out spectrum is compared with the char-LSTM baseline conditioned directly
on the raw 1024-d DreaMS embedding (sna.generate.SmilesGenerator).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sna.align import canonical, smiles_tanimoto
from sna.lsh import binned


def peak_matrix(mz, intensity, offsets, bins: int = 1000) -> np.ndarray:
    """Each spectrum as `bins` one-unit m/z bins summing to 1 - the raw input a from-scratch encoder sees."""
    out = np.zeros((len(offsets) - 1, bins), np.float32)
    for i in range(len(offsets) - 1):
        v = binned(mz[offsets[i]:offsets[i + 1]], intensity[offsets[i]:offsets[i + 1]], bins)
        out[i] = v / v.sum() if v.sum() else 0.0
    return out


def generate(model, z: torch.Tensor, batch: int = 512, max_len: int = 160, temperature: float = 0.0) -> list[str]:
    """One SMILES per spectrum, chunked: the attention matrix for a whole fold does not fit on the GPU."""
    return [s for i in range(0, len(z), batch)
            for s in model.generate(z[i:i + batch], max_len, temperature)]


def report(generated: list[str], truth: list[str]) -> tuple[float, float, float]:
    """(parse rate, mean Morgan Tanimoto of the parsed ones, exact match rate)."""
    parsed = [(canonical(g), t) for g, t in zip(generated, truth) if canonical(g)]
    return (len(parsed) / len(generated),
            float(np.mean([smiles_tanimoto(g, t) for g, t in parsed])) if parsed else 0.0,
            float(np.mean([g == t for g, t in parsed])) if parsed else 0.0)


def char_tensor(smiles: list[str], vocab: list[str], max_len: int) -> torch.Tensor:
    """SMILES as token ids, padded with END to a common width (END is index len(vocab))."""
    index = {c: i for i, c in enumerate(vocab)}
    width = min(max(map(len, smiles)), max_len)
    return torch.tensor([[index[c] for c in s[:width]] + [len(vocab)] * (width - len(s[:width]))
                         for s in smiles], device="cuda")


def _causal_mask(length: int, device) -> torch.Tensor:
    # this torch build needs an explicit mask: -inf above the diagonal keeps each token blind to
    # the characters that come after it.
    return torch.triu(torch.full((length, length), float("-inf"), device=device), 1)


def _decode(stream, embed, logits, max_len: int, vocab: list[str], end: int, temperature: float = 0.0) -> list[str]:
    """Extend `stream` one character at a time until every sequence emits END."""
    out, done = [[] for _ in range(len(stream))], torch.zeros(len(stream), dtype=torch.bool, device=stream.device)
    for _ in range(max_len):
        nxt = logits(stream)[:, -1]
        token = nxt.argmax(1) if temperature <= 0 else \
            torch.multinomial(F.softmax(nxt / temperature, 1), 1).squeeze(1)
        for i, t in enumerate(token.tolist()):
            if done[i]:
                continue
            if t == end:
                done[i] = True
            else:
                out[i].append(vocab[t])
        if bool(done.all()):
            break
        stream = torch.cat([stream, embed(token)[:, None, :]], dim=1)
    return ["".join(s) for s in out]


class Block(nn.Module):
    """One pre-norm causal transformer block."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        y, _ = self.attn(n, n, n, attn_mask=_causal_mask(x.shape[1], x.device))
        return (x + y) + self.ff(self.norm2(x + y))


class CrossBlock(nn.Module):
    """Self-attention over the characters, then attention over the latent memory, so every character
    - not only the first - can query the spectrum."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(hidden) for _ in range(3))
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        y, _ = self.self_attn(n, n, n, attn_mask=_causal_mask(x.shape[1], x.device))
        x = x + y
        z, _ = self.cross_attn(self.norm2(x), memory, memory)
        return x + z + self.ff(self.norm3(x + z))


class GenerativeEmbedding(nn.Module):
    """Spectrum embedding -> latent -> SMILES, trained end to end.

    The latent is the generative embedding: the decoder sees it as the first token of the stream
    and predicts every character of the SMILES from it.
    """

    def __init__(self, dim_in: int, vocab: list[str], latent: int = 128, hidden: int = 256,
                 layers: int = 4, heads: int = 4):
        super().__init__()
        self.vocab = vocab
        self.end = len(vocab)
        self.encoder = nn.Sequential(nn.Linear(dim_in, latent), nn.LayerNorm(latent), nn.GELU())
        self.reconstruct = nn.Linear(latent, dim_in)  # only used when the input is the raw spectrum
        self.seed = nn.Linear(latent, hidden)
        self.embed = nn.Embedding(self.end + 1, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads) for _ in range(layers)])
        self.out = nn.Linear(hidden, self.end + 1)

    def latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.encoder(z)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)

    def loss(self, z: torch.Tensor, chars: torch.Tensor, spectrum: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher forcing: the logits at position i predict character i+1 (END after the last)."""
        x = torch.cat([self.seed(self.latent(z))[:, None, :], self.embed(chars)], dim=1)
        target = torch.cat([chars, torch.full((len(chars), 1), self.end, device=chars.device)], dim=1)
        loss = F.cross_entropy(self._logits(x).reshape(-1, self.end + 1), target.reshape(-1))
        if spectrum is not None:
            loss = loss + F.mse_loss(self.reconstruct(self.latent(z)), spectrum)
        return loss

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, max_len: int = 160, temperature: float = 0.0) -> list[str]:
        """One SMILES per spectrum: greedy by default, sampled when temperature > 0."""
        return _decode(self.seed(self.latent(z))[:, None, :], self.embed, self._logits,
                       max_len, self.vocab, self.end, temperature)


class LatentDecoder(nn.Module):
    """SMILES decoder built for a latent that is already computed: it is expanded into `memory`
    tokens that the character stream attends to at every step.

    A single leading seed token is easy for the character pathway to route around - after a few
    characters the stream can predict the next one from itself. Cross-attention makes the latent a
    standing source of evidence instead of a one-off prefix.
    """

    def __init__(self, latent: int, vocab: list[str], hidden: int = 256, layers: int = 4,
                 heads: int = 4, memory: int = 8):
        super().__init__()
        self.vocab, self.end, self.memory = vocab, len(vocab), memory
        self.seed = nn.Linear(latent, memory * hidden)
        self.embed = nn.Embedding(self.end + 1, hidden)
        self.blocks = nn.ModuleList([CrossBlock(hidden, heads) for _ in range(layers)])
        self.out = nn.Linear(hidden, self.end + 1)

    def _memory(self, z: torch.Tensor) -> torch.Tensor:
        return self.seed(z).reshape(len(z), self.memory, -1)

    def _logits(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, memory)
        return self.out(x)

    def loss(self, z: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: the stream begins with the special token and position i predicts the
        character after it; the same special id is the stop target after the last character."""
        start = self.embed(torch.full((len(chars), 1), self.end, device=chars.device))
        target = torch.cat([chars, torch.full((len(chars), 1), self.end, device=chars.device)], dim=1)
        logits = self._logits(torch.cat([start, self.embed(chars)], dim=1), self._memory(z))
        return F.cross_entropy(logits.reshape(-1, self.end + 1), target.reshape(-1))

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, max_len: int = 160, temperature: float = 0.0) -> list[str]:
        memory = self._memory(z)
        start = self.embed(torch.full((len(z), 1), self.end, device=z.device))
        return _decode(start, self.embed, lambda x: self._logits(x, memory),
                       max_len, self.vocab, self.end, temperature)


def _train(model, z: torch.Tensor, smiles: list[str], vocab: list[str], epochs: int, batch: int,
           max_len: int, tag: str, spectra: torch.Tensor | None = None):
    chars = char_tensor(smiles, vocab, max_len)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(z), device=z.device)
        loss = 0.0
        for i in range(0, len(order), batch):
            r = order[i:i + batch]
            obj = (model.loss(z[r], chars[r], spectra[r]) if spectra is not None
                   else model.loss(z[r], chars[r]))
            opt.zero_grad(); obj.backward(); opt.step()
            loss += obj.item() * len(r)
        label = "next-character + reconstruction" if spectra is not None else "next-character"
        print(f"[{tag}] epoch {epoch} {label} loss {loss / len(z):.3f}", flush=True)
    return model.eval()


def train_generative(z: torch.Tensor, smiles: list[str], vocab: list[str], latent: int = 128,
                     epochs: int = 10, batch: int = 256, hidden: int = 256, layers: int = 4,
                     max_len: int = 160, spectra: torch.Tensor | None = None) -> GenerativeEmbedding:
    model = GenerativeEmbedding(z.shape[1], vocab, latent, hidden, layers).cuda()
    return _train(model, z, smiles, vocab, epochs, batch, max_len, "generative", spectra)


def train_latent_decoder(latent: torch.Tensor, smiles: list[str], vocab: list[str], hidden: int = 256,
                         layers: int = 4, memory: int = 8, epochs: int = 10, batch: int = 256,
                         max_len: int = 160) -> LatentDecoder:
    model = LatentDecoder(latent.shape[1], vocab, hidden, layers, memory).cuda()
    return _train(model, latent, smiles, vocab, epochs, batch, max_len, "latent decoder")
