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

import torch
from torch import nn
from torch.nn import functional as F


def char_tensor(smiles: list[str], vocab: list[str], max_len: int) -> torch.Tensor:
    """SMILES as token ids, padded with END to a common width (END is index len(vocab))."""
    index = {c: i for i, c in enumerate(vocab)}
    width = min(max(map(len, smiles)), max_len)
    return torch.tensor([[index[c] for c in s[:width]] + [len(vocab)] * (width - len(s[:width]))
                         for s in smiles], device="cuda")


class Block(nn.Module):
    """One pre-norm causal transformer block."""

    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # this torch build needs an explicit mask: -inf above the diagonal keeps each token
        # blind to the characters that come after it.
        length = x.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf"), device=x.device), 1)
        n = self.norm1(x)
        y, _ = self.attn(n, n, n, attn_mask=mask)
        return (x + y) + self.ff(self.norm2(x + y))


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

    def loss(self, z: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: the logits at position i predict character i+1 (END after the last)."""
        x = torch.cat([self.seed(self.latent(z))[:, None, :], self.embed(chars)], dim=1)
        target = torch.cat([chars, torch.full((len(chars), 1), self.end, device=chars.device)], dim=1)
        return F.cross_entropy(self._logits(x).reshape(-1, self.end + 1), target.reshape(-1))

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, max_len: int = 160, temperature: float = 0.0) -> list[str]:
        """One SMILES per spectrum: greedy by default, sampled when temperature > 0."""
        x = self.seed(self.latent(z))[:, None, :]
        out, done = [[] for _ in range(len(z))], torch.zeros(len(z), dtype=torch.bool, device=z.device)
        for _ in range(max_len):
            logits = self._logits(x)[:, -1]
            nxt = logits.argmax(1) if temperature <= 0 else \
                torch.multinomial(F.softmax(logits / temperature, 1), 1).squeeze(1)
            for i, token in enumerate(nxt.tolist()):
                if done[i]:
                    continue
                if token == self.end:
                    done[i] = True
                else:
                    out[i].append(self.vocab[token])
            if bool(done.all()):
                break
            x = torch.cat([x, self.embed(nxt)[:, None, :]], dim=1)
        return ["".join(s) for s in out]


def train_generative(z: torch.Tensor, smiles: list[str], vocab: list[str], latent: int = 128,
                     epochs: int = 10, batch: int = 256, hidden: int = 256, layers: int = 4,
                     max_len: int = 160) -> GenerativeEmbedding:
    model = GenerativeEmbedding(z.shape[1], vocab, latent, hidden, layers).cuda()
    chars = char_tensor(smiles, vocab, max_len)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(z), device=z.device)
        loss = 0.0
        for i in range(0, len(order), batch):
            r = order[i:i + batch]
            obj = model.loss(z[r], chars[r])
            opt.zero_grad(); obj.backward(); opt.step()
            loss += obj.item() * len(r)
        print(f"[generative] epoch {epoch} next-character loss {loss / len(z):.3f}", flush=True)
    return model.eval()
