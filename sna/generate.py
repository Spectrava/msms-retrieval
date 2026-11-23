"""Generate SMILES from DreaMS spectrum embeddings.

The DreaMS encoder is discriminative: its embedding cannot be inverted into a structure. This
module trains the smallest honest decoder - a character-level LSTM seeded with the spectrum
embedding - on labelled spectra, so an unseen spectrum can propose a SMILES. Quality is judged
against the nearest-neighbour retrieval baseline from the same embeddings: if the decoder cannot
beat "copy the closest training spectrum's molecule", the embedding carries no generative signal
beyond retrieval.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SmilesGenerator(nn.Module):
    """Character-level SMILES decoder conditioned on a spectrum embedding.

    The embedding is projected into the LSTM stream as the first "token"; the network then
    predicts the next character until it emits <end>.
    """

    def __init__(self, dim: int, vocab: list[str], hidden: int = 256, layers: int = 2):
        super().__init__()
        self.vocab = vocab
        self.end = len(vocab)
        self.embed = nn.Embedding(self.end + 1, hidden)
        self.proj = nn.Linear(dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, self.end + 1)

    def loss(self, z: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        """Teacher forcing: logits at position i predict character i+1 (END after the last)."""
        x = torch.cat([self.proj(z)[:, None, :], self.embed(chars)], dim=1)
        out, _ = self.lstm(x)
        logits = self.out(out)
        target = torch.cat([chars, torch.full((len(chars), 1), self.end, device=chars.device)], dim=1)
        return F.cross_entropy(logits.reshape(-1, self.end + 1), target.reshape(-1))

    @torch.inference_mode()
    def generate(self, z: torch.Tensor, max_len: int = 120, temperature: float = 0.0) -> list[str]:
        """Greedy (temperature 0) or sampled generation for a batch of embeddings."""
        h = c = None
        x = self.proj(z)[:, None, :]
        chars, done = [[] for _ in range(len(z))], torch.zeros(len(z), dtype=torch.bool, device=z.device)
        for _ in range(max_len):
            out, (h, c) = self.lstm(x, None if h is None else (h, c))
            logits = self.out(out[:, -1])
            nxt = logits.argmax(1) if temperature <= 0 else \
                torch.multinomial(F.softmax(logits / temperature, 1), 1).squeeze(1)
            for i in range(len(z)):
                if not done[i] and int(nxt[i]) != self.end:
                    chars[i].append(self.vocab[int(nxt[i])])
            done |= nxt == self.end
            x = self.embed(nxt[:, None])
            if bool(done.all()):
                break
        return ["".join(s) for s in chars]
