"""The seeded decoder learns to reproduce a SMILES and only emits vocabulary characters."""
import torch

from sna.generate import SmilesGenerator


def test_decoder_learns_and_generates():
    torch.manual_seed(0)
    vocab = sorted(set("CN=O"))
    model = SmilesGenerator(4, vocab, hidden=16, layers=1)
    z = torch.randn(8, 4)
    chars = torch.tensor([[vocab.index(c) for c in "CNO=O"] for _ in range(8)])
    first = model.loss(z, chars).item()
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(30):
        loss = model.loss(z, chars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first
    generated = model.generate(z, max_len=12)[0]
    assert all(c in vocab for c in generated)
