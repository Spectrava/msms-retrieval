"""The generative latent is invertible: decoding a training spectrum's latent returns its SMILES."""
import numpy as np
import torch

from sna.align import canonical
from sna.genembed import train_generative


def test_latent_reconstructs_its_own_training_molecules():
    unique = ["CCO", "CCOC", "CC(=O)O", "c1ccccc1", "CCNCC", "OCCO", "CCCl", "C(C)C", "C1CC1"]
    vocab = sorted({c for s in unique for c in s})
    base = np.random.default_rng(0).normal(size=(len(unique), 64)).astype(np.float32)
    z = torch.tensor(np.repeat(base, 20, axis=0), device="cuda")
    model = train_generative(z, [s for s in unique for _ in range(20)], vocab,
                            latent=64, epochs=60, batch=64, hidden=128, layers=2)
    out = model.generate(torch.tensor(base, device="cuda"), max_len=30)
    exact = sum(canonical(g) == canonical(s) for g, s in zip(out, unique)) / len(unique)
    assert exact >= 0.8, f"only {exact:.1%} of the training molecules were reproduced"
