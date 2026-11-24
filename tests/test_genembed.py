"""The generative latent is invertible: decoding a training spectrum's latent returns its SMILES."""
import numpy as np
import torch

from sna.align import canonical
from sna.genembed import stop_masked_target, train_generative, train_latent_decoder


def test_only_the_stop_token_is_scored_after_the_last_character():
    """Padding must not be scored, or the loss is dominated by teaching the model to emit END."""
    chars = torch.tensor([[1, 2, 3, 4, 4, 4], [5, 6, 4, 4, 4, 4]])
    expected = torch.tensor([[1, 2, 3, 4, -100, -100], [5, 6, 4, -100, -100, -100]])
    assert torch.equal(stop_masked_target(chars, 4), expected)


def test_latent_rebuilds_the_spectrum_it_was_given():
    """With the raw spectrum as input, the latent is checked against the observation, not just a label."""
    unique = ["CCO", "CCOC", "CC(=O)O", "c1ccccc1", "CCNCC", "OCCO", "CCCl", "C(C)C"]
    vocab = sorted({c for s in unique for c in s})
    base = np.random.default_rng(1).normal(size=(len(unique), 40)).astype(np.float32)
    z = torch.tensor(np.repeat(base, 8, axis=0), device="cuda")
    model = train_generative(z, [s for s in unique for _ in range(8)], vocab,
                             latent=32, epochs=60, batch=32, hidden=64, layers=2, spectra=z)
    b = torch.tensor(base, device="cuda")
    with torch.no_grad():
        rec = model.reconstruct(model.latent(b))
        cos = ((rec * b).sum(1) / (rec.norm(dim=1) * b.norm(dim=1) + 1e-9)).mean()
    assert cos > 0.9, f"reconstructed spectrum cosine {cos:.3f}"


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


def test_decoder_reads_a_latent_held_as_memory():
    """The cross-attention decoder writes the molecule from a latent it never sees as a character."""
    unique = ["CCO", "CCOC", "CC(=O)O", "c1ccccc1", "CCNCC", "OCCO", "CCCl", "C(C)C", "C1CC1"]
    vocab = sorted({c for s in unique for c in s})
    base = np.random.default_rng(3).normal(size=(len(unique), 64)).astype(np.float32)
    lat = torch.tensor(np.repeat(base, 20, axis=0), device="cuda")
    model = train_latent_decoder(lat, [s for s in unique for _ in range(20)], vocab,
                                 hidden=128, layers=2, memory=4, epochs=60, batch=64)
    out = model.generate(torch.tensor(base, device="cuda"), max_len=30)
    exact = sum(canonical(g) == canonical(s) for g, s in zip(out, unique)) / len(unique)
    assert exact >= 0.8, f"only {exact:.1%} of the training molecules were reproduced"
