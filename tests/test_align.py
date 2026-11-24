"""The ridge map recovers a linear relationship, the contrastive heads learn an isometry, a
rotation of the query space cannot change which indexed molecule is nearest, and the string
Tanimoto agrees with the bit-matrix one."""
import numpy as np

from sna.align import fit_contrastive, nearest, readout_ceiling, ridge, smiles_tanimoto, tanimoto


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(500, 8))
    W = rng.normal(size=(8, 6))
    m = z @ W + 0.01 * rng.normal(size=(500, 6))
    assert np.allclose(ridge(z, m, 1e-6), W, atol=1e-2)


def test_nearest_is_invariant_to_rotation():
    rng = np.random.default_rng(1)
    index = rng.normal(size=(20, 6))
    query = rng.normal(size=(5, 6))
    fp = rng.integers(2, size=(20, 8)).astype(np.uint8)
    truth = fp[:5]  # the true molecule for each query
    _, sims = nearest(query, index, fp, truth)
    R = np.linalg.qr(rng.normal(size=(6, 6)))[0]
    _, rotated = nearest(query @ R, index @ R, fp, truth)
    assert np.allclose(sims, rotated)
    assert tanimoto(fp[:1], fp[:1])[0] == 1.0
    assert tanimoto(np.zeros((1, 8), np.uint8), np.zeros((1, 8), np.uint8)) == 0.0


def test_nearest_chunking_matches_a_single_pass():
    rng = np.random.default_rng(2)
    index = rng.normal(size=(30, 6))
    query = rng.normal(size=(7, 6))
    fp = rng.integers(2, size=(30, 8)).astype(np.uint8)
    truth = fp[:7]
    _, chunked = nearest(query, index, fp, truth, chunk=2)
    _, whole = nearest(query, index, fp, truth)
    assert np.allclose(chunked, whole)


def test_contrastive_heads_learn_an_isometry():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(400, 8))
    m = z @ np.linalg.qr(rng.normal(size=(8, 6)))[0]  # an exact rotation into another space
    spectrum, molecule = fit_contrastive(z, m, hidden=128, out=32, epochs=10, batch=200)
    sims = molecule(m) @ spectrum(z).T
    assert (sims.argmax(0) == np.arange(400)).mean() > 0.9


def test_readout_ceiling_is_exact_when_the_readout_is_exact():
    """If the predicted fingerprints are the true ones, the lookup recovers the molecule."""
    fp = np.eye(3, dtype=np.uint8)
    assert readout_ceiling(fp, fp, fp) == 1.0
    assert readout_ceiling(np.roll(fp, 1, axis=0), fp, fp) == 0.0  # a consistently wrong readout


def test_smiles_tanimoto_matches_the_bit_matrices():
    from sna.align import fingerprints
    fp = fingerprints(["CCO", "CCOC", "xyz"])
    assert smiles_tanimoto("CCO", "CCO") == 1.0
    assert smiles_tanimoto("CCO", "xyz") == 0.0  # RDKit cannot parse it
    assert 0.0 < smiles_tanimoto("CCO", "CCOC") < 1.0
    assert abs(smiles_tanimoto("CCO", "CCOC") - tanimoto(fp[:1], fp[1:2])[0]) < 1e-9
