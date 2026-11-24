"""CKA is invariant to rotation and scale, Procrustes recovers a known rotation, CCA recovers a
linear map, SIGReg is small for a Gaussian batch and large for a collapsed one, the cosine Gram
has no diagonal, and retrieval/MRR score a constructed case correctly."""
import numpy as np
import torch

from sna.joint import (cca, cka, cosine_gram, fit_geometry, mean_by_group, pair_correlation, pca,
                       procrustes, retrieval, sigreg)


def test_replicates_pool_into_one_row_per_molecule():
    z = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    assert np.allclose(mean_by_group(z, np.array([0, 0, 1])), [[2.0, 0.0], [0.0, 2.0]])


def test_cka_is_invariant_to_rotation_and_scale():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 10))
    R = np.linalg.qr(rng.normal(size=(10, 10)))[0]
    assert abs(cka(a, a) - 1.0) < 1e-9
    assert abs(cka(a, a @ R) - 1.0) < 1e-9
    assert abs(cka(a, 7.3 * a) - 1.0) < 1e-9
    assert cka(a, rng.normal(size=(200, 10))) < 0.3


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 8))
    R = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    assert np.allclose(procrustes(a, a @ R), R, atol=1e-6)
    assert np.allclose(procrustes(a, a), np.eye(8), atol=1e-6)


def test_cca_recovers_a_linear_map():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(400, 6))
    W = rng.normal(size=(6, 4))
    b = a @ W
    map_, corr = cca(a, b, lam=1e-6)
    assert np.allclose(a @ map_, b, atol=1e-3)
    assert (corr[:4] > 0.99).all()


def test_pca_matching_keeps_the_top_variance():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 10))
    mean, comp = pca(x, 3)
    assert comp.shape == (3, 10)
    assert np.allclose(comp @ comp.T, np.eye(3), atol=1e-9)  # orthonormal directions
    projected = (x - mean) @ comp.T
    # three of ten equal-variance dimensions: about 30% of the energy, not all of it
    assert 0.2 < (np.linalg.norm(projected) / np.linalg.norm(x - mean)) ** 2 < 0.45


def test_cosine_gram_has_no_diagonal():
    g = cosine_gram(torch.randn(5, 3))
    assert torch.allclose(g.diagonal(), torch.zeros(5), atol=1e-6)
    assert torch.allclose(g, g.T)


def test_sigreg_separates_a_gaussian_from_a_collapsed_latent():
    torch.manual_seed(0)
    gaussian = sigreg(torch.randn(512, 32))
    collapsed = sigreg(torch.ones(512, 32))  # every sample the same vector
    assert collapsed > gaussian > 0


def test_retrieval_and_mrr_score_a_constructed_case():
    index = np.eye(3, dtype=np.float32)
    query = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    fp = np.eye(3, dtype=np.uint8)
    ids = np.array(["a", "b", "c"])
    score = retrieval(query, index, fp, fp[1:], ids, np.array(["b", "c"]))
    assert score["top1"] == 1.0 and score["mrr"] == 1.0
    wrong = retrieval(np.roll(query, 1, axis=1), index, fp, fp[1:], ids, np.array(["b", "c"]))
    assert wrong["top1"] == 0.0 and abs(wrong["mrr"] - 1 / 3) < 1e-9  # the true molecule is third


def test_pair_correlation_is_one_when_similarity_is_the_fingerprint_similarity():
    # nested bit sets of sizes 1,2,3,5,7: cosine = sqrt(Tanimoto) for every pair and no two
    # pairs share a value, so the two rank orders must agree exactly
    sizes = [1, 2, 3, 5, 7]
    fp = np.array([[1] * s + [0] * (7 - s) for s in sizes], dtype=np.uint8)
    rho = pair_correlation(fp.astype(np.float32), fp, fp, fp, queries=5, candidates=5)
    assert abs(rho - 1.0) < 1e-6


def test_geometry_projectors_pair_the_data_without_collapsing():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(400, 12))
    m = z @ np.linalg.qr(rng.normal(size=(12, 8)))[0]
    spectrum, molecule = fit_geometry(z, m, out=16, alpha=1.0, beta=0.1, epochs=40, batch=100, seed=0, verbose=False)
    hs, hm = spectrum(z), molecule(m)
    assert np.isfinite(hs).all() and np.allclose(np.linalg.norm(hs, axis=1), 1.0, atol=1e-6)
    sim = hs @ hm.T
    assert sim.diagonal().mean() > sim.mean() + 0.2  # the pairing term: a pair is more alike than a random pair
    assert hs.std() > 0.01  # not collapsed onto one direction