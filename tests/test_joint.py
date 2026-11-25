"""CKA is invariant to rotation and scale, Procrustes recovers a known rotation, CCA recovers a
linear map, SIGReg is small for a Gaussian batch and large for a collapsed one, the cosine Gram
has no diagonal, retrieval/MRR score a constructed case correctly, and both GW couplings - entropic
and hard assignment - recover a known relabeling of the same geometry, are equivariant under a
joint relabeling, and report diagnostics against a pairing that is no pairing at all."""
import numpy as np
import torch

from sna.joint import (cca, cka, cosine_distance, cosine_gram, fit_geometry, gw_coupling, gw_cost,
                       gw_diagnostics, hard_gw_coupling, mean_by_group, pair_correlation, pca,
                       procrustes, retrieval, retrieval_scores, sigreg, transport, unit)


def test_gw_coupling_is_doubly_stochastic():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(60, 6))
    C, D = cosine_distance(x), cosine_distance(x + rng.normal(size=(60, 6)) * 0.3)
    Pi = gw_coupling(C, D, lam=0.1, iters=20)
    assert np.allclose(Pi.sum(1), 1.0, atol=1e-6) and np.allclose(Pi.sum(0), 1.0, atol=1e-6)
    assert (Pi >= 0).all()
    # the two geometries differ only by noise, so the cheapest coupling still pairs each point
    # with itself.
    assert Pi.diagonal().mean() > 1 / 60
    assert gw_cost(C, D, Pi) < gw_cost(C, D, np.full((60, 60), 1 / 60))


def test_gw_coupling_recovers_a_known_relabeling():
    # D is C with its points relabelled: the only geometry-preserving correspondence is that
    # permutation, and GW should find it without ever seeing the coordinates.
    rng = np.random.default_rng(6)
    x = rng.normal(size=(80, 12))
    perm = rng.permutation(80)
    C, D = cosine_distance(x), cosine_distance(x[perm])
    Pi = gw_coupling(C, D, lam=0.01, iters=60)
    inv = np.argsort(perm)  # point i of C is point inv[i] of D
    assert (np.argmax(Pi, 1) == inv).all()
    assert gw_cost(C, D, Pi) < gw_cost(C, D, np.eye(80))


def _relabelled_geometries(n=120, dim=12, seed=6, noise=0.0):
    """Two distance matrices of one geometry with its points relabelled (and optionally perturbed),
    plus the inverse permutation that maps a point of C to its point in D."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    perm = rng.permutation(n)
    y = x[perm] + (rng.normal(size=(n, dim)) * noise if noise else 0)
    return cosine_distance(x), cosine_distance(y), np.argsort(perm)


def test_hard_gw_recovers_a_known_relabeling():
    C, D, inv = _relabelled_geometries(120)
    Pi = hard_gw_coupling(C, D, iters=20)
    assert np.allclose(Pi.sum(1), 1.0) and np.allclose(Pi.sum(0), 1.0)  # a permutation
    assert (Pi.argmax(1) == inv).all()
    assert gw_cost(C, D, Pi) < 1e-6  # an isometry has zero GW cost
    assert gw_cost(C, D, Pi) < gw_cost(C, D, np.eye(120))


def test_annealing_is_a_schedule_and_a_flat_one_is_a_no_op():
    # On a control where an isometry exists, cooling from a soft start should not end in a worse
    # basin than dropping straight into the hard one. On the real data it does not, and that is the
    # point of running both.
    C, D, _ = _relabelled_geometries(150, 20, seed=11, noise=0.5)
    cold = gw_coupling(C, D, lam=0.01, iters=30)
    annealed = gw_coupling(C, D, lam=0.01, iters=30, anneal_from=1.0)
    assert gw_cost(C, D, annealed) <= gw_cost(C, D, cold)
    assert np.allclose(gw_coupling(C, D, lam=0.01, iters=30, anneal_from=0.01), cold)


def test_entropic_gw_finds_a_correspondence_when_one_exists():
    # The capability control behind --gw-control: a geometry against a noisy copy of itself. If the
    # solver cannot recover a pairing that is there, a failure on real data says nothing about the
    # data. The hard assignment cannot, which is why it is a diagnostic and not a competitor.
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 30))
    C = cosine_distance(x); C /= C.mean()
    D = cosine_distance(x + rng.normal(size=(200, 30)) * 0.1); D /= D.mean()
    Pi = gw_coupling(C, D, lam=0.01, iters=30)
    assert (Pi.argmax(1) == np.arange(200)).mean() > 0.9
    assert gw_diagnostics(C, D, Pi)["true_pair_mass"] > 100


def test_couplings_are_equivariant_under_a_joint_relabeling():
    # Relabelling the second geometry must relabel the coupling and change nothing else: a solver
    # that depends on the order of the rows would report a different alignment for the same data.
    C, D, _ = _relabelled_geometries(120)
    q = np.random.default_rng(9).permutation(120)
    D2 = D[np.ix_(q, q)]
    for fit in (lambda a, b: gw_coupling(a, b, 0.01, 60), lambda a, b: hard_gw_coupling(a, b, 20)):
        Pi, Pi2 = fit(C, D), fit(C, D2)
        assert np.allclose(Pi2, Pi[:, q], atol=1e-6)
        assert abs(gw_cost(C, D2, Pi2) - gw_cost(C, D, Pi)) < 1e-6


def test_gw_cost_is_symmetric_in_the_two_geometries():
    C, D, _ = _relabelled_geometries(60, 8, seed=3)
    Pi = gw_coupling(C, D, lam=0.1, iters=20)
    assert abs(gw_cost(C, D, Pi) - gw_cost(D, C, Pi.T)) < 1e-9


def test_gw_diagnostics_are_relative_to_no_pairing_at_all():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(120, 10))
    C, D = cosine_distance(x), cosine_distance(x + rng.normal(size=(120, 10)) * 0.2)
    d = gw_diagnostics(C, D, gw_coupling(C, D, lam=0.01, iters=60))
    assert d["top1"] > 0.9 and d["true_pair_mass"] > 1.0  # 1.0 is what a random pairing gives
    assert d["cost"] < d["cost_uniform"]


def test_transport_with_an_identity_coupling_is_plain_cosine():
    rng = np.random.default_rng(7)
    q, idx = rng.normal(size=(10, 4)), rng.normal(size=(5, 4))
    assert np.allclose(transport(q, idx, np.eye(5)), unit(q) @ unit(idx).T, atol=1e-6)


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
    # retrieval is retrieval_scores on the score matrix, so a non-dot-product baseline can be
    # scored through the second one and must give the identical numbers
    assert retrieval_scores(unit(query) @ index.T, fp, fp[1:], ids, np.array(["b", "c"])) == score


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