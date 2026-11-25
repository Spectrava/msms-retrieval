"""Aligning a frozen spectrum space with a frozen molecule space.

Four baselines of increasing ambition, all fitted on the train fold only:

  CKA          - a number, not a model: how much of the molecule space the spectrum space
                 already contains, measured on matched pairs.
  CCA          - the best linear map, and the canonical correlations it achieves.
  Procrustes   - the best *rigid* map after PCA-matching the dimensions, so the comparison
                 says "can one space be rotated into the other" rather than "can any linear
                 map fit".
  GW           - an entropic Gromov-Wasserstein transport coupling between the two geometries:
                 no shared coordinates, only a correspondence that makes the distances inside
                 each space agree.
  geometry     - two trainable projectors into a shared space, optimised with a pairing term
                 plus a soft constraint that each modality keeps its own neighbourhood
                 structure (cosine Gram) and stays non-collapsed (SIGReg).

The geometry loss is soft on purpose: structural and spectral similarity are not the same
thing, so forcing them to agree exactly is a hypothesis, not a requirement.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def mean_by_group(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """One row per group: the mean of its members. Groups are 0..n-1 and every member is a
    positive of its group, so a compound with many replicate spectra counts once."""
    sums = np.zeros((int(group.max()) + 1, x.shape[1]), dtype=np.float64)
    np.add.at(sums, group, x)
    return (sums / np.bincount(group)[:, None]).astype(np.float32)


def cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear CKA between two matched embedding matrices (1.0 identical up to rotation and
    scale, 0.0 uncorrelated). A measurement, not a retrieval model: it says how much of the
    molecule space the spectrum space already contains, before any map is fitted."""
    a, b = a - a.mean(0), b - b.mean(0)
    return float(np.linalg.norm(a.T @ b) ** 2 / (np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)))


def _inv_sqrtm_sym(c: np.ndarray, lam: float) -> np.ndarray:
    w, v = np.linalg.eigh(c + lam * np.eye(c.shape[0]))
    return v @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-9))) @ v.T


def _sqrtm_sym(c: np.ndarray, lam: float = 0.0) -> np.ndarray:
    w, v = np.linalg.eigh(c + lam * np.eye(c.shape[0]))
    return v @ np.diag(np.sqrt(np.maximum(w, 0.0))) @ v.T


def pca(x: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Train-only dimension matching: mean and the top `dim` principal directions of x."""
    mean = x.mean(0)
    _, _, v = np.linalg.svd(x - mean, full_matrices=False)
    return mean, v[:dim]


def cca(a: np.ndarray, b: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Regularised CCA. Returns the linear map W with b ~ a @ W and the canonical correlations.

    Whitening both spaces and taking the SVD of the cross-Gram gives the canonical directions;
    the map carries the a-space into the b-space along them, weighted by the correlations.
    `lam` is a fraction of each space's mean variance, so it means the same thing whatever the
    scale of the embeddings is - without shrinkage the noise directions of a 1024-d latent are
    whitened to unit length and the correlations saturate at 1.0.
    """
    ca, cb = a.T @ a, b.T @ b
    la, lb = lam * ca.trace() / ca.shape[0], lam * cb.trace() / cb.shape[0]
    ia, ib = _inv_sqrtm_sym(ca, la), _sqrtm_sym(cb, lb)
    u, s, vt = np.linalg.svd((a @ ia).T @ (b @ _inv_sqrtm_sym(cb, lb)), full_matrices=False)
    return (ia @ u @ np.diag(s) @ vt @ ib).astype(np.float32), np.clip(s, 0.0, 1.0)


def procrustes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The orthogonal R minimising ||a R - b||_F. a and b must already share a dimension."""
    u, _, vt = np.linalg.svd(a.T @ b)
    return (u @ vt).astype(np.float32)


def gw_coupling(C: np.ndarray, D: np.ndarray, lam: float = 0.1, iters: int = 30) -> np.ndarray:
    """Entropic Gromov-Wasserstein coupling between two geometries C and D (symmetric distance
    matrices of the same size, uniform marginals).

    Unlike CKA/CCA/Procrustes this never asks the two spaces to share coordinates: it only asks
    that distances *within* each space match under the correspondence Pi, so it can align the
    intrinsic geometry of the spectrum latent with the intrinsic geometry of the molecule latent
    even when their ambient dimensions and scales are unrelated.

    Sinkhorn iterations on the linearised GW cost: Pi = exp((C Pi D) / lam), renormalised to be
    doubly stochastic (C and D are symmetric, so C Pi D is the cross term of GW(Pi)). lam is a
    pure temperature: the linearised cost is centred and divided by its own spread, so lam means
    the same thing for a pool of 100 as for a pool of 10000. lam=1 is a soft, diffuse coupling;
    lam=0.1 is close to a hard assignment.

    shortcut: each iteration is an n^3 matmul, so the pool has to stay in the low thousands;
    upgrade to a low-rank or block GW if a larger pool is ever needed.
    """
    n = len(C)
    C, D = C.astype(np.float64), D.astype(np.float64)
    Pi = np.full((n, n), 1.0 / n)
    for _ in range(iters):
        L = C @ Pi @ D
        # log-domain Sinkhorn: exp would underflow at small lam.
        logPi = (L - L.mean()) / (lam * max(L.std(), 1e-12))
        u, v = np.zeros((n, 1)), np.zeros((1, n))
        for _ in range(1000):
            u = -_logsumexp(logPi + v, 1)
            v = -_logsumexp(logPi + u, 0)
            Pi = np.exp(logPi + u + v)
            if abs(Pi.sum(1) - 1).max() < 1e-9 and abs(Pi.sum(0) - 1).max() < 1e-9:
                break  # Sinkhorn has converged to a doubly stochastic coupling
    return Pi


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))


def gw_cost(C: np.ndarray, D: np.ndarray, Pi: np.ndarray) -> float:
    """GW(Pi) = sum_ij sum_kl (C_ij - D_kl)^2 Pi_ik Pi_jl. For doubly stochastic Pi the squared
    terms are constant, so this is the cross term only."""
    C, D = C.astype(np.float64), D.astype(np.float64)
    return float((C * C).sum() + (D * D).sum() - 2.0 * (Pi * (C @ Pi @ D)).sum())


def cosine_distance(x: np.ndarray) -> np.ndarray:
    """1 - cosine, clipped so the tiny negative round-off cannot make a 'distance' negative."""
    d = 1.0 - unit(x) @ unit(x).T
    return np.clip(d, 0.0, None)


def transport(query: np.ndarray, index: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Apply a fitted coupling to new spectra: each query is the coupling-weighted mixture of the
    molecules of the pool spectra it is close to. Returns a distribution over the pool molecules,
    which retrieval() can score directly against an identity index.

    shortcut: nearest-neighbour barycentric extension of the coupling, not a GW solve with the
    query set as a third geometry; upgrade if the pool neighbourhoods ever get too sparse.
    """
    return unit(query) @ unit(index).T @ Pi


def cosine_gram(x: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix with the diagonal zeroed, so it can be compared entrywise
    without the self-similarity term dominating."""
    k = F.normalize(x, dim=1) @ F.normalize(x, dim=1).T
    return k - torch.eye(len(x), device=x.device, dtype=k.dtype)


def sigreg(x: torch.Tensor, projections: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
    """SIGReg (Balestriero et al. 2024): standardise each feature of the batch, project onto
    random directions, sort along the batch, and compare with the order statistics of a
    standard normal. A latent that has collapsed onto a few directions is far from Gaussian
    in every direction, so this is the anti-collapse term the pairing loss cannot provide."""
    x = (x - x.mean(0)) / x.std(0).clamp(min=1e-6)
    w = torch.randn(x.shape[1], projections, device=x.device, generator=generator)
    p = (x @ w).sort(dim=0).values
    p = (p - p.mean(0)) / p.std(0).clamp(min=1e-6)
    n = x.shape[0]
    quantile = torch.sqrt(torch.tensor(2.0, device=x.device)) * torch.erfinv(
        (2 * torch.arange(n, device=x.device, dtype=x.dtype) + 1) / n - 1)
    return (p - quantile[:, None]).abs().mean()


def fit_geometry(z: np.ndarray, m: np.ndarray, out: int = 256, alpha: float = 1.0, beta: float = 0.1,
                 epochs: int = 20, batch: int = 1024, seed: int = 0, lr: float = 1e-3, verbose: bool = True):
    """Two projectors into a shared space, trained with pairing + geometry + SIGReg.

    Returns (predict_spectrum, predict_molecule), each giving a unit vector in the shared space.
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    head = lambda dim: nn.Sequential(nn.Linear(dim, out), nn.GELU(), nn.Linear(out, out)).to(device)
    spectrum, molecule = head(z.shape[1]), head(m.shape[1])
    zt, mt = torch.tensor(z, dtype=torch.float32).to(device), torch.tensor(m, dtype=torch.float32).to(device)
    opt = torch.optim.AdamW(list(spectrum.parameters()) + list(molecule.parameters()), lr)
    for epoch in range(epochs):
        order = torch.randperm(len(zt), device=device)
        loss = pair = geom = reg = 0.0
        for i in range(0, len(zt), batch):
            r = order[i:i + batch]
            hs, hm = spectrum(zt[r]), molecule(mt[r])
            l = (F.normalize(hs, dim=1) - F.normalize(hm, dim=1)).pow(2).sum(1).mean()
            g = (cosine_gram(hs) - cosine_gram(zt[r])).pow(2).mean() + \
                (cosine_gram(hm) - cosine_gram(mt[r])).pow(2).mean()
            s = sigreg(hs, generator=gen) + sigreg(hm, generator=gen)
            obj = l + alpha * g + beta * s
            opt.zero_grad(); obj.backward(); opt.step()
            loss += obj.item() * len(r); pair += l.item() * len(r); geom += g.item() * len(r); reg += s.item() * len(r)
        n = len(zt)
        if verbose:
            print(f"[geometry] epoch {epoch} loss {loss/n:.4f} pair {pair/n:.4f} geometry {geom/n:.4f} sigreg {reg/n:.4f}", flush=True)
    spectrum.eval(); molecule.eval()

    def predict(net):
        def f(x):
            with torch.no_grad():
                return F.normalize(net(torch.tensor(x, dtype=torch.float32).to(device)), dim=1).cpu().numpy()
        return f
    return predict(spectrum), predict(molecule)


def retrieval_scores(score: np.ndarray, fp_index: np.ndarray, fp_query: np.ndarray,
                    ids_index: np.ndarray, ids_query: np.ndarray, chunk: int = 2000) -> dict:
    """The same metrics as retrieval, but for a precomputed (queries, index) score matrix, so
    baselines that are not dot products (precursor mass) can be scored the same way."""
    from sna.align import tanimoto
    top = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr = 0.0
    for i in range(0, len(score), chunk):
        order = np.argsort(-score[i:i + chunk], axis=1)  # per query, best index first
        for k in top:
            n = min(k, len(fp_index))
            top[k] += sum(tanimoto(fp_index[order[:, j]], fp_query[i:i + chunk]).sum() for j in range(n))
        match = ids_index[order] == ids_query[i:i + chunk][:, None]
        mrr += float(((1.0 / (match.argmax(1) + 1)) * match.any(1)).sum())
    return {f"top{k}": top[k] / (min(k, len(fp_index)) * len(score)) for k in top} | {"mrr": mrr / len(score)}


def retrieval(query: np.ndarray, index: np.ndarray, fp_index: np.ndarray, fp_query: np.ndarray,
              ids_index: np.ndarray, ids_query: np.ndarray, chunk: int = 2000) -> dict:
    """Mean Tanimoto of the top-1/5/10 cosine hits and the MRR of the exact molecule, for
    cosine retrieval of `query` against `index` (both unit-normalised), scored against Morgan
    fingerprints. top-k is a mean similarity, not a sum."""
    return retrieval_scores(unit(query) @ np.asarray(index, dtype=np.float64).T,
                           fp_index, fp_query, ids_index, ids_query, chunk)


def pair_correlation(query: np.ndarray, index: np.ndarray, fp_index: np.ndarray, fp_query: np.ndarray,
                     queries: int = 500, candidates: int = 2000, seed: int = 0) -> float:
    """Spearman rho between cosine similarity in the aligned space and Morgan Tanimoto over
    sampled (query, candidate) pairs: does the aligned space rank structure the way fingerprints do?"""
    from sna.align import tanimoto
    rng = np.random.default_rng(seed)
    q, c = rng.choice(len(query), queries), rng.choice(len(index), candidates)
    sim = unit(query)[q] @ unit(index)[c].T          # (queries, candidates)
    tan = np.array([tanimoto(fp_index[c], fp_query[i]) for i in q])  # one row per query
    a, b = sim.ravel(), tan.ravel()
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))
