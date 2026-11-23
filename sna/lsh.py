"""The Spectrava/molecule_lsh retrieval strategy: MinHash over binned peak intensities, ranked by
Jaccard.

Faithful to the org repo: m/z digitised into 1000 bins of 1 unit (np.digitize on
np.linspace(0, 1000, 999)), intensities normalised to sum 100, datasketch WeightedMinHash with 128
hashes and seed 42, and library spectra of the same SMILES merged into one peak set (makedbs.py's
generate_peak_sets). Similarity is the fraction of equal hashes; the LSH forest only cuts
candidates, ranking uses the exact Jaccard.
"""
from __future__ import annotations

import numpy as np
from datasketch import MinHashLSHForest, WeightedMinHashGenerator

BINS, HASHES, SEED = 1000, 128, 42


def binned(mz, intensity, bins=BINS) -> np.ndarray:
    vector = np.zeros(bins, np.float64)
    vector[np.digitize(mz, np.linspace(0, bins, bins - 1))] += intensity
    return vector


def signature(mz, intensity, wmg: WeightedMinHashGenerator):
    vector = binned(mz, intensity)
    return wmg.minhash(vector / vector.sum() * 100)


def hash_array(minhashes) -> np.ndarray:
    # WeightedMinHash hashvalues are (num_hashes, 2) pairs of (bin, min value); flatten so
    # equality of pairs is the same Jaccard datasketch's MinHash.jaccard computes. Keep the
    # float values: truncating to uint64 makes every value < 1 collide at 0.
    return np.array([mh.hashvalues.reshape(-1) for mh in minhashes], np.float64)


def jaccard(query_hashes: np.ndarray, library_hashes: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Exact Jaccard (fraction of equal hashes) between every query and library signature."""
    out = np.empty((len(query_hashes), len(library_hashes)), np.float32)
    for start in range(0, len(query_hashes), chunk):
        q = query_hashes[start:start + chunk]
        out[start:start + chunk] = (q[:, None, :] == library_hashes[None, :, :]).mean(-1)
    return out


def top_k(similarities: np.ndarray, k: int):
    order = np.argsort(-similarities, axis=1)[:, :k]
    return order, np.take_along_axis(similarities, order, axis=1)


def merge_peaks(spectra: list[dict]):
    """Union of (m/z, intensity) tuples, as makedbs.py does for one molecule."""
    peaks = {(float(m), float(v)) for s in spectra for m, v in zip(s["mz"], s["intensity"])}
    mz, intensity = zip(*sorted(peaks))
    return np.array(mz), np.array(intensity)


def forest(minhashes) -> MinHashLSHForest:
    lsh = MinHashLSHForest(num_perm=HASHES)
    for i, mh in enumerate(minhashes):
        lsh.add(str(i), mh)
    lsh.index()
    return lsh
