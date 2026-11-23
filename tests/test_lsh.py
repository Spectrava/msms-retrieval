"""The MinHash signature is exact for identical/disjoint spectra and the molecule merge unions peaks."""
import numpy as np
from datasketch import WeightedMinHashGenerator

from sna.lsh import BINS, HASHES, SEED, jaccard, merge_peaks, signature, top_k


def test_identical_spectra_match_and_disjoint_do_not():
    from sna.lsh import hash_array
    wmg = WeightedMinHashGenerator(BINS, HASHES, seed=SEED)
    a = hash_array([signature([100.0, 200.0], [1.0, 2.0], wmg)])
    b = hash_array([signature([100.0, 200.0], [2.0, 4.0], wmg)])  # same after intensity normalisation
    c = hash_array([signature([500.0, 600.0], [1.0, 1.0], wmg)])
    assert jaccard(a, b)[0, 0] == 1.0
    assert jaccard(a, c)[0, 0] < 0.2  # MinHash estimate: disjoint sets still collide on a few hashes


def test_merge_peaks_unions_without_duplicates():
    mz, intensity = merge_peaks([{"mz": [100.0, 101.0], "intensity": [1.0, 2.0]},
                                 {"mz": [101.0, 102.0], "intensity": [2.0, 3.0]}])
    assert list(mz) == [100.0, 101.0, 102.0] and list(intensity) == [1.0, 2.0, 3.0]


def test_top_k_orders_and_truncates():
    sim = np.array([[0.1, 0.9, 0.5], [0.2, 0.3, 0.8]])
    order, scores = top_k(sim, 2)
    assert order.tolist() == [[1, 2], [2, 1]]
    assert np.allclose(scores, [[0.9, 0.5], [0.8, 0.3]])
