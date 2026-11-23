"""Neighbour ordering and the MS1 join, on synthetic embeddings."""
import numpy as np
import pandas as pd

from sna.retrieval import match_table, ms1_context, neighbours


def test_neighbours_exclude_self_and_sort_by_similarity():
    vectors = np.array([[1, 0], [0, 1], [1, 1e-3], [1, -1e-3]], np.float32)
    order, sim = neighbours(vectors, 2)
    assert (order != np.arange(4)[:, None]).all()
    assert order[0].tolist() == [2, 3] and sim[0][0] >= sim[0][1]
    assert sim[0].max() > 0.999  # near-duplicate spectra of the same candidate score ~1


def test_match_table_carries_ms1_context():
    embeddings = pd.DataFrame({"candidate_id": ["a", "b"], "spectrum_index": [0, 1],
                               "precursor_mz": [100.0, 100.02], "rt": [10.0, 12.0], "n_peaks": [3, 3],
                               "embedding": [[1.0, 0.0], [1.0, 1e-6]]})
    traces = pd.DataFrame({"candidate_id": ["a", "b"], "source_sample": ["s1", "s2"],
                           "rt_start": [8.0, 11.0], "rt_end": [12.0, 13.0]})
    catalog = pd.DataFrame({"peak_id": ["a", "b"], "mz": [100.1, 100.2], "rt": [10.0, 12.0],
                            "source_intensity": [1e5, 2e5]})
    row = match_table(embeddings, ms1_context(traces, catalog), k=1).iloc[0]
    assert row.similarity > 0.999 and row.same_precursor
    assert row.query_sample == "s1" and row.match_ms1_mz == 100.2 and row.match_rt_start == 11.0
