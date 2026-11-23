"""ROI graph edges respect the sample flags, and the JEPA predicts a latent from its neighbours."""
import numpy as np
import pandas as pd
import torch

from sna.graph import GraphJEPA, components, degrees, jepa_loss, roi_graph


def _data():
    # a and b co-elute in sample 1, c matches them in sample 2, d has no MS2 at all
    embeddings = pd.DataFrame({"candidate_id": ["a", "a", "b", "c"],
                               "embedding": [[1.0, 0.0], [1.0, 0.1], [1.0, 0.1], [1.0, 0.2]]})
    traces = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"], "source_sample": ["s1", "s1", "s2", "s2"],
                           "rt_start": [1.0, 1.0, 2.0, 3.0], "rt_end": [5.0, 6.0, 7.0, 8.0]})
    catalog = pd.DataFrame({"peak_id": ["a", "b", "c", "d"], "mz": [100.0, 100.1, 100.0, 200.0],
                            "rt": [3.0, 4.0, 5.0, 6.0], "source_intensity": [1.0, 2.0, 3.0, 4.0]})
    return embeddings, traces, catalog


def _pairs(edges):
    return sorted(zip(edges.a, edges.b))


def test_sample_flags_select_the_edges():
    embeddings, traces, catalog = _data()
    _, edges = roi_graph(embeddings, traces, catalog, 0.9)
    assert _pairs(edges) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert list(edges.same_sample) == [True, False, False]
    _, only_within = roi_graph(embeddings, traces, catalog, 0.9, within_sample=True, between_samples=False)
    assert _pairs(only_within) == [("a", "b")]
    _, only_between = roi_graph(embeddings, traces, catalog, 0.9, within_sample=False, between_samples=True)
    assert _pairs(only_between) == [("a", "c"), ("b", "c")]
    try:
        roi_graph(embeddings, traces, catalog, 0.9, within_sample=False, between_samples=False)
    except ValueError:
        pass
    else:
        raise AssertionError("no link type enabled should raise")


def test_roi_without_ms2_is_isolated():
    embeddings, traces, catalog = _data()
    nodes, edges = roi_graph(embeddings, traces, catalog, 0.9)
    assert degrees(nodes, edges).tolist() == [2, 2, 2, 0]
    labels = components(nodes, edges)
    index = list(nodes.candidate_id).index
    assert labels[index("d")] not in labels[:3]
    assert int((labels == labels[index("a")]).sum()) == 3


def test_graph_jepa_lowers_the_latent_prediction_loss():
    embeddings, traces, catalog = _data()
    nodes, edges = roi_graph(embeddings, traces, catalog, 0.9)
    position = {c: i for i, c in enumerate(nodes.candidate_id)}
    z = torch.tensor(np.stack([e if e is not None else [0.0, 0.0] for e in nodes.embedding]), dtype=torch.float32)
    src = torch.tensor([position[c] for c in edges.a], dtype=torch.long)
    dst = torch.tensor([position[c] for c in edges.b], dtype=torch.long)
    connected = torch.tensor(nodes.has_ms2.to_numpy() & (degrees(nodes, edges) > 0))
    model = GraphJEPA(2, hidden=4, layers=1, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    first = jepa_loss(model, z, src, dst, connected).item()
    for _ in range(40):
        loss = jepa_loss(model, z, src, dst, connected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first
    assert np.allclose(model.predict(model(z, src, dst), src, dst).norm(dim=1)[connected].detach().numpy(), 1.0)
    latents = model(z, src, dst)[connected].detach().numpy()
    assert np.linalg.norm(latents - latents.mean(axis=0)) > 0.1, "the latents collapsed to one direction"
