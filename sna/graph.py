"""The ROI graph behind the graph JEPA: one node per ROI trace, an edge when two ROIs' MS2 spectra
match.

Only ROIs that actually have MS2 can be linked - the graph records what was measured, not what is
guessed. `within_sample` and `between_samples` turn the two kinds of edge on or off, so a run can
be about co-elution inside one acquisition or about matching the same compound across acquisitions.
A node's feature is the mean unit embedding of its MS2 spectra; an edge exists when two such
features are within `threshold` cosine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from sna.retrieval import ms1_context, unit

THRESHOLD = 0.9


def roi_graph(embeddings: pd.DataFrame, traces: pd.DataFrame, catalog: pd.DataFrame,
              threshold: float = THRESHOLD, within_sample: bool = True, between_samples: bool = True):
    """Return (nodes, edges). Nodes are every ROI trace; edges are undirected MS2-similarity links."""
    if not within_sample and not between_samples:
        raise ValueError("enable at least one of within_sample or between_samples")
    nodes = ms1_context(traces, catalog)
    ids = nodes.candidate_id.to_numpy()
    position = {c: i for i, c in enumerate(ids)}
    dim = len(embeddings.embedding.iloc[0])
    acc, count = np.zeros((len(ids), dim), np.float32), np.zeros(len(ids), np.int32)
    for candidate, vector in zip(embeddings.candidate_id, unit(np.stack(embeddings["embedding"]))):
        i = position[candidate]
        acc[i] += vector
        count[i] += 1
    features = acc / np.maximum(count, 1)[:, None]
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)

    # Only ROIs with MS2 can be linked, so the similarity matrix is over those, not all 42k traces.
    measured = count > 0
    slot = np.where(measured)[0]
    sim = features[slot] @ features[slot].T
    np.fill_diagonal(sim, -1.0)
    same = nodes["sample"].to_numpy()[slot]  # not nodes.sample: that is DataFrame.sample
    keep = sim >= threshold
    if within_sample and not between_samples:
        keep &= same[:, None] == same[None, :]
    elif between_samples and not within_sample:
        keep &= same[:, None] != same[None, :]
    i, j = np.where(np.triu(keep, 1))
    edges = pd.DataFrame({"a": ids[slot[i]], "b": ids[slot[j]], "similarity": sim[i, j],
                          "same_sample": same[i] == same[j]})
    nodes = nodes.assign(n_spectra=count, has_ms2=measured,
                        embedding=[v.tolist() if n else None for v, n in zip(features, count)])
    return nodes, edges


def components(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """Connected-component id of every node (isolated ROIs get their own)."""
    adjacency = {c: set() for c in nodes.candidate_id}
    for a, b in zip(edges.a, edges.b):
        adjacency[a].add(b)
        adjacency[b].add(a)
    label = {}
    for start in nodes.candidate_id:
        if start in label:
            continue
        stack, label[start] = [start], len(label)
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in label:
                    label[neighbour] = label[start]
                    stack.append(neighbour)
    return np.array([label[c] for c in nodes.candidate_id])


def degrees(nodes: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """Number of MS2-similarity links per node."""
    index = {c: i for i, c in enumerate(nodes.candidate_id)}
    degree = np.zeros(len(nodes), np.int32)
    for column in ("a", "b"):
        for candidate in edges[column]:
            degree[index[candidate]] += 1
    return degree


def neighbour_mean(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Mean of the neighbours of every node; isolated nodes get the zero vector."""
    agg = torch.zeros_like(x)
    agg.index_add_(0, dst, x[src])
    degree = torch.zeros(len(x), device=x.device)
    degree.index_add_(0, dst, torch.ones(len(src), device=x.device))
    return agg / degree.clamp(min=1)[:, None]


class GraphJEPA(nn.Module):
    """Message passing over the ROI graph plus a latent predictor: a ROI's frozen MS2 embedding is
    predicted from its neighbours' latents. JEPA because the prediction is made in latent space -
    no peaks or spectra are reconstructed.
    """

    def __init__(self, dim: int, hidden: int = 256, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        def block(out):
            return nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, out))
        self.layers = nn.ModuleList([block(dim) for _ in range(layers)])
        self.predictor = block(dim)

    def forward(self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Graph-informed latents: each ROI's latent plus a unit-norm message from its neighbours, so
        a ROI with no neighbours keeps its own embedding and a message never swamps it."""
        h = z
        for layer in self.layers:
            h = F.normalize(h + F.normalize(layer(neighbour_mean(h, src, dst)), dim=-1), dim=-1)
        return h

    def predict(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Latent of a ROI from its neighbours only - the ROI itself is not in the pool."""
        return F.normalize(self.predictor(neighbour_mean(h, src, dst)), dim=-1)


def jepa_loss(model: GraphJEPA, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
              connected: torch.Tensor, spread_weight: float = 0.1) -> torch.Tensor:
    """1 - cosine between the neighbour-predicted latent and the ROI's own embedding, plus a term that
    keeps the latents apart. Without it the loss is minimised by mapping every ROI to one vector,
    which is what the released run did (mean pairwise cosine 0.99). Only ROIs with MS2 and at least
    one neighbour enter the loss; isolated ROIs have nothing to predict from."""
    h = model(z, src, dst)
    error = 1 - (model.predict(h, src, dst) * z).sum(-1)
    latents = h[connected]
    n = latents.shape[0]
    collapse = 0.0 if n < 2 else ((latents @ latents.T).sum() - n) / (n * (n - 1))
    return (error + spread_weight * collapse)[connected].mean()
