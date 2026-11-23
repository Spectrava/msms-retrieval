"""Nearest-neighbour MS2 retrieval over pinned embeddings, joined back to the MS1 trace.

Similarity is cosine over the encoder's global vectors. A match is only useful if it carries the
RT window and apex m/z of the MS1 trace the spectrum was attached to, so that join is part of the
contract rather than an afterthought.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def unit(vectors) -> np.ndarray:
    v = np.asarray(vectors, np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def neighbours(vectors, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours of every spectrum, self excluded. Returns (indices, similarities)."""
    u = unit(vectors)
    sim = u @ u.T
    np.fill_diagonal(sim, -1.0)
    order = (-sim).argsort(1, kind="stable")[:, :k]
    return order, np.take_along_axis(sim, order, 1)


def ms1_context(traces: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: sample, RT window, and the apex m/z/RT/intensity of its MS1 trace."""
    apex = catalog.rename(columns={"peak_id": "candidate_id", "mz": "ms1_mz", "rt": "ms1_rt",
                                   "source_intensity": "ms1_intensity"})
    return (traces[["candidate_id", "source_sample", "rt_start", "rt_end"]]
            .rename(columns={"source_sample": "sample"})
            .merge(apex[["candidate_id", "ms1_mz", "ms1_rt", "ms1_intensity"]], on="candidate_id"))


def library_match(query: pd.DataFrame, library: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Top-k library spectra for every query. Spectra of different polarity are never compared -
    the encoder embeds positive and negative MS2 differently, so a cross-polarity cosine is
    meaningless. The library side carries the metadata the library actually states."""
    sim = unit(np.stack(query["embedding"])) @ unit(np.stack(library["embedding"])).T
    sim[query["polarity"].to_numpy()[:, None] != library["polarity"].to_numpy()[None, :]] = -1.0
    order = (-sim).argsort(1, kind="stable")[:, :k]
    sims = np.take_along_axis(sim, order, 1)
    rows = np.repeat(np.arange(len(query)), k)
    lib = library.iloc[order.ravel()].reset_index(drop=True)
    return pd.DataFrame({
        "query_candidate_id": query["candidate_id"].to_numpy()[rows],
        "query_spectrum": query["spectrum_index"].to_numpy()[rows],
        "query_precursor_mz": query["precursor_mz"].to_numpy()[rows],
        "query_rt": query["rt"].to_numpy()[rows],
        "library": lib["library"], "name": lib["name"], "formula": lib["formula"],
        "smiles": lib["smiles"], "match_precursor_mz": lib["precursor_mz"],
        "similarity": sims.ravel(),
        "same_precursor": np.abs(query["precursor_mz"].to_numpy()[rows] - lib["precursor_mz"].to_numpy()) < 0.05,
    })


def match_table(embeddings: pd.DataFrame, context: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """One row per (query spectrum, neighbour spectrum) with both sides' MS1 trace context."""
    order, sim = neighbours(np.stack(embeddings["embedding"]), k)
    rows = np.arange(len(embeddings))
    flat = np.repeat(rows, k), order.ravel()
    query = embeddings.iloc[flat[0]].reset_index(drop=True)
    match = embeddings.iloc[flat[1]].reset_index(drop=True)
    out = pd.DataFrame({
        "query_candidate_id": query["candidate_id"], "query_spectrum": query["spectrum_index"],
        "query_precursor_mz": query["precursor_mz"], "query_rt": query["rt"], "query_n_peaks": query["n_peaks"],
        "match_candidate_id": match["candidate_id"], "match_spectrum": match["spectrum_index"],
        "match_precursor_mz": match["precursor_mz"], "match_rt": match["rt"], "match_n_peaks": match["n_peaks"],
        "similarity": sim.ravel(),
        "same_precursor": np.abs(query["precursor_mz"].to_numpy() - match["precursor_mz"].to_numpy()) < 0.05,
    })
    ctx = context.set_index("candidate_id")
    for side in ("query", "match"):
        ids = out[f"{side}_candidate_id"].to_numpy()
        for column in ("sample", "rt_start", "rt_end", "ms1_mz", "ms1_rt", "ms1_intensity"):
            out[f"{side}_{column}"] = ctx.loc[ids, column].to_numpy()
    return out
