"""Retrieve the nearest MS2 spectra for every embedded spectrum and link the hits to MS1 traces.

    python scripts/ms2_retrieval.py --embeddings data/7600_ms2_embeddings.parquet \
        --traces /home/dmoi/projects/spectralens-jepa/data/exports/7600_traces.parquet \
        --catalog /home/dmoi/projects/spectralens-jepa/data/exports/7600_catalog.parquet \
        --out data/7600_ms2_matches.parquet --k 10

A hit is reported with the RT window and apex m/z of the MS1 trace the matched spectrum came from,
so a retrieval result can be checked against the chromatography that produced it.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sna.retrieval import library_match, match_table, ms1_context


def report(matches: pd.DataFrame, k: int) -> None:
    best = matches.iloc[np.arange(len(matches) // k) * k]  # the top neighbour of each query
    print(f"{len(best)} queries, {k} neighbours each")
    print(f"top-1 similarity: mean {best.similarity.mean():.3f}, median {best.similarity.median():.3f}, "
          f"max {best.similarity.max():.3f}")
    print(f"top-1 hits sharing the precursor m/z (+-0.05): {best.same_precursor.mean():.1%}")
    print(f"top-1 hits that are another spectrum of the same candidate: "
          f"{(best.query_candidate_id == best.match_candidate_id).mean():.1%}")
    print(f"top-1 hits from the same sample: {(best.query_sample == best.match_sample).mean():.1%}")
    strong = best[(best.query_candidate_id != best.match_candidate_id) & best.same_precursor]
    print(f"\n{len(strong)} distinct-candidate pairs with the same precursor m/z; strongest 10:")
    for row in strong.sort_values("similarity", ascending=False).head(10).itertuples():
        print(f"  {row.similarity:.4f}  {row.query_precursor_mz:.4f} @ {row.query_rt:.1f}s "
              f"(MS1 {row.query_ms1_mz:.4f}, {row.query_rt_start:.1f}-{row.query_rt_end:.1f}s) -> "
              f"{row.match_precursor_mz:.4f} @ {row.match_rt:.1f}s "
              f"(MS1 {row.match_ms1_mz:.4f}, {row.match_rt_start:.1f}-{row.match_rt_end:.1f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--traces", help="Spectrava traces Parquet (self-match mode only)")
    ap.add_argument("--catalog", help="Spectrava MS1 catalog Parquet (self-match mode only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--library", help="reference library embeddings from scripts/embed_library.py; "
                                      "queries then match against the library instead of each other")
    args = ap.parse_args()

    embeddings = pd.read_parquet(args.embeddings)
    if args.library:
        matches = library_match(embeddings, pd.read_parquet(args.library), args.k)
        best = matches.iloc[np.arange(len(matches) // args.k) * args.k]
        print(f"{len(best)} queries against {len(pd.read_parquet(args.library))} library spectra "
              f"(only same-polarity pairs compared)")
        print(f"top-1 similarity: mean {best.similarity.mean():.3f}, median {best.similarity.median():.3f}, "
              f"max {best.similarity.max():.3f}")
        print(f"queries whose top-1 hit reaches 0.9: {(best.similarity >= 0.9).mean():.1%}")
        print(f"top-1 hits sharing the precursor m/z (+-0.05): {best.same_precursor.mean():.1%}")
        strong = best[best.similarity >= 0.9]
        print(f"\n{len(strong)} queries with a top-1 library hit at >= 0.9; strongest 10:")
        for row in strong.sort_values("similarity", ascending=False).head(10).itertuples():
            print(f"  {row.similarity:.4f}  query {row.query_precursor_mz:.4f} @ {row.query_rt:.1f}s "
                  f"(candidate {row.query_candidate_id}) -> {row.name} [{row.formula}] "
                  f"precursor {row.match_precursor_mz:.4f}")
    else:
        if not (args.traces and args.catalog):
            ap.error("--traces and --catalog are needed when no --library is given")
        matches = match_table(embeddings, ms1_context(pd.read_parquet(args.traces), pd.read_parquet(args.catalog)), args.k)
        report(matches, args.k)
    matches.to_parquet(args.out)
    print(f"wrote {len(matches)} matches to {args.out}")


if __name__ == "__main__":
    main()
