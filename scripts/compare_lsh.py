"""Compare the MinHash/LSH retrieval of Spectrava/molecule_lsh with DreaMS cosine retrieval, on
exactly the same query spectra and the same reference library.

    python scripts/compare_lsh.py --traces ../spectralens-jepa/data/exports/7600_traces.parquet \
        --embeddings data/7600_ms2_embeddings.parquet --library-embeddings data/plasma_ms2_embeddings.parquet \
        --msp /mnt/data2/datasets/peaknet_data/mspfiles/BioMSMS-Pos-PlaSMA.msp \
        --msp /mnt/data2/datasets/peaknet_data/mspfiles/BioMSMS-Neg-PlaSMA.msp

Both methods see every peak: LSH bins all of them into 1000 m/z bins, while DreaMS truncates to its
128 most intense peaks. The LSH side merges library spectra of the same SMILES into one peak set
(molecule level, like makedbs.py); the cosine side aggregates per-spectrum scores to the molecule
with a max. Cross-polarity pairs are excluded in both.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasketch import WeightedMinHashGenerator

sys.path.insert(0, os.path.dirname(__file__))
from embed_ms2 import read_ms2

from sna.encoders.dreams import load_dreams
from sna.lsh import BINS, HASHES, SEED, forest, hash_array, jaccard, merge_peaks, signature, top_k
from sna.library import read_msp
from sna.retrieval import unit
from sna.spectra import POLARITIES, SpectraTable, arrays_from_spectra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--embeddings", required=True, help="query MS2 embeddings (embed_ms2.py)")
    ap.add_argument("--library-embeddings", required=True, help="library MS2 embeddings (embed_library.py)")
    ap.add_argument("--msp", action="append", required=True)
    ap.add_argument("--out", default="data/7600_vs_plasma_lsh.parquet")
    ap.add_argument("--ckpt", default="checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--perturb", type=int, default=500, help="queries for the peak-dropping robustness test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    queries = read_ms2(args.traces)
    library = [s for f in args.msp for s in read_msp(f)]
    qemb, lemb = np.stack(pd.read_parquet(args.embeddings)["embedding"]), np.stack(pd.read_parquet(args.library_embeddings)["embedding"])
    if len(queries) != len(qemb) or len(library) != len(lemb):
        raise SystemExit("embeddings do not match the spectra - rerun embed_ms2.py and embed_library.py")

    groups = {}
    for i, s in enumerate(library):
        groups.setdefault(s["smiles"] or s["name"] or str(i), []).append(i)
    keys = np.array(list(groups))
    mol_prec = np.array([library[i[0]]["precursor_mz"] for i in groups.values()])
    mol_pol = np.array([library[i[0]]["polarity"] for i in groups.values()])
    q_pol = np.array([s["polarity"] for s in queries])

    wmg = WeightedMinHashGenerator(BINS, HASHES, seed=SEED)
    t0 = time.time()
    lib_sig = [signature(*merge_peaks([library[i] for i in idxs]), wmg) for idxs in groups.values()]
    q_sig = [signature(s["mz"], s["intensity"], wmg) for s in queries]
    lib_h, q_h = hash_array(lib_sig), hash_array(q_sig)
    t_sig = time.time() - t0

    t0 = time.time()
    jac = jaccard(q_h, lib_h)
    cross = q_pol[:, None] != mol_pol[None, :]
    jac[cross] = -1.0
    t_jac = time.time() - t0

    cos = unit(qemb) @ unit(lemb).T
    spec_to_mol = np.zeros(len(library), np.int32)
    for mi, idxs in enumerate(groups.values()):
        spec_to_mol[idxs] = mi
    order = np.argsort(spec_to_mol, kind="stable")
    starts = np.concatenate(([0], np.flatnonzero(np.diff(spec_to_mol[order])) + 1))
    mol_cos = np.maximum.reduceat(cos[:, order], starts, axis=1)
    mol_cos[cross] = -1.0

    lsh_order, lsh_score = top_k(jac, args.k)
    emb_order, emb_score = top_k(mol_cos, args.k)
    same_top = np.array(keys[lsh_order[:, 0]]) == np.array(keys[emb_order[:, 0]])
    overlap = [len(set(lsh_order[i, :args.k]) & set(emb_order[i, :args.k])) for i in range(len(queries))]
    prec_hit_lsh = np.abs(mol_prec[lsh_order[:, 0]] - np.array([s["precursor_mz"] for s in queries])) < 0.05
    prec_hit_emb = np.abs(mol_prec[emb_order[:, 0]] - np.array([s["precursor_mz"] for s in queries])) < 0.05

    print(f"{len(queries)} queries vs {len(library)} library spectra -> {len(keys)} molecules "
          f"(LSH signatures in {t_sig:.1f}s, exact Jaccard of {len(queries)}x{len(keys)} in {t_jac:.1f}s)")
    for name, score, prec in (("MinHash/Jaccard", lsh_score[:, 0], prec_hit_lsh),
                              ("DreaMS cosine", emb_score[:, 0], prec_hit_emb)):
        print(f"  {name}: top-1 mean {score.mean():.3f} median {np.median(score):.3f} max {score.max():.3f}, "
              f"precursor match {prec.mean():.1%}")
    print(f"  same top-1 molecule: {same_top.mean():.1%}; mean shared molecules in top-{args.k}: {np.mean(overlap):.2f}")
    print(f"  top-1 score correlation: {np.corrcoef(lsh_score[:, 0], emb_score[:, 0])[0, 1]:.3f}")

    lshf = forest(lib_sig)
    candidates = [lshf.query(mh, 100) for mh in q_sig]
    recalled = [str(lsh_order[i, 0]) in c for i, c in enumerate(candidates)]
    print(f"  LSH forest (num_perm={HASHES}): {np.mean([len(c) for c in candidates]):.1f} candidates/query, "
          f"exact top-1 recalled {np.mean(recalled):.1%} of the time")

    # Self-test: can each method tell apart library molecules from each other?
    pos = [i for i, s in enumerate(library) if s["polarity"] == POLARITIES.index("positive")]
    self_sig = [signature(s["mz"], s["intensity"], wmg) for s in (library[i] for i in pos)]
    self_h = hash_array(self_sig)
    sim = jaccard(self_h, self_h)
    np.fill_diagonal(sim, -1.0)
    cos_self = unit(lemb[pos]) @ unit(lemb[pos]).T
    np.fill_diagonal(cos_self, -1.0)
    print(f"  library self-test (nearest other spectrum): Jaccard mean {sim.max(1).mean():.3f} "
          f"({(sim.max(1) >= 0.9).sum()} pairs >= 0.9) | cosine mean {cos_self.max(1).mean():.3f} "
          f"({(cos_self.max(1) >= 0.9).sum()} pairs >= 0.9)")

    # Robustness: drop 30% of each query's peaks and see if the top-1 molecule survives.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(queries), args.perturb, replace=False)
    perturbed = []
    for i in idx:
        s = queries[i]
        keep = rng.random(len(s["mz"])) >= 0.3
        perturbed.append(dict(s, mz=s["mz"][keep], intensity=s["intensity"][keep]))
    t0 = time.time()
    pert_h = hash_array([signature(s["mz"], s["intensity"], wmg) for s in perturbed])
    pert_lsh = top_k(jaccard(pert_h, lib_h), 1)[0][:, 0]
    encoder = load_dreams(args.ckpt).to(args.device).eval()
    table = SpectraTable(arrays_from_spectra(perturbed), device=args.device)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(perturbed), 256):
            batch = table.batch(torch.arange(start, min(start + 256, len(perturbed)), device=args.device),
                                encoder.max_peaks)
            vectors.append(encoder(batch)[0].float().cpu().numpy())
    pert_cos = np.maximum.reduceat((np.concatenate(vectors) @ unit(lemb).T)[:, order], starts, axis=1)
    pert_emb = top_k(pert_cos, 1)[0][:, 0]
    print(f"  perturbation (drop 30% of query peaks, {args.perturb} queries, {time.time() - t0:.1f}s): "
          f"top-1 molecule unchanged - Jaccard {np.mean(pert_lsh == lsh_order[idx, 0]):.1%}, "
          f"cosine {np.mean(pert_emb == emb_order[idx, 0]):.1%}")

    out = pd.DataFrame({
        "candidate_id": [s["candidate_id"] for s in queries],
        "spectrum_index": [s["spectrum_index"] for s in queries],
        "precursor_mz": [s["precursor_mz"] for s in queries],
        "n_peaks": [len(s["mz"]) for s in queries],
        "lsh_top": [keys[i] for i in lsh_order[:, 0]],
        "lsh_jaccard": lsh_score[:, 0],
        "emb_top": [keys[i] for i in emb_order[:, 0]],
        "emb_cosine": emb_score[:, 0],
        "agree": same_top,
        "topk_overlap": overlap,
    })
    out.to_parquet(args.out)
    print(f"wrote per-query comparison to {args.out}")


if __name__ == "__main__":
    main()
