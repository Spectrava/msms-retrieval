"""Joint molecular-MS/MS alignment benchmark.

Frozen DreaMS spectrum embeddings against a frozen molecule encoder (Mol-JEPA, CC BY-NC, or
MoLFormer XL as the commercially permissive fallback), aligned with the four baselines from
the brief - linear CKA, regularised CCA, orthogonal Procrustes, symmetric InfoNCE heads - plus
the geometry-preserving projector (pairing + cosine-Gram + SIGReg) and an entropic
Gromov-Wasserstein transport coupling between the two intrinsic geometries. With --synthetic a
third modality is added: rule-based in-silico spectra of the same molecules, so the alignment can
be fitted on synthetic pairs and tested on real held-out spectra.

Everything is fitted on the train fold; the reported fold is the held-out one. Retrieval is
scored against the *train* molecules, so the ceiling is the fingerprint similarity of the
nearest train molecule to the true one, and the shortcut baselines (precursor mass, random
pair) say how much of any gain is real.

    python scripts/align_joint.py --encoder moljepa --dims 256 --seeds 3
    python scripts/align_joint.py --encoder molformer --methods cka,cca,procrustes
    python scripts/align_joint.py --encoder molformer --methods gw --gw-sub 2000 --gw-lam 0.1
    python scripts/align_joint.py --encoder moljepa --methods gw --gw-method entropic,hard --gw-anneal 1.0
    python scripts/align_joint.py --encoder molformer --methods cka,cca,procrustes,infonce --synthetic
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from sna import paths
from sna.align import canonical, fit_contrastive, fingerprints, tanimoto_matrix
from sna.data import read_msg_tsv
from sna.joint import (cca, cka, cosine_distance, fit_geometry, gw_coupling, gw_diagnostics,
                       hard_gw_coupling, mean_by_group, pair_correlation, pca, procrustes, retrieval,
                       retrieval_scores, transport, unit)
from sna.towers import Tower

RDLogger.DisableLog("rdApp.error")


def molecule_embeddings(encoder: str, smiles: list[str], cache: Path) -> np.ndarray:
    """Pooled embeddings of a SMILES list through a frozen encoder, cached as float32."""
    if cache.exists():
        c = np.load(cache)
        if "smiles" in c.files and list(c["smiles"]) == smiles:
            return c["emb"].astype(np.float32)
    if encoder == "moljepa":
        from sna.moljepa import embed, load
        model = load("cuda")
        emb = embed(smiles, model, batch=256)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(*paths.TOWERS[encoder], trust_remote_code=True)
        model = Tower(encoder).cuda()
        emb = np.zeros((len(smiles), model.dim), dtype=np.float32)
        for i in range(0, len(smiles), 1024):
            x = tok(smiles[i:i + 1024], padding=True, return_tensors="pt").to("cuda")
            emb[i:i + 1024] = model(x["input_ids"], x["attention_mask"].bool(), pooled=True)[0][:, 0].float().cpu().numpy()
    np.savez(cache, smiles=np.array(smiles), emb=emb, encoder=encoder)
    return emb


def embed_spectra(data: dict, ckpt: str, rows=None) -> np.ndarray:
    """Frozen DreaMS embeddings of the rows of a CSR spectra dict, in row order."""
    from sna.encoders.dreams import load_dreams
    from sna.spectra import SpectraTable
    model = load_dreams(ckpt).cuda().eval()
    table = SpectraTable(data, rows=rows, device="cuda")
    with torch.no_grad():  # without it the autograd graph of 7 blocks per batch costs ~20 GB
        return np.concatenate([
            model(table.batch(torch.arange(i, min(i + 256, len(table)), device="cuda"), model.max_peaks))[0]
            .float().cpu().numpy()
            for i in range(0, len(table), 256)
        ])


def collapse_report(x: np.ndarray) -> tuple[float, float]:
    """Standard deviation and effective rank of a latent: how much of it is still there."""
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    p = s / s.sum()
    return float(x.std()), float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default=str(paths.MSG_TSV))
    parser.add_argument("--spectra", default="data/msg_generated.parquet.embeddings.npz")
    parser.add_argument("--encoder", default="moljepa", choices=("moljepa", "molformer", "chemberta"))
    parser.add_argument("--fold", default="val", choices=("val", "test"))
    parser.add_argument("--dims", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--methods", default="cka,cca,procrustes,infonce,geometry,gw")
    parser.add_argument("--gw-sub", type=int, default=2000, help="molecules in the GW pool (n^3 per iteration)")
    parser.add_argument("--gw-lam", default="0.1", help="entropic temperature(s) of the GW coupling; "
                                                         "a comma-separated list sweeps the temperature")
    parser.add_argument("--gw-iters", type=int, default=30)
    parser.add_argument("--gw-method", default="entropic,hard",
                        help="which GW couplings to fit: entropic (fixed lam), hard (Hungarian "
                             "assignment, no temperature); --gw-anneal adds the annealed entropic run")
    parser.add_argument("--gw-anneal", type=float, default=None,
                        help="start the entropic coupling at this lam and anneal geometrically to "
                             "--gw-lam; the annealed run is the one to compare against")
    parser.add_argument("--gw-control", action="store_true",
                        help="run the capability controls: each geometry against a noisy copy of "
                             "itself and against the Tanimoto geometry of the fingerprints")
    parser.add_argument("--gw-control-sub", type=int, default=1000,
                        help="molecules in the control pool (the controls are run per seed)")
    parser.add_argument("--synthetic", action="store_true",
                        help="add rule-based in-silico spectra: fit the alignment on synthetic pairs and "
                             "test it on real held-out spectra")
    parser.add_argument("--dreams", default="checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt")
    parser.add_argument("--out", default="results/metrics.csv")
    args = parser.parse_args()
    methods = args.methods.split(",")
    lams = [float(x) for x in str(args.gw_lam).split(",")]
    started = time.time()

    msg = read_msg_tsv(args.tsv)
    fold = np.array(msg["fold"])
    smiles = [canonical(s) for s in msg["smiles"]]
    ok = np.array([bool(s) for s in smiles])
    train_rows = np.where((fold == "train") & ok)[0]
    held_rows = np.where((fold == args.fold) & ok)[0]

    emb = np.load(args.spectra)
    z_train = emb["train"][np.where(ok[np.where(fold == "train")[0]])[0]]  # the cache is in fold order
    if args.fold in emb.files:
        z_held = emb[args.fold][np.where(ok[np.where(fold == args.fold)[0]])[0]]
    else:
        from sna.spectra import SpectraTable
        z_held = embed_spectra(msg, args.dreams, held_rows)

    unique = sorted({smiles[i] for i in train_rows})
    mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
    held_ids = np.array([smiles[i] for i in held_rows])
    assert not set(held_ids) & set(unique), f"{args.fold} molecules leak into the train index"

    m = molecule_embeddings(args.encoder, unique + list(held_ids), Path(f"data/molecule_embeddings_{args.encoder}.npz"))
    m_train, m_held = m[: len(unique)], m[len(unique):]
    fp = fingerprints(unique + list(held_ids))
    fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

    # one row per molecule: the mean spectrum of its train replicates, so a compound with many
    # spectra cannot dominate the batch and every replicate of a molecule is a positive
    z_unique = mean_by_group(z_train, mol_of_row)
    z_train_n, m_train_n = unit(z_unique), unit(m_train)

    ids_train = np.array(unique)
    rows = []
    def record(method, query, index, fp_index, ids_index, index_held=None, note=""):
        # Two pools: the held fold shares no molecule with the train fold, so retrieval against
        # train measures transfer to unseen molecules, and retrieval against the fold's own
        # molecules is the only place an exact match is possible. index_held is left empty for
        # baselines whose query and index are the same modality, where "nearest peer" is
        # trivially the query itself. index=None means `query` is already a (queries, index)
        # score matrix (the precursor-mass shortcut), which has no embedding space whose collapse
        # or rank correlation could be reported.
        if index is None:
            score = retrieval_scores(np.asarray(query, dtype=np.float32), fp_index, fp_held,
                                     ids_index, held_ids)
            std = erank = spearman = None
            pool = len(fp_index)
        else:
            query = unit(np.asarray(query, dtype=np.float32))
            index = unit(np.asarray(index, dtype=np.float32))
            score = retrieval(query, index, fp_index, fp_held, ids_index, held_ids)
            std, erank = collapse_report(query)
            spearman = pair_correlation(query, index, fp_index, fp_held)
            pool = len(index)
        peer = retrieval(query, unit(np.asarray(index_held, dtype=np.float32)),
                         fp_held, fp_held, held_ids, held_ids) if index_held is not None else None
        row = dict(encoder=args.encoder, fold=args.fold, method=method,
                   dim=1 if index is None else query.shape[1],
                   pool=pool,
                   top1=score["top1"], top5=score["top5"], top10=score["top10"],
                   top1_peer=peer["top1"] if peer else None,
                   exact_peer=peer["mrr"] if peer else None,
                   spearman=spearman, std=std, effective_rank=erank, note=note)
        rows.append(row)
        peer_txt = "" if peer is None else f"  peer exact {row['exact_peer']:.3f}"
        collapse = "" if std is None else f"  rho {spearman:.3f}  std {std:.3f}  erank {erank:.1f}"
        print(f"{method:<24} pool {pool:>6}  top1 {row['top1']:.3f}  top5 {row['top5']:.3f}  "
              f"top10 {row['top10']:.3f}{peer_txt}{collapse}")

    print(f"CKA(frozen spectrum space, frozen molecule space) = {cka(z_unique, m_train):.4f}")

    if "cca" in methods:
        W, corr = cca(z_unique.astype(np.float64), m_train.astype(np.float64), lam=1e-3)
        print(f"CCA: top canonical correlations {corr[:5].round(3)}, "
              f"{(corr > 0.9).sum()} of {len(corr)} above 0.9")
        record("cca", z_held @ W, m_train_n, fp_train, ids_train, m_held)
    if "procrustes" in methods:
        mean, comp = pca(z_unique, m_train.shape[1])
        R = procrustes((z_unique - mean) @ comp.T, m_train)
        record("procrustes", ((z_held - mean) @ comp.T) @ R, m_train_n, fp_train, ids_train, m_held)
    if "infonce" in methods:
        for seed in range(args.seeds):
            spectrum, molecule = fit_contrastive(z_unique, m_train, out=args.dims, seed=seed)
            record("infonce", spectrum(z_held), molecule(m_train), fp_train, ids_train,
                   molecule(m_held), f"seed {seed}")
    if "geometry" in methods:
        for alpha, beta in ((0.0, 0.0), (args.alpha, 0.0), (0.0, args.beta), (args.alpha, args.beta)):
            for seed in range(args.seeds):
                spectrum, molecule = fit_geometry(z_unique, m_train, out=args.dims, alpha=alpha, beta=beta,
                                                 epochs=args.epochs, seed=seed)
                record(f"geometry a{alpha} b{beta}", spectrum(z_held), molecule(m_train), fp_train,
                       ids_train, molecule(m_held), f"seed {seed}")
    if "gw" in methods:
        for seed in range(args.seeds):
            sub = np.random.default_rng(seed).choice(len(unique), args.gw_sub, replace=False)
            C, D = cosine_distance(z_unique[sub]), cosine_distance(m_train[sub])
            C, D = C / C.mean(), D / D.mean()
            n = len(sub)
            names = args.gw_method.split(",") + (["annealed"] if args.gw_anneal else [])
            for lam in lams:
                fit = {"entropic": lambda: gw_coupling(C, D, lam, args.gw_iters),
                       "annealed": lambda: gw_coupling(C, D, lam, args.gw_iters, anneal_from=args.gw_anneal),
                       "hard": lambda: hard_gw_coupling(C, D, args.gw_iters)}
                for name in names:
                    Pi = fit[name]()
                    d = gw_diagnostics(C, D, Pi)
                    print(f"[gw {name:<8}] seed {seed} lam {lam:<5} pool {n}: cost {d['cost']:>10.1f}  "
                          f"identity {d['cost_identity']:>10.1f}  uniform {d['cost_uniform']:>10.1f}  "
                          f"true-pair mass {d['true_pair_mass']:.2f}x uniform  top1 {d['top1']:.3f}")
                    record(f"gw {name} lam{lam}", transport(z_held, z_unique[sub], Pi), None,
                           fp_train[sub], ids_train[sub], note=f"seed {seed}")
            # Control: can this solver find a correspondence at all in geometries of this size and
            # scale? Pair each geometry with a noisy copy of itself (an isometry exists by
            # construction) and with the Tanimoto geometry of the fingerprints - chemistry, the
            # ground truth the latents are supposed to encode. If the true pairing is recovered
            # there but not between the two modalities, the mismatch is the data, not the transport.
            if args.gw_control:
                ctl = np.random.default_rng(seed).choice(len(unique), args.gw_control_sub, replace=False)
                A = cosine_distance(z_unique[ctl]); A /= A.mean()
                B = cosine_distance(m_train[ctl]); B /= B.mean()
                noise = np.random.default_rng(seed + 7).normal(size=m_train[ctl].shape) * 0.1
                Bn = cosine_distance(m_train[ctl] + noise); Bn /= Bn.mean()
                T = 1.0 - tanimoto_matrix(fp_train[ctl], fp_train[ctl]); T /= T.mean()
                for name, (X, Y) in (("self", (B, Bn)), ("molecule vs chemistry", (B, T)),
                                     ("spectrum vs chemistry", (A, T))):
                    for fit in ("entropic", "hard"):
                        Pi = (gw_coupling(X, Y, lam=lams[0], iters=args.gw_iters) if fit == "entropic"
                              else hard_gw_coupling(X, Y, args.gw_iters))
                        d = gw_diagnostics(X, Y, Pi)
                        print(f"[gw control {name:<21} {fit:<8}] pool {len(ctl)}: cost {d['cost']:>10.1f}  "
                              f"identity {d['cost_identity']:>10.1f}  uniform {d['cost_uniform']:>10.1f}  "
                              f"true-pair mass {d['true_pair_mass']:.2f}x uniform  top1 {d['top1']:.3f}")
            record("raw cosine, same pool", z_held, z_unique[sub], fp_train[sub], ids_train[sub],
                   note="identity coupling baseline")

    if args.synthetic:
        # Rule-based spectra are the only synthetic source that is independent of the encoder being
        # aligned: a decoder trained on real spectra makes the "synthetic" spectrum a function of the
        # molecule embedding, so any alignment measured through it is the decoder, not chemistry.
        from sna.synth import insilico_table
        names = unique + list(held_ids)
        cache = Path("data/synthetic_spectra.npz")
        if cache.exists() and list(np.load(cache)["smiles"]) == names:
            z_syn = np.load(cache)["emb"].astype(np.float32)
        else:
            syn = insilico_table(names)
            t0 = time.time()
            z_syn = embed_spectra(syn, args.dreams)
            np.savez(cache, smiles=np.array(names), emb=z_syn)
            print(f"[synthetic] {len(names)} rule-based spectra, mean "
                  f"{np.diff(syn['offsets']).mean():.1f} peaks, embedded in {time.time() - t0:.0f}s")
        z_syn_train, z_syn_held = z_syn[: len(unique)], z_syn[len(unique):]
        print(f"CKA(rule-based synthetic space, frozen molecule space) = {cka(z_syn_train, m_train):.4f}")
        print(f"CKA(real spectrum space, rule-based synthetic space) = {cka(z_unique, z_syn_train):.4f}")
        if "gw" in methods:
            # Does the rule-based space couple to the molecule space where the real one does not?
            # Same pool, same seed, same temperature as the real arm, so the two costs are directly
            # comparable; the identity coupling is the true pairing in both.
            for seed in range(args.seeds):
                sub = np.random.default_rng(seed).choice(len(unique), args.gw_sub, replace=False)
                M = cosine_distance(m_train[sub]); M /= M.mean()
                for name, Q in (("rule-based", z_syn_train[sub]), ("real spectra", z_unique[sub])):
                    X = cosine_distance(Q); X /= X.mean()
                    Pi = gw_coupling(X, M, lam=lams[0], iters=args.gw_iters)
                    d = gw_diagnostics(X, M, Pi)
                    print(f"[gw {name:<12} vs molecules] seed {seed} pool {len(sub)}: "
                          f"cost {d['cost']:>10.1f}  identity {d['cost_identity']:>10.1f}  "
                          f"uniform {d['cost_uniform']:>10.1f}  true-pair mass {d['true_pair_mass']:.2f}x "
                          f"top1 {d['top1']:.3f}")
                    record(f"gw {name} -> molecules", transport(z_syn_held, z_syn_train[sub], Pi)
                           if name == "rule-based" else transport(z_held, z_unique[sub], Pi),
                           None, fp_train[sub], ids_train[sub], note=f"seed {seed}")
                A = cosine_distance(z_unique[sub]); A /= A.mean()
                B = cosine_distance(z_syn_train[sub]); B /= B.mean()
                Pi = gw_coupling(A, B, lam=lams[0], iters=args.gw_iters)
                d = gw_diagnostics(A, B, Pi)
                print(f"[gw real vs rule-based  ] seed {seed} pool {len(sub)}: "
                      f"cost {d['cost']:>10.1f}  identity {d['cost_identity']:>10.1f}  "
                      f"uniform {d['cost_uniform']:>10.1f}  true-pair mass {d['true_pair_mass']:.2f}x "
                      f"top1 {d['top1']:.3f}")
        record("synthetic cosine", z_syn_held, z_syn_train, fp_train, ids_train,
               note="rule-based spectra, no alignment")
        if "cca" in methods:
            W_syn, corr_syn = cca(z_syn_train.astype(np.float64), m_train.astype(np.float64), lam=1e-3)
            record("cca fit on synthetic", z_held @ W_syn, m_train_n, fp_train, ids_train, m_held,
                   note="learned on rule-based spectra, tested on real")
            record("cca real-fit, synthetic query", z_syn_held @ W, m_train_n, fp_train, ids_train,
                   note="real map applied to rule-based spectra")
        if "procrustes" in methods:
            mean_s, comp_s = pca(z_syn_train, m_train.shape[1])
            R_syn = procrustes((z_syn_train - mean_s) @ comp_s.T, m_train)
            record("procrustes fit on synthetic", ((z_held - mean_s) @ comp_s.T) @ R_syn, m_train_n,
                   fp_train, ids_train, m_held, note="learned on rule-based spectra, tested on real")
        if "infonce" in methods:
            for seed in range(args.seeds):
                spectrum, molecule = fit_contrastive(z_syn_train, m_train, out=args.dims, seed=seed)
                record("infonce fit on synthetic", spectrum(z_held), molecule(m_train), fp_train,
                       ids_train, molecule(m_held), f"seed {seed}")

    record("raw spectrum cosine", z_held, z_train_n, fp_train, ids_train, note="no alignment")
    record("molecule space ceiling", m_held, m_train_n, fp_train, ids_train,
           note="what a perfect map could retrieve")
    mass = np.bincount(mol_of_row, weights=msg["precursor_mz"][train_rows]) / np.bincount(mol_of_row)
    mz_held = msg["precursor_mz"][held_rows]
    # |mass difference| is not a dot product, so it is passed as a score matrix; there is no
    # embedding space here, hence no peer pool and no collapse report.
    record("precursor mass only", -np.abs(mass[None, :] - mz_held[:, None]), None,
           fp_train, ids_train, note="shortcut baseline")
    rng = np.random.default_rng(0)
    record("random", rng.random((len(z_held), 8)), rng.random((len(unique), 8)), fp_train, ids_train,
           rng.random((len(held_ids), 8)), "floor")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("method")[["top1", "top5", "top10", "top1_peer", "exact_peer", "spearman"]].mean().round(3).to_string())
    print(f"wrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
