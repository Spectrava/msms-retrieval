"""Joint molecular-MS/MS alignment benchmark.

Frozen DreaMS spectrum embeddings against a frozen molecule encoder (Mol-JEPA, CC BY-NC, or
MoLFormer XL as the commercially permissive fallback), aligned with the four baselines from
the brief - linear CKA, regularised CCA, orthogonal Procrustes, symmetric InfoNCE heads - plus
the geometry-preserving projector (pairing + cosine-Gram + SIGReg).

Everything is fitted on the train fold; the reported fold is the held-out one. Retrieval is
scored against the *train* molecules, so the ceiling is the fingerprint similarity of the
nearest train molecule to the true one, and the shortcut baselines (precursor mass, random
pair) say how much of any gain is real.

    python scripts/align_joint.py --encoder moljepa --dims 256 --seeds 3
    python scripts/align_joint.py --encoder molformer --methods cka,cca,procrustes
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
from sna.align import canonical, fit_contrastive, fingerprints
from sna.data import read_msg_tsv
from sna.joint import (cca, cka, fit_geometry, mean_by_group, pair_correlation, pca, procrustes,
                       retrieval, unit)
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
    parser.add_argument("--methods", default="cka,cca,procrustes,infonce,geometry")
    parser.add_argument("--out", default="results/metrics.csv")
    args = parser.parse_args()
    methods = args.methods.split(",")
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
        from sna.encoders.dreams import load_dreams
        from sna.spectra import SpectraTable
        encoder = load_dreams("checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt").cuda().eval()
        table = SpectraTable(msg, device="cuda")
        z_held = np.concatenate([
            encoder(table.batch(torch.arange(i, min(i + 512, len(held_rows)), device="cuda"), encoder.max_peaks))[0]
            .float().cpu().numpy()
            for i in range(0, len(held_rows), 512)
        ])

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

    rows = []
    def record(method, mapped, index, note=""):
        mapped = unit(np.asarray(mapped, dtype=np.float32))
        index = unit(np.asarray(index, dtype=np.float32))
        score = retrieval(mapped, index, fp_train, fp_held, np.array(unique), held_ids)
        std, erank = collapse_report(mapped)
        row = dict(encoder=args.encoder, fold=args.fold, method=method, dim=mapped.shape[1],
                   top1=score["top1"], top5=score["top5"], top10=score["top10"], mrr=score["mrr"],
                   spearman=pair_correlation(mapped, index, fp_train, fp_held),
                   std=std, effective_rank=erank, note=note)
        rows.append(row)
        print(f"{method:<22} top1 {row['top1']:.3f}  top5 {row['top5']:.3f}  top10 {row['top10']:.3f}  "
              f"MRR {row['mrr']:.3f}  rho {row['spearman']:.3f}  std {std:.3f}  erank {erank:.1f}")

    print(f"CKA(frozen spectrum space, frozen molecule space) = {cka(z_unique, m_train):.4f}")

    if "cca" in methods:
        W, corr = cca(z_unique.astype(np.float64), m_train.astype(np.float64), lam=1e-3)
        print(f"CCA: top canonical correlations {corr[:5].round(3)}")
        record("cca", z_held @ W, m_train_n)
    if "procrustes" in methods:
        mean, comp = pca(z_unique, m_train.shape[1])
        R = procrustes((z_unique - mean) @ comp, m_train)
        record("procrustes", ((z_held - mean) @ comp) @ R, m_train_n)
    if "infonce" in methods:
        for seed in range(args.seeds):
            spectrum, molecule = fit_contrastive(z_unique, m_train, out=args.dims, seed=seed)
            record("infonce", spectrum(z_held), molecule(m_train), f"seed {seed}")
    if "geometry" in methods:
        for alpha, beta in ((0.0, 0.0), (args.alpha, 0.0), (0.0, args.beta), (args.alpha, args.beta)):
            for seed in range(args.seeds):
                spectrum, molecule = fit_geometry(z_unique, m_train, out=args.dims, alpha=alpha, beta=beta,
                                                 epochs=args.epochs, seed=seed)
                record(f"geometry a{alpha} b{beta}", spectrum(z_held), molecule(m_train), f"seed {seed}")

    record("raw spectrum cosine", z_held, z_train_n, "no alignment")
    record("molecule space ceiling", m_held, m_train, "what a perfect map could retrieve")
    mass = np.bincount(mol_of_row, weights=msg["precursor_mz"][train_rows]) / np.bincount(mol_of_row)
    record("precursor mass only", -np.abs(mass[None, :] - msg["precursor_mz"][held_rows][:, None]),
           np.ones((len(unique), 1)), "shortcut baseline")
    record("random", np.random.default_rng(0).random((len(z_held), 8)),
           np.random.default_rng(1).random((len(unique), 8)), "floor")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("method")[["top1", "top5", "top10", "mrr", "spearman"]].mean().round(3).to_string())
    print(f"wrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
