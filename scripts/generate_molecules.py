"""Train a SMILES decoder on MassSpecGym train spectra and generate molecules from embeddings.

    python scripts/generate_molecules.py --tsv data/MassSpecGym1.5.tsv \
        --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt \
        --out data/msg_generated.parquet --spectra-embeddings data/7600_ms2_embeddings.parquet

The spectrum embeddings are the frozen DreaMS vectors; only the decoder trains (both the embeddings
and the decoder weights are cached next to --out, so reruns are quick). Reports, on the official val
fold - whose molecules are disjoint from the train fold, so exact recovery is impossible by
construction: RDKit validity, mean Tanimoto to the true molecule, and the nearest-neighbour retrieval
baseline on the same embeddings (the honest floor for any decoder), plus two controls: reproduction
of the decoder's own training molecules, and generation from the embedding of a different spectrum.
Then generates a SMILES for every MS2 spectrum of the Spectrava export whose embeddings are given and
checks whether its mass is consistent with the observed precursor.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")  # the decoder emits many invalid SMILES; per-parse errors are noise

from sna.encoders.dreams import load_dreams
from sna.generate import SmilesGenerator
from sna.retrieval import unit
from sna.spectra import SpectraTable, read_msg_tsv


def embed(table: SpectraTable, encoder, batch: int) -> np.ndarray:
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(table), batch):
            idx = torch.arange(start, min(start + batch, len(table)), device=table.mask.device)
            vectors.append(encoder(table.batch(idx, encoder.max_peaks))[0].float().cpu().numpy())
    return np.concatenate(vectors)


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def neutral_mass(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcExactMolWt(mol) if mol else 0.0


def constrained(model, z, target, tol, beams, max_len, chunk=512):
    """Beam-search in chunks so the n*beams LSTM states stay inside memory."""
    out = []
    for start in range(0, len(z), chunk):
        out += model.generate_mass_constrained(z[start:start + chunk], target[start:start + chunk],
                                              tol=tol, beams=beams, max_len=max_len)
    return out


MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def tanimoto(a: str, b: str) -> float:
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if not ma or not mb:
        return 0.0
    return DataStructs.TanimotoSimilarity(MORGAN.GetFingerprint(ma), MORGAN.GetFingerprint(mb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--ckpt", default="checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt")
    ap.add_argument("--out", default="data/msg_generated.parquet")
    ap.add_argument("--spectra-embeddings", help="MS2 embeddings Parquet to generate molecules for")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=120)
    ap.add_argument("--mass-constrained", action="store_true",
                    help="beam-search the decoder under the MS1 precursor mass instead of decoding greedily")
    ap.add_argument("--beams", type=int, default=4, help="beam width for --mass-constrained")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    msg = read_msg_tsv(args.tsv)
    fold = msg["fold"]
    train, val = np.where(fold == "train")[0], np.where(fold == "val")[0]
    smiles = msg["smiles"]
    vocab = sorted({c for s in smiles[train] for c in s})
    index = {c: i for i, c in enumerate(vocab)}

    encoder = load_dreams(args.ckpt).to(args.device).eval()
    t0 = time.time()
    cache = f"{args.out}.embeddings.npz"
    if os.path.exists(cache):
        z_train, z_val = np.load(cache)["train"], np.load(cache)["val"]
        print(f"reusing cached embeddings for {len(train)} train + {len(val)} val spectra")
    else:
        z_train = embed(SpectraTable(msg, rows=train, device=args.device), encoder, args.batch)
        z_val = embed(SpectraTable(msg, rows=val, device=args.device), encoder, args.batch)
        np.savez(cache, train=z_train, val=z_val)
        print(f"embedded {len(train)} train + {len(val)} val spectra in {time.time() - t0:.0f}s")

    model = SmilesGenerator(z_train.shape[1], vocab, hidden=256, layers=2).to(args.device)
    weights = f"{args.out}.generator.pt"
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # Teacher forcing needs a fixed width: pad the short SMILES with END.
    width = min(max(map(len, smiles[train])), args.max_len)
    end = len(vocab)
    chars_all = torch.tensor([[index[c] for c in s[:width]] + [end] * (width - len(s[:width]))
                              for s in smiles[train]])
    t0 = time.time()
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=args.device))
        print(f"reusing the trained decoder from {weights}")
    else:
        for epoch in range(args.epochs):
            order = torch.randperm(len(train))
            total = 0.0
            for start in range(0, len(train), args.batch):
                rows = order[start:start + args.batch]
                loss = model.loss(torch.tensor(z_train[rows], device=args.device),
                                  chars_all[rows].to(args.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(rows)
            print(f"epoch {epoch:2d}  next-character loss {total / len(train):.3f}")
        torch.save(model.state_dict(), weights)

    model.eval()
    generated = model.generate(torch.tensor(z_val, device=args.device), args.max_len)
    valid = [canonical(s) for s in generated]
    truth = [canonical(s) for s in smiles[val]]
    nn = unit(z_val) @ unit(z_train).T
    baseline = smiles[train][nn.argmax(1)]
    baseline_can = [canonical(s) for s in baseline]
    n = len(val)
    gen_sim = [tanimoto(v, t) for v, t in zip(valid, truth) if v]
    nn_sim = [tanimoto(b, t) for b, t in zip(baseline_can, truth)]
    print(f"val fold ({n} spectra): generated SMILES parse {sum(v is not None for v in valid) / n:.1%}, "
          f"exact match {sum(v is not None and v == t for v, t in zip(valid, truth)) / n:.2%}, "
          f"mean Tanimoto to the true molecule {np.mean(gen_sim):.3f}")
    print(f"nearest-neighbour baseline: exact match {sum(b == t for b, t in zip(baseline_can, truth)) / n:.2%}, "
          f"mean Tanimoto {np.mean(nn_sim):.3f}")

    # Controls: does the decoder memorise training molecules, and does the embedding steer it at all?
    # A rolled embedding is the same decoder fed the wrong spectrum.
    sample = np.random.default_rng(0).choice(len(train), 2000)
    own = model.generate(torch.tensor(z_train[sample], device=args.device), args.max_len)
    print(f"control - training spectra: reproduces its own SMILES "
          f"{sum(canonical(g) == canonical(smiles[train][i]) for g, i in zip(own, sample)) / len(sample):.1%}")
    shifted = [canonical(s) for s in model.generate(torch.tensor(np.roll(z_val, 1), device=args.device), args.max_len)]
    shift_sim = [tanimoto(v, t) for v, t in zip(shifted, truth) if v]
    print(f"control - embedding of a different spectrum: mean Tanimoto {np.mean(shift_sim):.3f}")

    constrained_rows = {}
    if args.mass_constrained:
        # The MS1 precursor pins the neutral mass exactly, so on the val fold the target is the true
        # molecule's mass: that is the ceiling of what a mass constraint can buy.
        targets = [neutral_mass(t) for t in truth]
        res = constrained(model, torch.tensor(z_val, device=args.device), targets, 0.05,
                         args.beams, args.max_len)
        valid_c = [canonical(s) for s, _, _ in res]
        sim_c = [tanimoto(v, t) for v, t in zip(valid_c, truth) if v]
        print(f"mass-constrained (beams {args.beams}): parse {sum(v is not None for v in valid_c) / n:.1%}, "
              f"mass match {sum(m for _, m, _ in res) / n:.1%}, "
              f"exact match {sum(v is not None and v == t for v, t in zip(valid_c, truth)) / n:.2%}, "
              f"mean Tanimoto {np.mean(sim_c):.3f}")
        constrained_rows = {"generated_mass_constrained": [s for s, _, _ in res],
                            "mass_constrained_valid": [v is not None for v in valid_c],
                            "mass_constrained_matched": [m for _, m, _ in res]}

    out = pd.DataFrame({"identifier": msg["identifier"][val], "true_smiles": smiles[val],
                        "generated": generated, "generated_canonical": valid,
                        "valid": [v is not None for v in valid],
                        "exact": [v is not None and v == t for v, t in zip(valid, truth)],
                        "tanimoto": gen_sim + [0.0] * (n - len(gen_sim)),
                        "nn_baseline_smiles": baseline, "nn_baseline_tanimoto": nn_sim,
                        **constrained_rows})
    out.to_parquet(args.out)
    prov = {"encoder": "DreaMS (frozen)", "decoder": "character LSTM seeded with the embedding",
            "vocab": len(vocab), "maxLen": args.max_len, "epochs": args.epochs,
            "trainSpectra": len(train), "valSpectra": len(val), "device": args.device,
            "seconds": round(time.time() - t0, 1),
            "note": "Generated molecules are hypotheses from a discriminative embedding; the "
                    "nearest-neighbour baseline is the honest floor for any decoder."}
    with open(f"{args.out}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    if args.spectra_embeddings:
        emb = pd.read_parquet(args.spectra_embeddings)
        z = torch.tensor(np.stack(emb["embedding"]), dtype=torch.float32, device=args.device)
        gen = model.generate(z, args.max_len)
        proposals = pd.DataFrame({"candidate_id": emb["candidate_id"], "spectrum_index": emb["spectrum_index"],
                                  "precursor_mz": emb["precursor_mz"], "generated_smiles": gen,
                                  "valid": [canonical(s) is not None for s in gen]})
        mols = [Chem.MolFromSmiles(s) for s in gen]
        masses = [rdMolDescriptors.CalcExactMolWt(m) if m else 0.0 for m in mols]
        # Consistent with the observed precursor only if the neutral mass plus a proton or sodium
        # matches it; the adduct is unknown, so either is accepted.
        proposals["mass_matches_precursor"] = [abs(m + 1.00728 - p) < 0.05 or abs(m + 22.98922 - p) < 0.05
                                               for m, p in zip(masses, emb["precursor_mz"])]
        proposals.to_parquet("data/7600_generated_molecules.parquet")
        print(f"generated {len(gen)} molecule proposals for the 7600 spectra "
            f"({sum(proposals.valid):.0f} parseable, {sum(proposals.mass_matches_precursor):.0f} whose mass "
            f"matches the observed precursor) -> data/7600_generated_molecules.parquet")

        if args.mass_constrained:
            # The adduct is not recorded, so both plausible ones are tried and the matching one kept.
            prec = np.asarray(emb["precursor_mz"], dtype=float)
            best = {}
            for adduct, delta in (("[M+H]+", 1.00728), ("[M+Na]+", 22.98922)):
                for i, (smi, matched, exact) in enumerate(
                        constrained(model, z, prec - delta, 0.05, args.beams, args.max_len)):
                    if matched or i not in best:
                        best[i] = (smi, matched, exact, adduct)
            proposals["generated_mass_constrained"] = [best[i][0] for i in range(len(gen))]
            proposals["adduct_assumed"] = [best[i][3] for i in range(len(gen))]
            proposals["mass_matches_precursor_constrained"] = [best[i][1] for i in range(len(gen))]
            proposals.to_parquet("data/7600_generated_molecules.parquet")
            parseable = sum(canonical(s) is not None for s in proposals.generated_mass_constrained)
            print(f"mass-constrained: {parseable} parseable, "
                f"{sum(proposals.mass_matches_precursor_constrained)} match the observed precursor "
                f"(greedy: {sum(proposals.mass_matches_precursor)})")


if __name__ == "__main__":
    main()
