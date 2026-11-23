"""Train a generative MS2 embedding and generate structures from unseen spectra.

The DreaMS embedding is discriminative - it is trained to tell spectra apart, so nothing can be read
back out of it. Here it is pushed through an encoder into a bottleneck latent that a transformer
decoder is trained, jointly, to turn into a SMILES, so the latent is generative by construction. The
comparison is the char-LSTM decoder conditioned directly on the raw 1024-d embedding, and the
nearest-neighbour retrieval ceiling: whatever the embedding can retrieve, the generator cannot beat.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.error")  # generated SMILES often fail to parse; the errors are noise

from sna.align import canonical, smiles_tanimoto
from sna.data import read_msg_tsv
from sna.generate import SmilesGenerator
from sna.genembed import GenerativeEmbedding, char_tensor, train_generative

PROTON, SODIUM = 1.00728, 22.98922  # [M+H]+ and [M+Na]+ over a neutral monoisotopic mass


def report(generated: list[str], truth: list[str]) -> tuple[float, float, float]:
    parsed = [(canonical(g), t) for g, t in zip(generated, truth) if canonical(g)]
    return (len(parsed) / len(generated),
            float(np.mean([smiles_tanimoto(g, t) for g, t in parsed])) if parsed else 0.0,
            float(np.mean([g == t for g, t in parsed])) if parsed else 0.0)


def generate(model, z: torch.Tensor, batch: int = 512, max_len: int = 160, temperature: float = 0.0) -> list[str]:
    return [s for i in range(0, len(z), batch)
            for s in model.generate(z[i:i + batch], max_len, temperature)]


def chunked_loss(model, z: torch.Tensor, chars: torch.Tensor, batch: int = 512) -> float:
    """Next-character loss, chunked: the attention matrix for a whole fold does not fit on the GPU."""
    return float(np.mean([model.loss(z[i:i + batch], chars[i:i + batch]).item()
                         for i in range(0, len(z), batch)]))


def mass_consistent(smiles: list[str], precursor: np.ndarray) -> list[bool]:
    neutral = [rdMolDescriptors.CalcExactMolWt(m) if m else 0.0 for m in map(Chem.MolFromSmiles, smiles)]
    return [abs(m + PROTON - p) < 0.05 or abs(m + SODIUM - p) < 0.05 for m, p in zip(neutral, precursor)]


def retrieval_ceiling(z_train: torch.Tensor, z_val: torch.Tensor, train_smiles: list[str],
                      val_smiles: list[str], sample: int = 2000) -> float:
    """Tanimoto of the true val molecule to the molecule of its nearest training spectrum."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(val_smiles), sample)
    sims = []
    for i in idx:
        best = int((z_train @ z_val[i]).argmax())
        sims.append(smiles_tanimoto(train_smiles[best], val_smiles[i]))
    return float(np.mean(sims))


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default="data/MassSpecGym1.5.tsv")
    parser.add_argument("--spectra", default="data/msg_generated.parquet.embeddings.npz")
    parser.add_argument("--export", default="data/7600_ms2_embeddings.parquet")
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--load", help="skip training and evaluate a saved checkpoint instead")
    parser.add_argument("--candidates", type=int, default=1,
                        help="sample this many SMILES per export spectrum (greedy first, then temperature 1.0)")
    parser.add_argument("--out", default="data/generative_embedding.pt")
    args = parser.parse_args()
    started = time.time()

    msg = read_msg_tsv(args.tsv)
    fold = np.array(msg["fold"])
    train_rows, val_rows = np.where(fold == "train")[0], np.where(fold == "val")[0]
    train_smiles = [canonical(msg["smiles"][i]) for i in train_rows]
    val_smiles = [canonical(msg["smiles"][i]) for i in val_rows]
    vocab = sorted({c for s in train_smiles for c in s if c})

    emb = np.load(args.spectra)
    z_train = torch.tensor(emb["train"], device="cuda")
    z_val = torch.tensor(emb["val"], device="cuda")

    if args.load:
        ckpt = torch.load(args.load, map_location="cuda")
        vocab = ckpt["vocab"]
        model = GenerativeEmbedding(z_train.shape[1], vocab, ckpt["latent"], ckpt["hidden"],
                                    ckpt["layers"]).cuda()
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"loaded {args.load} (latent {ckpt['latent']})")
    else:
        model = train_generative(z_train, train_smiles, vocab, args.latent, args.epochs, args.batch,
                                 args.hidden, args.layers)
        torch.save({"vocab": vocab, "dim_in": z_train.shape[1], "latent": args.latent,
                    "hidden": args.hidden, "layers": args.layers, "model": model.state_dict()}, args.out)
        print(f"wrote {args.out} in {time.time() - started:.0f}s")

    with torch.no_grad():
        val_chars = char_tensor(val_smiles, vocab, 160)
        print(f"held-out next-character loss       {chunked_loss(model, z_val, val_chars):.3f}")
        sample = np.random.default_rng(0).choice(len(train_smiles), 2000)
        parse, sim, exact = report(generate(model, z_train[sample]), [train_smiles[i] for i in sample])
        print(f"self-generation, training spectra  {parse:.1%} parse, {sim:.3f} Tanimoto, {exact:.1%} exact")
        parse, sim, exact = report(generate(model, z_val), val_smiles)
        print(f"generation, held-out spectra       {parse:.1%} parse, {sim:.3f} Tanimoto, {exact:.1%} exact")
        wrong = z_val[np.random.default_rng(1).permutation(len(z_val))]
        parse, sim, exact = report(generate(model, wrong), val_smiles)
        print(f"generation, wrong spectrum         {parse:.1%} parse, {sim:.3f} Tanimoto, {exact:.1%} exact")
        print(f"retrieval ceiling (nearest train spectrum) {retrieval_ceiling(z_train, z_val, train_smiles, val_smiles):.3f} Tanimoto")

        if not args.load:
            baseline = SmilesGenerator(z_train.shape[1], vocab).cuda()
            chars = char_tensor(train_smiles, vocab, 160)
            opt = torch.optim.Adam(baseline.parameters(), 1e-3)
            for epoch in range(args.epochs):
                order = torch.randperm(len(z_train), device="cuda")
                loss = 0.0
                for i in range(0, len(order), args.batch):
                    r = order[i:i + args.batch]
                    obj = baseline.loss(z_train[r], chars[r])
                    opt.zero_grad(); obj.backward(); opt.step()
                    loss += obj.item() * len(r)
                print(f"[baseline] epoch {epoch} next-character loss {loss / len(z_train):.3f}", flush=True)
            baseline.eval()
            parse, sim, exact = report(generate(baseline, z_val), val_smiles)
            print(f"baseline (raw embedding, no latent) {parse:.1%} parse, {sim:.3f} Tanimoto, {exact:.1%} exact")

        export = pd.read_parquet(args.export)
        z_export = torch.tensor(np.array(export["embedding"].tolist(), dtype=np.float32), device="cuda")
        precursor = np.array(export["precursor_mz"], dtype=np.float64)
        stem = args.out.rsplit(".", 1)[0]
        rows = []
        for k in range(args.candidates):
            gen = generate(model, z_export, temperature=0.0 if k == 0 else 1.0)
            rows.append(pd.DataFrame({"candidate_id": export["candidate_id"], "precursor_mz": export["precursor_mz"],
                                      "generated_smiles": gen,
                                      "mass_matches_precursor": mass_consistent(gen, precursor)}))
        table = pd.concat(rows)
        print(f"{args.export}: {int(table.generated_smiles.map(canonical).notna().sum())} of {len(table)} proposals parse, "
              f"{int(table.groupby('candidate_id').mass_matches_precursor.any().sum())} of {len(export)} spectra have "
              f"at least one proposal whose mass fits the precursor")
        table.to_parquet(f"{stem}_generated.parquet")
        pd.DataFrame({"candidate_id": export["candidate_id"],
                      "generative_latent": [list(l) for l in model.latent(z_export).cpu().numpy()]}).to_parquet(
            f"{stem}_latents.parquet")


if __name__ == "__main__":
    main()
