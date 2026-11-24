"""Train the peak-sequence encoder and hand its latent to the same SMILES decoder.

The histogram substrates (raw bins, JEPA latent, DreaMS) all sit at or below their retrieval
ceiling, and a nonlinear readout of them buys nothing. This tests the other hypothesis: the
chemistry is lost by the 1000-bin histogram, so encode the peak list as a set of m/z tokens and
see whether a decoder can write structures from that.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from rdkit import RDLogger

from sna.align import canonical, fingerprints, nearest_neighbour_tanimoto
from sna.data import read_msg_tsv
from sna.genembed import generate, report, train_latent_decoder
from sna.peakseq import PeakEncoder, peak_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default="data/MassSpecGym1.5.tsv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--max-peaks", type=int, default=128)
    parser.add_argument("--drop", type=float, default=0.25)
    parser.add_argument("--decoder-epochs", type=int, default=10)
    parser.add_argument("--out", default="data/peakseq_encoder.pt")
    args = parser.parse_args()
    started = time.time()

    RDLogger.DisableLog("rdApp.error")
    msg = read_msg_tsv(args.tsv)
    fold = np.array(msg["fold"])
    train_rows, val_rows = np.where(fold == "train")[0], np.where(fold == "val")[0]
    train_smiles = [canonical(msg["smiles"][i]) for i in train_rows]
    val_smiles = [canonical(msg["smiles"][i]) for i in val_rows]
    vocab = sorted({c for s in train_smiles for c in s if c})
    train_fp, val_fp = fingerprints(train_smiles), fingerprints(val_smiles)

    tokens = torch.tensor(peak_tokens(msg["mz"], msg["intensity"], msg["offsets"],
                                     max_peaks=args.max_peaks), device="cuda")
    model = PeakEncoder(hidden=args.hidden, layers=args.layers, dim=args.latent,
                        max_peaks=args.max_peaks).cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(args.epochs):
        order = torch.randperm(len(tokens), device="cuda")
        loss = 0.0
        for i in range(0, len(order), args.batch):
            r = order[i:i + args.batch]
            obj = model.jepa_loss(tokens[r], args.drop)
            opt.zero_grad(); obj.backward(); opt.step()
            model.update_teacher()
            loss += obj.item() * len(r)
        print(f"[peakseq] epoch {epoch} masked-latent error {loss / len(tokens):.4f}", flush=True)
    torch.save({"latent": args.latent, "hidden": args.hidden, "layers": args.layers,
                "max_peaks": args.max_peaks, "model": model.state_dict()}, args.out)

    with torch.no_grad():
        lat_train, lat_val = model.latent(tokens[train_rows]), model.latent(tokens[val_rows])
    print(f"nearest-neighbour ceiling, peak-sequence latent: "
          f"{nearest_neighbour_tanimoto(lat_train, lat_val, train_fp, val_fp):.3f}")

    decoder = train_latent_decoder(lat_train, train_smiles, vocab, epochs=args.decoder_epochs)
    parse, sim, exact = report(generate(decoder, lat_val), val_smiles)
    print(f"generation from the peak-sequence latent  {parse:.1%} parse, {sim:.3f} Tanimoto, "
          f"{exact:.1%} exact")
    wrong = lat_val[np.random.default_rng(1).permutation(len(lat_val))][:2000]
    parse, sim, exact = report(generate(decoder, wrong), val_smiles[:2000])
    print(f"  same decoder, shuffled peak-sequence latent  {parse:.1%} parse, {sim:.3f} Tanimoto")
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
