"""Train a JEPA encoder on MS2, then test whether its latent is a better substrate for generation.

Masked latent prediction: the encoder sees a spectrum with contiguous m/z windows removed and must
predict, in latent space, the latent of the whole spectrum as scored by an EMA teacher. Nothing is
reconstructed and no molecule enters the loss - that is the JEPA part. The hypothesis is that an
encoder which has learned to fill in the peaks it was not shown carries more of the spectrum than a
discriminative one, and therefore feeds a structure decoder better.

The frozen latent is then handed to the same SMILES decoder used for the DreaMS and raw-bin
baselines, so the comparison is the substrate, not the decoder.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F

from sna.align import canonical, fingerprints, nearest_neighbour_tanimoto
from sna.data import read_msg_tsv
from sna.genembed import generate, peak_matrix, report, train_latent_decoder
from sna.jepa import JEPA, linear_probe, mask_windows


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-9)).mean())


def train_jepa(peaks: torch.Tensor, epochs: int, batch: int, hidden: int, dim: int,
               windows: int, fraction: float) -> JEPA:
    model = JEPA(peaks.shape[1], hidden, dim).cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(peaks), device=peaks.device)
        loss = 0.0
        for i in range(0, len(order), batch):
            r = order[i:i + batch]
            obj = model.loss(peaks[r], mask_windows(peaks[r], windows, fraction))
            opt.zero_grad(); obj.backward(); opt.step()
            model.update_teacher()
            loss += obj.item() * len(r)
        print(f"[jepa] epoch {epoch} masked-latent error {loss / len(peaks):.4f}", flush=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default="data/MassSpecGym1.5.tsv")
    parser.add_argument("--spectra", default="data/msg_generated.parquet.embeddings.npz")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--latent", type=int, default=256)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--decoder-epochs", type=int, default=10)
    parser.add_argument("--decoder-hidden", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--substrates", default="all",
                       help="comma separated: 'JEPA latent', 'raw bins', 'DreaMS embedding'")
    parser.add_argument("--out", default="data/jepa_encoder.pt")
    parser.add_argument("--decoder-out", default="data/jepa_decoder.pt")
    parser.add_argument("--latents", default="data/jepa_latents.npz")
    args = parser.parse_args()
    started = time.time()

    msg = read_msg_tsv(args.tsv)
    fold = np.array(msg["fold"])
    train_rows, val_rows = np.where(fold == "train")[0], np.where(fold == "val")[0]
    train_smiles = [canonical(msg["smiles"][i]) for i in train_rows]
    val_smiles = [canonical(msg["smiles"][i]) for i in val_rows]
    vocab = sorted({c for s in train_smiles for c in s if c})
    train_fp, val_fp = fingerprints(train_smiles), fingerprints(val_smiles)

    substrates = ["JEPA latent", "raw bins", "DreaMS embedding"] if args.substrates == "all" \
        else [s.strip() for s in args.substrates.split(",")]

    # the peak matrix is 776 MB on the GPU; only needed if a substrate is the bins themselves
    if "raw bins" in substrates or "JEPA latent" in substrates:
        peaks = torch.tensor(peak_matrix(msg["mz"], msg["intensity"], msg["offsets"]), device="cuda")
        z_train, z_val = peaks[train_rows], peaks[val_rows]
    else:
        z_train, z_val = None, None

    if "JEPA latent" in substrates:
        model = train_jepa(z_train, args.epochs, args.batch, args.hidden, args.latent, args.windows, args.fraction)
        torch.save({"latent": args.latent, "hidden": args.hidden, "model": model.state_dict()}, args.out)
        with torch.no_grad():
            lat_train, lat_val = model.latent(z_train), model.latent(z_val)
            np.savez(args.latents, train=lat_train.cpu().numpy(), val=lat_val.cpu().numpy())
            pred = F.normalize(model.predictor(model.encoder(mask_windows(z_val, args.windows, args.fraction))), dim=-1)
            target = F.normalize(model.teacher(z_val), dim=-1)
            print(f"held-out masked-latent error {float((1 - (pred * target).sum(1)).mean()):.4f}")
            print(f"latent predicts the spectrum (linear probe) cosine {cosine(linear_probe(lat_val, z_val), z_val):.3f}")
    else:
        lat = np.load(args.latents)
        lat_train, lat_val = torch.tensor(lat["train"], device="cuda"), torch.tensor(lat["val"], device="cuda")

    emb = np.load(args.spectra)
    dream = torch.tensor(emb["train"], device="cuda"), torch.tensor(emb["val"], device="cuda")

    for name, zt, zv in [("JEPA latent", lat_train, lat_val), ("raw bins", z_train, z_val),
                         ("DreaMS embedding", dream[0], dream[1])]:
        if name not in substrates:
            continue
        print(f"{name}: nearest-neighbour ceiling "
              f"{nearest_neighbour_tanimoto(zt, zv, train_fp, val_fp):.3f}")
        decoder = train_latent_decoder(zt, train_smiles, vocab, hidden=args.decoder_hidden,
                                       layers=args.decoder_layers, epochs=args.decoder_epochs)
        if name == "JEPA latent":
            torch.save({"latent": args.latent, "memory": decoder.memory, "vocab": vocab,
                        "decoder": decoder.state_dict()}, args.decoder_out)
        parse, sim, exact = report(generate(decoder, zv), val_smiles)
        print(f"generation from the {name}  {parse:.1%} parse, {sim:.3f} Tanimoto, {exact:.1%} exact")
        wrong = zv[np.random.default_rng(1).permutation(len(zv))][:2000]
        parse, sim, exact = report(generate(decoder, wrong), val_smiles[:2000])
        print(f"  same decoder, shuffled {name}  {parse:.1%} parse, {sim:.3f} Tanimoto, {exact:.1%} exact")
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
