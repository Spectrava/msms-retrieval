"""How much chemistry each latent carries, measured without a decoder.

Generation is capped by what a readout can recover from the latent. A ridge map from the latent to
the element counts and the Morgan fingerprint is the cheapest way to find that cap: if a linear
readout cannot recover the formula, no decoder can write the structure. The readout ceiling is
directly comparable to the generation numbers because it is the same score - Tanimoto to the true
molecule - reached by the best available lookup rather than by generation.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch import nn
from torch.nn import functional as F

from sna.align import canonical, fingerprints, readout_ceiling, ridge, tanimoto
from sna.data import read_msg_tsv
from sna.genembed import GenerativeEmbedding, peak_matrix

ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


def composition(smiles: list[str]) -> np.ndarray:
    """Element counts (implicit hydrogens included) plus the heavy-atom count per SMILES."""
    counts = np.zeros((len(smiles), len(ELEMENTS) + 1), dtype=np.float64)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        c = Counter(a.GetSymbol() for a in mol.GetAtoms())
        c["H"] += sum(a.GetNumImplicitHs() for a in mol.GetAtoms())
        for j, e in enumerate(ELEMENTS):
            counts[i, j] = c.get(e, 0)
        counts[i, -1] = mol.GetNumAtoms()
    return counts


def audit(name: str, z_train: np.ndarray, z_val: np.ndarray, train_fp: np.ndarray, val_fp: np.ndarray,
          train_counts: np.ndarray, val_counts: np.ndarray, lam: float) -> None:
    counts = z_val @ ridge(z_train, train_counts, lam)
    exact = float(np.all(np.round(counts) == val_counts, axis=1).mean())
    mae = float(np.abs(counts[:, -1] - val_counts[:, -1]).mean())
    bits = ((z_val @ ridge(z_train, train_fp.astype(np.float64), lam)) > 0.5).astype(np.uint8)
    print(f"{name:<18} formula exact {exact:.1%}  heavy atoms MAE {mae:.2f}  "
          f"readout fingerprint {tanimoto(bits, val_fp).mean():.3f}  "
          f"readout ceiling {readout_ceiling(bits, train_fp, val_fp):.3f}")


def mlp_readout(name: str, z_train: np.ndarray, z_val: np.ndarray, train_fp: np.ndarray, val_fp: np.ndarray,
                train_counts: np.ndarray, val_counts: np.ndarray, hidden: int = 512,
                epochs: int = 20) -> None:
    """Same readout with one hidden layer, to check whether the linear probe is the limit or the
    latent is. If a nonlinear readout cannot do better either, the chemistry is not in there."""
    zt = torch.tensor(z_train, device="cuda")
    zv = torch.tensor(z_val, device="cuda")
    bits = torch.tensor(train_fp, device="cuda")
    counts = torch.tensor(train_counts, device="cuda")
    model = nn.Sequential(nn.Linear(z_train.shape[1], hidden), nn.ReLU(),
                          nn.Linear(hidden, train_fp.shape[1] + train_counts.shape[1])).cuda()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(zt), device="cuda")
        loss = 0.0
        for i in range(0, len(order), 512):
            r = order[i:i + 512]
            out = model(zt[r])
            obj = F.binary_cross_entropy_with_logits(out[:, :train_fp.shape[1]], bits[r]) \
                + F.mse_loss(out[:, train_fp.shape[1]:], counts[r])
            opt.zero_grad(); obj.backward(); opt.step()
            loss += obj.item() * len(r)
        print(f"  [mlp {name}] epoch {epoch} loss {loss / len(zt):.4f}", flush=True)
    with torch.no_grad():
        out = model(zv).cpu().numpy()
    pred_bits = (out[:, :train_fp.shape[1]] > 0.5).astype(np.uint8)
    pred_counts = np.round(out[:, train_fp.shape[1]:])
    print(f"{name:<18} mlp formula exact {np.all(pred_counts == val_counts, axis=1).mean():.1%}  "
          f"heavy atoms MAE {np.abs(pred_counts[:, -1] - val_counts[:, -1]).mean():.2f}  "
          f"readout fingerprint {tanimoto(pred_bits, val_fp).mean():.3f}  "
          f"readout ceiling {readout_ceiling(pred_bits, train_fp, val_fp):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default="data/MassSpecGym1.5.tsv")
    parser.add_argument("--spectra", default="data/msg_generated.parquet.embeddings.npz")
    parser.add_argument("--latents", default="data/jepa_latents.npz")
    parser.add_argument("--autoencoder", default="data/generative_embedding_peaks.pt")
    parser.add_argument("--lambda", dest="lam", type=float, default=1e3)
    parser.add_argument("--mlp", action="store_true", help="also fit a one-hidden-layer readout")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")
    msg = read_msg_tsv(args.tsv)
    fold = np.array(msg["fold"])
    tr, va = np.where(fold == "train")[0], np.where(fold == "val")[0]
    train_smiles = [canonical(msg["smiles"][i]) for i in tr]
    val_smiles = [canonical(msg["smiles"][i]) for i in va]
    train_counts, val_counts = composition(train_smiles), composition(val_smiles)
    train_fp, val_fp = fingerprints(train_smiles), fingerprints(val_smiles)

    peaks = peak_matrix(msg["mz"], msg["intensity"], msg["offsets"])
    ck = torch.load(args.autoencoder)
    auto = GenerativeEmbedding(ck["dim_in"], ck["vocab"], ck["latent"], ck["hidden"], ck["layers"]).cuda()
    auto.load_state_dict(ck["model"])
    with torch.no_grad():
        ae = auto.latent(torch.tensor(peaks, device="cuda")).cpu().numpy()
    lat, emb = np.load(args.latents), np.load(args.spectra)

    substrates = [("raw bins", peaks[tr], peaks[va]),
                  ("JEPA latent", lat["train"], lat["val"]),
                  ("autoencoder latent", ae[tr], ae[va]),
                  ("DreaMS embedding", emb["train"], emb["val"])]
    print(f"{len(val_smiles)} held-out spectra, {len(train_smiles)} training spectra")
    for name, a, b in substrates:
        audit(name, a.astype(np.float64), b.astype(np.float64), train_fp, val_fp,
              train_counts, val_counts, args.lam)
    if args.mlp:
        for name, a, b in substrates:
            mlp_readout(name, a.astype(np.float32), b.astype(np.float32), train_fp, val_fp,
                        train_counts, val_counts)


if __name__ == "__main__":
    main()
