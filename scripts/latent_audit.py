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

from sna.align import canonical, fingerprints, ridge, tanimoto
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


def readout_ceiling(pred_fp: np.ndarray, train_fp: np.ndarray, val_fp: np.ndarray, chunk: int = 2000) -> float:
    """Tanimoto between the true molecule and the molecule of the training spectrum whose *predicted*
    fingerprint is closest to the query's - what a lookup on the readout could reach."""
    index = train_fp / (np.linalg.norm(train_fp, axis=1, keepdims=True) + 1e-9)
    query = pred_fp / (np.linalg.norm(pred_fp, axis=1, keepdims=True) + 1e-9)
    sims = np.empty(len(query), dtype=np.float64)
    for i in range(0, len(query), chunk):
        best = (index @ query[i:i + chunk].T).argmax(0)
        sims[i:i + chunk] = tanimoto(train_fp[best], val_fp[i:i + chunk])
    return float(sims.mean())


def audit(name: str, z_train: np.ndarray, z_val: np.ndarray, train_fp: np.ndarray, val_fp: np.ndarray,
          train_counts: np.ndarray, val_counts: np.ndarray, lam: float) -> None:
    counts = z_val @ ridge(z_train, train_counts, lam)
    exact = float(np.all(np.round(counts) == val_counts, axis=1).mean())
    mae = float(np.abs(counts[:, -1] - val_counts[:, -1]).mean())
    bits = ((z_val @ ridge(z_train, train_fp.astype(np.float64), lam)) > 0.5).astype(np.uint8)
    print(f"{name:<18} formula exact {exact:.1%}  heavy atoms MAE {mae:.2f}  "
          f"readout fingerprint {tanimoto(bits, val_fp).mean():.3f}  "
          f"readout ceiling {readout_ceiling(bits, train_fp, val_fp):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default="data/MassSpecGym1.5.tsv")
    parser.add_argument("--spectra", default="data/msg_generated.parquet.embeddings.npz")
    parser.add_argument("--latents", default="data/jepa_latents.npz")
    parser.add_argument("--autoencoder", default="data/generative_embedding_peaks.pt")
    parser.add_argument("--lambda", dest="lam", type=float, default=1e3)
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

    print(f"{len(val_smiles)} held-out spectra, {len(train_smiles)} training spectra")
    for name, a, b in [("raw bins", peaks[tr], peaks[va]),
                       ("JEPA latent", lat["train"], lat["val"]),
                       ("autoencoder latent", ae[tr], ae[va]),
                       ("DreaMS embedding", emb["train"], emb["val"])]:
        audit(name, a.astype(np.float64), b.astype(np.float64), train_fp, val_fp,
              train_counts, val_counts, args.lam)


if __name__ == "__main__":
    main()
