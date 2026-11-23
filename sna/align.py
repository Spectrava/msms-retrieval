"""Linear alignment between a frozen spectrum embedding and a frozen molecule tower.

The molecule side is MoLFormer/ChemBERTa as they are: no training, no adapter. A ridge map
W from the spectrum embedding to the molecule embedding is fitted on the train fold, which is
the cheapest possible test of whether the two spaces can be lined up at all.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def fingerprints(smiles: list[str]) -> np.ndarray:
    """Morgan bits as a uint8 matrix; all-zero for a SMILES RDKit cannot parse."""
    out = np.zeros((len(smiles), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol:
            out[i] = np.array(MORGAN.GetFingerprint(mol))
    return out


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    union = (a | b).sum(1)
    return np.divide((a & b).sum(1), union, out=np.zeros(len(a), dtype=np.float64), where=union > 0)


def smiles_tanimoto(a: str, b: str) -> float:
    """Morgan Tanimoto between two SMILES strings; 0.0 if either fails to parse."""
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if not ma or not mb:
        return 0.0
    return DataStructs.TanimotoSimilarity(MORGAN.GetFingerprint(ma), MORGAN.GetFingerprint(mb))


def ridge(z: np.ndarray, m: np.ndarray, lam: float) -> np.ndarray:
    """W minimising ||W z - m||^2 + lam ||W||^2, solved through the z-space Gram matrix."""
    return np.linalg.solve(z.T @ z + lam * np.eye(z.shape[1], dtype=np.float64), z.T @ m).astype(np.float32)


def fit_mlp(z: np.ndarray, m: np.ndarray, hidden: int = 1024, epochs: int = 20,
            batch: int = 2048, seed: int = 0) -> callable:
    """A two-layer map, the cheapest non-linear test, trained on cosine to the target embedding."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(z.shape[1], hidden), torch.nn.LayerNorm(hidden),
                              torch.nn.GELU(), torch.nn.Linear(hidden, m.shape[1])).cuda()
    zt, mt = torch.tensor(z, dtype=torch.float32).cuda(), torch.tensor(m, dtype=torch.float32).cuda()
    target = torch.nn.functional.normalize(mt, dim=1)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(zt), device="cuda")
        loss = 0.0
        for i in range(0, len(zt), batch):
            r = order[i:i + batch]
            pred = torch.nn.functional.normalize(net(zt[r]), dim=1)
            obj = (1 - (pred * target[r]).sum(1)).mean()
            opt.zero_grad(); obj.backward(); opt.step()
            loss += obj.item()
        if epoch % 5 == 0:
            print(f"[mlp] epoch {epoch} cosine loss {loss / len(range(0, len(zt), batch)):.4f}", flush=True)
    net.eval()

    def predict(x):
        with torch.no_grad():
            return net(torch.tensor(x, dtype=torch.float32).cuda()).cpu().numpy()
    return predict


def nearest(query: np.ndarray, index: np.ndarray, fp: np.ndarray, truth: np.ndarray,
            chunk: int = 2000) -> tuple[float, np.ndarray]:
    """Mean Tanimoto between the nearest indexed molecule and the true molecule, per query."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    sims = np.empty(len(q), dtype=np.float64)
    for i in range(0, len(q), chunk):
        idx = (index @ q[i:i + chunk].T).argmax(0)
        sims[i:i + chunk] = tanimoto(fp[idx], truth[i:i + chunk])
    return float(sims.mean()), sims


def probe_decoder(emb: np.ndarray, smiles: list[str], held: np.ndarray, held_smiles: list[str],
                  epochs: int = 10, batch: int = 512, max_len: int = 120) -> tuple[float, float, float]:
    """Train the char-LSTM on a latent and report how well it reconstructs unseen molecules.

    Returns (parse rate, mean Tanimoto on unseen molecules, exact-match rate on the training
    molecules themselves) - the last one is the decoder's own ceiling, so a low unseen score
    says the latent is not generative rather than that the decoder is broken.
    """
    from sna.generate import SmilesGenerator
    vocab = sorted({c for s in smiles for c in s})
    index = {c: i for i, c in enumerate(vocab)}
    end = len(vocab)
    width = min(max(map(len, smiles)), max_len)
    chars = torch.tensor([[index[c] for c in s[:width]] + [end] * (width - len(s[:width])) for s in smiles],
                         device="cuda")
    gen = SmilesGenerator(emb.shape[1], vocab, hidden=256, layers=2).cuda()
    E = torch.tensor(emb, dtype=torch.float32, device="cuda")
    opt = torch.optim.Adam(gen.parameters(), 1e-3)
    for epoch in range(epochs):
        order = torch.randperm(len(chars), device="cuda")
        loss = 0.0
        for i in range(0, len(order), batch):
            r = order[i:i + batch]
            obj = gen.loss(E[r], chars[r].cuda())
            opt.zero_grad(); obj.backward(); opt.step()
            loss += obj.item() * len(r)
        print(f"[probe] epoch {epoch} next-character loss {loss / len(chars):.3f}", flush=True)
    gen.eval()
    out = [canonical(s) for s in gen.generate(torch.tensor(held, dtype=torch.float32, device="cuda"), max_len)]
    truth = [canonical(s) for s in held_smiles]
    parsed = [(a, b) for a, b in zip(out, truth) if a]
    own = gen.generate(E[:2000], max_len)
    exact = sum(canonical(g) == t for g, t in zip(own, smiles[:2000])) / 2000
    return len(parsed) / len(out), float(np.mean([smiles_tanimoto(a, b) for a, b in parsed])), exact
