"""Consensus reranking: use the pool's own structure as a second opinion.

Cosine ranks each candidate independently. But the top-K hits of a query are not independent -
if they all agree on a structural class, a pool molecule at the centre of that cluster is a
better answer than one that merely scores high on its own. Score every candidate by cosine plus
how similar it is, in fingerprint space, to the set of molecules the query already liked.

    python scripts/consensus_probe.py
"""
from __future__ import annotations

import numpy as np

from sna.align import canonical, fingerprints, tanimoto_matrix
from sna.data import read_msg_tsv
from sna.joint import unit

msg = read_msg_tsv("data/MassSpecGym1.5.tsv")
fold = np.array(msg["fold"])
smiles = [canonical(s) for s in msg["smiles"]]
ok = np.array([bool(s) for s in smiles])
train_rows = np.where((fold == "train") & ok)[0]
held_rows = np.where((fold == "val") & ok)[0]

z = np.load("data/msg_generated.parquet.embeddings.npz")
z_train = z["train"][np.where(ok[train_rows])[0]]
z_held = z["val"][np.where(ok[held_rows])[0]]

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

X = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(X, mol_of_row, z_train)
X = unit(X / np.bincount(mol_of_row)[:, None])

S = np.empty((len(held_rows), len(unique)), dtype=np.float32)
for i in range(0, len(held_rows), 2000):
    S[i:i + 2000] = unit(z_held[i:i + 2000]) @ X.T

T = tanimoto_matrix(fp_train, fp_train).astype(np.float32)
print(f"pool Tanimoto matrix {T.shape}, mean {T.mean():.3f}")

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, score):
    sel = np.argsort(-score, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<44} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

report("cosine (baseline)", S)

for K in (20, 100, 500):
    top = np.argsort(-S, axis=1)[:, :K]
    cons = np.empty((len(held_rows), K), dtype=np.float32)
    for i in range(0, len(held_rows), 500):
        idx = top[i:i + 500]
        cons[i:i + 500] = T[idx[:, :, None], idx[:, None, :]].mean(2)
    for lam in (0.1, 0.3, 1.0):
        new = S.copy()
        np.add.at(new, (np.arange(len(held_rows))[:, None], top), lam * cons)
        report(f"cosine + {lam:g} * consensus over top-{K}", new)
