"""Optimise the metric we actually score on.

Retrieval is scored by Tanimoto on 2048-bit Morgan fingerprints, but every scorer here ranks by
cosine in a 1024-d frozen embedding - a different quantity. Fit a model that predicts the query's
fingerprint from its embedding, then rank by Tanimoto between the predicted fingerprint and the
real fingerprints of the pool. The pool fingerprints are known exactly, so only the query side is
predicted.

Two models, both cheap: ridge regression to a soft fingerprint, and the same thresholded to a
real bit vector. Compare against raw cosine.

    python scripts/fingerprint_probe.py
"""
from __future__ import annotations

import numpy as np

from sna.align import canonical, fingerprints
from sna.data import read_msg_tsv
from sna.joint import unit

msg = read_msg_tsv("data/MassSpecGym1.5.tsv")
fold = np.array(msg["fold"])
smiles = [canonical(s) for s in msg["smiles"]]
ok = np.array([bool(s) for s in smiles])
train_rows = np.where((fold == "train") & ok)[0]
held_rows = np.where((fold == "val") & ok)[0]

z = np.load("data/msg_generated.parquet.embeddings.npz")
z_train = unit(z["train"][np.where(ok[train_rows])[0]])
z_held = unit(z["val"][np.where(ok[held_rows])[0]])

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]

fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

# one training vector per pool molecule: the mean embedding of its spectra
X = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(X, mol_of_row, z_train)
X /= np.bincount(mol_of_row)[:, None]
X = unit(X)

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, fp_pred):
    # Tanimoto between a predicted (soft) fingerprint and real ones, vectorised
    inter = fp_pred @ fp_train.T
    union = fp_pred.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
    sel = np.argsort(-inter / union, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<40} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

C = np.empty((len(held_rows), len(unique)), dtype=np.float32)
for i in range(0, len(held_rows), 2000):
    C[i:i + 2000] = z_held[i:i + 2000] @ X.T
sel = np.argsort(-C, axis=1)[:, :5]
t = tan(sel)
print(f"{'raw cosine (baseline)':<40} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

# Ridge: one 1024x1024 solve shared by all 2048 bits, closed form, no per-bit classifiers.
lam = 1.0
G = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32)
W = np.linalg.solve(G, X.T @ fp_train.astype(np.float32))
pred = np.clip(z_held @ W, 0, 1)
report("ridge -> soft fingerprint", pred)
report("ridge -> thresholded at 0.5", (pred > 0.5).astype(np.float32))

# How good is the fingerprint prediction itself?
held_pred = np.clip(z_held @ W, 0, 1)
exact = np.mean((held_pred > 0.5) == (fp_held > 0.5))
print(f"bit accuracy of the ridge fingerprint model: {exact:.3f}")
print(f"mean predicted bits {held_pred.sum(1).mean():.0f} vs true {fp_held.sum(1).mean():.0f}")

