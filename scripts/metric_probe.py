"""The metric, not the encoder: cosine on a space whose mean is not zero is dominated by
that mean. Try centring, whitening, and PCA truncation of the frozen DreaMS embedding.

    python scripts/metric_probe.py
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

unique = sorted({smiles[i] for i in train_rows})
idx = {s: i for i, s in enumerate(unique)}
mol_of_row = np.array([idx[smiles[i]] for i in train_rows])
fp = fingerprints(unique + [smiles[i] for i in held_rows])
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

z = np.load("data/msg_generated.parquet.embeddings.npz")
zt = unit(z["train"][np.where(ok[train_rows])[0]])
zh = unit(z["val"][np.where(ok[held_rows])[0]])
P = np.zeros((len(unique), zt.shape[1]), dtype=np.float32)
np.add.at(P, mol_of_row, zt)
P /= np.bincount(mol_of_row)[:, None]

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, Q, R):
    out = np.empty((len(Q), len(R)), dtype=np.float32)
    for i in range(0, len(Q), 2000):
        out[i:i + 2000] = Q[i:i + 2000] @ R.T
    sel = np.argsort(-out, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<44} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

mean = P.mean(0)
print(f"mean embedding norm {np.linalg.norm(mean):.3f}; "
      f"cos(query, mean) mean {np.mean(zh @ (mean / np.linalg.norm(mean))):.3f}")

report("plain cosine", unit(zh), unit(P))
report("centred cosine", unit(zh - mean), unit(P - mean))

C = (P - mean).T @ (P - mean) / len(P)
evals, evecs = np.linalg.eigh(C)
print(f"top eigenvalues {evals[-6:][::-1].round(4)}; "
      f"rank where eigenvalue > 1e-6: {int((evals > 1e-6).sum())}")

for k in (50, 100, 200, 400, 800):
    V = evecs[:, -k:]
    report(f"PCA top-{k} cosine", unit((zh - mean) @ V), unit((P - mean) @ V))

whiten = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-9))) @ evecs.T
report("whitened (Mahalanobis) cosine", unit((zh - mean) @ whiten), unit((P - mean) @ whiten))

# Tanimoto between the *predicted* fingerprint and the pool, but ranking by a
# symmetric score: how similar is the query's predicted fingerprint to the pool's.
G = P.T @ P + np.eye(P.shape[1], dtype=np.float32)
W = np.linalg.solve(G, P.T @ fp_train.astype(np.float32))
pq = np.clip((zh - mean) @ W, 0, 1)
inter = pq @ fp_train.T
union = pq.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
print(f"{'fingerprint head (centred)':<44} top1 {np.max(inter / union, axis=1).mean():.4f}")
