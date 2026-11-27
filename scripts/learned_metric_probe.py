"""Learn the metric instead of the fingerprint.

Fit a symmetric bilinear form B so that z_q^T B z_p approximates the Tanimoto similarity
between the two molecules, then retrieve with it. Closed form: with Z the pool matrix,
minimising ||Z B Z^T - T||_F gives B = (Z^T Z)^-1 Z^T T Z (Z^T Z)^-1.

    python scripts/learned_metric_probe.py
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
P = unit(P / np.bincount(mol_of_row)[:, None])

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, S):
    sel = np.argsort(-S, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<44} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

S_cos = np.empty((len(zh), len(unique)), dtype=np.float32)
for i in range(0, len(zh), 2000):
    S_cos[i:i + 2000] = zh[i:i + 2000] @ P.T
report("cosine", S_cos)

T = tanimoto_matrix(fp_train, fp_train).astype(np.float32)
A = P.T @ P
Ainv = np.linalg.solve(A + 1e-3 * np.eye(A.shape[1], dtype=np.float32), np.eye(A.shape[1], dtype=np.float32))
ZT = P.T @ T                       # [d, n]
B = Ainv @ (ZT @ P) @ Ainv
B = (B + B.T) / 2
print(f"||Z B Z^T - T||_F / ||T||_F = {np.linalg.norm(P @ B @ P.T - T) / np.linalg.norm(T):.3f}")

S_B = np.empty((len(zh), len(unique)), dtype=np.float32)
for i in range(0, len(zh), 2000):
    S_B[i:i + 2000] = (zh[i:i + 2000] @ B) @ P.T
report("learned bilinear metric", S_B)
for w in (0.25, 0.5, 0.75):
    report(f"blend {1 - w:.2f} cosine + {w:.2f} learned metric", (1 - w) * S_cos + w * S_B)
