"""Sweep the blend weight and add graph consensus on top of the best combination.

    python scripts/blend_sweep.py
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

n_peaks = np.diff(msg["offsets"])
def meta(rows):
    onehot = np.zeros((len(rows), 3), dtype=np.float32)
    onehot[np.arange(len(rows)), msg["adduct"][rows]] = 1
    inst = np.zeros((len(rows), 4), dtype=np.float32)
    inst[np.arange(len(rows)), msg["instrument"][rows]] = 1
    return np.column_stack([n_peaks[rows], np.log1p(n_peaks[rows]), msg["precursor_mz"][rows],
                            msg["ce"][rows], msg["ce_valid"][rows].astype(np.float32),
                            onehot, inst]).astype(np.float32)

M_train, M_held = meta(train_rows), meta(held_rows)
M_train_u = np.zeros((len(unique), M_train.shape[1]), dtype=np.float32)
np.add.at(M_train_u, mol_of_row, M_train)
M_train_u /= np.bincount(mol_of_row)[:, None]
scale = M_train_u.std(0); scale[scale == 0] = 1.0

def cos(q, P_):
    out = np.empty((len(q), len(P_)), dtype=np.float32)
    for i in range(0, len(q), 2000):
        out[i:i + 2000] = unit(q[i:i + 2000]) @ P_.T
    return out

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, S):
    sel = np.argsort(-S, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<52} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

Q_train = np.hstack([P, M_train_u / scale]).astype(np.float32)
Q_held = np.hstack([zh, M_held / scale]).astype(np.float32)
G = Q_train.T @ Q_train + np.eye(Q_train.shape[1], dtype=np.float32)
W = np.linalg.solve(G, Q_train.T @ fp_train.astype(np.float32))
p = np.clip(Q_held @ W, 0, 1)
inter = p @ fp_train.T
union = p.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
S_fp = inter / union
S_cos = cos(zh, P)

T_pool = tanimoto_matrix(fp_train, fp_train)

for w in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
    report(f"blend {1 - w:.1f} cosine + {w:.1f} fingerprint head", (1 - w) * S_cos + w * S_fp)

best = 0.7
S = (1 - best) * S_cos + best * S_fp
for K, lam in [(20, 0.1), (20, 0.3), (50, 0.1), (50, 0.3)]:
    top = np.argsort(-S, axis=1)[:, :K]
    cons = T_pool[top].mean(1)
    report(f"best blend + consensus over top-{K}, lam={lam}", S + lam * cons)
