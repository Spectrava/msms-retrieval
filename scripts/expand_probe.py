"""Query expansion: retrieve, average the top-K embeddings, retrieve again.

A classic training-free trick. If the frozen embedding is noisy, the centroid of the first
retrieval is a less noisy version of the query. Test whether iterating helps, and how far.

    python scripts/expand_probe.py
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
z_train = z["train"][np.where(ok[train_rows])[0]]
q = unit(z["val"][np.where(ok[held_rows])[0]])

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

z_unique = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(z_unique, mol_of_row, z_train)
z_unique /= np.bincount(mol_of_row)[:, None]
z_unique = unit(z_unique)


def score(qv):
    out = np.empty((len(q), len(unique)), dtype=np.float32)
    for i in range(0, len(qv), 2000):
        out[i:i + 2000] = unit(qv[i:i + 2000]) @ z_unique.T
    return out


def report(name, s):
    sel = np.argsort(-s, axis=1)[:, :5]
    t = np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
        np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)
    print(f"{name:<34} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")


S = score(q)
report("no expansion", S)
for k in (5, 20, 50, 200):
    sel = np.argsort(-S, axis=1)[:, :k]
    q2 = z_unique[sel].mean(1)
    report(f"expand, mean of top-{k}", score(q2))
    # keep the original query in the mix
    for w in (0.25, 0.5):
        report(f"expand top-{k}, weight {w} of original", score(w * q + (1 - w) * unit(q2)))
