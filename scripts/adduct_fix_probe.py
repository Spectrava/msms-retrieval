"""Correct the sodium adduct before retrieving.

The ceiling probe found that [M+Na]+ queries have a *higher* oracle Tanimoto than [M+H]+ queries
(0.666 vs 0.552 for long SMILES) but cosine finds them in the top 5 only 2.2% of the time against
9.1% for [M+H]+. The embedding is adduct-dominated: the two adducts of the same molecule sit at
cosine 0.559. So fit a linear map from the [M+Na]+ embedding to the [M+H]+ embedding on train
pairs, apply it to held [M+Na]+ queries, and see whether retrieval for those queries improves.

    python scripts/adduct_fix_probe.py
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
z_held = z["val"][np.where(ok[held_rows])[0]]
add_train, add_held = msg["adduct"][train_rows], msg["adduct"][held_rows]

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

# Pair the two adducts of the same molecule in the training fold.
by_mol = {}
for pos, r in enumerate(train_rows):
    by_mol.setdefault(mol_of_row[pos], {})[add_train[pos]] = pos
pairs = [(v[1], v[2]) for v in by_mol.values() if 1 in v and 2 in v]
print(f"{len(pairs)} training molecules measured as both [M+H]+ and [M+Na]+")

A = z_train[[p[0] for p in pairs]]  # [M+H]+
B = z_train[[p[1] for p in pairs]]  # [M+Na]+
# least squares B -> A, with a bias term
X = np.concatenate([B, np.ones((len(B), 1))], axis=1)
W, *_ = np.linalg.lstsq(X, A, rcond=None)

def apply(v):
    return np.concatenate([v, np.ones((len(v), 1))], axis=1) @ W

def pool(embeddings):
    z_unique = np.zeros((len(unique), embeddings.shape[1]), dtype=np.float32)
    np.add.at(z_unique, mol_of_row, embeddings)
    z_unique /= np.bincount(mol_of_row)[:, None]
    return unit(z_unique)

def report(name, q, p):
    s = np.empty((len(held_rows), len(unique)), dtype=np.float32)
    for i in range(0, len(held_rows), 2000):
        s[i:i + 2000] = unit(q[i:i + 2000]) @ p.T
    sel = np.argsort(-s, axis=1)[:, :5]
    t = np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
        np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)
    na = add_held == 2
    print(f"{name:<32} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}  "
          f"[M+Na]+ only: top1 {t[na, 0].mean():.4f} top5 {t[na].mean(1).mean():.4f}")

P_plain = pool(z_train)
z_fixed = z_train.copy()
z_fixed[add_train == 2] = apply(z_train[add_train == 2])
P_fixed = pool(z_fixed)

report("no correction", z_held, P_plain)
q = z_held.copy()
q[add_held == 2] = apply(z_held[add_held == 2])
report("query corrected", q, P_plain)
report("query and pool corrected", q, P_fixed)

# How good is the map? Held-out check on held pairs.
by_mol_h = {}
for r, m, a in zip(held_rows, [unique.index(smiles[i]) for i in held_rows], add_held):
    by_mol_h.setdefault(m, {})[a] = r
hp = [(v[1], v[2]) for v in by_mol_h.values() if 1 in v and 2 in v]
pred = apply(z_held[[p[1] for p in hp]])
true = unit(z_held[[p[0] for p in hp]])
print(f"corrected vs true [M+H]+ cosine: {np.mean(np.sum(pred * true, axis=1)):.4f} "
      f"(uncorrected [M+Na]+ vs [M+H]+: {np.mean(np.sum(unit(z_held[[p[1] for p in hp]]) * true, axis=1)):.4f})")
