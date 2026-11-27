"""The two adducts of the same molecule sit at cosine 0.559 in the DreaMS space, yet the pool
averages them into one vector. Does pooling by adduct instead help?

  A. mean over all spectra of the molecule - what every benchmark here uses
  B. mean over spectra of the molecule with the *same* adduct as the query
  C. the best-scoring spectrum of the molecule (max instead of mean)

    python scripts/adduct_pool_probe.py
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
z_held = unit(z["val"][np.where(ok[held_rows])[0]])

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
add_train, add_held = msg["adduct"][train_rows], msg["adduct"][held_rows]

fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]


def report(name, mol_sel):
    t = np.minimum(fp_train[mol_sel], fp_held[:, None, :]).sum(-1) / \
        np.maximum(fp_train[mol_sel], fp_held[:, None, :]).sum(-1)
    print(f"{name:<34} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")


def top5(pool, pool_mol):
    sel = np.empty((len(held_rows), 5), dtype=np.int32)
    for i in range(0, len(held_rows), 2000):
        s = z_held[i:i + 2000] @ pool.T
        sel[i:i + 2000] = np.argsort(-s, axis=1)[:, :5]
    return pool_mol[sel]


# A. mean over all spectra of the molecule
z_mean = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(z_mean, mol_of_row, z_train)
z_mean /= np.bincount(mol_of_row)[:, None]
report("A. mean over all spectra", top5(unit(z_mean), np.arange(len(unique))))

# B. mean per (molecule, adduct); each query is scored only against its own adduct
keys = {(m, a) for m, a in zip(mol_of_row, add_train)}
codes = sorted(keys)
code_of = {k: i for i, k in enumerate(codes)}
z_add = np.zeros((len(codes), z_train.shape[1]), dtype=np.float32)
np.add.at(z_add, [code_of[(m, a)] for m, a in zip(mol_of_row, add_train)], z_train)
z_add /= np.bincount([code_of[(m, a)] for m, a in zip(mol_of_row, add_train)])[:, None]
sel_b = np.zeros((len(held_rows), 5), dtype=np.int32)
for a in (1, 2):
    cols = np.array([code_of[k] for k in codes if k[1] == a])
    s = np.empty((len(held_rows), len(cols)), dtype=np.float32)
    for i in range(0, len(held_rows), 2000):
        s[i:i + 2000] = z_held[i:i + 2000] @ unit(z_add[cols]).T
    rows = np.where(add_held == a)[0]
    sel_b[rows] = cols[np.argsort(-s[rows], axis=1)[:, :5]]
report("B. adduct-matched pool", np.array([codes[i][0] for i in sel_b.ravel()]).reshape(sel_b.shape))

# C. best-scoring spectrum of the molecule (max pooling instead of mean)
z_tr = unit(z_train)
order = np.argsort(mol_of_row, kind="stable")
starts = np.concatenate([[0], np.where(np.diff(mol_of_row[order]) > 0)[0] + 1])
best = np.empty((len(held_rows), len(unique)), dtype=np.float32)
for i in range(0, len(held_rows), 500):
    s = z_held[i:i + 500] @ z_tr[order].T
    best[i:i + 500] = np.maximum.reduceat(s, starts, axis=1)
report("C. max over spectra per molecule", np.argsort(-best, axis=1)[:, :5])
