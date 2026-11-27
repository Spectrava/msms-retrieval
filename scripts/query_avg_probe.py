"""The val fold is 19429 spectra but only 3386 molecules.

Every held query is one spectrum of a compound that has, on average, 5.7 spectra in the fold.
The pool side already averages across spectra; the query side does not. Average the query, and
see whether the extra spectra of the same compound buy anything.

    python scripts/query_avg_probe.py
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

X = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(X, mol_of_row, z_train)
X = unit(X / np.bincount(mol_of_row)[:, None])

held_mol = np.array([unique.index(s) if s in unique else -1 for s in held_smiles])
groups = {}
for i, s in enumerate(held_smiles):
    groups.setdefault(s, []).append(i)
sizes = np.array([len(v) for v in groups.values()])
print(f"{len(groups)} held molecules, spectra per molecule: "
      f"mean {sizes.mean():.1f} median {np.median(sizes):.0f} max {sizes.max()}")

def tan(sel, rows):
    return np.minimum(fp_train[sel], fp_held[rows, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[rows, None, :]).sum(-1)

def report(name, S, rows=None):
    if rows is None:
        rows = np.arange(S.shape[0])
    sel = np.argsort(-S, axis=1)[:, :5]
    t = tan(sel, rows)
    print(f"{name:<44} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

def cos(q):
    out = np.empty((len(q), len(unique)), dtype=np.float32)
    for i in range(0, len(q), 2000):
        out[i:i + 2000] = unit(q[i:i + 2000]) @ X.T
    return out

report("one spectrum per query (baseline)", cos(z_held), np.arange(len(held_rows)))

mol_list = list(groups)
q_avg = np.array([z_held[groups[s]].mean(0) for s in mol_list])
rows = [groups[s][0] for s in mol_list]
report("averaged over all spectra of the compound", cos(q_avg), rows)

# ridge fingerprint head on the averaged query
W = np.linalg.solve(X.T @ X + np.eye(X.shape[1], dtype=np.float32),
                    X.T @ fp_train.astype(np.float32))
def fp_rank(q):
    p = np.clip(unit(q) @ W, 0, 1)
    inter = p @ fp_train.T
    union = p.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
    return inter / union

report("ridge fingerprint head, single spectrum", fp_rank(z_held), np.arange(len(held_rows)))
report("ridge fingerprint head, averaged query", fp_rank(q_avg), rows)
