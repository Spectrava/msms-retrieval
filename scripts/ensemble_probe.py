"""Two frozen encoders see the same spectra: DreaMS (1024-d) and the JEPA encoder (256-d).
Ensembling them is the cheapest possible gain - no training, no new data.

Sweeps: cosine on each alone, the average of the two similarities, and rank fusion.

    python scripts/ensemble_probe.py
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

dr = np.load("data/msg_generated.parquet.embeddings.npz")
je = np.load("data/jepa_latents.npz")
enc = {"dreams": (dr["train"], dr["val"]), "jepa": (je["train"], je["val"])}

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]

fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

S = {}
for name, (z_train, z_held) in enc.items():
    z_unique = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
    np.add.at(z_unique, mol_of_row, z_train)
    z_unique /= np.bincount(mol_of_row)[:, None]
    z_unique = unit(z_unique)
    s = np.empty((len(held_rows), len(unique)), dtype=np.float32)
    for i in range(0, len(held_rows), 2000):
        s[i:i + 2000] = unit(z_held[i:i + 2000]) @ z_unique.T
    S[name] = s
    print(f"{name}: dim {z_train.shape[1]}")

def score(name, sel):
    t = np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)
    return t[:, 0].mean(), t.mean(1).mean()

def report(name, s):
    sel = np.argsort(-s, axis=1)[:, :5]
    a, b = score(name, sel)
    print(f"{name:<32} top1 {a:.4f}  top5 {b:.4f}")

report("DreaMS alone", S["dreams"])
report("JEPA alone", S["jepa"])
for w in (0.25, 0.5, 0.75):
    report(f"score fusion {w} dreams + {1-w} jepa", w * S["dreams"] + (1 - w) * S["jepa"])

def rank(s):
    order = np.argsort(-s, axis=1)
    r = np.zeros_like(s, dtype=np.float32)
    np.put(r, order.ravel(), np.tile(np.arange(s.shape[1], dtype=np.float32), s.shape[0]))
    return s.shape[1] - r  # higher is better, so report's argsort(-score) works


r1, r2 = rank(S["dreams"]), rank(S["jepa"])
for w in (0.25, 0.5, 0.75):
    report(f"rank fusion {w} dreams + {1-w} jepa", w * r1 + (1 - w) * r2)

# How much do the two encoders disagree about the top-5?
t1 = np.argsort(-S["dreams"], axis=1)[:, :5]
t2 = np.argsort(-S["jepa"], axis=1)[:, :5]
overlap = [len(set(a) & set(b)) for a, b in zip(t1, t2)]
print(f"mean overlap between the two top-5 lists: {np.mean(overlap):.2f} of 5")
