"""Where is the headroom? Raw cosine gets 0.166 top-1 Tanimoto; the oracle over the same pool is 0.505.

Three cheap questions, nothing fitted:
  1. Is the true molecule even inside the window a scorer can reach?
  2. Do the queries that fail share a property we are throwing away - peak count,
     collision energy, instrument, adduct?
  3. How much of the oracle is reachable with formula knowledge alone?

    python scripts/ceiling_probe.py
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

emb = np.load("data/msg_generated.parquet.embeddings.npz")
z_train = emb["train"][np.where(ok[train_rows])[0]]
z_held = emb["val"][np.where(ok[held_rows])[0]]

unique = sorted({smiles[i] for i in train_rows})
idx = {s: i for i, s in enumerate(unique)}
mol_of_row = np.array([idx[smiles[i]] for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
z_unique = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(z_unique, mol_of_row, z_train)
z_unique /= np.bincount(mol_of_row)[:, None]

fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

S = np.empty((len(held_rows), len(unique)), dtype=np.float32)
for i in range(0, len(held_rows), 2000):
    S[i:i + 2000] = unit(z_held[i:i + 2000]) @ z_unique.T

# The oracle candidate: the pool molecule with the best chemistry for this query.
best = np.empty((len(held_rows), 5), dtype=np.int32)
best_t = np.empty((len(held_rows), 5), dtype=np.float32)
for i in range(0, len(held_rows), 1000):
    T = tanimoto_matrix(fp_held[i:i + 1000], fp_train)
    order = np.argsort(T, axis=1)[:, -5:]
    best[i:i + 1000] = order
    best_t[i:i + 1000] = np.take_along_axis(T, order, axis=1)
rank_best = (S > S[np.arange(len(held_rows)), best[:, -1]][:, None]).sum(1)

print(f"held {len(held_rows)} spectra, pool {len(unique)} molecules")
print(f"oracle top-1 Tanimoto {best_t[:, -1].mean():.4f}, top-5 mean {best_t.mean(1).mean():.4f}")
print(f"where cosine puts the oracle top-1: rank median {np.median(rank_best):.0f}, "
      f"in top-5 {np.mean(rank_best < 5):.1%}, top-20 {np.mean(rank_best < 20):.1%}, "
      f"top-100 {np.mean(rank_best < 100):.1%}, top-500 {np.mean(rank_best < 500):.1%}")

n_peaks = np.diff(msg["offsets"])[held_rows]
ce, instrument, adduct = msg["ce"][held_rows], msg["instrument"][held_rows], msg["adduct"][held_rows]

def group(name, values, labels):
    print(f"\nby {name}")
    for v in labels:
        m = values == v
        if m.sum() >= 50:
            print(f"  {str(v):<12} n={m.sum():>6}  oracle top-1 in cosine top-5 {np.mean(rank_best[m] < 5):.3f}  "
                  f"oracle top-1 Tanimoto {best_t[m, -1].mean():.3f}")

group("peak count", np.digitize(n_peaks, [5, 10, 20, 40, 80]), [0, 1, 2, 3, 4, 5])
group("collision energy", np.round(ce), sorted(set(np.round(ce))))
group("instrument", instrument, range(len(msg["instrument"]) and 4))
group("adduct", adduct, sorted(set(adduct)))

top5 = np.argsort(-S, axis=1)[:, :5]
t5 = lambda sel: np.mean(np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) /
                         np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1))
print(f"\ncosine top-5 mean Tanimoto {t5(top5):.3f} against the oracle {best_t.mean(1).mean():.3f}")
print(f"cosine reaches {t5(top5) / best_t.mean(1).mean():.0%} of the achievable top-5 mean")
