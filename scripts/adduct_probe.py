"""The ceiling probe says cosine puts the best pool match at median rank 1342 of 25046.
Two things that look fixable and are cheap to test:

  1. [M+Na]+ is 24% of the held spectra and cosine finds its best match 2.5% of the time
     against 4.9% for [M+H]+. Is that the molecules or the adduct?
  2. Peak count predicts success (1.7% for <5 peaks, 8.8% for 80+). Is the embedding
     throwing away the low-intensity tail, or are sparse spectra just less informative?

    python scripts/adduct_probe.py
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

adduct_all = msg["adduct"]
peaks_all = np.diff(msg["offsets"])
adduct = adduct_all[held_rows]
n_peaks = peaks_all[held_rows]

# Same molecule measured as [M+H]+ and as [M+Na]+: the mass differs by 20.98 Da, the
# fragmentation should not. Compare the two embeddings directly.
held_pos = {r: j for j, r in enumerate(held_rows)}
by_mol = {}
for r, s in zip(held_rows, held_smiles):
    by_mol.setdefault(s, []).append(r)
pairs = [(held_pos[a], held_pos[b]) for s, rows in by_mol.items() if len(rows) >= 2
         for a in rows for b in rows if adduct_all[a] == 1 and adduct_all[b] == 2]
print(f"{len(pairs)} held molecules measured both as [M+H]+ and [M+Na]+")
if pairs:
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    cos = (unit(z_held[a]) * unit(z_held[b])).sum(1)
    print(f"cosine between the two adducts of the SAME molecule: mean {cos.mean():.4f}, "
          f"median {np.median(cos):.4f}")

# Does the pool average wash the adduct difference out?
pool_adduct = msg["adduct"][train_rows]
per_mol_adduct = np.zeros((len(unique), 2), dtype=np.int64)
np.add.at(per_mol_adduct, mol_of_row, np.eye(2, dtype=np.int64)[pool_adduct - 1])
both = (per_mol_adduct[:, 0] > 0) & (per_mol_adduct[:, 1] > 0)
print(f"pool molecules with spectra of both adducts: {both.sum()}")

# Retrieval quality per adduct, controlling for molecule complexity.
S = np.empty((len(held_rows), len(unique)), dtype=np.float32)
for i in range(0, len(held_rows), 2000):
    S[i:i + 2000] = unit(z_held[i:i + 2000]) @ z_unique.T
best = np.empty((len(held_rows), 5), dtype=np.int32)
best_t = np.empty((len(held_rows), 5), dtype=np.float32)
for i in range(0, len(held_rows), 1000):
    T = tanimoto_matrix(fp_held[i:i + 1000], fp_train)
    order = np.argsort(T, axis=1)[:, -5:]
    best[i:i + 1000] = order
    best_t[i:i + 1000] = np.take_along_axis(T, order, axis=1)
rank_best = (S > S[np.arange(len(held_rows)), best[:, -1]][:, None]).sum(1)

heavy = np.array([len(s) for s in held_smiles])  # crude size proxy
print("\nadduct, split by query size (heavy-atom proxy = SMILES length)")
for a in (1, 2):
    m = adduct == a
    for lo, hi in [(0, 30), (30, 50), (50, 80), (80, 1000)]:
        s = m & (heavy >= lo) & (heavy < hi)
        if s.sum() >= 50:
            print(f"  adduct {a}  len {lo}-{hi}  n={s.sum():>5}  "
                  f"oracle in cosine top-5 {np.mean(rank_best[s] < 5):.3f}  "
                  f"oracle Tanimoto {best_t[s, -1].mean():.3f}")

print("\npeak count, split by adduct")
for a in (1, 2):
    m = adduct == a
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 1000)]:
        s = m & (n_peaks >= lo) & (n_peaks < hi)
        if s.sum() >= 50:
            print(f"  adduct {a}  {lo:>3}-{hi:<4} peaks  n={s.sum():>5}  "
                  f"oracle in cosine top-5 {np.mean(rank_best[s] < 5):.3f}  "
                  f"oracle Tanimoto {best_t[s, -1].mean():.3f}")
