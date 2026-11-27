"""Peak-list cosine: the signal the frozen embedding throws away.

Mass spectrometry practice compares spectra by shared fragments. DreaMS compresses each spectrum
to 1024 numbers; the raw peak list is still on disk. Build a sparse (spectrum, m/z bin) intensity
matrix, take the cosine between a query and every train spectrum, aggregate to the molecule with a
max, and ask whether it adds anything to the embedding - and whether the two agree.

    python scripts/peak_probe.py
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from sna.align import canonical, fingerprints
from sna.data import read_msg_tsv

BIN = 0.1  # Da per fragment bin

msg = read_msg_tsv("data/MassSpecGym1.5.tsv")
fold = np.array(msg["fold"])
smiles = [canonical(s) for s in msg["smiles"]]
ok = np.array([bool(s) for s in smiles])
train_rows = np.where((fold == "train") & ok)[0]
held_rows = np.where((fold == "val") & ok)[0]

off = msg["offsets"]
n_peaks = off[1:] - off[:-1]
bin_id = np.floor(msg["mz"] / BIN).astype(np.int64)
n_bins = int(bin_id.max()) + 1

def sparse(rows):
    idx = np.repeat(np.arange(len(rows), dtype=np.int64), n_peaks[rows])
    mz_sel = np.concatenate([bin_id[off[r]:off[r + 1]] for r in rows])
    i_sel = np.concatenate([msg["intensity"][off[r]:off[r + 1]] for r in rows])
    M = sp.csr_matrix((i_sel, (idx, mz_sel)), shape=(len(rows), n_bins), dtype=np.float32)
    norm = np.sqrt(M.multiply(M).sum(1)).A1
    norm[norm == 0] = 1.0
    return M, norm

Q, qn = sparse(held_rows)
T, tn = sparse(train_rows)

unique = sorted({smiles[i] for i in train_rows})
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

best = np.full((len(held_rows), len(unique)), -1.0, dtype=np.float32)
order = np.argsort(mol_of_row, kind="stable")
starts = np.concatenate([[0], np.where(np.diff(mol_of_row[order]) > 0)[0] + 1])
T_sorted = T[order]
for i in range(0, len(held_rows), 1000):
    s = (Q[i:i + 1000] @ T_sorted.T).toarray() / (qn[i:i + 1000, None] * tn[order][None, :])
    best[i:i + 1000] = np.maximum.reduceat(s, starts, axis=1)

z = np.load("data/msg_generated.parquet.embeddings.npz")
z_train = z["train"][np.where(ok[train_rows])[0]]
z_held = z["val"][np.where(ok[held_rows])[0]]
z_held /= np.linalg.norm(z_held, axis=1, keepdims=True)
z_unique = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(z_unique, mol_of_row, z_train)
z_unique /= np.bincount(mol_of_row)[:, None]
z_unique /= np.linalg.norm(z_unique, axis=1, keepdims=True)
C = np.empty((len(held_rows), len(unique)), dtype=np.float32)
for i in range(0, len(held_rows), 2000):
    C[i:i + 2000] = z_held[i:i + 2000] @ z_unique.T

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

sel_p = np.argsort(-best, axis=1)[:, :5]
sel_c = np.argsort(-C, axis=1)[:, :5]
tp, tc = tan(sel_p), tan(sel_c)
print(f"peak-list cosine      top1 {tp[:, 0].mean():.4f}  top5 {tp.mean(1).mean():.4f}")
print(f"embedding cosine      top1 {tc[:, 0].mean():.4f}  top5 {tc.mean(1).mean():.4f}")
print(f"same top-1 molecule: {np.mean(sel_p[:, 0] == sel_c[:, 0]):.1%}")
print(f"top-5 overlap: {np.mean([len(set(sel_p[i]) & set(sel_c[i])) for i in range(0, len(held_rows), 500)]) / 5:.1%}")
for w in (0.25, 0.5, 0.75):
    tb = tan(np.argsort(-(w * C + (1 - w) * best), axis=1)[:, :5])
    print(f"blend {w:.2f} embedding / {1 - w:.2f} peaks  top1 {tb[:, 0].mean():.4f}  top5 {tb.mean(1).mean():.4f}")
