"""Everything that helped, in one place, against the same held fold.

    python scripts/stack_all_probe.py
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

unique = sorted({smiles[i] for i in train_rows})
idx = {s: i for i, s in enumerate(unique)}
mol_of_row = np.array([idx[smiles[i]] for i in train_rows])
fp = fingerprints(unique + [smiles[i] for i in held_rows])
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]

encoders = {}
for name, path in (("dreams", "data/msg_generated.parquet.embeddings.npz"),
                   ("jepa", "data/jepa_latents.npz")):
    z = np.load(path)
    encoders[name] = (unit(z["train"][np.where(ok[train_rows])[0]]),
                      unit(z["val"][np.where(ok[held_rows])[0]]))

def pooled(z_train):
    out = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
    np.add.at(out, mol_of_row, z_train)
    return unit(out / np.bincount(mol_of_row)[:, None])

def meta(rows):
    n_peaks = np.diff(msg["offsets"])[rows]
    onehot = np.zeros((len(rows), 3), dtype=np.float32)
    onehot[np.arange(len(rows)), msg["adduct"][rows]] = 1
    inst = np.zeros((len(rows), 4), dtype=np.float32)
    inst[np.arange(len(rows)), msg["instrument"][rows]] = 1
    return np.column_stack([n_peaks, np.log1p(n_peaks), msg["precursor_mz"][rows],
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
        out[i:i + 2000] = unit(q[i:i + 2000]) @ unit(P_).T
    return out

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, S):
    sel = np.argsort(-S, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<56} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

def head(Q_train, Q_held, lam=1.0):
    G = Q_train.T @ Q_train + lam * np.eye(Q_train.shape[1], dtype=np.float32)
    W = np.linalg.solve(G, Q_train.T @ fp_train.astype(np.float32))
    p = np.clip(Q_held @ W, 0, 1)
    inter = p @ fp_train.T
    union = p.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
    return inter / union

def feats(q, m):
    return np.hstack([unit(q), m / scale]).astype(np.float32)

d, dh = encoders["dreams"]
j, jh = encoders["jepa"]
P_d, P_j = pooled(d), pooled(j)
S_d, S_j = cos(dh, P_d), cos(jh, P_j)
S_ens = 0.8 * S_d + 0.2 * S_j

Q_train = feats(np.hstack([P_d, P_j]), M_train_u)
Q_held = feats(np.hstack([dh, jh]), M_held)
S_fp = head(Q_train, Q_held, 1.0)

report("baseline cosine (last run)", S_d)
report("score ensemble 0.8 DreaMS + 0.2 JEPA", S_ens)
report("ridge fingerprint head, both encoders + metadata", S_fp)
for w in (0.2, 0.4, 0.6, 0.8):
    report(f"blend {1 - w:.1f} ensemble cosine + {w:.1f} fingerprint head",
           (1 - w) * S_ens + w * S_fp)

# one vector per held compound: the mean of its spectra
groups = {}
for r, s in enumerate(smiles[i] for i in held_rows):
    groups.setdefault(s, []).append(r)
members = list(groups.values())
rep = np.array([m[0] for m in members])
fp_held_avg = fp_held[rep]
d_avg = np.array([dh[m].mean(0) for m in members])
j_avg = np.array([jh[m].mean(0) for m in members])
C_avg = cos(d_avg, P_d)
S_fp_avg = head(Q_train, feats(np.hstack([d_avg, j_avg]), M_held[rep]), 1.0)
for w in (0.2, 0.4, 0.6, 0.8):
    sel = np.argsort(-((1 - w) * C_avg + w * S_fp_avg), axis=1)[:, :5]
    t = np.minimum(fp_train[sel], fp_held_avg[:, None, :]).sum(-1) / \
        np.maximum(fp_train[sel], fp_held_avg[:, None, :]).sum(-1)
    print(f"{'averaged spectra of one compound, ' + f'{1 - w:.1f} cosine + {w:.1f} head':<56} "
          f"top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}  ({len(members)} compounds)")
