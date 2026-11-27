"""Stack everything that helped, and add the free metadata the encoder throws away.

Signals tried so far, on the same pool and held fold:
  baseline cosine                       0.1838 / 0.1715
  ensemble with the JEPA encoder        0.1881
  max-pooling the pool                  0.1867
  consensus reranking                   0.1881
  ridge fingerprint head                0.1890 / 0.1851
  averaged query + fingerprint head     0.1938 / 0.1887

This script fits the fingerprint head on query features that include acquisition metadata
(peak count, precursor, adduct, collision energy, instrument) and checks whether the gains
stack or are the same gain counted twice.

    python scripts/stack_probe.py
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
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
fp = fingerprints(unique + held_smiles)
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
    return np.column_stack([
        n_peaks, np.log1p(n_peaks), msg["precursor_mz"][rows], msg["ce"][rows],
        msg["ce_valid"][rows].astype(np.float32), onehot, inst,
    ]).astype(np.float32)

M_train, M_held = meta(train_rows), meta(held_rows)

def cos(q, P):
    out = np.empty((len(q), len(P)), dtype=np.float32)
    for i in range(0, len(q), 2000):
        out[i:i + 2000] = unit(q[i:i + 2000]) @ P.T
    return out

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, S):
    sel = np.argsort(-S, axis=1)[:, :5]
    t = tan(sel)
    print(f"{name:<52} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

def head(Q_train, Q_held, lam=1.0):
    """Ridge from query features to a soft fingerprint, then Tanimoto against the pool.
    Fitted on the train fold only - the held spectra never see their own molecule."""
    G = Q_train.T @ Q_train + lam * np.eye(Q_train.shape[1], dtype=np.float32)
    W = np.linalg.solve(G, Q_train.T @ fp_train.astype(np.float32))
    p = np.clip(Q_held @ W, 0, 1)
    inter = p @ fp_train.T
    union = p.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
    return inter / union

def feats(q, m):
    # embedding on the unit sphere, metadata scaled to unit variance so both blocks matter
    scale = m.std(0)
    scale[scale == 0] = 1.0
    return np.hstack([unit(q), m / scale]).astype(np.float32)

d, dh = encoders["dreams"]
j, jh = encoders["jepa"]
P_d = pooled(d)

report("baseline cosine", cos(dh, P_d))

M_train_u = np.zeros((len(unique), M_train.shape[1]), dtype=np.float32)
np.add.at(M_train_u, mol_of_row, M_train)
M_train_u /= np.bincount(mol_of_row)[:, None]

for lam in (1.0, 10.0, 100.0):
    report(f"ridge head, DreaMS + metadata, lam={lam:g}", head(feats(P_d, M_train_u), feats(dh, M_held), lam))

report("ridge head, DreaMS only, lam=1", head(P_d, unit(dh)))

