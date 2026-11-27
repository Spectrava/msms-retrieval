"""Stack the signals that individually helped: two encoders, max-pooled pool scores,
and a ridge model that turns an embedding into a soft fingerprint.

Everything is fit on the training fold only and scored on the validation fold.

    python scripts/combine_probe.py
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

# spectra of each pool molecule, in one group, for max-pooling the scores

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
    print(f"{name:<48} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}")

d, dh = encoders["dreams"]
j, jh = encoders["jepa"]
S_d, S_j = cos(dh, pooled(d)), cos(jh, pooled(j))

report("baseline: DreaMS, mean pool", S_d)
for w in (0.1, 0.2, 0.3):
    report(f"ensemble {1-w:.2g} DreaMS + {w:.2g} JEPA (mean pool)", (1 - w) * S_d + w * S_j)

def ridge_head(q, P):
    W = np.linalg.solve(P.T @ P + np.eye(P.shape[1], dtype=np.float32),
                        P.T @ fp_train.astype(np.float32))
    fp_pred = np.clip(unit(q) @ W, 0, 1)
    inter = fp_pred @ fp_train.T
    union = fp_pred.sum(1, keepdims=True) + fp_train.sum(1)[None, :] - inter
    return inter / union

S_fp = ridge_head(dh, pooled(d))
report("ridge fingerprint head, DreaMS", S_fp)
for w in (0.25, 0.5, 0.75):
    report(f"blend {1-w:.2g} cosine + {w:.2g} fingerprint", (1 - w) * S_d + w * S_fp)
