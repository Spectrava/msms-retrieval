"""Class-aware retrieval: predict the HEAL ladder (element counts, rings, unsaturation,
mass) from the spectrum embedding, then score a pool molecule by how well its true
classes agree with the predicted ones.

    python scripts/class_probe.py
"""
from __future__ import annotations

import numpy as np

from sna.align import canonical, descriptors, fingerprints, formulas
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
held_smiles = [smiles[i] for i in held_rows]
fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]
all_formula = np.array(formulas(smiles))
pos = {s: i for i, s in enumerate(smiles)}
formula_train = np.array([all_formula[pos[s]] for s in unique])
formula_held = np.array([all_formula[i] for i in held_rows])

D_train = descriptors(unique)
D_held = descriptors(held_smiles)

z = np.load("data/msg_generated.parquet.embeddings.npz")
zt = unit(z["train"][np.where(ok[train_rows])[0]])
zh = unit(z["val"][np.where(ok[held_rows])[0]])
P = np.zeros((len(unique), zt.shape[1]), dtype=np.float32)
np.add.at(P, mol_of_row, zt)
P = unit(P / np.bincount(mol_of_row)[:, None])

def cos(q, P_):
    out = np.empty((len(q), len(P_)), dtype=np.float32)
    for i in range(0, len(q), 2000):
        out[i:i + 2000] = q[i:i + 2000] @ P_.T
    return out

def tan(sel):
    return np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) / \
           np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1)

def report(name, S):
    sel = np.argsort(-S, axis=1)[:, :5]
    t = tan(sel)
    f5 = np.mean((formula_train[sel] == formula_held[:, None]).any(1))
    print(f"{name:<52} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}  formula@5 {f5:.4f}")

S_cos = cos(unit(zh), P)
report("baseline cosine", S_cos)

# ridge from the embedding to the class ladder, fitted on the train pool only
G = P.T @ P + np.eye(P.shape[1], dtype=np.float32)
W = np.linalg.solve(G, P.T @ D_train)
pred = zh @ W                                   # [held, 14]
# scale each class to unit variance so the agreement term is comparable across levels
sd = D_train.std(0)
sd[sd == 0] = 1.0
agree = np.empty((len(zh), len(unique)), dtype=np.float32)
for i in range(0, len(zh), 1000):
    # [chunk, pool, 14] -> mean over levels of clipped agreement
    d = np.abs(pred[i:i + 1000, None, :] - D_train[None, :, :]) / sd
    agree[i:i + 1000] = np.clip(1.0 - d, 0, 1).mean(2)

exact_formula = (formula_held[:, None] == formula_train[None, :]).astype(np.float32)
print(f"oracle formula@5 ceiling {np.mean((formula_held[:, None] == formula_train[None, :]).sum(1) > 0):.4f}")

for lam in (0.1, 0.3, 0.5, 1.0):
    report(f"cosine + {lam} class agreement", S_cos + lam * agree)
for lam in (0.1, 0.3, 0.5):
    report(f"cosine + {lam} class agreement + {lam} exact formula", S_cos + lam * (agree + exact_formula))

# how good is the predicted mass level on its own?
print(f"mean absolute error of the predicted nominal mass "
      f"{np.abs(pred[:, -2] - D_held[:, -2]).mean():.1f} Da, "
      f"predicted exact mass {np.abs(pred[:, -1] - D_held[:, -1]).mean():.1f} Da")
