"""Can anything be gained by combining the signals we already have, without training?

Raw cosine is the best single scorer (top1 0.184) and the precursor mass is the best formula
scorer (formula@5 0.051). This probe sweeps the obvious ways of putting them together - a hard
mass window, a soft mass penalty, rank fusion, two-stage filter-then-rank - and reports whether
any of them beats either alone. Nothing is fitted, so a gain here is free and a loss says the
combination is a dead end before we spend GPU time on it.

Each query is scored only over the union of its 500 best cosine candidates and its 500 nearest
masses: a top-5 answer from any convex combination of the two signals must come from that set,
and the full 19429 x 25046 matrices do not fit in memory several times over.

    python scripts/fusion_probe.py
"""
from __future__ import annotations

import numpy as np

from sna.align import canonical, exact_mass, fingerprints, formulas, tanimoto_matrix
from sna.data import read_msg_tsv
from sna.joint import unit

K = 500  # per signal; the candidate set is the union of the two top-K lists

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
pos = {s: i for i, s in enumerate(smiles)}
mol_of_row = np.array([unique.index(smiles[i]) for i in train_rows])
held_smiles = [smiles[i] for i in held_rows]
z_unique = np.zeros((len(unique), z_train.shape[1]), dtype=np.float32)
np.add.at(z_unique, mol_of_row, z_train)
z_unique /= np.bincount(mol_of_row)[:, None]

fp = fingerprints(unique + held_smiles)
fp_train, fp_held = fp[: len(unique)], fp[len(unique):]
all_formula = np.array(formulas(list(smiles)))
formula_train = np.array([all_formula[pos[s]] for s in unique])
formula_held = np.array([all_formula[pos[s]] for s in held_smiles])
mass_train = exact_mass(unique).astype(np.float32)
mass_held = exact_mass(held_smiles).astype(np.float32)
peaks = np.diff(msg["offsets"])

print(f"pool {len(unique)} molecules, {len(held_rows)} held spectra")
print(f"peaks per spectrum: mean {peaks.mean():.1f}, median {np.median(peaks):.0f}, max {peaks.max()}")
print(f"distinct formulas in pool {len(set(formula_train))}, "
      f"{np.mean((formula_train[None, :] == formula_held[:, None]).any(1)):.1%} of queries have a congener in the pool")

# Candidate set per query: union of the cosine top-K and the mass-nearest top-K.
S = np.empty((len(held_rows), 2 * K), dtype=np.float32)
dm = np.empty((len(held_rows), 2 * K), dtype=np.float32)
cand = np.empty((len(held_rows), 2 * K), dtype=np.int32)
for i in range(0, len(held_rows), 1000):
    s = (unit(z_held[i:i + 1000]) @ z_unique.T).astype(np.float32)
    m = np.abs(mass_train[None, :] - mass_held[i:i + 1000, None])
    a = np.argpartition(-s, K, axis=1)[:, :K]
    b = np.argpartition(m, K, axis=1)[:, :K]
    idx = np.concatenate([a, b], axis=1)
    cand[i:i + 1000] = idx
    S[i:i + 1000] = np.take_along_axis(s, idx, axis=1)
    dm[i:i + 1000] = np.take_along_axis(m, idx, axis=1)
same_formula = formula_train[cand] == formula_held[:, None]
print(f"candidate set: {2 * K} per query, {same_formula.sum()} share the query formula "
      f"({np.mean(same_formula.any(1)):.1%} of queries have at least one)")

# The oracle: the best any ranking could do with this pool. Everything below it is headroom.
oracle = np.empty((len(fp_held), 5))
for i in range(0, len(fp_held), 1000):
    oracle[i:i + 1000] = np.sort(tanimoto_matrix(fp_held[i:i + 1000], fp_train), axis=1)[:, -5:]
print(f"ORACLE (best possible with this pool) top1 {oracle[:, -1].mean():.4f}  top5 {oracle.mean(1).mean():.4f}")


def report(name, score, k=5):
    order = np.argsort(-score, axis=1)[:, :k]
    sel = cand[np.arange(len(cand))[:, None], order]
    t = (np.minimum(fp_train[sel], fp_held[:, None, :]).sum(-1) /
         np.maximum(fp_train[sel], fp_held[:, None, :]).sum(-1))
    f5 = (formula_train[sel] == formula_held[:, None]).mean()
    print(f"{name:<38} top1 {t[:, 0].mean():.4f}  top5 {t.mean(1).mean():.4f}  formula@5 {f5:.4f}")


report("raw cosine", S)
report("precursor mass only", -dm)
for w in (0.5, 1, 2, 5, 10, 20, 50):
    report(f"cosine, mass window +/-{w} Da", S * (dm <= w))
for sigma in (1, 5, 20):
    for lam in (0.1, 0.5, 1.0):
        report(f"cosine - {lam} (dm/{sigma})^2", S - lam * (dm / sigma) ** 2, k=5)

# Rank fusion: the two signals disagree about which molecule is right, so average their ranks.
def rank(score):
    order = np.argsort(-score, axis=1)
    r = np.zeros_like(score, dtype=np.float32)
    np.put(r, order.ravel(), np.tile(np.arange(score.shape[1], dtype=np.float32), score.shape[0]))
    return r


cos_rank, mass_rank = rank(S), rank(-dm)
for w in (0.1, 0.2, 0.3, 0.5):
    report(f"rank fusion cosine {1-w} + mass {w}", (1 - w) * cos_rank + w * mass_rank)

# Two-stage: keep the mass-consistent candidates, then rank those by cosine.
for w in (1, 5, 10):
    report(f"mass window +/-{w} then cosine", np.where(dm <= w, S, -1.0))

# How good is the mass window as a recall filter for the true formula?
for w in (1, 5, 10, 20):
    hit = (same_formula & (dm <= w)).sum(1)
    print(f"  window +/-{w} Da: {(hit > 0).mean():.1%} of queries contain >=1 true-formula candidate, "
          f"mean {hit.mean():.1f} per query")
