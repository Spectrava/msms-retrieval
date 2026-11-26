"""Is the monoisotopic mass already linearly readable from the frozen embeddings?

Ridge probe: embedding -> neutral exact mass, fitted on the train fold and scored on the val fold.
Reported as R^2 and the median absolute error in Da. If a linear map recovers mass to well under
1 Da the frozen spaces already carry it, and training on it buys little.

    python scripts/mass_probe.py
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdMolDescriptors

from sna.align import ridge

msg = pd.read_csv("data/MassSpecGym1.5.tsv", sep="\t")
train = np.where(msg["fold"].values == "train")[0]
val = np.where(msg["fold"].values == "val")[0]
exact = np.array([rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s)) for s in msg["smiles"]])

# Does the file's own parent_mass agree with RDKit? If it does, mass is free and needs no encoder.
d = exact[val] - msg["parent_mass"].values[val]
print(f"file parent_mass vs RDKit exact: median |diff| {np.median(np.abs(d)):.4f} Da, "
      f"max {np.abs(d).max():.4f}")

emb = np.load("data/msg_generated.parquet.embeddings.npz")
z_train, z_val = emb["train"], emb["val"]
mass = exact[val]


def report(name, zt, zv):
    w = ridge(zt, exact[train], lam=1e-3)
    err = zv @ w - mass
    r2 = 1 - np.sum(err ** 2) / np.sum((mass - mass.mean()) ** 2)
    print(f"{name:34s} R2 {r2:.4f}  median |err| {np.median(np.abs(err)):.3f} Da  "
          f"<=0.05 {np.mean(np.abs(err) <= 0.05):.1%}  <=1 Da {np.mean(np.abs(err) <= 1):.1%}")


print(f"mass range {mass.min():.1f}-{mass.max():.1f} Da, mean {mass.mean():.1f}")
report("DreaMS spectrum embedding (1024-d)", z_train, z_val)
report("DreaMS embedding norm only", np.linalg.norm(z_train, axis=1)[:, None],
       np.linalg.norm(z_val, axis=1)[:, None])

mj = np.load("data/molecule_embeddings_moljepa.npz")
index = {s: i for i, s in enumerate(mj["smiles"])}
# The Mol-JEPA cache holds train + test SMILES, not the val fold, so that side is scored on test.
def rows_of(side):
    r = [x for x in np.where(msg["fold"].values == side)[0] if msg["smiles"].values[x] in index]
    return mj["emb"][[index[msg["smiles"].values[x]] for x in r]], exact[r]

zt, mt = rows_of("train")
zv, mv = rows_of("test")
print(f"Mol-JEPA covers {len(mt)} train / {len(mv)} test molecules")
w = ridge(zt, mt, lam=1e-3)
err = zv @ w - mv
r2 = 1 - np.sum(err ** 2) / np.sum((mv - mv.mean()) ** 2)
print(f"{'Mol-JEPA molecule embedding (512-d)':34s} R2 {r2:.4f}  median |err| "
      f"{np.median(np.abs(err)):.3f} Da  <=0.05 {np.mean(np.abs(err) <= 0.05):.1%}  "
      f"<=1 Da {np.mean(np.abs(err) <= 1):.1%}")

from sna.align import fingerprints
fp = fingerprints(list(msg["smiles"]))
report("Morgan fingerprint of the molecule (floor)", fp[train], fp[val])
