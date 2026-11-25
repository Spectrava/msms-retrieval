"""Rule-based in-silico MS2 peak lists from a SMILES.

Every peak is a deterministic consequence of the structure: the precursor, one peak per
single-bond cleavage of an acyclic bond, and the neutral losses the formula can support. No
learned weights.

That matters. A decoder trained on real spectra turns a "synthetic" spectrum into a function of
the molecule embedding, so any alignment measured through it is the decoder, not chemistry, and
the retrieval metrics would be self-consistent by construction. Rules are the only synthetic
source that is independent of the encoder being aligned.

shortcut: intensities are all equal and only single-bond cleavages of heavy-atom bonds are
considered, so this is a substructure probe, not a fragmentation model; an element missing from
MONO falls back to its average atomic mass. Upgrade to a learned generator (MolGen, Fpocket)
only if the rule-based signal turns out to be worth extending.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from rdkit import Chem

PROTON = 1.00727646
# name -> (mass, atoms the molecule must have to lose it)
LOSSES = (("H2O", 18.010565, {"O": 1, "H": 2}),
          ("NH3", 17.026549, {"N": 1, "H": 3}),
          ("CO2", 43.989829, {"C": 1, "O": 2}),
          ("CO", 27.994915, {"C": 1, "O": 1}))
MIN_FRAGMENT = 14.0  # below this the peak is CH3 or smaller and carries no structure
# monoisotopic masses; an element missing here falls back to its average atomic mass
MONO = {"H": 1.00782503207, "C": 12.0, "N": 14.00307400443, "O": 15.99491461956,
        "F": 18.99840301515, "S": 31.97207069, "Cl": 34.968852682, "Br": 78.91833761,
        "I": 126.904473, "P": 30.97376163, "Na": 22.989769282, "Si": 28.085576,
        "B": 11.009305, "Se": 73.921178, "As": 74.921595}


def _mono(atom) -> float:
    return MONO.get(atom.GetSymbol(), atom.GetMass())


def insilico_ms2(smiles: str) -> dict:
    """One spectrum as {mz, intensity, precursor_mz} for [M+H]+, peaks sorted by m/z."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mz": np.zeros(0, np.float32), "intensity": np.zeros(0, np.float32),
               "precursor_mz": 0.0}
    mol = Chem.AddHs(mol)
    precursor = sum(_mono(a) for a in mol.GetAtoms()) + PROTON
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())

    peaks = [precursor]
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if mol.GetAtomWithIdx(i).GetSymbol() == "H" or mol.GetAtomWithIdx(j).GetSymbol() == "H":
            continue  # breaking a C-H bond loses only a hydrogen, which carries no structure
        rw = Chem.RWMol(mol)
        rw.RemoveBond(i, j)
        frags = Chem.GetMolFrags(rw.GetMol())
        if len(frags) != 2:
            continue  # the cut did not separate the molecule
        for idxs in frags:
            m = sum(_mono(mol.GetAtomWithIdx(k)) for k in idxs)
            if m >= MIN_FRAGMENT:
                peaks.append(m + PROTON)  # charge here, the other fragment leaves as a radical
    for _, mass, needed in LOSSES:
        if all(counts.get(el, 0) >= n for el, n in needed.items()):
            peaks.append(precursor - mass)

    peaks = np.sort(np.unique(np.round(np.array(peaks, np.float32), 3)))
    return {"mz": peaks, "intensity": np.ones(len(peaks), np.float32), "precursor_mz": float(precursor)}


def insilico_table(smiles: list[str]) -> dict:
    """The SpectraTable-shaped CSR arrays for a list of SMILES, ready for the frozen encoder."""
    spectra = [insilico_ms2(s) for s in smiles]
    counts = [len(s["mz"]) for s in spectra]
    return dict(
        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        mz=np.concatenate([s["mz"] for s in spectra] or [np.zeros(0, np.float32)]),
        intensity=np.concatenate([s["intensity"] for s in spectra] or [np.zeros(0, np.float32)]),
        precursor_mz=np.array([s["precursor_mz"] for s in spectra], np.float32),
        adduct=np.full(len(spectra), 1, np.int64),      # [M+H]+
        charge=np.ones(len(spectra), np.int64),
        polarity=np.full(len(spectra), 1, np.int64),   # positive
        instrument=np.full(len(spectra), 3, np.int64),  # Unknown
        ce=np.zeros(len(spectra), np.float32), ce_valid=np.zeros(len(spectra), bool),
    )
