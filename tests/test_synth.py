"""The rule-based generator is deterministic, its precursor is the exact [M+H]+ mass, no peak is
above it, every peak is a fragment or a neutral loss the formula can support, and a different
molecule gives a different peak list."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from sna.synth import MIN_FRAGMENT, PROTON, insilico_ms2, insilico_table


def test_precursor_is_the_exact_mass():
    for smi in ("CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "[Na+].[Cl-]"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        got = insilico_ms2(smi)["precursor_mz"]
        assert abs(got - rdMolDescriptors.CalcExactMolWt(mol) - PROTON) < 1e-3, smi


def test_peaks_sorted_below_the_precursor():
    sp = insilico_ms2("CCOC(=O)C1=CC=CC=C1")
    assert sp["mz"].size > 1 and (sp["mz"][:-1] < sp["mz"][1:]).all()
    assert abs(sp["mz"].max() - sp["precursor_mz"]) < 1e-3
    assert (sp["mz"] >= MIN_FRAGMENT).all()
    # the benzoyl cation C6H5C=O+ (106.041) comes from cleaving the acyl-phenyl bond
    assert np.isclose(sp["mz"], 106.041).any(), sp["mz"]


def test_losses_only_when_the_formula_supports_them():
    assert np.isclose(insilico_ms2("CC(=O)Oc1ccccc1C(=O)O")["mz"], 181.0495 - 18.0106).any()
    assert not np.isclose(insilico_ms2("c1ccccc1")["mz"], 79.0545 - 18.0106).any()


def test_different_molecules_differ():
    assert set(insilico_ms2("CCO")["mz"]) != set(insilico_ms2("CCCO")["mz"])
    assert len(insilico_ms2("c1ccccc1")["mz"]) == 1  # no acyclic bond to break, no losses


def test_bad_smiles_gives_an_empty_spectrum():
    sp = insilico_ms2("not a smiles")
    assert sp["mz"].size == 0 and sp["precursor_mz"] == 0.0


def test_table_is_a_valid_csr():
    t = insilico_table(["CCO", "CCCC", "not a smiles"])
    assert len(t["precursor_mz"]) == 3 and t["offsets"][0] == 0
    assert t["offsets"][-1] == len(t["mz"]) == len(t["intensity"])
    assert np.diff(t["offsets"]).min() >= 0
    assert (t["mz"] >= 0).all() and (t["intensity"] <= 1).all()
