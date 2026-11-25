"""The seeded decoder learns to reproduce a SMILES and only emits vocabulary characters."""
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from sna.generate import SmilesGenerator


def test_decoder_learns_and_generates():
    torch.manual_seed(0)
    vocab = sorted(set("CN=O"))
    model = SmilesGenerator(4, vocab, hidden=16, layers=1)
    z = torch.randn(8, 4)
    chars = torch.tensor([[vocab.index(c) for c in "CNO=O"] for _ in range(8)])
    first = model.loss(z, chars).item()
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(30):
        loss = model.loss(z, chars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first
    generated = model.generate(z, max_len=12)[0]
    assert all(c in vocab for c in generated)


def test_mass_constrained_generation_hits_the_precursor_and_refuses_to_exceed_it():
    """The beam search must land on the target mass, and a target too light for the vocabulary
    must force it to stop early rather than emit an impossible molecule."""
    torch.manual_seed(0)
    vocab = sorted(set("CO"))
    model = SmilesGenerator(4, vocab, hidden=16, layers=1)
    z = torch.randn(6, 4)
    chars = torch.tensor([[vocab.index(c) for c in "CCO"] for _ in range(6)])
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(40):
        loss = model.loss(z, chars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ethanol = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles("CCO"))
    out = model.generate_mass_constrained(z, ethanol, beams=4, max_len=20)
    assert all(smi and matched and abs(m - ethanol) <= 0.05 for smi, matched, m in out)

    methane = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles("C"))  # 16.0313
    out = model.generate_mass_constrained(z, methane, beams=1, max_len=20)
    assert all(smi == "C" and matched for smi, matched, m in out)  # a second heavy atom is too heavy

    out = model.generate_mass_constrained(z, 500.0, beams=4, max_len=20)
    assert all(not matched for smi, matched, m in out)  # nothing in this vocabulary can reach 500 Da


def test_mass_constraint_tracks_rings_aromatics_and_halogens():
    """Chlorobenzene exercises aromatic atoms, ring-closure digits and the two-character Cl, whose
    masses and valences the beam search tracks by hand."""
    torch.manual_seed(1)
    vocab = sorted(set("Clc1"))
    model = SmilesGenerator(4, vocab, hidden=16, layers=1)
    z = torch.randn(4, 4)
    chars = torch.tensor([[vocab.index(c) for c in "Clc1ccccc1"] for _ in range(4)])
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(60):
        loss = model.loss(z, chars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    chlorobenzene = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles("Clc1ccccc1"))  # 112.0528
    out = model.generate_mass_constrained(z, chlorobenzene, beams=4, max_len=20)
    assert all(smi == "Clc1ccccc1" and matched for smi, matched, m in out)

    benzene = rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles("c1ccccc1"))  # 78.1116
    out = model.generate_mass_constrained(z, benzene, beams=4, max_len=20)
    assert all(m <= benzene + 0.05 for smi, matched, m in out)  # the chlorine is pruned before it is emitted
