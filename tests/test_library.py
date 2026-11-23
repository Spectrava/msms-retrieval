"""MSP parsing and polarity-aware retrieval against a reference library."""
import numpy as np
import pandas as pd

from sna.library import read_msp
from sna.retrieval import library_match

MSP = """NAME: PC 34:1; level-1
PRECURSORMZ: 782.577
PRECURSORTYPE: [M+H]+
FORMULA: C52H104NO5P
SMILES: CCCCCCCCC
IONMODE: Positive
Num Peaks: 3
782.577\t100
555.42\t40
341.25\t20

NAME: alanine
EXACTMASS: 89.047
IONMODE: Positive
Num Peaks: 2
85\t3158
89\t964

NAME: FA 18:0
PRECURSORMZ: 284.24
IONMODE: Negative
Num Peaks: 2
283.24\t100
241.2\t30
"""


def test_read_msp_keeps_ms2_and_skips_precursorless_entries(tmp_path):
    path = tmp_path / "lib.msp"
    path.write_text(MSP)
    spectra = read_msp(path)
    assert [s["name"] for s in spectra] == ["PC 34:1; level-1", "FA 18:0"]  # alanine is EI, no precursor
    assert spectra[0]["precursor_mz"] == 782.577 and len(spectra[0]["mz"]) == 3
    assert spectra[0]["polarity"] == 1 and spectra[1]["polarity"] == 2  # POLARITIES indices
    assert spectra[0]["adduct"] == 1 and spectra[1]["adduct"] == 0  # [M+H]+ known, other unknown


def test_library_match_never_crosses_polarity():
    query = pd.DataFrame({"candidate_id": ["q"], "spectrum_index": [0], "precursor_mz": [284.24],
                          "rt": [1.0], "polarity": [2], "embedding": [[1.0, 0.0]]})
    library = pd.DataFrame({"library": ["l", "l"], "name": ["identical but positive", "same polarity"],
                            "formula": ["", ""], "smiles": ["", ""], "precursor_mz": [284.24, 284.24],
                            "polarity": [1, 2], "embedding": [[1.0, 0.0], [1.0, 0.0]]})
    matches = library_match(query, library, k=2)
    assert len(matches) == 2  # k per query
    top = matches.iloc[0]
    assert top["name"] == "same polarity" and np.isclose(top["similarity"], 1.0) and top["same_precursor"]
    assert matches.iloc[1]["similarity"] < 0  # the identical cross-polarity entry is excluded, not ranked


def test_library_match_returns_k_hits_per_query():
    query = pd.DataFrame({"candidate_id": ["q1", "q2"], "spectrum_index": [0, 1], "precursor_mz": [100.0, 200.0],
                          "rt": [1.0, 2.0], "polarity": [1, 1], "embedding": [[1.0, 0.0], [0.0, 1.0]]})
    library = pd.DataFrame({"library": ["l"] * 3, "name": ["a", "b", "c"], "formula": [""] * 3,
                            "smiles": [""] * 3, "precursor_mz": [100.0, 200.05, 300.0], "polarity": [1, 1, 1],
                            "embedding": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]})
    matches = library_match(query, library, k=3)
    assert len(matches) == 6
    assert matches.iloc[0]["name"] == "a" and matches.iloc[3]["name"] == "c"
    assert not matches.iloc[1]["same_precursor"]  # 100.0 vs 200.05 is not the same precursor
