"""Reference MS2 libraries in NIST MSP format.

An MSP entry is a header of KEY: value lines, "Num Peaks: N", then N peak lines. Entries without
PRECURSORMZ (GCMS/EI spectra) are not MS2 - DreaMS needs a precursor - so they are skipped rather
than embedded with a made-up one. Fields the library does not carry stay unknown.
"""
from __future__ import annotations

import numpy as np

from sna.spectra import ADDUCTS, POLARITIES


def read_msp(path) -> list[dict]:
    """One dict per MS2 spectrum in the file, in file order."""
    spectra, block = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                block.append(line)
            elif block:
                spectra.append(_parse(block))
                block = []
    if block:
        spectra.append(_parse(block))
    return [s for s in spectra if s is not None]


def _parse(block: list[str]) -> dict | None:
    header = {}
    i = 0
    while i < len(block) and not block[i].lower().startswith("num peaks"):
        key, _, value = block[i].partition(":")
        header[key.strip().upper()] = value.strip()
        i += 1
    if "PRECURSORMZ" not in header or i == len(block):
        return None  # EI spectrum or malformed entry: no precursor, not MS2
    try:
        n = int(block[i].partition(":")[2])
    except ValueError:
        return None
    mz, intensity = [], []
    for line in block[i + 1:i + 1 + n]:
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                peak, value = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if peak > 0 and value > 0:
                mz.append(peak), intensity.append(value)
    if not mz:
        return None
    ionmode = header.get("IONMODE", "").lower()
    return dict(
        name=header.get("NAME", ""), precursor_mz=float(header["PRECURSORMZ"]),
        formula=header.get("FORMULA", ""), smiles=header.get("SMILES", ""),
        inchikey=header.get("INCHIKEY", ""), ontology=header.get("ONTOLOGY", ""),
        rt=float(header.get("RETENTIONTIME", 0.0) or 0.0),
        polarity=POLARITIES.index(ionmode) if ionmode in POLARITIES else 0,
        adduct=ADDUCTS.index(header.get("PRECURSORTYPE", "")) if header.get("PRECURSORTYPE") in ADDUCTS else 0,
        mz=np.asarray(mz, np.float32), intensity=np.asarray(intensity, np.float32),
    )