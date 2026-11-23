"""Embed a reference MS2 library (NIST MSP) with the pinned DreaMS checkpoint.

    python scripts/embed_library.py --msp /mnt/data2/datasets/peaknet_data/mspfiles/BioMSMS-Pos-PlaSMA.msp \
        --out data/plasma_ms2_embeddings.parquet --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt

Writes one row per library spectrum (library, spectrum_index, name, formula, smiles, inchikey,
precursor_mz, polarity, n_peaks, truncated, embedding) plus a provenance sidecar. Entries without
a precursor (GCMS/EI) are skipped, not embedded with a made-up one; identifications are taken as
the library states them and are not verified here.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from embed_ms2 import sha256_file  # scripts/ is on sys.path when this runs as a script
from sna.encoders.dreams import load_dreams
from sna.library import read_msp
from sna.spectra import SpectraTable, arrays_from_spectra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--msp", required=True, nargs="+", help="one or more MSP library files")
    ap.add_argument("--out", required=True, help="output embeddings Parquet")
    ap.add_argument("--ckpt", required=True, help="pinned encoder checkpoint")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    encoder = load_dreams(args.ckpt).to(args.device).eval()
    spectra, skipped = [], 0
    for path in args.msp:
        entries = read_msp(path)
        for s in entries:
            s["library"] = os.path.basename(path)
        spectra.extend(entries)
        entries_count = sum(1 for line in open(path) if line.strip().lower().startswith("num peaks"))
        skipped += entries_count - len(entries)
    if not spectra:
        raise SystemExit(f"no MS2 spectra (with a precursor) in {args.msp}")

    table = SpectraTable(arrays_from_spectra(spectra), device=args.device)
    vectors, t0 = [], time.time()
    with torch.inference_mode():
        for start in range(0, len(spectra), args.batch):
            idx = torch.arange(start, min(start + args.batch, len(spectra)), device=args.device)
            vectors.append(encoder(table.batch(idx, encoder.max_peaks))[0].float().cpu().numpy())
    emb = np.concatenate(vectors)

    out = pd.DataFrame({
        "library": [s["library"] for s in spectra],
        "spectrum_index": list(range(len(spectra))),
        "name": [s["name"] for s in spectra],
        "formula": [s["formula"] for s in spectra],
        "smiles": [s["smiles"] for s in spectra],
        "inchikey": [s["inchikey"] for s in spectra],
        "ontology": [s["ontology"] for s in spectra],
        "precursor_mz": [s["precursor_mz"] for s in spectra],
        "polarity": [s["polarity"] for s in spectra],
        "n_peaks": [len(s["mz"]) for s in spectra],
        "truncated": [len(s["mz"]) > encoder.max_peaks for s in spectra],
        "embedding": [v.tolist() for v in emb],
    })
    out.to_parquet(args.out)
    prov = {
        "encoder": "DreaMS", "checkpoint": args.ckpt, "checkpointSha256": sha256_file(args.ckpt),
        "dim": int(emb.shape[1]), "maxPeaks": int(encoder.max_peaks), "device": args.device,
        "torch": torch.__version__, "files": args.msp, "spectra": len(spectra),
        "skippedNoPrecursor": skipped, "truncatedSpectra": int(out["truncated"].sum()),
        "seconds": round(time.time() - t0, 2),
        "note": "Compound names/formulas/SMILES are as stated by the library (annotation levels "
                "vary, e.g. 'Formula predicted'); they are not verified here.",
    }
    with open(f"{args.out}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))
    print(f"wrote {len(out)} library embeddings to {args.out}")


if __name__ == "__main__":
    main()
