"""Encode MS2 spectra from a Spectrava traces Parquet with a pinned encoder checkpoint.

Input rows are the Spectrava export shape: candidate_id, source_sample, ms2_json (a JSON array
of spectra, or [None]). Only spectra that exist are encoded; nothing is synthesised and no
metadata is guessed - DreaMS reads only m/z, intensity and the precursor, so the adduct/CE/
instrument fields are set to the vocabulary's "unknown" entries rather than invented.

    python scripts/embed_ms2.py --traces data/7600_traces.parquet --out data/7600_ms2_embeddings.parquet \
        --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt

Writes one row per spectrum (candidate_id, spectrum_index, precursor_mz, n_peaks, embedding)
plus a provenance sidecar next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from sna.encoders.dreams import load_dreams
from sna.spectra import POLARITIES, SpectraTable, arrays_from_spectra


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ms2(traces_path: str) -> list[dict]:
    """One dict per MS2 spectrum that actually exists, in file order."""
    spectra = []
    for row in pd.read_parquet(traces_path)[["candidate_id", "ms2_json"]].itertuples():
        for i, spec in enumerate(json.loads(row.ms2_json or "[]")):
            if not spec:
                continue
            mz = np.asarray(spec.get("mz", []), np.float32)
            intensity = np.asarray(spec.get("intensity", []), np.float32)
            if mz.size == 0 or mz.size != intensity.size:
                continue
            keep = intensity > 0
            if not keep.any():
                continue
            spectra.append(dict(candidate_id=row.candidate_id, spectrum_index=i,
                               precursor_mz=float(spec.get("precursorMz") or 0.0), rt=float(spec.get("rt") or 0.0),
                               mz=mz[keep], intensity=intensity[keep],
                               polarity=POLARITIES.index(spec["polarity"]) if spec.get("polarity") in POLARITIES else 0))
    return spectra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True, help="Spectrava traces Parquet (candidate_id, ms2_json)")
    ap.add_argument("--out", required=True, help="output embeddings Parquet")
    ap.add_argument("--ckpt", required=True, help="pinned encoder checkpoint")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    encoder = load_dreams(args.ckpt).to(args.device).eval()
    spectra = read_ms2(args.traces)
    if not spectra:
        raise SystemExit(f"no MS2 spectra in {args.traces}")
    table = SpectraTable(arrays_from_spectra(spectra), device=args.device)

    vectors, t0 = [], time.time()
    with torch.inference_mode():
        for start in range(0, len(spectra), args.batch):
            idx = torch.arange(start, min(start + args.batch, len(spectra)), device=args.device)
            batch = table.batch(idx, encoder.max_peaks)  # no augment: inference only
            vectors.append(encoder(batch)[0].float().cpu().numpy())
    emb = np.concatenate(vectors)

    out = pd.DataFrame({
        "candidate_id": [s["candidate_id"] for s in spectra],
        "spectrum_index": [s["spectrum_index"] for s in spectra],
        "precursor_mz": [s["precursor_mz"] for s in spectra],
        "rt": [s["rt"] for s in spectra],
        "polarity": [s["polarity"] for s in spectra],
        "n_peaks": [len(s["mz"]) for s in spectra],
        "truncated": [len(s["mz"]) > encoder.max_peaks for s in spectra],
        "embedding": [v.tolist() for v in emb],
    })
    out.to_parquet(args.out)
    prov = {
        "encoder": "DreaMS", "checkpoint": args.ckpt, "checkpointSha256": sha256_file(args.ckpt),
        "dim": int(emb.shape[1]), "maxPeaks": int(encoder.max_peaks), "device": args.device,
        "torch": torch.__version__, "spectra": len(spectra), "truncatedSpectra": int(out["truncated"].sum()),
        "seconds": round(time.time() - t0, 2),
        "note": "MS2 spectra are attached to candidates by an m/z/RT proximity hypothesis, not a "
                "confirmed precursor-fragment assignment; embeddings inherit that uncertainty.",
    }
    with open(f"{args.out}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))
    print(f"wrote {len(out)} embeddings to {args.out}")


if __name__ == "__main__":
    main()
