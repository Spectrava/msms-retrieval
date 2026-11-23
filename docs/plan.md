# Plan: from single MS2 spectra to graph-informed representations and molecule inference

The repo started as the paper's retrieval harness (`sna`). What we are adding is a path from
real LC-MS/MS acquisitions in the Spectrava workspace to representations that can be retrieved,
linked to chromatography, and eventually used to infer molecular structure.

## Stage 1 - single-peak MS2 representations (done)

- `sna/encoders/dreams.py` ports the released DreaMS encoder and loads the MassSpecGym checkpoint
  (`checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt`, SHA-256 recorded per run). The encoder
  reads only m/z, intensity and the precursor; the global vector is token 0 (the precursor token)
  after the final LayerNorm. Upstream DreaMS takes the embedding from the last feed-forward block
  instead, so these vectors are **not** interchangeable with published DreaMS scores.
- `scripts/embed_ms2.py` turns a Spectrava traces Parquet into one embedding per MS2 spectrum,
  writing a provenance sidecar (checkpoint hash, dim, max peaks, truncation count, device).
- `sna/retrieval.py` + `scripts/ms2_retrieval.py` do cosine nearest-neighbour retrieval over those
  embeddings and join every hit back to the MS1 trace it came from (sample, RT window, apex m/z,
  apex intensity).

Measured on the 7600 export (2,736 spectra on 2,431 candidates): top-1 similarity mean 0.827,
34.3% of top-1 hits share the precursor m/z within 0.05, 14.6% come from the same sample. The
strongest pairs are the same lipid seen in adjacent ROI traces - the retrieval behaves as expected.

Open items for this stage:
- Only 2,431 of 42,354 candidates have MS2. OT_raw (~12.8k MS2 spectra) is not exported yet.
- Spectra are truncated to 128 peaks (DreaMS' data format); 1,294 of 2,736 were truncated.

## Stage 2 - retrieval against reference libraries

Retrieval is only useful if the query can hit something outside its own run.
- Reference sources: MassSpecGym 1.5 TSV (already the harness input) and the MSP libraries under
  `/mnt/data2/datasets/peaknet_data/mspfiles` (BioMSMS-Pos/Neg-PlaSMA lipid MS2, GCMS databases).
- The MSP files are up to ~400 MB and exceed the frontend's `MSP_LIMITS`, so parsing must happen in
  the backend (`backend/spectrava_api/`), producing a validated, immutable versioned library.
- Index: one Parquet of (library_id, spectrum_id, embedding) plus the metadata the library actually
  carries. No guessed adducts, instruments or collision energies - missing values stay "unknown".

## Stage 3 - graph-informed representations

The single-spectrum embedding ignores everything the workspace already knows. The graph is:
MS1 ROI trace - co-elution - MS1 ROI trace, trace - isolation window - MS2 spectrum,
trace - isotope/adduct - trace, spectrum - reference hit - molecule.

Two cheap steps before any new model:
1. **Context features, not new weights**: pool embeddings of spectra attached to the same trace or
   to co-eluting traces, and compare against the reference-hit pool. This is a retrieval query, not
   training, and it is auditable.
2. **JEPA-style pretraining on the graph**: predict the representation of a trace's MS2 from the
   representation of its co-eluting neighbours (latent prediction, no pixel/peak reconstruction).
   Start with the existing `sna/model.py` retrieval head; only add a graph encoder if the pooled
   baseline in step 1 is already close.

Honesty rule: trace-to-spectrum edges are proximity hypotheses. Any graph model must carry the
edge's m/z/RT tolerance and the count of spectra behind it in its provenance, and a representation
must never be trained on an edge it is later asked to confirm.

Done for step 1 and 2: `sna/graph.py` pools each trace's MS2 embeddings into one unit feature and
draws an edge when two such features reach a cosine threshold; `--within-sample` and
`--between-samples` flag the two edge kinds on or off (both off raises). `scripts/graph_jepa.py`
trains a JEPA head that predicts a ROI's frozen embedding from its neighbours' latents. On the 7600
export at threshold 0.9: 163 edges (7 within-sample, 156 between samples) over 2,431 MS2-measured
ROIs, largest component 12, loss 1.05 -> 0.21. The loss needs a collapse penalty: unpenalised, the
head maps every ROI to one vector (mean pairwise cosine 0.99); penalised, graph latents stay
separated (0.001) and the top-1 apex-m/z agreement is 29.0% vs 31.8% frozen. With only 163 edges
this is a proof of mechanism, not a trained model - OT_raw MS2 and reference-library edges are what
make the graph worth training on.

## Stage 4 - retrieval database

- Storage: DuckDB over Parquet (same shape as the backend), tables `spectra`, `embeddings`,
  `traces`, `edges`, `runs` (checkpoint + provenance).
- Every query returns the neighbour, its similarity, and the trace context, so a reviewer can see
  the chromatography behind a hit in the workbench.
- Versioning: an embedding is keyed by (dataset version, checkpoint SHA-256, max_peaks). Re-running
  with a different checkpoint produces a new version, never an in-place overwrite.

## Stage 5 - molecular structure inference (separate module)

Retrieval gives candidate reference spectra; inferring structure from them is a different problem
and stays out of `sna`:
- rank candidate structures from reference hits (inchikey/SMILES where the library has them),
- report the evidence chain: query spectrum -> reference hit -> structure, with similarity and the
  trace that produced the query,
- never present a predicted structure as an identification without the annotation workflow that
  a reviewer confirms.

A first probe (Stage 5, done): a character LSTM conditioned on the frozen DreaMS embedding, trained
on MassSpecGym train and scored on val, reaches mean Tanimoto 0.155 to the true molecule - below the
0.187 of nearest-neighbour retrieval on the same embeddings, and well above 0.064 from a random
embedding. Structure inference from these embeddings is therefore not ready: retrieval plus the
library's own structures is the honest path, and any generated SMILES must be labelled a hypothesis.

## Gate before any of this reaches the UI

Nothing from this repo enters the Spectrava frontend until it comes through `validateBundle` with
`backendView` provenance. Embeddings are outputs of a pinned checkpoint, not scientific facts.
