# Separability Is Not Alignability

Retrieval experiments for *Separability Is Not Alignability: Representations for
Spectrum-to-Molecule Retrieval*. This release covers retrieval training and evaluation;
encoder pretraining, spectrum simulation, and geometry analyses are outside its scope.

## Setup

Use Python 3.11 or newer and CUDA GPUs. The paper runs used four A100 80 GB GPUs
(one for JESTR); using a different GPU count changes the training recipe.

```bash
pip install -e .              # encoder + MS2 scripts
pip install -e '.[train]'     # also the MassSpecGym training and evaluation harness
cp .env.example .env
```

Fill in `.env` with the MassSpecGym 1.5 TSV, candidate-list directory, output directory,
and checkpoints for the encoders you want to use. Simulated experiments also need
`SNA_AUTONMOL`. Inputs must already be available locally.

Download the molecule towers once:

```bash
python -c "from huggingface_hub import snapshot_download as d; from sna.paths import TOWERS; [d(name, revision=rev) for name, rev in TOWERS.values()]"
```

Prepare real data and pooled molecule embeddings:

```bash
python scripts/prepare.py
CUDA_VISIBLE_DEVICES=0 python -m sna.towers molformer
CUDA_VISIBLE_DEVICES=0 python -m sna.towers chemberta
```

For simulated experiments, also run `python scripts/prepare.py --sim`.
Preparation reuses existing outputs; use a new output directory when changing inputs.
The full prepared data occupies about 29 GB, plus 42 GB for the simulated store.

## Run

```bash
sna list
sna show frozen_msjepa_B
sna run frozen_msjepa_B
sna run jestr
sna run fresh_A_lr1e-4 --set data.epochs=400 --as fresh_A_lr1e-4_ep400
sna all --timeout-hours 12
sna report
```

`run` executes one configuration; JESTR runs both regularized and unregularized arms.
`all` runs unfinished configurations in dependency order and stops on failure.
`--timeout-hours` limits each run; the default, `0`, disables the timeout.
Failed runs restart from scratch. `--force` replaces an existing run.

Configurations are in [sna/experiments.py](sna/experiments.py), with shared defaults in
[sna/config.py](sna/config.py). The paper's validation-selected carrier and fresh learning
rate are fixed in the registry. A configuration with `init` requires that run's completed
checkpoint; all other configurations can run independently.

Outputs go to `$SNA_OUT/runs/<name>/`: configuration, runtime information, logs,
TensorBoard events, checkpoint, test metrics, and per-spectrum ranks. `sna report`
writes `results.json` with molecule-bootstrap intervals and selected paired comparisons
for ladder rungs, training lengths, and JESTR.

Run `pytest tests` after preparing the data and configuring checkpoint paths.
Dependencies are pinned to the tested environment in [pyproject.toml](pyproject.toml).

## MS2 spectra from Spectrava exports

`scripts/embed_ms2.py` encodes the MS2 spectra attached to a Spectrava traces Parquet with the
released DreaMS checkpoint, and `scripts/ms2_retrieval.py` retrieves the nearest spectra and joins
each hit back to the MS1 trace (sample, RT window, apex m/z) that produced it:

```bash
python scripts/embed_ms2.py --traces data/7600_traces.parquet --out data/7600_ms2_embeddings.parquet \
    --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt
python scripts/ms2_retrieval.py --embeddings data/7600_ms2_embeddings.parquet \
    --traces data/7600_traces.parquet --catalog data/7600_catalog.parquet \
    --out data/7600_ms2_matches.parquet --k 10
```

The spectra are attached to candidates by an m/z/RT proximity hypothesis, so every embedding and
match carries that caveat in its provenance. The staged plan for graph-informed representations and
structure inference is in [docs/plan.md](docs/plan.md).

## Reference MS2 libraries

`sna/library.py` reads NIST MSP files (entries without a precursor are skipped; metadata the file
does not carry is recorded as unknown, never guessed), `scripts/embed_library.py` encodes them with
the same DreaMS checkpoint, and `scripts/ms2_retrieval.py --library` matches sample spectra against
the library, comparing only same-polarity pairs:

```bash
python scripts/embed_library.py --msp BioMSMS-Pos-PlaSMA.msp BioMSMS-Neg-PlaSMA.msp \
    --out data/plasma_ms2_embeddings.parquet \
    --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt
python scripts/ms2_retrieval.py --embeddings data/7600_ms2_embeddings.parquet \
    --library data/plasma_ms2_embeddings.parquet --out data/7600_vs_plasma.parquet --k 5
```

Against the PlaSMA plant-metabolite library (3,948 spectra) the 2,736 spectra from the 7600 export
match weakly: top-1 similarity mean 0.53, none reaching 0.9, and even queries whose precursor sits
within 0.05 of a library precursor stay around 0.4-0.65. A control (drop 30% of the peaks of
library spectra, then retrieve) recovers the original spectrum as top-1 for 95.7% of the queries,
so the encoder is robust; the weak cross-library scores reflect the data - the MS2 attachment is a
proximity hypothesis and the 7600 spectra are denser and noisier than the library entries.

## MinHash/LSH baseline (Spectrava/molecule_lsh)

`sna/lsh.py` reproduces the org repo's strategy exactly: m/z digitised into 1000 bins of 1 unit,
intensities normalised to sum 100, datasketch WeightedMinHash (128 hashes, seed 42), library spectra
of the same SMILES merged into one peak set, ranking by exact Jaccard (fraction of equal hash
pairs); the `MinHashLSHForest` only cuts candidates. `scripts/compare_lsh.py` runs both this and
DreaMS cosine on the same spectra (cosine aggregated to the molecule with a max):

```bash
python scripts/compare_lsh.py --traces 7600_traces.parquet --embeddings data/7600_ms2_embeddings.parquet \
    --library-embeddings data/plasma_ms2_embeddings.parquet --msp BioMSMS-Pos-PlaSMA.msp BioMSMS-Neg-PlaSMA.msp
```

On the 7600 spectra vs the PlaSMA library the two strategies disagree: Jaccard top-1 averages 0.40,
cosine 0.54; they pick the same top-1 molecule only 3.3% of the time and their top-1 scores are
uncorrelated (-0.11). The LSH forest returns ~85 candidates per query but misses the exact top-1
about half the time (54% recall). In the library self-test cosine separates related spectra better
(nearest-other mean 0.66, 48 pairs >= 0.9) than Jaccard (0.44, 14 pairs >= 0.9). Dropping 30% of a
query's peaks keeps its top-1 molecule for 41% of queries under Jaccard and 34% under cosine.
Jaccard counts binned-peak overlap - dense, noisy spectra accumulate bins that match almost
anything - while the embedding captures peak-pattern shape. Cosine is the better ranker; LSH is the
cheaper candidate filter (no GPU, no model, signatures in seconds).

## ROI graph and graph JEPA

`sna/graph.py` builds the ROI graph: one node per trace, an edge when two traces' pooled MS2
embeddings reach a cosine threshold. Only traces that actually have MS2 can be linked, and
`--within-sample` / `--no-within-sample` and `--between-samples` / `--no-between-samples` choose
which kind of edge to draw. `scripts/graph_jepa.py` then trains a small JEPA head that predicts a
ROI's frozen MS2 embedding from its neighbours' latents:

```bash
python scripts/graph_jepa.py --embeddings data/7600_ms2_embeddings.parquet \
    --traces data/7600_traces.parquet --catalog data/7600_catalog.parquet \
    --out data/7600_graph_latents.parquet --threshold 0.9
```

On the 7600 export at threshold 0.9: 2,431 of 42,354 ROIs have MS2, 163 edges (7 within-sample,
156 between samples), largest component 12 ROIs, latent prediction loss 1.05 -> 0.21. The loss
carries a collapse penalty (mean pairwise cosine of the latents): without it the model maps every
ROI to one vector. The output Parquet has one row per ROI with degree, component and graph latent
(null for ROIs without MS2), plus `.edges.parquet` and `.provenance.json`.

## Generating molecules from the embeddings

`sna/generate.py` and `scripts/generate_molecules.py` train the smallest honest decoder on top of the
frozen DreaMS embedding: a character-level LSTM that takes the embedding as its first token and
predicts SMILES characters, trained on the MassSpecGym train fold and scored on the val fold.

```bash
python scripts/generate_molecules.py --tsv data/MassSpecGym1.5.tsv \
    --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt \
    --out data/msg_generated.parquet --spectra-embeddings data/7600_ms2_embeddings.parquet
```

On the 19,429 val spectra (val molecules are disjoint from train, so an exact match is impossible by
construction): 73.0% of generated SMILES parse, mean Tanimoto to the true molecule 0.155. The
nearest-neighbour retrieval baseline on the same embeddings scores 0.187, and feeding the decoder the
embedding of a *different* spectrum gives 0.064 - the embedding does steer the decoder, but not
enough to beat plain retrieval. On training spectra the decoder reproduces its own molecule 26.9% of
the time, so part of what it "knows" is memorisation. For the 2,736 MS2 spectra of the 7600 export it
proposes 1,804 parseable structures, and only 31 of them have a monoisotopic mass consistent with the
observed precursor (±0.05, as [M+H]+ or [M+Na]+).

The DreaMS embedding is discriminative: it is a good retrieval key and a weak generative one.
Generated SMILES are hypotheses to be reviewed, never identifications.

## Aligning the spectrum space with a molecule tower

`scripts/align_spaces.py` fits a map from the DreaMS embedding to the pooled embedding of the
matching molecule in a frozen SMILES tower (MoLFormer XL, `--tower chemberta` for the other) and
measures what the alignment buys:

```bash
python scripts/align_spaces.py --tsv data/MassSpecGym1.5.tsv --map linear   # or --map mlp, --map contrastive, --probe
```

| | linear map | two-layer MLP | contrastive heads |
|---|---|---|---|
| cosine(mapped, true molecule) | 0.817 | 0.820 | 0.375 |
| same-space random molecule | 0.727 | 0.715 | 0.034 |
| nearest train molecule | 0.159 | 0.177 | 0.173 |
| raw spectrum nearest neighbour | 0.184 | 0.184 | 0.184 |
| ceiling: nearest train molecule to the *true* molecule | 0.386 | 0.386 | 0.386 |
| nearest among the val molecules themselves | 0.176 (0.9% exact) | 0.199 (1.5% exact) | 0.247 (5.7% exact) |

The regression maps look good in absolute terms (0.817 against a 0.727 random-molecule floor), but the pooled MoLFormer
space is anisotropic: the nearest *other* molecule to any given molecule sits at cosine 0.888 (Tanimoto 0.456). The
mapped latent is only 0.09 above the floor, so it is further from the molecule it came from than that molecule is from
its nearest neighbour - which is why retrieval does not improve. The contrastive heads win differently: they throw away most of the space (a random pair is only
0.034) so the true molecule stands out by a factor of eleven, and they recover the exact molecule 5.7% of the time -
but the directions they keep are the discriminative ones, so retrieving a *similar* molecule is still no better than
cosine in the raw spectrum space.

`--probe` trains the same char-LSTM on the molecule latents instead of the spectra: 35.3% of the
outputs parse, mean Tanimoto 0.177 on unseen molecules, and 0.5% exact reproduction of the molecules
it was trained on. A pooled transformer embedding is not a generative latent space, so the
bottleneck is the latent, not the decoder.

The graph autoencoder in [mcunow/graph-matching](https://github.com/mcunow/graph-matching) has the
right shape - latent plus one-shot graph decoder with a permutation-invariant loss - but its decoder
is fixed to 9 heavy atoms and C/N/O/F, and only 2.4% of MassSpecGym molecules have ≤9 heavy atoms
(2.1% also contain nothing else). It cannot cover this chemistry without rewriting the decoder.

An aligned space is therefore useful for *ranking* candidate structures, not for producing them.

## License

[MIT](LICENSE). Upstream acknowledgments and license notices are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). External models and datasets retain their
own licenses.
