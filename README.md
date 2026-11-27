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

### Pinning the mass with MS1

The precursor mass is known, so `SmilesGenerator.generate_mass_constrained` beam-searches the decoder
instead of decoding greedily. Each prefix carries the heavy-atom mass it has emitted and the free
valence left on those atoms, so a prefix is dropped as soon as it is already heavier than the
precursor allows, or can no longer reach it, or cannot be a SMILES at all (`((`, `--`, a ring closed
twice, unbalanced brackets). `<end>` is always a candidate, so a beam can close when the greedy
continuation is infeasible.

```bash
python scripts/generate_molecules.py --tsv data/MassSpecGym1.5.tsv \
    --ckpt checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt \
    --out data/msg_generated.parquet --spectra-embeddings data/7600_ms2_embeddings.parquet \
    --mass-constrained --beams 16
```

On the val fold this lifts the parse rate from 73.0% to 100% and the share of proposals whose exact
mass equals the true molecule's from 0% to 17.6% - but mean Tanimoto stays at 0.138, still under the
0.187 retrieval baseline, and only 1% of the mass-matched molecules even have the right formula. A
monoisotopic mass pinned to ±0.05 Da admits many formulas: C32H41N5O5 (575.3108) and C25H45N5O10
(575.3166) are 0.006 apart. Widening the beam buys more mass hits and no structure - beams 4/16/32 on
1,024 val spectra give 8.5%/24.8%/36.1% mass matches at mean Tanimoto 0.089/0.101/0.102.

For the 2,736 MS2 spectra of the 7600 export the constraint is what makes the proposals usable at all:
986 of them are consistent with the observed precursor as [M+H]+ or [M+Na]+, against 31 from greedy
decoding. They are still hypotheses - the mass narrows the space, it does not identify a molecule.

The DreaMS embedding is discriminative: it is a good retrieval key and a weak generative one.
Generated SMILES are hypotheses to be reviewed, never identifications.

## Training a generative embedding

`sna/genembed.py` and `scripts/gen_embed.py` build the embedding generatively instead of
retrieval-wise: the frozen DreaMS spectrum embedding is pushed through an encoder into a bottleneck
latent, and a four-layer causal transformer is trained *jointly* to write the SMILES of that
spectrum from the latent. The latent is invertible by construction - it is the first token of the
decoder's stream - so the test is whether the spectrum space carries enough structure to be written
back out, not whether a similar spectrum can be found.

```bash
python scripts/gen_embed.py --tsv data/MassSpecGym1.5.tsv --latent 256 --epochs 10 \
    --candidates 8 --out data/generative_embedding_latent256.pt
```

Mean Tanimoto between the generated and the true molecule:

| latent | training spectra | held-out spectra | wrong spectrum | retrieval ceiling |
|---|---|---|---|---|
| 128 | 0.664 | 0.165 | - | 0.187 |
| 256 | 0.697 | 0.164 | 0.109 | 0.188 |

87% of training and 64% of held-out generated SMILES parse; 41% of training spectra reproduce their
own molecule exactly (held-out molecules are disjoint from training, so an exact match is
impossible by construction). The latent is carrying something - decoding the *wrong* spectrum's
latent scores 0.109 against 0.164 - but its width is not the limit (128 and 256 are
indistinguishable), and generation never reaches the retrieval ceiling: copying the nearest training
spectrum's molecule scores 0.188. The model memorises the training set (next-character loss 0.041
train, 0.545 held-out) and on unseen molecules writes a plausible structure no better than the
char-LSTM conditioned on the raw embedding (0.159).

Sampling only helps as a filter: with eight temperature-1.0 candidates per spectrum, 9,748 of
21,888 proposals parse, and 116 of the 2,736 MS2 spectra in the 7600 export have at least one
proposal whose monoisotopic mass fits the observed precursor (±0.05, as [M+H]+ or [M+Na]+), against
21 for the greedy generation alone. That is a shortlist for a reviewer, not an identification.

### Training the encoder from the spectrum itself

`--from-peaks` drops DreaMS entirely: the input is the spectrum binned into 1000 one-unit m/z bins,
the encoder is a single linear layer, and the latent is additionally scored on rebuilding those bins,
so it is checked against the observation and not only against a label.

| latent | training spectra | held-out spectra | wrong spectrum | retrieval ceiling | rebuilds the spectrum (cosine) |
|---|---|---|---|---|---|
| 256 | 0.456 | 0.120 | 0.103 | 0.114 | 0.462 |
| 512 | 0.501 | 0.118 | 0.099 | 0.114 | 0.573 |

Doubling the latent buys reconstruction (0.462 -> 0.573) and nothing for generation (0.120 -> 0.118):
generation sits exactly at the ceiling of the representation it is built on (nearest training
spectrum, 0.114), and the char-LSTM on the same bins scores 0.115. The raw peak vector is a weaker
substrate than the DreaMS embedding - its ceiling is 0.114 against 0.188 - so a generative decoder
fitted on top of it cannot reach what the frozen features already support.

The honest reading: the limit is the substrate, not the decoder. Generation tracks the retrieval
ceiling of whatever features it is built on, and neither a wider latent nor a spectrum-reconstruction
objective breaks through it. If generation is the goal, the encoder has to be trained on a
structure-conditioned objective - predict the spectrum from the structure - rather than a decoder
being fitted onto a discriminative one.

## A JEPA encoder for the raw spectrum

`sna/jepa.py` and `scripts/train_jepa.py` train an encoder the JEPA way: contiguous m/z windows are
deleted from the binned spectrum, and the encoder must predict, in latent space, the latent an EMA
teacher would have assigned to the *whole* spectrum. Nothing is reconstructed and no molecule enters
the loss - the encoder is scored only on filling in the peaks it was not shown.

```bash
python scripts/train_jepa.py --tsv data/MassSpecGym1.5.tsv --epochs 20 --latent 256 --hidden 512
```

Held-out masked-latent error 0.092. A linear probe from the latent back to the 1000 bins reaches
cosine 0.640 - better than the latents that were explicitly trained to reconstruct the spectrum
(0.462 and 0.573 above) - and its nearest-neighbour ceiling is 0.130, against 0.114 for the raw bins
and 0.188 for DreaMS.

### A decoder that reads the latent as memory

`sna.genembed.LatentDecoder` expands the latent into eight memory tokens that every character
position cross-attends to, rather than handing it to the stream as a single leading token: after a
few characters a causal decoder can predict the next character from the characters alone, so a
one-token prefix stops mattering. Generation (greedy or sampled) is one code path shared with the
seed-token baseline.

The loss is computed only over the real characters and the stop token. Scoring the trailing padding
too made the training loss read 0.057 while the stop position cost 11.4 and the character positions
4.95 - the decoder had learned to emit `END` everywhere and, at inference, produced runs of `C`.

Same decoder, same hyperparameters, three substrates (19,429 held-out spectra):

| substrate | dim | parses | Tanimoto | wrong spectrum | ceiling |
|---|---|---|---|---|---|
| JEPA latent | 256 | 81.7% | 0.122 | 0.094 | 0.130 |
| raw 1000 bins | 1000 | 82.5% | 0.122 | 0.095 | 0.114 |
| DreaMS embedding | 1024 | 62.3% | 0.171 | 0.094 | 0.188 |

Masked latent prediction does make a better latent for retrieval and for reconstructing the
spectrum, but it does not make a better latent for writing a molecule: generation from the JEPA
latent is indistinguishable from generation from the raw bins it was trained on, and both now sit at
or just above their own ceiling. DreaMS still generates the closest molecules, though fewer of them
parse. The shuffled-latent column is the chemistry-prior floor: a decoder given the wrong spectrum
still writes a plausible structure, and anything within ~0.03 of it is not information from the
spectrum.

### How much chemistry a latent actually carries

`scripts/latent_audit.py` skips the decoder entirely: a ridge map from each latent to the element
counts and the 2048-bit Morgan fingerprint of the matching molecule, scored on held-out spectra.

| substrate | formula exact | heavy atoms MAE | readout fingerprint | readout ceiling |
|---|---|---|---|---|
| raw 1000 bins | 0.0% | 33.6 | 0.002 | 0.126 |
| JEPA latent | 0.0% | 13.6 | 0.113 | 0.108 |
| autoencoder latent | 0.0% | 11.7 | 0.133 | 0.123 |
| DreaMS embedding | 0.0% | 5.3 | 0.183 | 0.161 |

A random pair of molecules in this set scores 0.112, so a linear readout recovers barely more than
chance - and the raw bins recover less than chance. λ from 1 to 10,000 changes the number by 0.003,
so this is not underfitting. No latent, discriminative or generative, holds the molecule in a form a
linear map can read; the decoder's 0.171 is the first thing in the pipeline that does better than a
linear probe.

Replacing the ridge with a 512-hidden readout (`--mlp`) changes nothing that matters: DreaMS moves
from 0.183 to 0.188 fingerprint and 0.161 to 0.178 ceiling, the other three stay within 0.006 of
their linear scores and none beats the 0.112 random floor. The chemistry is not in these
embeddings in a form any readout can reach, so the encoder - not the decoder - is the thing to
change.

### Decoder capacity is not the bottleneck

Repeating the DreaMS arm with a 512-hidden, six-layer decoder (10 epochs) gives 51.8% parse and
0.167 Tanimoto against the 256-hidden, four-layer decoder's 62.3% and 0.171 - the bigger model is
slightly worse, and both sit under the 0.188 ceiling. Training more parameters on the same substrate
buys nothing; the substrate is the limit.

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

## Matching the two geometries, and filling in the spectra space

`scripts/align_joint.py` is the joint benchmark: frozen DreaMS spectra against a frozen molecule
tower (`--encoder moljepa|molformer|chemberta`), scored against the 25046 unique train molecules.

```bash
python scripts/align_joint.py --tsv data/MassSpecGym1.5.tsv --encoder moljepa --dims 256 --seeds 3
python scripts/align_joint.py --encoder molformer --methods gw --gw-sub 2000 --gw-lam 0.1 --synthetic
```

### Gromov-Wasserstein: the geometries do not match

CKA, CCA and Procrustes all assume the two spaces share a coordinate system. An entropic
Gromov-Wasserstein coupling (`sna/joint.py: gw_coupling`) does not: it only asks for a
correspondence `Pi` under which *distances inside* each space agree, so it can match a 1024-d
spectrum latent to a 512-d molecule latent without ever naming a shared axis.

Mol-JEPA, pool 2000 train molecules, held-out test fold, `lam=0.01`, 2 seeds (each Sinkhorn sweep
is `O(n^3)`, so the pool stays small):

```bash
python scripts/align_joint.py --tsv data/MassSpecGym1.5.tsv --encoder moljepa --fold test \
  --methods gw --gw-sub 2000 --gw-lam 0.01 --gw-anneal 1.0 --gw-method entropic,hard --seeds 2
```

| coupling | GW cost (seed 0 / seed 1) | mass on the true pair | top-1 of the coupling | retrieval top-1 |
|---|---|---|---|---|
| identity (spectrum ↔ its own molecule) | 221971 / 221414 | 1.00x uniform | 1.000 | 0.154 / 0.159 |
| uniform | 273635 / 269131 | 1.00x uniform | — | — |
| entropic, `lam=0.01` | 132365 / 124447 | 0.58x / 0.23x | 0.000 / 0.001 | 0.084 / 0.087 |
| annealed 1.0 → 0.01 | 132309 / 127336 | 0.27x / 0.26x | 0.000 / 0.000 | 0.082 / 0.090 |
| hard (Hungarian, no temperature) | 202156 / 193653 | 2 / 3 fixed points | 0.001 / 0.002 | 0.091 / 0.088 |

The minimum of the GW cost is nowhere near the true pairing: the cheapest correspondence found is
40% cheaper than pairing each spectrum with its own molecule, and it puts *less* mass on the true
pairs than a random pairing does. Transporting through it retrieves worse than plain cosine in the
same pool (0.086 against 0.156).

Annealing is not the escape hatch it was reported to be. Cooling `lam` from 1.0 to 0.01 helps on
one seed (132309 against 132365) and hurts on the other (127336 against 124447): the schedule finds
a different basin, not a better one. The hard-assignment coupling - the Hungarian algorithm on the
linearised cross term, warm-started from uniform rather than from the identity, since the identity
is the hypothesis being tested - is the only variant that concentrates mass (a permutation with 2
or 3 fixed points), and it is still half again as expensive as the entropic optimum.

The same battery run against geometries where the answer is known (`--gw-control`, Mol-JEPA, test
fold, pool 1000, seed 0, `lam=0.01`) says where the method is capable and where it is not:

```bash
python scripts/align_joint.py --tsv data/MassSpecGym1.5.tsv --encoder moljepa --fold test \
  --methods gw --gw-sub 50 --gw-lam 0.01 --gw-method entropic,hard \
  --gw-control --gw-control-sub 1000 --seeds 1
```

| geometry pair | entropic cost | identity | uniform | mass on the true pair | coupling top-1 | hard cost | hard top-1 |
|---|---|---|---|---|---|---|---|
| molecule latent vs a noisy copy of itself | 20480 | 12233 | 64385 | 220.0x | 0.890 | 39089 | 0.011 |
| molecule latent vs Tanimoto chemistry | 39965 | 40702 | 53449 | 0.68x | 0.000 | 43277 | 0.001 |
| spectrum latent vs Tanimoto chemistry | 17581 | 19667 | 25974 | 0.84x | 0.000 | 20377 | 0.000 |

Read down the first column: the coupling recovers a correspondence when one exists. Given a
geometry and a perturbed copy of itself it puts 220x uniform mass on the right molecule and 89% of
rows argmax to it. So the failure on the real spectrum-molecule pair is the data, not the transport.

The other two rows say how much chemistry each latent carries, and that is encoder-dependent. With
MoLFormer XL (pool 500) pairing the molecule latent with the Tanimoto geometry of its own fingerprint
is cheaper than any coupling the solver finds (10293 against 11734) and still concentrates 2.3x on
the true pairs; with Mol-JEPA it is not (40702 against 39965) and the coupling puts *less* mass on
the true pairs than a random pairing does. In both encoders the spectrum latent carries less: its
identity is within 12% of the fixed point and its true-pair mass is below chance.

The controls also bound what "the GW optimum" means. The linearised Sinkhorn iteration is a local
search, not a minimiser: on the self control its fixed point costs 67% more than the identity it
just recovered. The hard Hungarian variant is weaker still - it recovers an exact isometry in the
unit test but not a noisy one (top-1 0.011, cost 39089 against 12233). It is a diagnostic, not a
competitor.

So the true molecule pair is not the minimum of the GW cost between these two latent geometries:
the frozen spectrum space and the frozen molecule space are not isomorphic at the level of
molecule pairs.

The geometry-preserving projector is the one fitted map that does order pairs better: pairing plus
the cosine-Gram term plus SIGReg gives a Spearman of 0.263 between latent proximity and Tanimoto
similarity, against 0.142 for raw cosine - and pairing alone, without the geometry term, gives
-0.059. It buys that ordering by throwing most of the space away (effective rank 56 against 676),
and better ordering still does not translate into retrieval (0.148 top-1 against 0.184): the gain is
in the middle of the ranking, not at the top.

### Rule-based synthetic spectra do not help

Populating the sparse spectra space needs a generator that is independent of the encoder being
aligned. A decoder trained on real spectra - the char-LSTM already in `data/` - makes a
"synthetic" spectrum a function of the molecule embedding, so every alignment measured through it
is the decoder, not chemistry. `sna/synth.py` is therefore pure rules: the `[M+H]+` precursor, one
peak per acyclic single-bond cleavage at the fragment's monoisotopic mass plus a proton, and the
H₂O/NH₃/CO₂/CO losses the formula can support. 44475 spectra, mean 21.5 peaks (real MassSpecGym
spectra average ~36), generated in 14 s per 5000 molecules; 814 of 5000 molecules collide into an
identical peak set, so isomers with the same bond-cleavage pattern are indistinguishable by rules.

| | fitted on real (MoLFormer XL) | fitted on synthetic |
|---|---|---|
| CKA with the molecule space | 0.295 | 0.233 |
| nearest train molecule (raw cosine) | 0.184 | 0.197 |
| CCA | 0.159 | 0.110 |
| Procrustes | 0.151 | 0.107 |
| contrastive heads | 0.166 | 0.100 |

CKA between the real and the synthetic spectrum spaces is 0.600, so they are far from
interchangeable, and a map fitted on the rule spectra transfers badly to real held-out spectra: it
is below the raw-cosine baseline, and the contrastive head's Spearman correlation with Tanimoto
similarity goes *negative* (-0.054, against 0.008 for a random pairing). The one thing the rules
do show is that the frozen DreaMS encoder reads structure out of a minimal peak list - the
synthetic query retrieves the nearest train molecule at 0.197, above the 0.184 of real spectra and
about half the 0.386 structure-only ceiling. That is a probe of the encoder, not a training signal.

The GW coupling says the same thing from the other side (MoLFormer XL, val fold, pool 2000, seed 0,
`lam=0.01`). Coupling the rule-based spectrum geometry to the molecule geometry costs 158457 and puts
1.72x uniform mass on the true molecule - no better than the real-spectrum coupling (155936, 1.92x) -
and retrieving through either gives 0.106 / 0.107 against 0.152 for plain cosine in the same pool. But
real and rule-based spectra couple to *each other* cleanly: 126831 against an identity of 153984, with
3.27x of the mass on the right molecule. The two spectrum spaces are close to isomorphic; neither is
isomorphic to the molecule space. More spectra of the first kind cannot repair the second mismatch.

The test fold says the same thing (Mol-JEPA, 3 seeds): raw cosine 0.185, CCA 0.153, Procrustes
0.159, contrastive heads 0.162 (7.3% exact against the held fold's own molecules), geometry
projector 0.145 with rho 0.268, structure-only ceiling 0.343, precursor mass 0.132, random 0.100.

**Recommendation:** do not fine-tune on synthetic pairs. The bottleneck is not the number of
spectra but the objective - a frozen encoder that carries chemistry in an anisotropic,
non-isometric space. If synthetic data is added later, it has to come from a physical
fragmentation model (Fpocket, MolGen), be validated against real spectra, and be used only as
augmentation alongside real pairs, never as the alignment target.

## Putting the precursor mass into the joint space

Mass is the one quantity both sides share exactly, so the obvious next step is to train the joint
latent so that it *contains* it. Before training anything, `scripts/mass_probe.py` asks whether the
frozen spaces already carry it: a ridge map from the 1024-d DreaMS spectrum embedding to the neutral
exact mass of the molecule reaches R² 0.771 on the val fold but a median error of **39.6 Da** - only
1.3% of molecules land within a dalton. The Mol-JEPA molecule side is worse (R² 0.584, 51.5 Da) and
the embedding norm alone is worse than predicting the mean (R² -1.32). The mass is not in the
embeddings; it is in the TSV, whose `parent_mass` agrees with RDKit's exact mass to 0.0001 Da. Mass
is therefore free as a filter and expensive as a representation.

`fit_geometry` (`sna/joint.py`) takes a fourth term accordingly: one *shared* readout on the joint
latent - an MLP, because a single linear axis cannot pin a 130 Da spread to a dalton - must return
the mass from the spectrum side *and* from the molecule side, so the two have to agree. With
`--mass-target classes` the target is not one number but a coarse-to-fine ladder of molecular
classes (`sna/align.py: descriptors`): element counts, heavy atoms, rings, degree of unsaturation,
nominal mass, exact mass. The latent is then rewarded for getting a *class* right before it is
rewarded for getting the molecule right, and retrieval is scored with `formula@5` - the fraction of
the top-5 hits that share the query's molecular formula.

```bash
python scripts/align_joint.py --tsv data/MassSpecGym1.5.tsv --encoder moljepa \
  --methods geometry --mass-gamma 0.1,1.0,10.0 --mass-target classes --seeds 1
```

Mol-JEPA, val fold, 25046 train molecules, 256-d projectors, seed 0. `formula@5` is the fraction of
the top-5 hits that share the query's molecular formula; chance is 0.00003 and the ceiling is 0.186,
because only 18.6% of held-out molecules have a congener in the train pool.

| target of the readout | weight | retrieval top-1 | exact match against the fold's own molecules | formula@5 | Spearman rho with Tanimoto | median mass error, train / held |
|---|---|---|---|---|---|---|
| none (pairing + geometry + SIGReg) | - | 0.149 | 0.030 | 0.0006 | 0.268 | - |
| exact mass, linear head, 20 epochs | 0.1 | 0.150 | 0.034 | - | 0.296 | - / 31.2 Da |
| exact mass, linear head, 20 epochs | 1.0 | 0.149 | 0.028 | - | 0.321 | - / 24.2 Da |
| exact mass, linear head, 20 epochs | 10.0 | 0.139 | 0.014 | - | 0.380 | - / 24.8 Da |
| exact mass, MLP head, 60 epochs | 0 | 0.144 | 0.024 | - | 0.230 | - |
| exact mass, MLP head, 60 epochs | 1.0 | 0.150 | 0.028 | - | 0.252 | 5.8 / 27.2 Da |
| exact mass, MLP head, 60 epochs | 10.0 | 0.153 | 0.022 | - | 0.347 | 3.1 / 23.0 Da |
| class ladder, 20 epochs | 1.0 | **0.159** | 0.034 | 0.0009 | 0.331 | 11.5 / 30.1 Da |
| class ladder, 20 epochs | 10.0 | 0.157 | 0.021 | 0.0007 | **0.425** | 10.4 / 33.1 Da |
| class ladder without the mass columns, 20 epochs | 1.0 | 0.157 | 0.032 | 0.0009 | 0.335 | - |
| class ladder without the mass columns, 20 epochs | 10.0 | 0.157 | 0.022 | 0.0008 | 0.431 | - |
| raw spectrum cosine | - | 0.184 | - | 0.0024 | 0.142 | - |
| random | - | 0.097 | 0.010 | 0.00003 | 0.008 | - |

Giving the head real capacity fixes the collapse: with the MLP the top-1 no longer falls apart at a
weight of 10 (0.153 instead of 0.139), and the readout can drive the *training* masses to 3 Da. It
cannot make them generalise: the same fit is still 23 Da off on molecules it has never seen, and only
2.6% of held-out molecules land within a dalton. Memorising 25046 masses is easy; predicting a mass
from a spectrum embedding is not, because the embedding does not contain the mass.

The class ladder is the version that works. Predicting element counts, heavy atoms, rings,
unsaturation, nominal mass and exact mass - instead of one number that has to be exact - lifts top-1
from 0.149 to 0.159 and the rank correlation with structure from 0.268 to 0.425, the largest rho in
the whole benchmark. Rewarding the latent for getting a *class* right is worth more than rewarding it
for getting a dalton right.

The ablation says where that gain comes from: delete the two mass columns from the ladder and top-1
is 0.157 with rho 0.431 - the same as with them. It is the chemistry classes doing the work, not the
mass.

It does not, however, make the space group congeners: formula@5 stays at 0.0009, below the raw
spectrum cosine (0.0024) and fifty times below simply ranking by the precursor mass (0.051).

Mass as a score, with no training at all - `cos + 2w t_query t_candidate`, which for a fixed query is
the same ranking as `cos - w (t_query - t_candidate)^2`, the hard filter with the cut replaced by a
graded penalty:

| how mass enters the score | top-1 | formula@5 |
|---|---|---|
| cosine + soft mass axis, w=0.1 | 0.186 | 0.0025 |
| cosine + soft mass axis, w=1.0 | 0.167 | 0.0029 |
| cosine + soft mass axis, w=10.0 | 0.125 | 0.0014 |
| cosine, candidates within 1 Da | 0.141 | 0.0155 |
| precursor mass only | 0.134 | 0.0505 |

A whisper of mass (w=0.1) nudges top-1 from 0.184 to 0.186; a hard 1 Da cut drops it to 0.141 and a
strong mass term to 0.125.

The mass term makes the latent mass-aware without making it mass-exact: rho with Tanimoto climbs from
0.268 to 0.380 as the weight rises, even though retrieval never catches the raw cosine.

**Recommendation:** two jobs, two uses. For *analogue* retrieval - "which compound is it like" - mass
belongs in the training target as a class ladder and not in the score: the ladder is the only fit that
beats the plain pairing baseline, while a hard "candidates within 1 Da of the precursor" screen *lowers*
top-1 from 0.184 to 0.141, because the structurally closest train molecules usually do not share the
precursor. For *identification* - "which compound is this" - the raw precursor mass is still the best
class filter we measured, and no learned latent comes close.

## How much headroom is there, and what actually closes it

`scripts/ceiling_probe.py` measures the gap before any model is fitted. On the same 25046-molecule
pool and the 19429 held val spectra, the *oracle* - rank the pool by true Tanimoto to the query -
scores top-1 0.505 and top-5 0.441. Raw cosine puts that same molecule at median rank **1342** of
25046, and only 4.2% of the time in the top 5. So the loss is not the ranking rule; it is that the
embedding does not know which pool molecule is closest.

The failures are not random. The molecule the oracle would pick reaches the cosine top 5 for 1.7% of
spectra with fewer than 5 peaks and 8.8% of spectra with 80 or more; 3.0% at collision energy 0 eV
against 17.6% at 6 eV; 4.9% for `[M+H]+` against 2.0% for `[M+Na]+` - even though the `[M+Na]+`
queries have a *higher* oracle Tanimoto (0.611 against 0.471), so they are easier chemistry and the
embedding is simply worse at them.

Everything we tried on top of cosine, all training-free or closed-form, in `scripts/`:

| attempt | top-1 | top-5 |
|---|---|---|
| raw cosine (the baseline) | 0.1838 | 0.1715 |
| score ensemble, 0.8 DreaMS + 0.2 JEPA | 0.1886 | 0.1728 |
| whitened (Mahalanobis) cosine | 0.1884 | 0.1734 |
| centred cosine | 0.1839 | 0.1716 |
| cosine on the top-400 PCA components | 0.1830 | 0.1709 |
| max-pool the spectra of a molecule instead of averaging | 0.1867 | 0.1722 |
| adduct-matched pool (compare like with like) | 0.1852 | 0.1719 |
| linear `[M+Na]+` -> `[M+H]+` map fitted on 4576 train pairs | 0.1779 | 0.1682 |
| raw peak-list cosine (fragment overlap) alone | 0.1484 | 0.1341 |
| consensus reranking, `cos + lambda * mean Tanimoto to the top-K hits` | 0.1881 | 0.1797 |
| class-ladder agreement term | 0.1825 | 0.1703 |
| learned symmetric bilinear metric `Z B Z^T` | 0.1627 | 0.1590 |
| ridge head: embedding -> soft 2048-bit fingerprint, rank by Tanimoto | 0.1890 | 0.1851 |
| the same head with both encoders and acquisition metadata | 0.1914 | 0.1874 |
| **blend 0.2 cosine + 0.8 fingerprint head** | **0.1988** | **0.1893** |

The fingerprint head is the only thing that moves the needle, and it moves it by about one and a half
Tanimoto points. It is a single closed-form ridge solve, `W = (Q^T Q + I)^-1 Q^T fp_train`, so it is
cheap and it transfers. The blend sweep is flat between 0.4 and 0.8 head weight, which means the two
signals are largely redundant rather than complementary - the head is not finding a different set of
molecules, it is ordering the same ones slightly better.

Two things that look promising and are not. Consensus reranking (a candidate scores higher if the
other top hits look like it) *lowers* the best blend, because the top hits are already the answer.
A learned metric fitted to the Tanimoto matrix reproduces it to 37% relative error and then retrieves
*worse* than cosine, because the fit spends its capacity on the bulk of the matrix rather than on the
top of each row.

The formula question is a separate, harder one. The ceiling for formula@5 is 0.1857 - only 18.6% of
held queries have a congener in the pool at all. Ranking by precursor mass alone reaches 0.0505;
adding a predicted-formula bonus reaches 0.1766, but only when the bonus uses the *true* formula, so
it is an oracle, not a method. The ridge head's mass prediction is off by 111 Da on average, which is
not a formula error but a sign that a linear map from a 1024-d spectrum embedding cannot recover mass.

**Recommendation:** stop tuning the score. The blend is within a point of everything else we tried,
and the oracle is 30 points above it. The gap is in the encoder, not the ranking.


## License

[MIT](LICENSE). Upstream acknowledgments and license notices are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). External models and datasets retain their
own licenses.
