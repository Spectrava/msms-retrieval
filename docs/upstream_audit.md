# Upstream audit: every frozen model we align against

Checked 2026-11-24. Nothing here is vendored into the repository: weights are downloaded by the
runtime into the Hugging Face cache, and the DreaMS checkpoint sits in `checkpoints/` untracked.
Each entry records the exact revision used, so a run can be reproduced and the licence is known
before the model is used rather than after.

| Model | Pinned revision | Licence | Params | Output used | How it is called |
|---|---|---|---|---|---|
| DreaMS (MassSpecGym embedding model) | code `dbec3a0b514a99e5056cfccde4559fda8cfe8129` (2026-04-27), weights Zenodo [10997887](https://zenodo.org/records/10997887) | code MIT, weights CC-BY-4.0 | - | 1024-d global vector, token 0 after the final LayerNorm | `sna/encoders/dreams.py` ports the architecture; `load_dreams(checkpoint)` |
| MoLFormer-XL-both-10pct | `361063d0ad524ef77cf39b08469f6be770dc550f` | apache-2.0 | 46,805,760 | 768-d pooler output (mean over attended tokens) | `sna/towers.py` `Tower("molformer")` |
| ChemBERTa-zinc-base-v1 | `761d6a18cf99db371e0b43baf3e2d21b3e865a20` | **none declared on the Hub page** | ~38 M | 384-d `<s>` token state | `sna/towers.py` `Tower("chemberta")` |
| Mol-JEPA | `8098738a73e219c36347094ce7ffa340e888b04d` (HEAD) | MIT | 45,406,772 | 512-d CLS token (`output="cls"`); 13x512 latent modality embeddings available | `sna/moljepa.py` `load()` / `embed(smiles, model)` |

## Mol-JEPA: the licence changed under the same weights

The repository has seven commits. Revision `4c912b450175f31b5ba913a5dc921c03b27b985a` ("Update
license to CC BY-NC 4.0", 3 months ago) declares `cc-by-nc-4.0`; the current HEAD
`8098738a73e219c36347094ce7ffa340e888b04d` declares `mit`. `model.safetensors` is byte-identical
between the two revisions:

```
a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14  model.safetensors   (both revisions)
```

So the weights are the same and only the stated licence differs. We pin the HEAD revision, which
is the permissive one, and record the hash so a later change of weights is detectable. If the
upstream ever reverts to CC BY-NC, MoLFormer XL (apache-2.0) is the fallback - and it is never
substituted silently: the encoder name is written into `data/molecule_embeddings_<encoder>.npz`
and into every row of the metrics CSV.

Caveat: a licence change by a model uploader is not a legal opinion. The Mol-JEPA weights were
trained on 5M molecules and the model card gives no training-data licence; for any commercial
deployment this needs a check with the upstream authors (Flogrammer / Boehringer Ingelheim).

## Checkpoint hashes

- `checkpoints/DreaMS_embedding_model_MassSpecGym.ckpt` - SHA-256
  `d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832`.
- Mol-JEPA `model.safetensors` - SHA-256 `a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14`.

## Loading notes

- Mol-JEPA ships custom modelling code written against transformers 4.50.3. Under transformers 5
  `from_pretrained` requires `all_tied_weights_keys`, which the upstream class does not define, so
  `sna/moljepa.py` sets it to `{}` on `PreTrainedModel` (correct: the checkpoint has no shared
  weights). Without the shim, both `AutoModel.from_pretrained` and `molfeat.MolJEPATransformer`
  fail with `'MolJEPAModel' object has no attribute 'all_tied_weights_keys'`.
- Mol-JEPA's graph encoder pools with `max` over atoms; the CLS token is the model's own global
  readout. We use CLS, not the flattened latent modalities, so the molecule space is 512-d against
  DreaMS' 1024-d - CKA and CCA do not need equal dimensions, Procrustes does, which is why it
  first PCA-reduces the spectrum space to 512.
- DreaMS embeddings in `data/msg_generated.parquet.embeddings.npz` are the token-0 global vector,
  which is **not** the vector the upstream DreaMS paper publishes (upstream reads the last
  feed-forward block). Do not compare these numbers to published DreaMS scores.
