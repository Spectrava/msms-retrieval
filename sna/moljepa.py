"""Mol-JEPA as a frozen molecule encoder.

Pinned to the current HEAD of Flogrammer/Mol-JEPA, whose model card declares MIT. An
earlier revision (4c912b4) declared CC BY-NC 4.0 for the *same* weight file (SHA-256
a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14); the licence change is
what makes this encoder usable, and it is recorded in docs/upstream_audit.md. MoLFormer XL
(apache-2.0) in `sna/towers.py` is the fallback, and it is never used silently: the encoder
name is written into every cache and every metrics row.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel

REPO, REVISION = "Flogrammer/Mol-JEPA", "8098738a73e219c36347094ce7ffa340e888b04d"
WEIGHTS_SHA256 = "a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14"  # 45,406,772 F32 params
# shortcut: the upstream custom class predates transformers 5 and does not declare
# all_tied_weights_keys, which from_pretrained now requires; an empty dict is correct
# because the checkpoint ships every weight unshared.
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}


def load(device="cuda") -> torch.nn.Module:
    return AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).requires_grad_(False).eval().to(device)


@torch.no_grad()
def embed(smiles: list[str], model, output: str = "cls", batch: int = 256) -> np.ndarray:
    """Graph-level molecule embeddings: the CLS token (512) or the flattened latent modalities."""
    dim = 512 if output == "cls" else 512 * 13
    out = np.zeros((len(smiles), dim), dtype=np.float32)
    for i in range(0, len(smiles), batch):
        chunk = [s for s in smiles[i:i + batch] if s]
        if not chunk:
            continue
        o = model(chunk)
        vec = o.cls if output == "cls" else o.embeddings
        out[i:i + len(chunk)] = vec.float().cpu().numpy().reshape(len(chunk), -1)
    return out
