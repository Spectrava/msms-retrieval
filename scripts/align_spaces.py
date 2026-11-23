"""Can the frozen spectrum embedding be lined up with a frozen molecule tower's space?

Fits a ridge map W from the DreaMS spectrum embedding to the pooled MoLFormer embedding of the
matching molecule on the train fold, then asks on the val fold:
  - how close W z lands to the true molecule's embedding (cosine),
  - which train molecule is nearest to W z and how similar it is to the true molecule,
  - the ceiling: the nearest train molecule to the *true* molecule's embedding.
The ceiling is what a perfect alignment could retrieve, so it is the bar the mapped space is
judged against - and the bar any decoder would have to work under.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from sna import paths
from sna.align import canonical, fingerprints, fit_contrastive, fit_mlp, nearest, probe_decoder, ridge, tanimoto
from sna.data import read_msg_tsv
from sna.towers import Tower


def embed_smiles(tower: str, smiles: list[str], batch: int = 1024) -> np.ndarray:
    """Pooled embeddings of a list of SMILES through a frozen tower (float16, like build_cache)."""
    repo, revision = paths.TOWERS[tower]
    tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
    model = Tower(tower).cuda()
    out = np.zeros((len(smiles), model.dim), dtype=np.float16)
    for i in range(0, len(smiles), batch):
        x = tok(smiles[i:i + batch], padding=True, return_tensors="pt").to("cuda")
        pooled, _ = model(x["input_ids"], x["attention_mask"].bool(), pooled=True)
        out[i:i + batch] = pooled[:, 0].half().cpu().numpy()
        print(f"[{tower}] {min(i + batch, len(smiles)):,}/{len(smiles):,}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tsv", default=str(paths.MSG_TSV))
    parser.add_argument("--out", default=None)
    parser.add_argument("--tower", default="molformer", choices=sorted(paths.TOWERS))
    parser.add_argument("--lam", type=float, default=1e3)
    parser.add_argument("--map", choices=("linear", "mlp", "contrastive"), default="linear")
    parser.add_argument("--probe", action="store_true", help="train the char-LSTM on the molecule latents")
    args = parser.parse_args()
    args.out = args.out or f"data/alignment_{args.tower}_{args.map}.npz"
    rng = np.random.default_rng(0)

    msg = read_msg_tsv(args.tsv)
    fold = np.array(msg["fold"])
    # the cached spectrum embeddings are stored in fold order, so everything stays positional
    train_rows = np.where(fold == "train")[0]
    val_rows = np.where(fold == "val")[0]
    train_smiles = [canonical(msg["smiles"][i]) for i in train_rows]
    val_smiles = [canonical(msg["smiles"][i]) for i in val_rows]
    train_smiles = [s for s in train_smiles if s]
    keep = [i for i, s in enumerate(val_smiles) if s]
    val_smiles = [val_smiles[i] for i in keep]
    val_id = np.array(msg["identifier"])[val_rows[keep]]

    unique = sorted(set(train_smiles))
    mol_id = {s: i for i, s in enumerate(unique)}
    mol_of_row = np.array([mol_id[s] for s in train_smiles])

    emb = np.load("data/msg_generated.parquet.embeddings.npz")
    z_train, z_val = emb["train"], emb["val"]

    cache = Path(f"data/molecule_embeddings_{args.tower}.npz")
    started = time.time()
    if cache.exists():
        c = np.load(cache)
        assert list(c["unique"]) == unique, f"{cache} is for a different molecule set"
        m_train, m_val = c["m_train"].astype(np.float32), c["m_val"].astype(np.float32)
    else:
        m_train = embed_smiles(args.tower, unique)
        m_val = embed_smiles(args.tower, val_smiles)
        np.savez(cache, unique=np.array(unique), m_train=m_train, m_val=m_val)

    fp = fingerprints(unique + val_smiles)
    fp_train, fp_val = fp[: len(unique)], fp[len(unique):]

    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    m_train_n, m_val_n = unit(m_train), unit(m_val)
    z_val_n = unit(z_val)
    # one mean spectrum per unique train molecule, so the index and the fingerprints line up
    sums = np.zeros((len(unique), z_train.shape[1]), dtype=np.float64)
    np.add.at(sums, mol_of_row, z_train)
    z_unique_n = unit(sums / np.bincount(mol_of_row)[:, None])

    W = None
    if args.map == "linear":
        W = ridge(z_train.astype(np.float64), m_train[mol_of_row], args.lam)
        mapped, index, index_val = unit(z_val @ W), m_train_n, m_val_n
    elif args.map == "mlp":
        mapped, index, index_val = unit(fit_mlp(z_train, m_train[mol_of_row])(z_val)), m_train_n, m_val_n
    else:
        spectrum, molecule = fit_contrastive(z_train, m_train[mol_of_row])
        mapped, index, index_val = unit(spectrum(z_val)), unit(molecule(m_train)), unit(molecule(m_val))

    align_cos = float((mapped * index_val).sum(1).mean())
    # the same-space noise floor: a mapped spectrum against a random molecule's adapted vector
    aligned_floor = float((mapped * index_val[rng.integers(len(index_val), size=len(mapped))]).sum(1).mean())
    retrieval, sims = nearest(mapped, index, fp_train, fp_val)
    ceiling, _ = nearest(m_val_n, m_train_n, fp_train, fp_val)
    raw, _ = nearest(z_val_n, z_unique_n, fp_train, fp_val)
    # alignment quality against the val molecules themselves: the ceiling here is 1.0, so this
    # isolates the map from the fact that no train molecule is the answer
    self_retrieval, self_sims = nearest(mapped, index_val, fp_val, fp_val)
    exact = float((self_sims == 1.0).mean())
    random = float(tanimoto(fp_train[rng.integers(len(fp_train), size=2000)],
                            fp_val[rng.integers(len(fp_val), size=2000)]).mean())

    print(f"cosine(mapped, true molecule)        {align_cos:.3f}   (random molecule in the same space: {aligned_floor:.3f})")
    print(f"mapped nearest train molecule        {retrieval:.3f} Tanimoto")
    print(f"raw spectrum nearest neighbour       {raw:.3f} Tanimoto")
    print(f"ceiling (true molecule's nearest)    {ceiling:.3f} Tanimoto")
    print(f"nearest among val molecules          {self_retrieval:.3f} Tanimoto, {exact:.1%} exact")
    print(f"random pair                          {random:.3f} Tanimoto")
    if args.probe:
        seen = {}
        for s, e in zip(val_smiles, m_val):
            seen.setdefault(s, e)
        val_unique = sorted(seen)
        parse, sim, exact = probe_decoder(m_train, unique, np.array([seen[s] for s in val_unique]), val_unique)
        print(f"decoder on molecule latents: {parse:.1%} parse, {sim:.3f} mean Tanimoto on unseen "
              f"molecules, {exact:.1%} exact on its own training molecules")
    np.savez(args.out, W=W, tower=args.tower, lam=args.lam, identifier=val_id,
            smiles=np.array(val_smiles), mapped=mapped, sims=sims,
            ceiling=ceiling, raw=raw, align_cos=align_cos, aligned_floor=aligned_floor, random=random)
    print(f"wrote {args.out} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
