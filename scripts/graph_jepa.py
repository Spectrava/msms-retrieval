"""Build the ROI graph from MS2 similarity and train the graph JEPA on it.

Edges are drawn only between ROIs that both have MS2; --within-sample and --between-samples turn
the two kinds of link on or off. Writes the graph-informed latents, the edge list, and provenance.

    python scripts/graph_jepa.py --embeddings data/7600_ms2_embeddings.parquet \
        --traces data/7600_traces.parquet --catalog data/7600_catalog.parquet \
        --out data/7600_roi_latents.parquet --threshold 0.9
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from sna.graph import THRESHOLD, GraphJEPA, components, degrees, jepa_loss, roi_graph
from sna.retrieval import neighbours


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings", required=True, help="MS2 embeddings Parquet from scripts/embed_ms2.py")
    ap.add_argument("--traces", required=True, help="Spectrava traces Parquet")
    ap.add_argument("--catalog", required=True, help="Spectrava MS1 catalog Parquet")
    ap.add_argument("--out", required=True, help="output Parquet of ROI latents")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="minimum cosine for an edge")
    ap.add_argument("--within-sample", action="store_true", default=True, help="link ROIs in the same sample")
    ap.add_argument("--no-within-sample", dest="within_sample", action="store_false")
    ap.add_argument("--between-samples", action="store_true", default=True, help="link ROIs across samples")
    ap.add_argument("--no-between-samples", dest="between_samples", action="store_false")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--spread-weight", type=float, default=0.1, help="penalty on latents collapsing to one direction")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    embeddings = pd.read_parquet(args.embeddings)
    nodes, edges = roi_graph(embeddings, pd.read_parquet(args.traces), pd.read_parquet(args.catalog),
                             args.threshold, args.within_sample, args.between_samples)
    label = components(nodes, edges)
    sizes = np.bincount(label)
    print(f"{len(nodes)} ROIs, {int(nodes.has_ms2.sum())} with MS2; {len(edges)} edges "
          f"({int(edges.same_sample.sum())} within-sample, {int((~edges.same_sample).sum())} between samples)")
    print(f"components: {len(sizes)}, largest {sizes.max()} ROIs, {int((sizes == 1).sum())} isolated")

    linked = nodes[nodes.has_ms2]  # only ROIs with MS2 have a feature to link
    degree = degrees(nodes, edges)
    degree_of = dict(zip(nodes.candidate_id, degree))
    position = {c: i for i, c in enumerate(linked.candidate_id)}
    z = torch.tensor(np.stack(linked.embedding), dtype=torch.float32, device=args.device)
    src = torch.tensor([position[c] for c in edges.a], device=args.device)
    dst = torch.tensor([position[c] for c in edges.b], device=args.device)
    connected = torch.tensor([degree_of[c] > 0 for c in linked.candidate_id], device=args.device)

    model = GraphJEPA(z.shape[1], args.hidden, args.layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    t0 = time.time()
    for epoch in range(args.epochs):
        loss = jepa_loss(model, z, src, dst, connected, args.spread_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  latent prediction loss {loss.item():.4f}")

    with torch.inference_mode():
        latents = model(z, src, dst).cpu().numpy()

    latent = {c: v.tolist() for c, v in zip(linked.candidate_id, latents)}
    out = nodes.assign(degree=[degree_of[c] for c in nodes.candidate_id], component=label,
                       graph_embedding=[latent.get(c) for c in nodes.candidate_id])
    out.drop(columns=["embedding"]).to_parquet(args.out)
    edges.to_parquet(f"{args.out}.edges.parquet")
    prov = {
        "encoder": "DreaMS (frozen)", "graph": "MS2 cosine similarity", "threshold": args.threshold,
        "withinSample": args.within_sample, "betweenSamples": args.between_samples,
        "nodes": len(nodes), "edges": len(edges), "linkedRois": int(connected.sum()),
        "layers": args.layers, "hidden": args.hidden, "epochs": args.epochs, "lr": args.lr,
        "spreadWeight": args.spread_weight,
        "device": args.device, "torch": torch.__version__, "seconds": round(time.time() - t0, 2),
        "note": "Edges are MS2-similarity hypotheses between ROI traces, not confirmed precursor "
                "relationships; ROIs without MS2 have no edges and no latent.",
    }
    with open(f"{args.out}.provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    apex = np.array(linked.ms1_mz)
    for name, vectors in (("frozen", np.stack(linked.embedding)), ("graph", latents)):
        order, _ = neighbours(vectors, 1)
        same = np.abs(apex[order[:, 0]] - apex) < 0.05
        print(f"{name} latents: top-1 neighbour shares the apex m/z for {same.mean():.1%} of linked ROIs")
    print(f"wrote {len(out)} ROI latents to {args.out}")


if __name__ == "__main__":
    main()
