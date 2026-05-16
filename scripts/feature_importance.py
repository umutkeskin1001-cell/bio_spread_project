from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bio_spread.data.dataset import SequenceDataset, load_normalizers, sequence_collate
from bio_spread.data.loader import get_device
from bio_spread.data.snapshot import load_taxonomy_vocab
from bio_spread.models import create_model
from bio_spread.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_importance")


def integrated_gradients(
    model: nn.Module,
    static: torch.Tensor,
    seq: torch.Tensor,
    mask: torch.Tensor,
    taxonomy_idxs: torch.Tensor | None,
    baseline_frac: float = 0.0,
    n_steps: int = 50,
) -> torch.Tensor:
    model.eval()
    B, L, F = seq.shape
    seq_baseline = seq * baseline_frac
    static_baseline = static * baseline_frac

    scaled_seqs = torch.stack([seq_baseline + (float(i) / n_steps) * (seq - seq_baseline) for i in range(n_steps + 1)], dim=0)
    scaled_statics = torch.stack([static_baseline + (float(i) / n_steps) * (static - static_baseline) for i in range(n_steps + 1)], dim=0)

    grads_seq = []
    grads_static = []
    for i in range(n_steps + 1):
        s = scaled_statics[i].requires_grad_(True)
        q = scaled_seqs[i].requires_grad_(True)
        out = model(s, q, mask, taxonomy_idxs)

        grad_outputs = torch.ones_like(out.hazard_logits)
        g = torch.autograd.grad(out.hazard_logits, [s, q], grad_outputs=grad_outputs, create_graph=False, retain_graph=False)

        grads_static.append(g[0].detach())
        grads_seq.append(g[1].detach())

    avg_grads_static = torch.stack(grads_static).mean(dim=0)
    avg_grads_seq = torch.stack(grads_seq).mean(dim=0)

    ig_static = (static - static_baseline) * avg_grads_static
    ig_seq = (seq - seq_baseline) * avg_grads_seq
    ig_seq = (ig_seq * mask.unsqueeze(-1))

    return ig_static, ig_seq


def main():
    cfg = load_config("config/default.yaml")
    device = get_device()
    feature_dir = Path("data/features")
    model_path = "artifacts/BS_20260516_092852/best_model.pt"

    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")
    taxonomy_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(taxonomy_vocab)

    val_df = seq_df.filter((pl.col("split") == "val") & (pl.col("observed") == 1.0))
    val_bids = val_df["backbone_id"].unique().to_list()
    if not val_bids:
        logger.error("No val backbones with observed=1")
        return

    max_seq_len = cfg.model.max_seq_len
    ds = SequenceDataset(
        val_df, val_bids, max_seq_len,
        normalizer=(seq_means, seq_stds), static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    collate = lambda b: sequence_collate(b, max_seq_len)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate)

    n_static = ds.items[0]["static"].size(-1)
    n_snapshot = ds.items[0]["seq"].size(-1)
    model = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab if use_taxonomy else None)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    from bio_spread.constants import SNAPSHOT_FEATURE_COLS, STATIC_COLS

    all_ig_static = []
    all_ig_seq = []
    model.eval()

    for batch in loader:
        static = batch["static"].to(device)
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        taxonomy_idxs = batch.get("taxonomy")
        if taxonomy_idxs is not None:
            taxonomy_idxs = taxonomy_idxs.to(device)

        ig_s, ig_q = integrated_gradients(model, static, seq, mask, taxonomy_idxs)
        all_ig_static.append(ig_s.detach().cpu().numpy())
        all_ig_seq.append(ig_q.detach().cpu().numpy())

    ig_static_arr = np.concatenate(all_ig_static, axis=0)
    ig_seq_arr = np.concatenate(all_ig_seq, axis=0)

    static_importance = {}
    for i, col in enumerate(STATIC_COLS):
        importances = np.abs(ig_static_arr[:, i])
        static_importance[col] = {
            "mean": float(importances.mean()),
            "std": float(importances.std()),
        }
    ranked_static = sorted(static_importance.items(), key=lambda x: x[1]["mean"], reverse=True)
    logger.info("=== Static Feature Importance (IG) ===")
    for col, imp in ranked_static:
        logger.info("  %30s: mean=%.6f, std=%.6f", col, imp["mean"], imp["std"])

    seq_importance = {}
    for i, col in enumerate(SNAPSHOT_FEATURE_COLS):
        vals = np.abs(ig_seq_arr[:, :, i]).flatten()
        seq_importance[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }
    ranked_seq = sorted(seq_importance.items(), key=lambda x: x[1]["mean"], reverse=True)
    logger.info("=== Snapshot Feature Importance (IG) ===")
    for col, imp in ranked_seq:
        logger.info("  %30s: mean=%.6f, std=%.6f", col, imp["mean"], imp["std"])

    out_path = Path("artifacts/feature_importance.json")
    result = {"static": dict(ranked_static), "snapshot": dict(ranked_seq)}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved feature importance to %s", out_path)


if __name__ == "__main__":
    main()
