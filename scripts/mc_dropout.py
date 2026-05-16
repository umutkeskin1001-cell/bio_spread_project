"""
MC Dropout uncertainty estimation for BioSpread.

Computes predictive uncertainty by running multiple forward passes
with dropout enabled at inference time.
"""
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from bio_spread.data.dataset import (
    SequenceDataset,
    load_normalizers,
    sequence_collate,
)
from bio_spread.data.snapshot import load_taxonomy_vocab
from bio_spread.models import create_model
from bio_spread.utils.config import load_config

logger = logging.getLogger(__name__)


def enable_dropout(model):
    """Enable dropout layers for Monte Carlo sampling at inference."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(
    model, loader, device, n_samples=30, use_cold_scaler=False,
):
    """
    Run MC Dropout inference.

    Args:
        model: BioSpreadModel
        loader: DataLoader
        device: torch device
        n_samples: number of MC forward passes
        use_cold_scaler: if True, use cold-start Platt scaler

    Returns:
        dict of per-backbone predictions with uncertainty
    """
    model.eval()
    enable_dropout(model)

    all_probs = {h: [] for h in range(3)}
    all_targets = {h: [] for h in range(3)}
    all_bids = []

    for batch in loader:
        static = batch["static"].to(device)
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        lengths = batch["seq_len"].to(device)
        targets = batch["hazard"].to(device)
        taxonomy_idxs = batch.get("taxonomy")
        if taxonomy_idxs is not None:
            taxonomy_idxs = taxonomy_idxs.to(device)
        bids = batch["backbone_ids"]

        B = static.size(0)

        # MC Dropout samples
        mc_probs = []
        for _ in range(n_samples):
            out = model(static, seq, mask, taxonomy_idxs)
            mc_probs.append(torch.sigmoid(out.hazard_logits))  # (B, 3)
        mc_probs = torch.stack(mc_probs, dim=0)  # (n_samples, B, 3)

        mean_probs = mc_probs.mean(dim=0)  # (B, 3)
        std_probs = mc_probs.std(dim=0)  # (B, 3)

        idx = (lengths - 1).clamp(min=0)
        for h in range(3):
            for b in range(B):
                t = targets[b, idx[b], h].item()
                if t >= 0:
                    all_probs[h].append({
                        "bid": bids[b],
                        "mean": mean_probs[b, h].item(),
                        "std": std_probs[b, h].item(),
                        "ci95_lower": (mean_probs[b, h] - 1.96 * std_probs[b, h]).item(),
                        "ci95_upper": (mean_probs[b, h] + 1.96 * std_probs[b, h]).item(),
                    })
                    all_targets[h].append(t)

        all_bids.extend(bids)

    results = {}
    for h in range(3):
        if all_probs[h]:
            results[f"horizon_{h+1}"] = all_probs[h]

    results["n_backbones"] = len(set(all_bids))
    results["n_samples"] = len(all_bids)
    results["mc_samples"] = n_samples

    return results


def compute_mc_dropout(
    model_path: str,
    config_path: str = "config/default.yaml",
    feature_dir: str = "data/features",
    n_samples: int = 30,
    split_name: str = "val",
):
    cfg = load_config(config_path)
    feature_dir = Path(feature_dir)

    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    seq_df = seq_df.filter(pl.col("observed") == 1.0)

    with open(feature_dir / "split.json") as f:
        split = json.load(f)

    tax_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(tax_vocab)

    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")

    backbone_ids = split[split_name]
    subset_df = seq_df.filter(pl.col("backbone_id").is_in(backbone_ids))

    ds = SequenceDataset(
        subset_df, backbone_ids, cfg.model.max_seq_len,
        normalizer=(seq_means, seq_stds), static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size,
        collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len),
    )

    n_static = ds.items[0]["static"].size(-1)
    n_snapshot = ds.items[0]["seq"].size(-1)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab=tax_vocab if use_taxonomy else None)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    results = mc_dropout_predict(model, loader, device, n_samples=n_samples)

    output_path = Path("artifacts/mc_dropout_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))

    # Summary stats
    print("\n" + "=" * 50)
    print("  MC DROPOUT UNCERTAINTY")
    print("=" * 50)
    print(f"  Split:       {split_name}")
    print(f"  Backbones:   {results['n_backbones']}")
    print(f"  MC samples:  {results['mc_samples']}")
    for h in range(3):
        key = f"horizon_{h+1}"
        if key in results:
            stds = [r["std"] for r in results[key]]
            print(f"  H{h+1}: mean std={np.mean(stds):.4f} +/- {np.std(stds):.4f}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--feature-dir", default="data/features")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()
    compute_mc_dropout(args.model_path, args.config, args.feature_dir, args.n_samples, args.split)
