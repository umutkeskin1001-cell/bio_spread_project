"""
External validation for BioSpread on held-out data.
"""
import json
import logging
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from bio_spread.data.dataset import SequenceDataset, load_normalizers, sequence_collate
from bio_spread.data.snapshot import load_taxonomy_vocab
from bio_spread.models import create_model
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config

logger = logging.getLogger(__name__)

def evaluate_external(
    model_path: str,
    external_data_path: str,
    config_path: str = "config/default.yaml",
    feature_dir: str = "data/features",
):
    cfg = load_config(config_path)
    feature_dir = Path(feature_dir)

    ext_df = pl.read_csv(external_data_path, separator="\t")
    ext_bids = ext_df["backbone_id"].unique().to_list()
    logger.info("Loaded %d external backbones from %s", len(ext_bids), external_data_path)

    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")
    tax_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(tax_vocab)

    ext_df = ext_df.filter(pl.col("observed") == 1.0)
    ext_bids = ext_df["backbone_id"].unique().to_list()

    ds = SequenceDataset(
        ext_df, ext_bids, cfg.model.max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
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

    trainer = BioSpreadTrainer(model, device=device, calibrate=False)
    platt_path = Path(model_path).parent / "platt.pt"
    if platt_path.exists():
        platt_state = torch.load(platt_path, map_location=device, weights_only=True)
        for h in range(3):
            key = f"scaler_h{h}"
            if key in platt_state:
                trainer.platt_scalers[h].load_state_dict(platt_state[key])
        logger.info("Loaded per-horizon Platt scalers from %s", platt_path)

    metrics = trainer.evaluate(loader)

    results = {
        "n_backbones": len(ext_bids),
        "n_samples": metrics.get("n", 0),
        "roc_auc": metrics.get("roc_auc", 0.0),
        "pr_auc": metrics.get("pr_auc", 0.0),
        "f1": metrics.get("f1", 0.0),
        "ece": metrics.get("ece", 0.0),
        "roc_auc_h1": metrics.get("roc_auc_h1", 0.0),
        "roc_auc_h2": metrics.get("roc_auc_h2", 0.0),
        "roc_auc_h3": metrics.get("roc_auc_h3", 0.0),
    }

    output_path = Path("artifacts/external_validation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 50)
    print("  EXTERNAL VALIDATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:20s} = {v}")
    print("=" * 50)

    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--external-data", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--feature-dir", default="data/features")
    args = parser.parse_args()
    evaluate_external(args.model_path, args.external_data, args.config, args.feature_dir)
