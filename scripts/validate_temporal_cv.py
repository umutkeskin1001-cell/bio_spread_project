"""
Expanding Window Temporal Cross-Validation for Sovereign-X Pro.

Uses the SovereignSequenceDataset and SovereignX model from the current codebase.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from bio_spread_reborn.utils.config import load_config
from bio_spread_reborn.data.dataset import (
    SovereignSequenceDataset, sequence_collate, SequenceBatchSampler,
    fit_normalizers, fit_static_normalizers, load_normalizers,
    SNAPSHOT_FEATURE_COLS, STATIC_COLS,
)
from bio_spread_reborn.data.snapshot import build_sequences, disjoint_backbone_split, load_taxonomy_vocab
from bio_spread_reborn.models.sovereign import SovereignX
from bio_spread_reborn.models.trainer import SovereignXTrainer

logger = logging.getLogger(__name__)


def expanding_window_cv(
    config_path="config/default.yaml",
    feature_dir="data/sovereign_features",
    min_train_years=5,
):
    """Run expanding window CV on Sovereign-X Pro.

    Uses precomputed sequences and split from ``feature_dir``.
    """
    cfg = load_config(config_path)
    feature_dir = Path(feature_dir)

    # Load precomputed sequences
    df = pl.read_csv(
        feature_dir / "sequences.tsv",
        separator="\t",
        infer_schema_length=50_000,
        null_values=["", "None", "null", "NULL", "NaN"],
    )
    df = df.filter(pl.col("observed") > 0)
    logger.info("Loaded %d sequences with %d backbones",
                len(df), df["backbone_id"].n_unique())

    # Load split
    with open(feature_dir / "split.json") as f:
        split = json.load(f)
    train_ids = set(split["train"])
    val_ids = set(split["val"])

    # Taxonomy
    tax_vocab_path = feature_dir / "taxonomy_vocab.json"
    tax_vocab = None
    if tax_vocab_path.exists():
        tax_vocab = load_taxonomy_vocab(tax_vocab_path)

    # Normalizers (precomputed on train only)
    norm_path = feature_dir / "normalizers.npz"
    static_norm_path = feature_dir / "static_normalizers.npz"
    if norm_path.exists():
        normalizer = load_normalizers(norm_path)
    else:
        normalizer = (np.zeros(len(SNAPSHOT_FEATURE_COLS)), np.ones(len(SNAPSHOT_FEATURE_COLS)))
    if static_norm_path.exists():
        static_normalizer = load_normalizers(static_norm_path)
    else:
        static_normalizer = (np.zeros(len(STATIC_COLS)), np.ones(len(STATIC_COLS)))

    # Determine years
    years = sorted(df["year"].unique().to_list())
    if len(years) < min_train_years + 1:
        logger.error("Need at least %d years for CV, only have %d",
                     min_train_years + 1, len(years))
        return []

    aucs = []
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        val_year = years[i]

        train_df = df.filter(pl.col("year").is_in(train_years))
        val_df = df.filter(pl.col("year") == val_year)

        if train_df.is_empty() or val_df.is_empty():
            continue

        # Build datasets
        train_ds = SovereignSequenceDataset(
            train_df, list(train_ids), max_seq_len=cfg.model.max_seq_len,
            normalizer=normalizer, static_normalizer=static_normalizer,
            use_taxonomy=tax_vocab is not None,
        )
        val_ds = SovereignSequenceDataset(
            val_df, list(val_ids), max_seq_len=cfg.model.max_seq_len,
            normalizer=normalizer, static_normalizer=static_normalizer,
            use_taxonomy=tax_vocab is not None,
        )

        if len(train_ds) == 0 or len(val_ds) == 0:
            continue

        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, shuffle=True,
            collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len),
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.training.batch_size,
            collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len),
        )

        # Initialize model
        n_static = train_ds[0]["static"].size(0)
        n_snapshot = train_ds[0]["seq"].size(1)
        tax_vocab_sizes = None
        if tax_vocab:
            tax_vocab_sizes = [len(v) for v in tax_vocab.values()]

        model = SovereignX(
            n_static=n_static,
            n_snapshot=n_snapshot,
            taxonomy_vocab_sizes=tax_vocab_sizes,
            static_dim=cfg.model.static_dim,
            temporal_dim=cfg.model.temporal_dim,
            hidden_dim=cfg.model.gru_hidden,
            num_layers=cfg.model.gru_layers,
            n_hazard=cfg.model.n_hazard_steps,
            max_seq_len=cfg.model.max_seq_len,
            dropout=cfg.model.dropout,
        )

        trainer = SovereignXTrainer(
            model, device=device,
            lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
            epochs=cfg.training.epochs, patience=cfg.training.patience,
            warmup_epochs=cfg.training.warmup_epochs,
            grad_clip=cfg.training.grad_clip,
            lambda_count=cfg.training.lambda_count,
            lambda_rank=cfg.training.lambda_rank,
            lambda_cold=cfg.training.lambda_cold,
            temporal_masking_prob=cfg.training.temporal_masking_prob,
            calibrate=cfg.training.calibrate,
        )

        trainer.fit(train_loader, val_loader)

        metrics = trainer.evaluate(val_loader)
        aucs.append(metrics["roc_auc"])
        logger.info(
            "Fold %d: train years %s..%d -> val year %d | AUC = %.4f",
            i - min_train_years + 1,
            min(train_years), max(train_years), val_year,
            metrics["roc_auc"],
        )

    print("\n" + "=" * 50)
    print("  EXPANDING WINDOW CV RESULTS")
    print("=" * 50)
    if aucs:
        print(f"  Folds:        {len(aucs)}")
        print(f"  AUCs:         {[f'{a:.4f}' for a in aucs]}")
        print(f"  Mean AUC:     {np.mean(aucs):.4f}")
        print(f"  Std AUC:      {np.std(aucs):.4f}")

        if np.std(aucs) > 0.03:
            print("\n  WARNING: High variance (std > 0.03).")
            print("  Model may be unstable for production deployment.")
        else:
            print("  Model is stable (std <= 0.03). OK for production.")
    else:
        print("  No folds completed.")
    print("=" * 50 + "\n")

    return aucs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    expanding_window_cv()
