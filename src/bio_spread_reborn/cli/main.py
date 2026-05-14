"""
Sovereign-X: Thin CLI. Three commands: sovereign-prepare, train, evaluate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from bio_spread_reborn.data.dataset import (
    SequenceBatchSampler,
    SovereignSequenceDataset,
    fit_normalizers,
    fit_static_normalizers,
    load_normalizers,
    save_normalizers,
    sequence_collate,
)
from bio_spread_reborn.data.snapshot import (
    TAXONOMY_COLS,
    build_sequences,
    build_taxonomy_vocab,
    disjoint_backbone_split,
    load_taxonomy_vocab,
    save_taxonomy_vocab,
)
from bio_spread_reborn.models.sovereign import SovereignX
from bio_spread_reborn.models.trainer import SovereignXTrainer
from bio_spread_reborn.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sovereign")


@click.group()
def cli():
    """Sovereign-X: Dual-expert temporal hazard model for plasmid spread."""


@cli.command()
@click.option("--config", default="config/default.yaml")
@click.option("--output-dir", default="data/sovereign_features")
def sovereign_prepare(config: str, output_dir: str):
    """Build sequences + disjoint backbone split + save normalizers."""
    cfg = load_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pl.read_csv(cfg.data.backbones_path, separator="\t")
    if "resolved_year" in raw.columns and "year" not in raw.columns:
        raw = raw.rename({"resolved_year": "year"})
    meta = raw.unique(subset=["backbone_id"])
    split_year = cfg.data.split_year

    # Disjoint backbone split
    train_bids, val_bids, test_bids = disjoint_backbone_split(
        raw,
        split_year,
        val_frac=cfg.data.val_backbone_frac,
        test_frac=cfg.data.test_backbone_frac,
    )

    seq_kwargs = {
        "horizon": cfg.data.spread_horizon,
        "min_snapshots": 1,
        "require_country_history": cfg.data.require_country_history,
    }

    all_bids = {}
    for bid in train_bids:
        all_bids[bid] = "train"
    for bid in val_bids:
        all_bids[bid] = "val"
    for bid in test_bids:
        all_bids[bid] = "test"

    # Build taxonomy vocabulary from ALL raw data
    taxonomy_vocab = build_taxonomy_vocab(raw)
    save_taxonomy_vocab(taxonomy_vocab, output_dir / "taxonomy_vocab.json")

    sequences = build_sequences(raw, meta, set(all_bids.keys()), taxonomy_vocab=taxonomy_vocab, **seq_kwargs)
    sequences = sequences.with_columns(
        pl.col("backbone_id")
        .map_elements(lambda bid: all_bids.get(bid, "unknown"), return_dtype=pl.Utf8)
        .alias("split")
    )

    # Fit normalizers on TRAIN only (leakage prevention)
    train_seq = sequences.filter(pl.col("split") == "train")
    seq_means, seq_stds = fit_normalizers(train_seq)
    static_means, static_stds = fit_static_normalizers(train_seq)
    save_normalizers(output_dir / "normalizers.npz", seq_means, seq_stds)
    save_normalizers(output_dir / "static_normalizers.npz", static_means, static_stds)

    sequences.write_csv(output_dir / "sequences.tsv", separator="\t")
    with open(output_dir / "split.json", "w") as f:
        json.dump(
            {
                "train": list(train_bids),
                "val": list(val_bids),
                "test": list(test_bids),
                "split_year": split_year,
            },
            f,
            indent=2,
        )

    has_tax = all(c in sequences.columns for c in TAXONOMY_COLS)
    logger.info(
        "Prepared %d sequences (%d train, %d val, %d test) [taxonomy=%s]",
        len(sequences),
        sequences.filter(pl.col("split") == "train").shape[0],
        sequences.filter(pl.col("split") == "val").shape[0],
        sequences.filter(pl.col("split") == "test").shape[0],
        "yes" if has_tax else "no",
    )


@cli.command()
@click.option("--config", default="config/default.yaml")
@click.option("--feature-dir", default="data/sovereign_features")
def train(config: str, feature_dir: str):
    """Train Sovereign-X model."""
    cfg = load_config(config)
    feature_dir = Path(feature_dir)

    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    taxonomy_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(taxonomy_vocab)

    train_df = seq_df.filter((pl.col("split") == "train") & (pl.col("observed") == 1.0))
    val_df = seq_df.filter((pl.col("split") == "val") & (pl.col("observed") == 1.0))

    # Split val into early-stop (70%), cal-temp (15%), cal-cold (15%)
    val_bids = val_df["backbone_id"].unique().to_list()
    np.random.seed(cfg.training.seed)
    np.random.shuffle(val_bids)
    n_val = len(val_bids)
    n_cal_temp = max(1, n_val // 7)
    n_cal_cold = max(1, n_val // 7)
    cal_temp_bids = set(val_bids[:n_cal_temp])
    cal_cold_bids = set(val_bids[n_cal_temp : n_cal_temp + n_cal_cold])
    early_stop_bids = set(val_bids[n_cal_temp + n_cal_cold :])

    # Load normalizers fitted on train
    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")

    max_seq_len = cfg.model.max_seq_len

    train_ds = SovereignSequenceDataset(
        train_df,
        train_df["backbone_id"].unique().to_list(),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    val_ds = SovereignSequenceDataset(
        val_df.filter(pl.col("backbone_id").is_in(early_stop_bids)),
        list(early_stop_bids),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    cal_temp_ds = SovereignSequenceDataset(
        val_df.filter(pl.col("backbone_id").is_in(cal_temp_bids)),
        list(cal_temp_bids),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    cal_cold_ds = SovereignSequenceDataset(
        val_df.filter(pl.col("backbone_id").is_in(cal_cold_bids)),
        list(cal_cold_bids),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty training or validation dataset")

    sampler = SequenceBatchSampler(
        [item["seq_len"] for item in train_ds.items],
        batch_size=cfg.training.batch_size,
    )

    def _collate(b):
        return sequence_collate(b, max_seq_len)

    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, collate_fn=_collate)
    cal_temp_loader = (
        DataLoader(cal_temp_ds, batch_size=cfg.training.batch_size, collate_fn=_collate)
        if len(cal_temp_ds) > 0
        else None
    )
    cal_cold_loader = (
        DataLoader(cal_cold_ds, batch_size=cfg.training.batch_size, collate_fn=_collate)
        if len(cal_cold_ds) > 0
        else None
    )

    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)

    # Build taxonomy vocab sizes list for model
    tax_vocab_sizes = None
    if use_taxonomy:
        tax_vocab_sizes = [
            len(v)
            for v in [
                taxonomy_vocab.get("TAXONOMY_phylum", {}),
                taxonomy_vocab.get("TAXONOMY_class", {}),
                taxonomy_vocab.get("TAXONOMY_order", {}),
                taxonomy_vocab.get("TAXONOMY_family", {}),
                taxonomy_vocab.get("genus", {}),
            ]
        ]

    model = SovereignX(
        n_static=n_static,
        n_snapshot=n_snapshot,
        taxonomy_vocab_sizes=tax_vocab_sizes,
        taxonomy_embed_dim=cfg.model.taxonomy_embed_dim,
        static_dim=cfg.model.static_dim,
        temporal_dim=cfg.model.temporal_dim,
        hidden_dim=cfg.model.gru_hidden,
        num_layers=cfg.model.gru_layers,
        n_hazard=cfg.model.n_hazard_steps,
        dropout=cfg.model.dropout,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    trainer = SovereignXTrainer(
        model,
        device=device,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        warmup_epochs=cfg.training.warmup_epochs,
        grad_clip=cfg.training.grad_clip,
        lambda_count=cfg.training.lambda_count,
        lambda_rank=cfg.training.lambda_rank,
        lambda_cold=cfg.training.lambda_cold,
        lambda_all=cfg.training.lambda_all,
        lambda_gate=cfg.training.lambda_gate,
        temporal_masking_prob=cfg.training.temporal_masking_prob,
        gaussian_noise_std=cfg.training.gaussian_noise_std,
        gate_entropy_target=cfg.training.gate_entropy_target,
        calibrate=cfg.training.calibrate,
        calibrate_cold=cfg.training.calibrate_cold,
    )

    logger.info(
        "Starting Sovereign-X training on %s (%d train, %d val, %d cal-temp, %d cal-cold, taxonomy=%s)",
        device,
        len(train_ds),
        len(val_ds),
        len(cal_temp_ds) if cal_temp_ds else 0,
        len(cal_cold_ds) if cal_cold_ds else 0,
        use_taxonomy,
    )
    artifact_dir = trainer.fit(train_loader, val_loader, cal_loader=cal_temp_loader, cold_cal_loader=cal_cold_loader)
    logger.info("Training complete. Best model in %s", artifact_dir)


@cli.command()
@click.option("--model-path", required=True)
@click.option("--config", default="config/default.yaml")
@click.option("--feature-dir", default="data/sovereign_features")
@click.option("--output-path", default=None)
def evaluate(model_path: str, config: str, feature_dir: str, output_path: str):
    """Evaluate Sovereign-X on temporal and cold-start splits."""
    cfg = load_config(config)
    feature_dir = Path(feature_dir)
    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    with open(feature_dir / "split.json") as f:
        split = json.load(f)

    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    results = {}
    for mode, flag in [("temporal", "val"), ("cold_start", "test")]:
        bids = split.get(flag, [])
        if not bids:
            results[mode] = {"error": f"no {flag} backbones"}
            continue
        mode_df = seq_df.filter(pl.col("backbone_id").is_in(set(bids)) & (pl.col("observed") == 1.0))
        mode_bids = mode_df["backbone_id"].unique().to_list()
        if not mode_bids:
            results[mode] = {"error": "empty after filtering"}
            continue

        max_seq_len = cfg.model.max_seq_len

        taxonomy_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
        use_taxonomy = bool(taxonomy_vocab)

        ds = SovereignSequenceDataset(
            mode_df,
            mode_bids,
            max_seq_len,
            normalizer=(seq_means, seq_stds),
            static_normalizer=(static_means, static_stds),
            use_taxonomy=use_taxonomy,
        )

        def _eval_collate(b):
            return sequence_collate(b, max_seq_len)

        loader = DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            collate_fn=_eval_collate,
        )

        n_static = ds.items[0]["static"].size(-1)
        n_snapshot = ds.items[0]["seq"].size(-1)
        tax_vocab_sizes = None
        if use_taxonomy:
            tax_vocab_sizes = [
                len(v)
                for v in [
                    taxonomy_vocab.get("TAXONOMY_phylum", {}),
                    taxonomy_vocab.get("TAXONOMY_class", {}),
                    taxonomy_vocab.get("TAXONOMY_order", {}),
                    taxonomy_vocab.get("TAXONOMY_family", {}),
                    taxonomy_vocab.get("genus", {}),
                ]
            ]
        model = SovereignX(
            n_static=n_static,
            n_snapshot=n_snapshot,
            taxonomy_vocab_sizes=tax_vocab_sizes,
            taxonomy_embed_dim=cfg.model.taxonomy_embed_dim,
            static_dim=cfg.model.static_dim,
            temporal_dim=cfg.model.temporal_dim,
            hidden_dim=cfg.model.gru_hidden,
            num_layers=cfg.model.gru_layers,
            n_hazard=cfg.model.n_hazard_steps,
            dropout=cfg.model.dropout,
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)

        # Load appropriate Platt scaler
        platt_path = Path(model_path).with_name("platt.pt")
        platt_cold_path = Path(model_path).with_name("platt_cold.pt")
        trainer = SovereignXTrainer(model, device=device, calibrate=False)
        use_cold = mode == "cold_start"
        if use_cold and platt_cold_path.exists():
            trainer.cold_platt_scaler.load_state_dict(
                torch.load(platt_cold_path, map_location=device, weights_only=True)
            )
            logger.info("Loaded cold-start Platt scaler")
        elif platt_path.exists():
            trainer.platt_scaler.load_state_dict(torch.load(platt_path, map_location=device, weights_only=True))
        metrics = trainer.evaluate(loader, use_cold_scaler=use_cold)
        results[mode] = {
            "n_backbones": len(mode_bids),
            "n_snapshots": len(mode_df),
            "roc_auc": metrics.get("roc_auc", 0.0),
            "pr_auc": metrics.get("pr_auc", 0.0),
            "f1": metrics.get("f1", 0.0),
            "ece": metrics.get("ece", 0.0),
            "positive_rate": metrics.get("positive_rate", 0.0),
            "n_eval": metrics.get("n", 0),
        }

    output = {"model_path": model_path, "split_year": cfg.data.split_year, "results": results}
    out_path = output_path or str(Path(model_path).with_name("evaluation.json"))
    Path(out_path).write_text(json.dumps(output, indent=2))
    click.echo(f"Evaluation saved to {out_path}")


if __name__ == "__main__":
    cli()
