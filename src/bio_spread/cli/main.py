from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from bio_spread.constants import TAXONOMY_COLS
from bio_spread.data.loader import get_device
from bio_spread.data.dataset import (
    SequenceBatchSampler,
    SequenceDataset,
    fit_normalizers,
    fit_static_normalizers,
    load_normalizers,
    save_normalizers,
    sequence_collate,
)
from bio_spread.data.snapshot import (
    build_categorical_vocabs,
    build_sequences,
    build_taxonomy_vocab,
    load_categorical_vocabs,
    load_taxonomy_vocab,
    random_backbone_split,
    save_categorical_vocabs,
    save_taxonomy_vocab,
)
from bio_spread.models import create_model
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biospread")


@click.group()
def cli():
    ...


@cli.command()
@click.option("--config", default="config/default.yaml")
@click.option("--output-dir", default="data/features")
def prepare(config: str, output_dir: str):
    cfg = load_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pl.read_csv(cfg.data.backbones_path, separator="\t")
    if "resolved_year" in raw.columns and "year" not in raw.columns:
        raw = raw.rename({"resolved_year": "year"})
    meta = raw.unique(subset=["backbone_id"])
    split_year = cfg.data.split_year

    train_bids, val_bids, test_bids = random_backbone_split(
        raw,
        train_frac=1.0 - cfg.data.val_backbone_frac - cfg.data.test_backbone_frac,
        val_frac=cfg.data.val_backbone_frac,
        seed=42,
    )
    _ = split_year  # kept for compatibility

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

    taxonomy_vocab = build_taxonomy_vocab(raw.filter(pl.col("backbone_id").is_in(train_bids)))
    save_taxonomy_vocab(taxonomy_vocab, output_dir / "taxonomy_vocab.json")

    categorical_vocabs = build_categorical_vocabs(
        raw.filter(pl.col("backbone_id").is_in(train_bids))
    )
    save_categorical_vocabs(categorical_vocabs, output_dir / "categorical_vocabs.json")

    sequences = build_sequences(
        raw, meta, set(all_bids.keys()),
        taxonomy_vocab=taxonomy_vocab,
        categorical_vocabs=categorical_vocabs,
        **seq_kwargs,
    )
    sequences = sequences.with_columns(
        pl.col("backbone_id")
        .map_elements(lambda bid: all_bids.get(bid, "unknown"), return_dtype=pl.Utf8)
        .alias("split")
    )

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
@click.option("--feature-dir", default="data/features")
def train(config: str, feature_dir: str):
    cfg = load_config(config)
    feature_dir = Path(feature_dir)

    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    taxonomy_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    use_taxonomy = bool(taxonomy_vocab)

    categorical_vocabs = load_categorical_vocabs(feature_dir / "categorical_vocabs.json")
    use_categorical = bool(categorical_vocabs)

    train_df = seq_df.filter((pl.col("split") == "train") & (pl.col("observed") == 1.0))
    val_df = seq_df.filter((pl.col("split") == "val") & (pl.col("observed") == 1.0))

    val_bids = val_df["backbone_id"].unique().to_list()
    np.random.seed(cfg.training.seed)
    np.random.shuffle(val_bids)
    n_val = len(val_bids)
    n_cal_temp = max(1, n_val // 7)
    n_cal_cold = max(1, n_val // 7)
    cal_temp_bids = set(val_bids[:n_cal_temp])
    cal_cold_bids = set(val_bids[n_cal_temp : n_cal_temp + n_cal_cold])
    early_stop_bids = set(val_bids[n_cal_temp + n_cal_cold :])

    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")

    max_seq_len = cfg.model.max_seq_len

    train_ds = SequenceDataset(
        train_df,
        train_df["backbone_id"].unique().to_list(),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
        use_categorical=use_categorical,
    )
    val_ds = SequenceDataset(
        val_df.filter(pl.col("backbone_id").is_in(early_stop_bids)),
        list(early_stop_bids),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
        use_categorical=use_categorical,
    )
    cal_temp_ds = SequenceDataset(
        val_df.filter(pl.col("backbone_id").is_in(cal_temp_bids)),
        list(cal_temp_bids),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
        use_categorical=use_categorical,
    )
    cal_cold_ds = SequenceDataset(
        val_df.filter(pl.col("backbone_id").is_in(cal_cold_bids)),
        list(cal_cold_bids),
        max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
        use_categorical=use_categorical,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Empty training or validation dataset")

    sampler = SequenceBatchSampler(
        len(train_ds.items),
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

    model = create_model(n_static, n_snapshot, cfg.model,
                         taxonomy_vocab=taxonomy_vocab if use_taxonomy else None,
                         categorical_vocabs=categorical_vocabs if use_categorical else None)
    device = get_device()

    trainer = BioSpreadTrainer(
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
        lambda_kd=cfg.training.lambda_kd,
        lambda_all=cfg.training.lambda_all,
        lambda_gate=cfg.training.lambda_gate,
        temporal_masking_prob=cfg.training.temporal_masking_prob,
        gaussian_noise_std=cfg.training.gaussian_noise_std,
        gate_entropy_target=cfg.training.gate_entropy_target,
        calibrate=cfg.training.calibrate,
        calibrate_cold=cfg.training.calibrate_cold,
        use_adaptive_loss=cfg.training.use_adaptive_loss,
        use_hard_negative_mining=cfg.training.use_hard_negative_mining,
        use_curriculum=cfg.training.use_curriculum,
    )

    logger.info(
        "Starting BioSpread training on %s (%d train, %d val, %d cal-temp, %d cal-cold, taxonomy=%s)",
        device,
        len(train_ds),
        len(val_ds),
        len(cal_temp_ds) if cal_temp_ds else 0,
        len(cal_cold_ds) if cal_cold_ds else 0,
        use_taxonomy,
    )
    artifact_dir = trainer.fit(train_loader, val_loader, cal_loader=cal_temp_loader, cold_cal_loader=cal_cold_loader)
    logger.info("Training complete. Best model in %s", artifact_dir)


def _eval_split(
    seq_df: pl.DataFrame, bids: list[str], model_path: str, cfg, device: str,
    seq_means: np.ndarray, seq_stds: np.ndarray,
    static_means: np.ndarray, static_stds: np.ndarray,
    taxonomy_vocab: dict, categorical_vocabs: dict,
    platt_path: Path, platt_cold_path: Path,
    force_temporal_mask: bool = False, use_cold_scaler: bool = False,
) -> dict:
    mode_df = seq_df.filter(pl.col("backbone_id").is_in(set(bids)) & (pl.col("observed") == 1.0))
    mode_bids = mode_df["backbone_id"].unique().to_list()
    if not mode_bids:
        return {"error": "empty after filtering", "n_backbones": len(bids), "n_snapshots": 0}

    use_taxonomy = bool(taxonomy_vocab)
    use_categorical = bool(categorical_vocabs)
    max_seq_len = cfg.model.max_seq_len
    ds = SequenceDataset(
        mode_df, mode_bids, max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
        use_categorical=use_categorical,
    )
    loader = DataLoader(
        ds, batch_size=cfg.training.batch_size,
        collate_fn=lambda b: sequence_collate(b, max_seq_len),
    )

    n_static = ds.items[0]["static"].size(-1)
    n_snapshot = ds.items[0]["seq"].size(-1)
    model = create_model(n_static, n_snapshot, cfg.model,
                         taxonomy_vocab if use_taxonomy else None,
                         categorical_vocabs if use_categorical else None)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    trainer = BioSpreadTrainer(model, device=device, calibrate=False)
    if use_cold_scaler and platt_cold_path.exists():
        st = torch.load(platt_cold_path, map_location=device, weights_only=True)
        for h in range(3):
            if f"scaler_h{h}" in st:
                trainer.cold_platt_scalers[h].load_state_dict(st[f"scaler_h{h}"])
    elif platt_path.exists():
        st = torch.load(platt_path, map_location=device, weights_only=True)
        for h in range(3):
            if f"scaler_h{h}" in st:
                trainer.platt_scalers[h].load_state_dict(st[f"scaler_h{h}"])

    metrics = trainer.evaluate(loader, use_cold_scaler=use_cold_scaler,
                               force_temporal_mask=force_temporal_mask)
    return {
        "n_backbones": len(mode_bids),
        "n_snapshots": len(mode_df),
        "roc_auc": metrics.get("roc_auc", 0.0),
        "roc_auc_h1": metrics.get("roc_auc_h1", 0.0),
        "roc_auc_h2": metrics.get("roc_auc_h2", 0.0),
        "roc_auc_h3": metrics.get("roc_auc_h3", 0.0),
        "pr_auc": metrics.get("pr_auc", 0.0),
        "f1": metrics.get("f1", 0.0),
        "ece": metrics.get("ece", 0.0),
        "positive_rate": metrics.get("positive_rate", 0.0),
        "n_eval": metrics.get("n", 0),
        "roc_auc_ci_low": metrics.get("roc_auc_ci_low", 0.0),
        "roc_auc_ci_high": metrics.get("roc_auc_ci_high", 0.0),
    }


@cli.command()
@click.option("--model-path", required=True)
@click.option("--config", default="config/default.yaml")
@click.option("--feature-dir", default="data/features")
@click.option("--output-path", default=None)
def evaluate(model_path: str, config: str, feature_dir: str, output_path: str):
    cfg = load_config(config)
    feature_dir = Path(feature_dir)
    seq_df = pl.read_csv(feature_dir / "sequences.tsv", separator="\t")
    with open(feature_dir / "split.json") as f:
        split = json.load(f)

    seq_means, seq_stds = load_normalizers(feature_dir / "normalizers.npz")
    static_means, static_stds = load_normalizers(feature_dir / "static_normalizers.npz")

    device = get_device()
    platt_path = Path(model_path).with_name("platt.pt")
    platt_cold_path = Path(model_path).with_name("platt_cold.pt")
    taxonomy_vocab = load_taxonomy_vocab(feature_dir / "taxonomy_vocab.json")
    categorical_vocabs = load_categorical_vocabs(feature_dir / "categorical_vocabs.json")

    results = {}

    val_bids = split.get("val", [])
    test_bids = split.get("test", [])

    # temporal: val backbones (seen during training, full temporal features)
    if val_bids:
        r = _eval_split(seq_df, val_bids, model_path, cfg, device,
                        seq_means, seq_stds, static_means, static_stds,
                        taxonomy_vocab, categorical_vocabs, platt_path, platt_cold_path,
                        force_temporal_mask=False, use_cold_scaler=False)
        results["temporal"] = r
        logger.info("Temporal (val, n=%d): AUC=%.4f F1=%.4f ECE=%.4f",
                     r.get("n_eval", 0), r.get("roc_auc", 0), r.get("f1", 0), r.get("ece", 0))

    # unseen: test backbones (NEVER seen during training, full temporal features)
    if test_bids:
        r = _eval_split(seq_df, test_bids, model_path, cfg, device,
                        seq_means, seq_stds, static_means, static_stds,
                        taxonomy_vocab, categorical_vocabs, platt_path, platt_cold_path,
                        force_temporal_mask=False, use_cold_scaler=False)
        results["unseen"] = r
        logger.info("Unseen (test, n=%d): AUC=%.4f F1=%.4f ECE=%.4f",
                     r.get("n_eval", 0), r.get("roc_auc", 0), r.get("f1", 0), r.get("ece", 0))

    # cold: test backbones (NEVER seen, temporal features MASKED)
    if test_bids:
        r = _eval_split(seq_df, test_bids, model_path, cfg, device,
                        seq_means, seq_stds, static_means, static_stds,
                        taxonomy_vocab, categorical_vocabs, platt_path, platt_cold_path,
                        force_temporal_mask=True, use_cold_scaler=True)
        results["cold"] = r
        logger.info("Cold-start (test, n=%d): AUC=%.4f F1=%.4f ECE=%.4f",
                     r.get("n_eval", 0), r.get("roc_auc", 0), r.get("f1", 0), r.get("ece", 0))

    output = {"model_path": model_path, "split_year": cfg.data.split_year, "results": results}
    out_path = output_path or str(Path(model_path).with_name("evaluation.json"))
    Path(out_path).write_text(json.dumps(output, indent=2))
    click.echo(f"Evaluation saved to {out_path}")


if __name__ == "__main__":
    cli()
