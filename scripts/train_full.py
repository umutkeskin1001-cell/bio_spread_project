"""
BioSpread: Full training script with comprehensive evaluation.

Supports:
  --dry-run     Quick smoke test (1 epoch, small batch)
  --seed N      Reproducibility seed
  --ensemble N  Train N models with different seeds and aggregate
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import polars as pl
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from bio_spread.data.dataset import (
    SequenceBatchSampler,
    SequenceDataset,
    fit_normalizers,
    fit_static_normalizers,
    sequence_collate,
)
from bio_spread.data.snapshot import (
    disjoint_backbone_split,
    load_taxonomy_vocab,
)
from bio_spread.models import create_model
from bio_spread.models.components import predict_with_uncertainty
from bio_spread.models.trainer import BioSpreadTrainer
from bio_spread.utils.config import load_config, set_seed
from bio_spread.utils.metrics import expected_calibration_error as _ece

logger = logging.getLogger(__name__)


def compute_per_horizon_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, horizon: int,
    fixed_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute full metrics for a single hazard horizon.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        horizon: Horizon label (1-indexed).
        fixed_threshold: If provided, use this threshold for F2-optimal
            metrics instead of searching on the current split.  Pass the
            threshold found on a validation split to avoid leakage.
    """
    metrics: Dict[str, float] = {}
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        metrics[f"h{horizon}_roc_auc"] = 0.5
        metrics[f"h{horizon}_pr_auc"] = float(y_true.mean()) if len(y_true) > 0 else 0.0
        metrics[f"h{horizon}_f1"] = 0.0
        metrics[f"h{horizon}_precision"] = 0.0
        metrics[f"h{horizon}_recall"] = 0.0
        metrics[f"h{horizon}_brier"] = float((y_true - y_prob) ** 2).mean() if len(y_true) > 0 else 0.0
        metrics[f"h{horizon}_ece"] = 0.0
        metrics[f"h{horizon}_tp"] = 0
        metrics[f"h{horizon}_fp"] = 0
        metrics[f"h{horizon}_tn"] = 0
        metrics[f"h{horizon}_fn"] = 0
        metrics[f"h{horizon}_fnr"] = 0.0
        metrics[f"h{horizon}_fpr"] = 0.0
        metrics[f"h{horizon}_f2_threshold"] = 0.5
        metrics[f"h{horizon}_f2_f1"] = 0.0
        metrics[f"h{horizon}_f2_precision"] = 0.0
        metrics[f"h{horizon}_f2_recall"] = 0.0
        metrics[f"h{horizon}_f2_tp"] = 0
        metrics[f"h{horizon}_f2_fp"] = 0
        metrics[f"h{horizon}_f2_tn"] = 0
        metrics[f"h{horizon}_f2_fn"] = 0
        metrics[f"h{horizon}_n"] = len(y_true)
        return metrics

    metrics[f"h{horizon}_roc_auc"] = float(roc_auc_score(y_true, y_prob))
    metrics[f"h{horizon}_pr_auc"] = float(average_precision_score(y_true, y_prob))
    metrics[f"h{horizon}_brier"] = float(brier_score_loss(y_true, y_prob))

    y_pred_default = (y_prob > 0.5).astype(int)
    metrics[f"h{horizon}_f1"] = float(f1_score(y_true, y_pred_default, zero_division=0))
    metrics[f"h{horizon}_precision"] = float(precision_score(y_true, y_pred_default, zero_division=0))
    metrics[f"h{horizon}_recall"] = float(recall_score(y_true, y_pred_default, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default, labels=[0, 1]).ravel()
    metrics[f"h{horizon}_tp"] = int(tp)
    metrics[f"h{horizon}_fp"] = int(fp)
    metrics[f"h{horizon}_tn"] = int(tn)
    metrics[f"h{horizon}_fn"] = int(fn)
    metrics[f"h{horizon}_fnr"] = float(fn / max(fn + tp, 1))
    metrics[f"h{horizon}_fpr"] = float(fp / max(fp + tn, 1))

    if fixed_threshold is not None:
        best_t = fixed_threshold
    else:
        thresholds = np.linspace(0.01, 0.99, 99)
        f2_scores = [fbeta_score(y_true, y_prob > t, beta=2, zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f2_scores))
        best_t = float(thresholds[best_idx])

    y_pred_f2 = (y_prob > best_t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, y_pred_f2, labels=[0, 1]).ravel()
    metrics[f"h{horizon}_f2_threshold"] = best_t
    metrics[f"h{horizon}_f2_f1"] = float(f1_score(y_true, y_pred_f2, zero_division=0))
    metrics[f"h{horizon}_f2_precision"] = float(precision_score(y_true, y_pred_f2, zero_division=0))
    metrics[f"h{horizon}_f2_recall"] = float(recall_score(y_true, y_pred_f2, zero_division=0))
    metrics[f"h{horizon}_f2_tp"] = int(tp2)
    metrics[f"h{horizon}_f2_fp"] = int(fp2)
    metrics[f"h{horizon}_f2_tn"] = int(tn2)
    metrics[f"h{horizon}_f2_fn"] = int(fn2)

    metrics[f"h{horizon}_ece"] = _ece(y_true, y_prob)
    metrics[f"h{horizon}_n"] = len(y_true)
    return metrics


def compute_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, list]:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.any():
            bin_accs.append(float(y_true[mask].mean()))
            bin_confs.append(float(y_prob[mask].mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bins[i] + bins[i + 1]) / 2))
            bin_counts.append(0)
    return {"bin_edges": bins.tolist(), "accuracy": bin_accs, "confidence": bin_confs, "counts": bin_counts}


def evaluate_model(
    trainer: BioSpreadTrainer,
    loader: DataLoader,
    label: str = "",
    fixed_thresholds: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Comprehensive evaluation on a single split.

    Args:
        trainer: Fitted trainer.
        loader: DataLoader for the split.
        label: Human-readable label for the split.
        fixed_thresholds: Optional mapping ``{horizon: threshold}`` found
            on a held-out validation set.  When provided, F2-optimal
            metrics on the evaluated split use these thresholds instead
            of searching on the split itself, preventing leakage.
    """
    trainer.model.eval()
    all_probs = {h: [] for h in range(3)}
    all_targets = {h: [] for h in range(3)}
    all_bids = []

    with torch.no_grad():
        for batch in loader:
            static = batch["static"].to(trainer.device)
            seq = batch["seq"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            lengths = batch["seq_len"].to(trainer.device)
            targets = batch["hazard"].to(trainer.device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(trainer.device)

            out = trainer.model(static, seq, mask, taxonomy_idxs)
            probs = trainer._calibrated_probs(out.hazard_logits)

            idx = (lengths - 1).clamp(min=0)
            B = static.size(0)
            for h in range(3):
                last_h = probs[range(B), h].cpu().numpy()
                last_t = targets[range(B), idx, h].cpu().numpy()
                valid = last_t >= 0
                if valid.any():
                    all_probs[h].append(last_h[valid])
                    all_targets[h].append(last_t[valid])

            all_bids.extend(batch["backbone_ids"])

    metrics: Dict[str, Any] = {"label": label, "n_samples": 0}
    for h in range(3):
        if all_probs[h]:
            y_prob = np.concatenate(all_probs[h])
            y_true = np.concatenate(all_targets[h])
            ft = fixed_thresholds.get(h + 1) if fixed_thresholds else None
            h_metrics = compute_per_horizon_metrics(y_true, y_prob, h + 1, fixed_threshold=ft)
            metrics.update(h_metrics)
            metrics[f"h{h+1}_calibration"] = compute_calibration_curve(y_true, y_prob)
            if h == 2:
                metrics["n_samples"] = len(y_true)

    metrics["backbone_ids"] = all_bids

    mc_mean, mc_std = None, None
    if all_probs[2]:
        first_batch = next(iter(loader))
        s = first_batch["static"].to(trainer.device)
        q = first_batch["seq"].to(trainer.device)
        m = first_batch["mask"].to(trainer.device)
        t = first_batch.get("taxonomy")
        if t is not None:
            t = t.to(trainer.device)
        mc_mean, mc_std = predict_with_uncertainty(trainer.model, s, q, m, t, n_samples=10)
        metrics["mc_dropout_mean_std"] = {
            "mean": mc_mean.cpu().numpy().tolist(),
            "std": mc_std.cpu().numpy().tolist(),
        }

    return metrics


def compute_cold_start_metrics(
    trainer: BioSpreadTrainer,
    test_loader: DataLoader,
    test_backbone_ids: List[str],
    fixed_thresholds: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Evaluate cold-start performance on test set backbones using cold-start Platt scaler."""
    trainer.model.eval()
    all_probs = {h: [] for h in range(3)}
    all_targets = {h: [] for h in range(3)}

    with torch.no_grad():
        for batch in test_loader:
            static = batch["static"].to(trainer.device)
            seq = batch["seq"].to(trainer.device)
            mask = batch["mask"].to(trainer.device)
            lengths = batch["seq_len"].to(trainer.device)
            targets = batch["hazard"].to(trainer.device)
            taxonomy_idxs = batch.get("taxonomy")
            if taxonomy_idxs is not None:
                taxonomy_idxs = taxonomy_idxs.to(trainer.device)

            B = static.size(0)
            temporal_mask = torch.ones(B, dtype=torch.bool, device=trainer.device)
            out = trainer.model(static, seq, mask, taxonomy_idxs, temporal_mask=temporal_mask)
            probs = trainer._calibrated_probs(out.hazard_logits, use_cold_scaler=True)

            idx = (lengths - 1).clamp(min=0)
            for h in range(3):
                last_h = probs[range(B), h].cpu().numpy()
                last_t = targets[range(B), idx, h].cpu().numpy()
                valid = last_t >= 0
                if valid.any():
                    all_probs[h].append(last_h[valid])
                    all_targets[h].append(last_t[valid])

    cold_metrics = {}
    for h in range(3):
        if all_probs[h]:
            y_prob = np.concatenate(all_probs[h])
            y_true = np.concatenate(all_targets[h])
            ft = fixed_thresholds.get(h + 1) if fixed_thresholds else None
            cold_metrics.update(compute_per_horizon_metrics(y_true, y_prob, h + 1, fixed_threshold=ft))
            cold_metrics[f"h{h+1}_calibration"] = compute_calibration_curve(y_true, y_prob)
    return cold_metrics


def run_single_training(
    cfg,
    seed: int,
    dry_run: bool = False,
    artifact_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single training run and return all metrics."""
    set_seed(seed)
    logger.info(f"=== Training run seed={seed} {'[DRY RUN]' if dry_run else ''} ===")

    feature_dir = Path(cfg.data.feature_dir)
    seq_path = feature_dir / "sequences.tsv"
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequences not found at {seq_path}. Run `bio-spread prepare` first.")

    seq_df = pl.read_csv(seq_path, separator="\t")
    seq_df = seq_df.filter(pl.col("observed") > 0)
    all_backbone_ids = sorted(seq_df["backbone_id"].unique().to_list())

    split_path = feature_dir / "split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        train_ids = set(split["train"])
        val_ids = set(split["val"])
        test_ids = set(split["test"])
    else:
        raw_path = Path(cfg.data.backbones_path)
        raw_df = pl.read_csv(raw_path, separator="\t")
        train_ids, val_ids, test_ids = disjoint_backbone_split(
            raw_df, cfg.data.split_year, cfg.data.val_backbone_frac, cfg.data.test_backbone_frac,
        )
        split = {"train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)}
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=2))

    train_ids = sorted(train_ids & set(all_backbone_ids))
    val_ids = sorted(val_ids & set(all_backbone_ids))
    test_ids = sorted(test_ids & set(all_backbone_ids))

    train_df = seq_df.filter(pl.col("backbone_id").is_in(train_ids))
    seq_means, seq_stds = fit_normalizers(train_df)
    static_means, static_stds = fit_static_normalizers(train_df)

    norm_path = feature_dir / "normalizers.npz"
    static_norm_path = feature_dir / "static_normalizers.npz"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(norm_path, means=seq_means, stds=seq_stds)
    np.savez_compressed(static_norm_path, means=static_means, stds=static_stds)

    tax_vocab_path = feature_dir / "taxonomy_vocab.json"
    tax_vocab = load_taxonomy_vocab(tax_vocab_path) if tax_vocab_path.exists() else {}
    use_taxonomy = bool(tax_vocab)

    train_ds = SequenceDataset(
        seq_df, train_ids, cfg.model.max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    val_ds = SequenceDataset(
        seq_df, val_ids, cfg.model.max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )
    test_ds = SequenceDataset(
        seq_df, test_ids, cfg.model.max_seq_len,
        normalizer=(seq_means, seq_stds),
        static_normalizer=(static_means, static_stds),
        use_taxonomy=use_taxonomy,
    )

    n_static = train_ds.items[0]["static"].size(-1)
    n_snapshot = train_ds.items[0]["seq"].size(-1)

    batch_size = 8 if dry_run else cfg.training.batch_size
    epochs = 1 if dry_run else cfg.training.epochs

    train_sampler = SequenceBatchSampler(len(train_ds.items), batch_size, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: sequence_collate(b, cfg.model.max_seq_len),
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model = create_model(n_static, n_snapshot, cfg.model, taxonomy_vocab=tax_vocab if use_taxonomy else None)
    model.to(device)

    trainer = BioSpreadTrainer(
        model,
        device=device,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        epochs=epochs,
        patience=cfg.training.patience if not dry_run else 1,
        warmup_epochs=min(cfg.training.warmup_epochs, epochs),
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

    artifact_dir = trainer.fit(train_loader, val_loader)
    if artifact_subdir:
        artifact_dir = Path("artifacts") / artifact_subdir
        artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Evaluation ===")
    val_metrics = evaluate_model(trainer, val_loader, label="val")

    # Extract F2 thresholds from val set to avoid leakage on test set
    val_thresholds: Dict[int, float] = {}
    for h in range(1, 4):
        t_key = f"h{h}_f2_threshold"
        if t_key in val_metrics:
            val_thresholds[h] = val_metrics[t_key]

    test_metrics = evaluate_model(trainer, test_loader, label="test", fixed_thresholds=val_thresholds)

    cold_metrics: Dict[str, Any] = {}
    if test_ids:
        cold_metrics = compute_cold_start_metrics(trainer, test_loader, test_ids, fixed_thresholds=val_thresholds)

    all_metrics = {
        "seed": seed,
        "dry_run": dry_run,
        "val": val_metrics,
        "test": test_metrics,
        "cold_start": cold_metrics,
    }

    metrics_path = artifact_dir / "full_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_path}")

    return all_metrics


def print_summary_table(metrics: Dict[str, Any], label: str = ""):
    """Print a formatted summary table of key metrics."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{'=' * 70}")
    print(f"  {prefix}EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    for split_name in ["val", "test"]:
        if split_name not in metrics:
            continue
        m = metrics[split_name]
        print(f"\n  --- {split_name.upper()} ---")
        print(f"  {'Horizon':<10} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'ECE':>7} {'Brier':>7} {'FNR':>7}")
        for h in range(1, 4):
            auc = m.get(f"h{h}_roc_auc", 0.0)
            pr = m.get(f"h{h}_pr_auc", 0.0)
            f1 = m.get(f"h{h}_f1", 0.0)
            ece = m.get(f"h{h}_ece", 0.0)
            brier = m.get(f"h{h}_brier", 0.0)
            fnr = m.get(f"h{h}_fnr", 0.0)
            print(f"  H{h:<9} {auc:>7.4f} {pr:>7.4f} {f1:>7.4f} {ece:>7.4f} {brier:>7.4f} {fnr:>7.4f}")

        h3_tp = m.get("h3_tp", 0)
        h3_fp = m.get("h3_fp", 0)
        h3_tn = m.get("h3_tn", 0)
        h3_fn = m.get("h3_fn", 0)
        print("\n  Confusion Matrix (H3, threshold=0.5):")
        print(f"    TP={h3_tp}  FP={h3_fp}")
        print(f"    FN={h3_fn}  TN={h3_tn}")

        h3_f2_t = m.get("h3_f2_threshold", 0.5)
        h3_f2_f1 = m.get("h3_f2_f1", 0.0)
        h3_f2_prec = m.get("h3_f2_precision", 0.0)
        h3_f2_rec = m.get("h3_f2_recall", 0.0)
        h3_f2_tp = m.get("h3_f2_tp", 0)
        h3_f2_fp = m.get("h3_f2_fp", 0)
        h3_f2_tn = m.get("h3_f2_tn", 0)
        h3_f2_fn = m.get("h3_f2_fn", 0)
        print(f"\n  Confusion Matrix (H3, optimal F2 threshold={h3_f2_t:.2f}):")
        print(f"    TP={h3_f2_tp}  FP={h3_f2_fp}")
        print(f"    FN={h3_f2_fn}  TN={h3_f2_tn}")
        print(f"    F1={h3_f2_f1:.4f}  Precision={h3_f2_prec:.4f}  Recall={h3_f2_rec:.4f}")

    if "cold_start" in metrics and metrics["cold_start"]:
        cm = metrics["cold_start"]
        print("\n  --- COLD-START (Test) ---")
        print(f"  {'Horizon':<10} {'AUC':>7} {'PR-AUC':>7} {'F1':>7} {'ECE':>7}")
        for h in range(1, 4):
            auc = cm.get(f"h{h}_roc_auc", 0.0)
            pr = cm.get(f"h{h}_pr_auc", 0.0)
            f1 = cm.get(f"h{h}_f1", 0.0)
            ece = cm.get(f"h{h}_ece", 0.0)
            print(f"  H{h:<9} {auc:>7.4f} {pr:>7.4f} {f1:>7.4f} {ece:>7.4f}")

    print(f"\n{'=' * 70}")


@click.command()
@click.option("--config", default="config/default.yaml", help="Path to config YAML")
@click.option("--dry-run", is_flag=True, help="Quick smoke test (1 epoch, small batch)")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--ensemble", default=1, help="Number of models to train with different seeds")
@click.option("--output-dir", default="artifacts", help="Directory to save artifacts")
def main(config: str, dry_run: bool, seed: int, ensemble: int, output_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(config)

    if ensemble > 1:
        all_results = []
        for i in range(ensemble):
            run_seed = seed + i
            subdir = f"ensemble_seed_{run_seed}"
            result = run_single_training(cfg, run_seed, dry_run=dry_run, artifact_subdir=subdir)
            all_results.append(result)

        ensemble_metrics = {"ensemble_size": ensemble, "seeds": [seed + i for i in range(ensemble)], "runs": all_results}

        if len(all_results) > 1:
            avg_metrics = {"val": {}, "test": {}, "cold_start": {}}
            for split in ["val", "test", "cold_start"]:
                keys = set()
                for r in all_results:
                    if split in r:
                        keys.update(r[split].keys())
                for k in keys:
                    vals = [r.get(split, {}).get(k) for r in all_results]
                    vals = [v for v in vals if v is not None and isinstance(v, (int, float))]
                    if vals:
                        avg_metrics[split][f"{k}_mean"] = float(np.mean(vals))
                        avg_metrics[split][f"{k}_std"] = float(np.std(vals))

            ensemble_metrics["averages"] = avg_metrics
            print_summary_table(avg_metrics, label=f"ENSEMBLE (N={ensemble})")

        output_path = Path(output_dir) / "ensemble_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ensemble_metrics, f, indent=2, default=str)
        logger.info(f"Ensemble metrics saved to {output_path}")
    else:
        result = run_single_training(cfg, seed, dry_run=dry_run)
        print_summary_table(result)


if __name__ == "__main__":
    main()
