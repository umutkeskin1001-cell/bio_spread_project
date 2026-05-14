"""
LightGBM baseline for BioSpread Sovereign-X Pro.

Builds a flat feature matrix from SovereignSequenceDataset snapshots
using the precomputed sequences.tsv. Provides a simple sklearn baseline
for comparison with the deep learning model.
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score

from bio_spread_reborn.data.dataset import (
    SNAPSHOT_FEATURE_COLS, STATIC_COLS, load_normalizers,
)

logger = logging.getLogger(__name__)


def main(config_path="config/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    feature_dir = Path(cfg["data"]["feature_dir"])
    seq_path = feature_dir / "sequences.tsv"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequences not found at {seq_path}. "
            "Run `bio-spread sovereign_prepare` first."
        )

    df = pl.read_csv(seq_path, separator="\t")
    df = df.filter(pl.col("observed") > 0)
    logger.info("Loaded %d observed snapshots (%d backbones)",
                len(df), df["backbone_id"].n_unique())

    # Load normalizers
    norm_path = feature_dir / "normalizers.npz"
    if norm_path.exists():
        s_means, s_stds = load_normalizers(norm_path)
    else:
        s_means, s_stds = np.zeros(len(SNAPSHOT_FEATURE_COLS)), np.ones(len(SNAPSHOT_FEATURE_COLS))

    static_norm_path = feature_dir / "static_normalizers.npz"
    if static_norm_path.exists():
        st_means, st_stds = load_normalizers(static_norm_path)
    else:
        st_means, st_stds = np.zeros(len(STATIC_COLS)), np.ones(len(STATIC_COLS))

    # Build flat feature matrix from normalized snapshot features + static features
    features = []
    targets = []
    for row in df.iter_rows(named=True):
        snap_vec = np.array([row.get(c, 0.0) for c in SNAPSHOT_FEATURE_COLS], dtype=np.float32)
        snap_vec = (snap_vec - s_means) / s_stds

        static_vec = np.array([row.get(c, 0.0) for c in STATIC_COLS], dtype=np.float32)
        static_vec = (static_vec - st_means) / st_stds

        full_vec = np.concatenate([snap_vec, static_vec])
        features.append(full_vec)

        # Use hazard_3 as the binary target
        h3 = row.get("hazard_3", -1.0)
        targets.append(h3)

    features = np.array(features)
    targets = np.array(targets)

    # Filter valid targets
    valid = targets >= 0
    features = features[valid]
    targets = targets[valid].astype(int)

    logger.info("Feature matrix: %d samples x %d dims (%.3f positive)",
                len(targets), features.shape[1], targets.mean())

    # Load split
    split_path = feature_dir / "split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        df_valid = df.filter(pl.col("observed") > 0)
        train_mask = df_valid["backbone_id"].is_in(split["train"]).to_numpy()
        val_mask = df_valid["backbone_id"].is_in(split["val"]).to_numpy()
        # Apply valid target filter
        train_mask = train_mask[valid]
        val_mask = val_mask[valid]

        if train_mask.sum() > 0 and val_mask.sum() > 0:
            X_train, y_train = features[train_mask], targets[train_mask]
            X_val, y_val = features[val_mask], targets[val_mask]
        else:
            logger.warning("Split not applicable for filtered data, falling back to temporal split")
            train_mask = None
    else:
        train_mask = None

    if train_mask is None:
        # Temporal split
        years = df.filter(pl.col("observed") > 0)["year"].to_numpy()[valid]
        split_year = cfg["data"]["split_year"]
        train_idx = years < split_year
        val_idx = years >= split_year
        X_train, y_train = features[train_idx], targets[train_idx]
        X_val, y_val = features[val_idx], targets[val_idx]

    logger.info("Train: %d (%.3f pos) | Val: %d (%.3f pos)",
                len(y_train), y_train.mean(), len(y_val), y_val.mean())

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        logger.error("Single-class split detected. Cannot train.")
        return 0.5

    # Train LightGBM
    import lightgbm as lgb

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=1000,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_pred)

    print(f"\n{'=' * 40}")
    print(f"  LIGHTGBM BASELINE RESULTS")
    print(f"{'=' * 40}")
    print(f"  Temporal AUC:     {auc:.4f}")
    print(f"  PR AUC:           {pr_auc:.4f}")
    print(f"  Val positive rate: {y_val.mean():.3f}")
    print(f"  Features:         {X_train.shape[1]} dims")
    print(f"  Best iteration:   {model.best_iteration_}")
    print(f"{'=' * 40}\n")

    # Save model
    import pickle
    model_path = Path("best_model_lgb.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", model_path)

    # Save metrics
    metrics = {
        "model": "lightgbm",
        "temporal_auc": float(auc),
        "temporal_pr_auc": float(pr_auc),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "positive_rate_val": float(y_val.mean()),
        "feature_dim": int(X_train.shape[1]),
    }
    metrics_path = feature_dir / "lgb_baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", metrics_path)

    return auc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
