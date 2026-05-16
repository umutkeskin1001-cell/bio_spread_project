"""
LightGBM baseline for BioSpread.

Builds a flat feature matrix from SequenceDataset snapshots
using the precomputed sequences.tsv. Provides a simple sklearn baseline
for comparison with the deep learning model.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from bio_spread.constants import SNAPSHOT_FEATURE_COLS, STATIC_COLS

logger = logging.getLogger(__name__)


def main(config_path="config/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    feature_dir = Path(cfg["data"]["feature_dir"])
    seq_path = feature_dir / "sequences.tsv"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequences not found at {seq_path}. "
            "Run `bio-spread prepare` first."
        )

    df = pl.read_csv(seq_path, separator="\t")
    df = df.filter(pl.col("observed") > 0)
    logger.info("Loaded %d observed snapshots (%d backbones)",
                len(df), df["backbone_id"].n_unique())

    # Build flat feature matrix from snapshot features + static features.
    # Features in sequences.tsv are already normalized by dataset.py, so no
    # additional normalization is needed here.
    features = []
    targets = []
    for row in df.iter_rows(named=True):
        # Features in sequences.tsv are already normalized, use directly
        snap_vec = np.array([row.get(c, 0.0) for c in SNAPSHOT_FEATURE_COLS], dtype=np.float32)
        static_vec = np.array([row.get(c, 0.0) for c in STATIC_COLS], dtype=np.float32)

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

    # Load split -- backbone-disjoint split is required
    split_path = feature_dir / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"split.json required for backbone-disjoint split: {split_path}"
        )

    with open(split_path) as f:
        split = json.load(f)
    df_valid = df.filter(pl.col("observed") > 0)
    train_mask = df_valid["backbone_id"].is_in(split["train"]).to_numpy()
    val_mask = df_valid["backbone_id"].is_in(split["val"]).to_numpy()
    # Apply valid target filter
    train_mask = train_mask[valid]
    val_mask = val_mask[valid]

    if not (train_mask.sum() > 0 and val_mask.sum() > 0):
        logger.error("Split not applicable for filtered data. Check split.json.")
        return 0.5

    X_train, y_train = features[train_mask], targets[train_mask]
    X_val, y_val = features[val_mask], targets[val_mask]

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
    print("  LIGHTGBM BASELINE RESULTS")
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
