"""
Sovereign-X: Clean metrics. No dead code.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Standard binary classification metrics. Clean and minimal."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob > 0.5).astype(int)

    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(y_true.mean())

    metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["specificity"] = float(tn / max(tn + fp, 1))
    metrics["balanced_accuracy"] = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    metrics["positive_rate"] = float(y_true.mean())
    return metrics


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE for binary classification using adaptive binning.

    Uses confidence-weighted binning: each bin spans equal probability mass
    rather than equal-width intervals, ensuring bins contain roughly
    the same number of samples.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32).clip(0, 1)
    if len(y_true) < n_bins:
        return 0.0
    # Adaptive bin edges based on probability quantiles
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    ids = np.digitize(y_prob, bin_edges, right=False) - 1
    ids = ids.clip(0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = ids == i
        if mask.any():
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)
