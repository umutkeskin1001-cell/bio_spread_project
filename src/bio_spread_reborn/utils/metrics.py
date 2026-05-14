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
    """ECE for binary classification."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if len(y_true) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(y_prob, bins, right=True) - 1, 0, n_bins - 1)
    ece = 0.0
    for i in range(n_bins):
        mask = ids == i
        if mask.any():
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return float(ece)
