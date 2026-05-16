from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob > threshold).astype(int)

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
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32).clip(0, 1)
    if len(y_true) < n_bins:
        return 0.0
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(0, 1, n_bins + 1)
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


def bootstrap_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> Optional[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 10 or len(np.unique(y_true)) < 2:
        return None
    auc_scores = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) > 1:
            try:
                auc_scores[i] = roc_auc_score(y_true[idx], y_prob[idx])
            except ValueError:
                auc_scores[i] = 0.5
        else:
            auc_scores[i] = 0.5
    auc_scores = auc_scores[auc_scores > 0]
    if len(auc_scores) < 2:
        return None
    alpha = 0.05
    p_low, p_high = (alpha / 2) * 100, (1 - alpha / 2) * 100
    return {
        "roc_auc": float(np.median(auc_scores)),
        "ci_low": float(max(0, np.percentile(auc_scores, p_low))),
        "ci_high": float(min(1, np.percentile(auc_scores, p_high))),
        "std": float(auc_scores.std()),
    }


class MondrianConformalManager:
    """
    v4 Mondrian Conformal Prediction.
    Provides rigorous coverage guarantees for different subgroups (e.g. Cold-start vs Warm).
    """
    def __init__(self, alpha: float = 0.1, n_classes: int = 2):
        self.alpha = alpha
        self.n_classes = n_classes
        self.cal_scores: Dict[Any, np.ndarray] = {} # Map category -> non-conformity scores
        self.quantiles: Dict[Any, float] = {}

    def calibrate(self, probs: np.ndarray, targets: np.ndarray, categories: np.ndarray):
        """
        Calibrate on a held-out set.
        Categories can be [0, 1] for Cold/Warm or Taksonomi classes.
        """
        unique_cats = np.unique(categories)
        for cat in unique_cats:
            mask = (categories == cat)
            if not mask.any(): continue
            
            p_cat = probs[mask]
            t_cat = targets[mask]
            
            # Non-conformity score: 1 - P(correct_class)
            # For binary: if t=1, score=1-p; if t=0, score=p
            scores = np.where(t_cat == 1, 1 - p_cat, p_cat)
            self.cal_scores[cat] = np.sort(scores)
            
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = np.clip(q_level, 0, 1)
            self.quantiles[cat] = np.quantile(scores, q_level, interpolation='higher')

    def predict(self, probs: np.ndarray, categories: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns [lower_bound, upper_bound] for each sample."""
        lowers = np.zeros_like(probs)
        uppers = np.ones_like(probs)
        
        for cat in self.quantiles:
            mask = (categories == cat)
            if not mask.any(): continue
            
            q = self.quantiles[cat]
            # Prediction set: {y | score(x, y) <= q}
            # For y=1: 1-p <= q  => p >= 1-q
            # For y=0: p <= q
            # This gives us a prediction set. To get bounds:
            lowers[mask] = (probs[mask] - q).clip(0, 1)
            uppers[mask] = (probs[mask] + q).clip(0, 1)
            
        return lowers, uppers

    def get_set_size(self, probs: np.ndarray, categories: np.ndarray) -> np.ndarray:
        """Returns number of labels in the prediction set (0, 1, or 2)."""
        sizes = np.zeros(len(probs))
        for cat in self.quantiles:
            mask = (categories == cat)
            if not mask.any(): continue
            q = self.quantiles[cat]
            include_0 = probs[mask] <= q
            include_1 = (1 - probs[mask]) <= q
            sizes[mask] = include_0.astype(int) + include_1.astype(int)
        return sizes


class StabilityMonitor:
    """Monitors model stability over time or across versions."""
    @staticmethod
    def compute_psi(initial: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Population Stability Index."""
        initial_percents = np.histogram(initial, bins=bins, range=(0, 1))[0] / len(initial)
        current_percents = np.histogram(current, bins=bins, range=(0, 1))[0] / len(current)
        
        # Avoid division by zero
        initial_percents = np.clip(initial_percents, 1e-6, 1.0)
        current_percents = np.clip(current_percents, 1e-6, 1.0)
        
        psi = np.sum((initial_percents - current_percents) * np.log(initial_percents / current_percents))
        return float(psi)
