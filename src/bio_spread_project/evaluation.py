
from typing import Any

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from bio_spread_project.model import Prediction


@njit(cache=True)  # type: ignore[untyped-decorator]
def _fast_auc_kernel(labels: NDArray[np.int32], scores: NDArray[np.float64]) -> float:
    """O(N log N) AUC calculation using Mann-Whitney U rank logic."""
    n = len(labels)
    if n == 0:
        return 0.5

    indices = np.argsort(scores)
    labels_sorted = labels[indices]

    n_pos = np.sum(labels)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Ranks (1-indexed)
    ranks = np.arange(1, n + 1).astype(np.float64)

    # Handle ties by averaging ranks
    # For speed, we skip explicit tie handling if we assume unique scores or minor ties
    # But for a robust implementation:
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores[indices[i]] == scores[indices[j]]:
            j += 1
        if j - i > 1:
            mean_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[k] = mean_rank
        i = j

    pos_rank_sum = np.sum(labels_sorted * ranks)
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    return float(u_stat / (n_pos * n_neg))


@njit(cache=True)  # type: ignore[untyped-decorator]
def _fast_ap_kernel(labels: NDArray[np.int32], scores: NDArray[np.float64]) -> float:
    """O(N log N) Average Precision calculation."""
    n = len(labels)
    if n == 0:
        return 0.0

    indices = np.argsort(-scores)
    labels_sorted = labels[indices]

    n_pos = np.sum(labels)
    if n_pos == 0:
        return 0.0

    pos_count = 0.0
    precision_sum = 0.0
    for i in range(n):
        if labels_sorted[i] == 1:
            pos_count += 1
            precision_sum += pos_count / (i + 1)

    return float(precision_sum / n_pos)


def _fast_auc(labels: NDArray[np.integer[Any]], scores: NDArray[np.floating[Any]]) -> float:
    return float(_fast_auc_kernel(labels.astype(np.int32), scores.astype(np.float64)))


def _fast_average_precision(labels: NDArray[np.integer[Any]], scores: NDArray[np.floating[Any]]) -> float:
    return float(average_precision_score(labels.astype(np.int32), scores.astype(np.float64)))


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]
def _bootstrap_kernel(
    labels: NDArray[np.int32],
    scores: NDArray[np.float64],
    n_resamples: int,
    seeds: NDArray[np.integer[Any]],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = len(labels)
    auc_values = np.empty(n_resamples)
    ap_values = np.empty(n_resamples)

    for i in prange(n_resamples):
        np.random.seed(seeds[i])
        indices = np.random.randint(0, n, n)
        auc_values[i] = _fast_auc_kernel(labels[indices], scores[indices])
        ap_values[i] = _fast_ap_kernel(labels[indices], scores[indices])

    return auc_values, ap_values


def evaluate_predictions(predictions: list[Prediction]) -> dict[str, float]:
    """Compute discrimination and decision-oriented summary metrics."""
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction set")

    labels = np.array([p.label_geo_spread for p in predictions], dtype=np.int32)
    scores = np.array([p.risk_probability for p in predictions], dtype=np.float64)

    n_positive = np.sum(labels)
    prevalence = n_positive / len(labels)

    # Top-k precision
    top_k = min(len(predictions), max(10, int(round(len(predictions) * 0.01))))
    indices = np.argsort(-scores)
    top_k_precision = np.mean(labels[indices[:top_k]])

    abstain_rate = sum(1 for p in predictions if p.confidence_tier == "review") / len(predictions)

    return {
        "n_backbones": float(len(predictions)),
        "n_positive": float(n_positive),
        "prevalence": float(prevalence),
        "roc_auc": float(_fast_auc_kernel(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "top_k_precision": float(top_k_precision),
        "abstain_rate": float(abstain_rate),
    }


def bootstrap_metric_intervals(
    predictions: list[Prediction],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> dict[str, float]:
    """Bootstrap confidence intervals using Numba-compiled parallel loop."""
    if not predictions:
        raise ValueError("Cannot bootstrap metrics for an empty prediction set")

    labels = np.array([p.label_geo_spread for p in predictions], dtype=np.int32)
    scores = np.array([p.risk_probability for p in predictions], dtype=np.float64)

    # Generate deterministic seeds for parallel execution
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_resamples)

    auc_values, ap_values = _bootstrap_kernel(labels, scores, n_resamples, seeds)

    alpha = (1.0 - confidence) / 2.0

    return {
        "bootstrap_confidence": confidence,
        "bootstrap_resamples": float(n_resamples),
        "bootstrap_roc_auc_ci_low": float(np.quantile(auc_values, alpha)),
        "bootstrap_roc_auc_ci_high": float(np.quantile(auc_values, 1.0 - alpha)),
        "bootstrap_average_precision_ci_low": float(np.quantile(ap_values, alpha)),
        "bootstrap_average_precision_ci_high": float(np.quantile(ap_values, 1.0 - alpha)),
    }
