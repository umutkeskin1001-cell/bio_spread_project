from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClassificationMetrics:
    n_backbones: int
    n_positive: int
    prevalence: float
    roc_auc: float
    average_precision: float
    top_k_precision: float
    abstain_rate: float
    expected_calibration_error: float
    brier_score: float
    calibration_bins: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def labels_probabilities_from_predictions(predictions: list[Any]) -> tuple[NDArray[np.int64], NDArray[np.float64], list[str]]:
    labels = np.array([int(p.label_geo_spread) for p in predictions], dtype=np.int64)
    probabilities = np.array([float(p.risk_probability) for p in predictions], dtype=np.float64)
    tiers = [str(p.confidence_tier) for p in predictions]
    return labels, probabilities, tiers


def _calibration_summary(labels: NDArray[np.int64], probs: NDArray[np.float64], *, bins: int = 10) -> dict[str, Any]:
    total = len(probs)
    brier = float(np.mean((probs - labels) ** 2))
    boundaries = np.linspace(0, 1, bins + 1)
    indices = np.digitize(probs, boundaries[1:-1])
    ece = 0.0
    payload: list[dict[str, Any]] = []
    for i in range(bins):
        mask = indices == i
        count = int(np.sum(mask))
        if count == 0:
            payload.append(
                {
                    "bin_start": float(boundaries[i]),
                    "bin_end": float(boundaries[i + 1]),
                    "mean_prediction": None,
                    "observed_rate": None,
                    "count": 0,
                }
            )
            continue
        mean_pred = float(np.mean(probs[mask]))
        obs = float(np.mean(labels[mask]))
        ece += (count / total) * abs(mean_pred - obs)
        payload.append(
            {
                "bin_start": float(boundaries[i]),
                "bin_end": float(boundaries[i + 1]),
                "mean_prediction": mean_pred,
                "observed_rate": obs,
                "count": count,
            }
        )
    return {"expected_calibration_error": float(ece), "brier_score": brier, "calibration_bins": payload}


def compute_metrics(
    *,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    confidence_tiers: list[str] | None = None,
    bins: int = 10,
) -> ClassificationMetrics:
    if len(labels) == 0:
        raise ValueError("labels is empty")
    n_positive = int(np.sum(labels))
    prevalence = float(n_positive / len(labels))
    top_k = min(len(labels), max(10, int(round(len(labels) * 0.01))))
    ordered = np.argsort(-probabilities)
    top_k_precision = float(np.mean(labels[ordered[:top_k]]))
    abstain_rate = 0.0
    if confidence_tiers:
        abstain_rate = float(sum(1 for t in confidence_tiers if t == "review") / len(confidence_tiers))
    cal = _calibration_summary(labels, probabilities, bins=bins)
    return ClassificationMetrics(
        n_backbones=len(labels),
        n_positive=n_positive,
        prevalence=prevalence,
        roc_auc=_fast_auc(labels, probabilities),
        average_precision=_fast_average_precision(labels, probabilities),
        top_k_precision=top_k_precision,
        abstain_rate=abstain_rate,
        expected_calibration_error=float(cal["expected_calibration_error"]),
        brier_score=float(cal["brier_score"]),
        calibration_bins=list(cal["calibration_bins"]),
    )


def bootstrap_metrics(
    *,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    aps: list[float] = []
    n = len(labels)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        p = probabilities[idx]
        if int(np.sum(y)) == 0 or int(np.sum(y)) == len(y):
            continue
        aucs.append(_fast_auc(y, p))
        aps.append(_fast_average_precision(y, p))
    if not aucs or not aps:
        return {}
    alpha = (1.0 - confidence) / 2.0
    return {
        "bootstrap_confidence": float(confidence),
        "bootstrap_resamples": float(n_resamples),
        "bootstrap_roc_auc_ci_low": float(np.quantile(aucs, alpha)),
        "bootstrap_roc_auc_ci_high": float(np.quantile(aucs, 1.0 - alpha)),
        "bootstrap_average_precision_ci_low": float(np.quantile(aps, alpha)),
        "bootstrap_average_precision_ci_high": float(np.quantile(aps, 1.0 - alpha)),
    }


def _fast_auc(labels: NDArray[np.integer[Any]], scores: NDArray[np.floating[Any]]) -> float:
    y = labels.astype(np.int64)
    s = scores.astype(np.float64)
    n_positive = int(np.sum(y))
    n = len(y)
    n_negative = n - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_labels = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], n]
    # Average ranks handle ties exactly like sklearn's rank-based AUC while
    # staying in a tight NumPy kernel for bootstrap inner loops.
    average_ranks = (starts + 1 + ends) / 2.0
    positives_per_group = np.add.reduceat(sorted_labels, starts)
    positive_rank_sum = float(np.dot(average_ranks, positives_per_group))
    auc = (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    return float(auc)


def _fast_average_precision(labels: NDArray[np.integer[Any]], scores: NDArray[np.floating[Any]]) -> float:
    y = labels.astype(np.int64)
    s = scores.astype(np.float64)
    n_positive = int(np.sum(y))
    if n_positive == 0:
        return 0.0
    if n_positive == len(y):
        return 1.0 if int(y[0]) == 1 else 0.0
    order = np.argsort(s, kind="mergesort")[::-1]
    sorted_scores = s[order]
    sorted_labels = y[order]
    threshold_idxs = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    tps = np.cumsum(sorted_labels, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    precision = tps / (tps + fps)
    recall = tps / n_positive
    recall_step = recall - np.r_[0.0, recall[:-1]]
    return float(np.sum(recall_step * precision))


def evaluate_predictions(predictions: list[Any]) -> dict[str, float]:
    if not predictions:
        raise ValueError("Cannot evaluate an empty prediction set")
    labels, probabilities, tiers = labels_probabilities_from_predictions(predictions)
    metrics = compute_metrics(labels=labels, probabilities=probabilities, confidence_tiers=tiers)
    return {
        "n_backbones": float(metrics.n_backbones),
        "n_positive": float(metrics.n_positive),
        "prevalence": float(metrics.prevalence),
        "roc_auc": float(metrics.roc_auc),
        "average_precision": float(metrics.average_precision),
        "top_k_precision": float(metrics.top_k_precision),
        "abstain_rate": float(metrics.abstain_rate),
    }


def bootstrap_metric_intervals(
    predictions: list[Any],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> dict[str, float]:
    if not predictions:
        raise ValueError("Cannot bootstrap metrics for an empty prediction set")
    labels, probabilities, _ = labels_probabilities_from_predictions(predictions)
    return bootstrap_metrics(
        labels=labels,
        probabilities=probabilities,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def calibration_summary(predictions: list[Any], *, bins: int = 5) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot calibrate an empty prediction set")
    labels, probabilities, _ = labels_probabilities_from_predictions(predictions)
    metrics = compute_metrics(labels=labels, probabilities=probabilities, bins=bins)
    return {
        "expected_calibration_error": float(metrics.expected_calibration_error),
        "brier_score": float(metrics.brier_score),
        "n_bins": float(bins),
        "calibration_bins": metrics.calibration_bins,
    }
