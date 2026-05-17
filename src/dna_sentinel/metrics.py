from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


def binary_metrics(y_true: list[float] | np.ndarray, y_prob: list[float] | np.ndarray, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    out: dict[str, float] = {}
    if len(np.unique(y)) < 2:
        out[f"{prefix}_auroc"] = 0.5
        out[f"{prefix}_auprc"] = float(y.mean()) if y.size else 0.0
    else:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y, p))
        out[f"{prefix}_auprc"] = float(average_precision_score(y, p))
    out[f"{prefix}_brier"] = float(brier_score_loss(y, p)) if y.size else 0.0
    out[f"{prefix}_ece"] = expected_calibration_error(y, p)
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    if y.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return ece


def multiclass_metrics(y_true: list[int] | np.ndarray, logits: np.ndarray, prefix: str) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    pred = logits.argmax(axis=1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
