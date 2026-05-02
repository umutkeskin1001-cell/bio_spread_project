from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class ConformalPredictor:
    def __init__(self, model: Any, calib_df: np.ndarray, calib_labels: np.ndarray, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        calib_scores = model.predict_proba(calib_df)[:, 1]
        labels = np.asarray(calib_labels)
        nonconformity = np.where(labels == 1, 1 - calib_scores, calib_scores)
        self.qhat = float(np.quantile(nonconformity, 1 - alpha)) if len(nonconformity) else 0.5

    def predict_with_set(self, X: np.ndarray) -> dict[str, np.ndarray]:
        proba = self.model.predict_proba(X)[:, 1]
        lower = np.where(proba >= 1 - self.qhat, 1, 0)
        upper = np.ones_like(proba)
        return {"risk_probability": proba, "lower": lower, "upper": upper}


def compute_alarm_score(prob: np.ndarray, interval_width: np.ndarray, knownness: np.ndarray) -> NDArray[np.float64]:
    known = np.clip(np.asarray(knownness, dtype=float), 0.0, 1.0)
    out = np.asarray(prob, dtype=float) * (1.0 + np.asarray(interval_width, dtype=float)) * (1.0 + (1.0 - known))
    return np.asarray(out, dtype=np.float64)
