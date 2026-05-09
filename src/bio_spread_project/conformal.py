from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class ConformalPredictor:
    def __init__(self, model: Any, calib_df: NDArray[Any], calib_labels: NDArray[Any], alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        labels = np.asarray(calib_labels)
        if hasattr(model, "predict_proba"):
            calib_scores = model.predict_proba(calib_df)[:, 1]
        else:
            raw_scores = model.predict(calib_df)
            calib_scores = 1.0 / (1.0 + np.exp(-raw_scores))
        nonconformity = np.where(labels == 1, 1 - calib_scores, calib_scores)
        self.qhat = float(np.quantile(nonconformity, 1 - alpha)) if len(nonconformity) else 0.5

    def predict_with_set(self, X: NDArray[Any]) -> dict[str, NDArray[Any]]:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[:, 1]
        else:
            raw_scores = self.model.predict(X)
            proba = 1.0 / (1.0 + np.exp(-raw_scores))
        lower = np.where(proba >= 1 - self.qhat, 1, 0)
        upper = np.ones_like(proba)
        return {"risk_probability": proba, "lower": lower, "upper": upper}


def compute_alarm_score(prob: NDArray[Any], interval_width: NDArray[Any], knownness: NDArray[Any]) -> NDArray[np.float64]:
    known = np.clip(np.asarray(knownness, dtype=float), 0.0, 1.0)
    out = np.asarray(prob, dtype=float) * (1.0 + np.asarray(interval_width, dtype=float)) * (1.0 + (1.0 - known))
    return np.asarray(out, dtype=np.float64)
