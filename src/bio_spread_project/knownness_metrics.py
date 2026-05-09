from __future__ import annotations

import numpy as np

from bio_spread_project.metrics import calibration_summary, evaluate_predictions
from bio_spread_project.model import Prediction


def knownness_slice_metrics(
    predictions: list[Prediction],
    *,
    quantile: float = 0.20,
    min_samples: int = 20,
) -> dict[str, float | None]:
    if not predictions:
        return {
            "knownness_slice_threshold": None,
            "knownness_slice_n": None,
            "knownness_slice_prevalence": None,
            "knownness_slice_roc_auc": None,
            "knownness_slice_average_precision": None,
            "knownness_slice_expected_calibration_error": None,
            "knownness_slice_brier_score": None,
            "knownness_slice_top_k_precision": None,
            "knownness_slice_abstain_rate": None,
        }
    knownness = np.asarray([float(p.knownness_score) for p in predictions], dtype=np.float64)
    threshold = float(np.quantile(knownness, quantile))
    subset = [p for p in predictions if float(p.knownness_score) <= threshold]
    payload: dict[str, float | None] = {
        "knownness_slice_threshold": threshold,
        "knownness_slice_n": float(len(subset)),
    }
    if len(subset) < min_samples:
        payload.update(
            {
                "knownness_slice_prevalence": None,
                "knownness_slice_roc_auc": None,
                "knownness_slice_average_precision": None,
                "knownness_slice_expected_calibration_error": None,
                "knownness_slice_brier_score": None,
                "knownness_slice_top_k_precision": None,
                "knownness_slice_abstain_rate": None,
            }
        )
        return payload
    base = evaluate_predictions(subset)
    cal = calibration_summary(subset)
    payload.update(
        {
            "knownness_slice_prevalence": float(base.get("prevalence", 0.0)),
            "knownness_slice_roc_auc": float(base["roc_auc"]) if base.get("roc_auc") is not None else None,
            "knownness_slice_average_precision": float(base["average_precision"]) if base.get("average_precision") is not None else None,
            "knownness_slice_expected_calibration_error": float(cal.get("expected_calibration_error", 0.0)),
            "knownness_slice_brier_score": float(cal.get("brier_score", 0.0)),
            "knownness_slice_top_k_precision": float(base.get("top_k_precision", 0.0)),
            "knownness_slice_abstain_rate": float(base.get("abstain_rate", 0.0)),
        }
    )
    return payload

