from __future__ import annotations

import numpy as np

from bio_spread_project.low_knownness_enhancement import enhance_low_knownness_predictions
from bio_spread_project.model import Prediction


def _mk_prediction(idx: int, prob: float, label: int, knownness: float) -> Prediction:
    return Prediction(
        model_name="m",
        backbone_id=f"bb{idx}",
        risk_probability=float(prob),
        confidence_tier="medium",
        label_geo_spread=int(label),
        knownness_score=float(knownness),
        n_new_countries_future=0,
        explanation="",
        alarm_score=float(abs(prob - 0.5)),
        meta={"alarm_score": float(abs(prob - 0.5))},
    )


def test_low_knownness_enhancement_preserves_global_metrics_within_guardrail() -> None:
    rng = np.random.default_rng(42)
    preds: list[Prediction] = []
    for i in range(220):
        known = float(rng.uniform(0.3, 0.95))
        label = int(rng.random() < 0.45)
        prob = float(np.clip(0.2 + 0.6 * label + rng.normal(0, 0.20), 0.001, 0.999))
        if known < 0.52:
            prob = float(np.clip(prob + (0.15 if label == 0 else -0.10), 0.001, 0.999))
        preds.append(_mk_prediction(i, prob, label, known))

    enhanced, summary = enhance_low_knownness_predictions(preds)
    assert len(enhanced) == len(preds)
    if summary.enabled:
        assert summary.global_auc_after >= summary.global_auc_before - 0.0031
        assert summary.global_ap_after >= summary.global_ap_before - 0.0051

