import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from bio_spread_project.metrics import _fast_auc, _fast_average_precision, bootstrap_metrics, evaluate_predictions
from bio_spread_project.model import Prediction


def _prediction(index: int, label: int, score: float) -> Prediction:
    return Prediction(
        model_name="test",
        backbone_id=f"bb_{index}",
        risk_probability=score,
        confidence_tier="medium",
        label_geo_spread=label,
        knownness_score=1.0,
        n_new_countries_future=label,
        explanation="",
    )


def test_fast_auc_and_average_precision_match_sklearn_with_ties():
    labels = np.array([1, 1, 0, 0], dtype=int)
    scores = np.array([0.8, 0.5, 0.8, 0.1], dtype=float)

    assert _fast_auc(labels, scores) == pytest.approx(float(roc_auc_score(labels, scores)))
    assert _fast_average_precision(labels, scores) == pytest.approx(float(average_precision_score(labels, scores)))


def test_fast_metrics_handle_single_class_without_sklearn_errors():
    labels = np.array([1, 1, 1], dtype=int)
    scores = np.array([0.2, 0.7, 0.9], dtype=float)

    assert _fast_auc(labels, scores) == 0.5
    assert _fast_average_precision(labels, scores) == 1.0


def test_evaluate_predictions_uses_vectorized_metric_path_for_large_inputs():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=50_000)
    scores = rng.random(50_000)
    predictions = [_prediction(index, int(label), float(score)) for index, (label, score) in enumerate(zip(labels, scores))]

    metrics = evaluate_predictions(predictions)

    assert metrics["roc_auc"] == pytest.approx(float(roc_auc_score(labels, scores)))
    assert metrics["average_precision"] == pytest.approx(float(average_precision_score(labels, scores)))


def test_bootstrap_metrics_use_fast_rank_kernels():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=1_000, dtype=np.int64)
    scores = rng.random(1_000)

    intervals = bootstrap_metrics(labels=labels, probabilities=scores, n_resamples=1_000)

    assert intervals["bootstrap_roc_auc_ci_low"] < intervals["bootstrap_roc_auc_ci_high"]
    assert intervals["bootstrap_average_precision_ci_low"] < intervals["bootstrap_average_precision_ci_high"]
