import numpy as np

from dna_sentinel.utils import false_positive_summary, task_score


def test_task_score_equal_weight():
    metrics = {"mobility_balanced_accuracy": 0.7, "amr_auroc": 0.9, "expansion_auroc": 0.8}
    assert task_score(metrics) == (0.7 + 0.9 + 0.8) / 3


def test_false_positive_summary_reports_rates_and_quantiles():
    summary = false_positive_summary(
        mobility_probs=np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]]),
        amr_probs=np.array([0.1, 0.8]),
        expansion_probs=np.array([0.2, 0.9]),
        risk_scores=np.array([0.1, 0.8]),
        threshold=0.5,
    )
    assert summary["false_mobile_rate"] == 0.5
    assert summary["false_amr_rate"] == 0.5
    assert summary["false_expansion_rate"] == 0.5
    assert summary["risk_q50"] == 0.45
    assert summary["risk_mean"] == 0.45
