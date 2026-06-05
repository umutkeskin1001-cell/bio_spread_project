
import numpy as np

from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.utils import LabeledSequence, evaluate_records, false_positive_summary, task_score


def test_task_score_equal_weight():
    assert task_score({"mobility_balanced_accuracy": 0.7, "amr_auroc": 0.9, "expansion_auroc": 0.8}) == (0.7 + 0.9 + 0.8) / 3


def test_false_positive_summary_reports_rates_and_quantiles():
    mob = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
    s = false_positive_summary(mob, np.array([0.1, 0.8]), np.array([0.2, 0.9]),
                                np.array([0.1, 0.8]))
    assert s["false_mobile_rate"] == 0.5 and s["risk_q50"] == 0.45


def test_evaluate_records_returns_task_metrics():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, n_layers=1, max_windows=56))
    records = [
        LabeledSequence("a", "ATGCGT" * 40, 0, 0, 0),
        LabeledSequence("b", "CGTATG" * 40, 1, 1, 1),
        LabeledSequence("c", "GGCATA" * 40, 2, 0, 1),
    ]
    m = evaluate_records(model, records)
    assert "mobility_balanced_accuracy" in m and "amr_auroc" in m and "expansion_auroc" in m


def test_false_positive_summary_empty():
    s = false_positive_summary(np.empty((0, 3)), np.array([]), np.array([]), np.array([]))
    assert isinstance(s, dict)
