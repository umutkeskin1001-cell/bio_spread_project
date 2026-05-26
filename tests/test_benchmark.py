import json
from pathlib import Path

import numpy as np
import torch
from click.testing import CliRunner

from dna_sentinel.cli import cli
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.utils import LabeledSequence, evaluate_records, false_positive_summary, task_score


def test_task_score_equal_weight():
    assert task_score({"mobility_balanced_accuracy": 0.7, "amr_auroc": 0.9, "expansion_auroc": 0.8}) == (0.7 + 0.9 + 0.8) / 3


def test_false_positive_summary_reports_rates_and_quantiles():
    s = false_positive_summary(np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1]]), np.array([0.1, 0.8]), np.array([0.2, 0.9]), np.array([0.1, 0.8]))
    assert s["false_mobile_rate"] == 0.5 and s["risk_q50"] == 0.45


def test_benchmark_reports_nonplasmid_stress_summary(tmp_path: Path):
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, n_layers=1, max_windows=56))
    ckpt = tmp_path / "cassiopeia.pt"
    out = tmp_path / "report.json"
    model.save(ckpt)
    feat = {"features": torch.randn(4, 56, 2728), "masks": torch.ones(4, 56, dtype=torch.bool)}
    lab = {"mobility": torch.tensor([0, 1, 2, 0], dtype=torch.long), "amr": torch.tensor([0.0, 1.0, 0.0, 1.0]), "expansion": torch.tensor([0.0, 1.0, 0.0, 1.0])}
    for split in ("val", "test", "heldout_test"):
        torch.save(feat, tmp_path / f"{split}_features.pt")
        torch.save(lab, tmp_path / f"{split}_labels.pt")
    torch.save(feat, tmp_path / "nonplasmid_control_features.pt")
    torch.save({"mobility": torch.zeros(4, dtype=torch.long), "amr": torch.zeros(4), "expansion": torch.zeros(4)}, tmp_path / "nonplasmid_control_labels.pt")
    result = CliRunner().invoke(cli, ["benchmark", "--checkpoint", str(ckpt), "--data-dir", str(tmp_path), "--out", str(out)])
    assert result.exit_code == 0
    report = json.loads(out.read_text())
    assert "false_mobile_rate" in report["splits"]["nonplasmid_control"]


def test_evaluate_records_returns_task_metrics():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, n_layers=1, max_windows=56))
    records = [LabeledSequence("a", "ATGCGT" * 40, 0, 0, 0), LabeledSequence("b", "CGTATG" * 40, 1, 1, 1), LabeledSequence("c", "GGCATA" * 40, 2, 0, 1)]
    m = evaluate_records(model, records)
    assert "mobility_balanced_accuracy" in m and "amr_auroc" in m and "expansion_auroc" in m


def test_false_positive_summary_empty():
    s = false_positive_summary(np.empty((0, 3)), np.array([]), np.array([]), np.array([]))
    assert isinstance(s, dict)
