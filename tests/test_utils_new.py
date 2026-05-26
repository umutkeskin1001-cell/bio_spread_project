"""Tests for experiment tracking, config validation, and new utilities."""

import numpy as np
import pytest
import torch

from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.utils import (
    CassiopeiaExperiment,
    ConfigError,
    DNASequenceAugmentation,
    LabeledSequence,
    ValidationError,
    bootstrap_ci,
    set_seed,
    validate_dna,
)


def test_set_seed_deterministic():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.allclose(a, b)


def test_validate_dna_valid():
    assert validate_dna("ACGT" * 10) == "ACGT" * 10


def test_validate_dna_too_long():
    with pytest.raises(ValidationError, match="too long"):
        validate_dna("A" * 500_000, max_len=1000)


def test_validate_dna_empty():
    with pytest.raises(ValidationError, match="empty sequence"):
        validate_dna("   \t\n  ")


def test_validate_dna_cleans_whitespace():
    assert validate_dna("A C G T") == "ACGT"


def test_config_error():
    with pytest.raises(ConfigError):
        raise ConfigError("test error")


class TestCassiopeiaExperiment:
    def test_create_experiment(self, tmp_path):
        exp = CassiopeiaExperiment("test_run", base_dir=str(tmp_path), config={"lr": 0.001})
        assert exp.dir.exists() and (tmp_path / exp.dir.name).exists()

    def test_log_metrics(self, tmp_path):
        exp = CassiopeiaExperiment("metrics_test", base_dir=str(tmp_path))
        exp.log_metrics(1, train_loss=0.5, val_acc=0.9)
        assert len(exp.history) == 1 and exp.history[0]["train_loss"] == 0.5

    def test_save_checkpoint(self, tmp_path):
        exp = CassiopeiaExperiment("ckpt_test", base_dir=str(tmp_path))
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, n_canonical_features=100, frp_out_dim=16, max_windows=8))
        p = tmp_path / "model.pt"
        model.save(p)
        cp = exp.save_checkpoint(p)
        assert cp.exists()

    def test_compare(self, tmp_path):
        e1 = CassiopeiaExperiment("run1", base_dir=str(tmp_path))
        e1.save_report({"acc": 0.9}, "report")
        e2 = CassiopeiaExperiment("run2", base_dir=str(tmp_path))
        e2.save_report({"acc": 0.8}, "report")
        c = CassiopeiaExperiment.compare(e1, e2)
        assert len(c) == 2


class TestDNASequenceAugmentation:
    def test_noop_when_not_training(self):
        aug = DNASequenceAugmentation(mutation_rate=0.5)
        records = [LabeledSequence("a", "ACGT" * 10, 0, 0, 0)]
        assert aug(records, training=False) == records

    def test_mutation_changes_sequence(self):
        aug = DNASequenceAugmentation(mutation_rate=0.5)
        records = [LabeledSequence("a", "ACGT" * 100, 0, 0, 0)]
        result = aug(records, training=True)
        assert result[0].dna != records[0].dna or result == records

    def test_truncation_shortens(self):
        aug = DNASequenceAugmentation(truncation_rate=1.0)
        records = [LabeledSequence("a", "ACGT" * 100, 0, 0, 0)]
        result = aug(records, training=True)
        assert len(result[0].dna) < len(records[0].dna) or result == records


def test_bootstrap_ci():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
    from sklearn.metrics import roc_auc_score
    point, lo, hi = bootstrap_ci(y, p, roc_auc_score, n_resamples=100)
    assert 0 <= lo <= point <= hi <= 1
