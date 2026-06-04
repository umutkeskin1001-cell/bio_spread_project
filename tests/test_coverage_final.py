"""Final coverage push - target remaining gaps."""

import json
import tempfile
import os
from pathlib import Path
import torch
import numpy as np

# ── train.py direct function coverage ─────────────────────────────

from dna_sentinel.train import _epoch_indices, _build_optimizer, _focal_bce, evaluate


def test_epoch_indices_random():
    gen = torch.Generator().manual_seed(42)
    data = {"mobility": torch.randint(0, 3, (20,)), "amr": torch.randint(0, 2, (20,)).float(),
            "expansion": torch.randint(0, 2, (20,)).float()}
    idx = _epoch_indices(10, data, {"balanced_sampling": False}, gen)
    assert idx.shape == (10,)
    assert idx.max() < 20


def test_epoch_indices_balanced():
    gen = torch.Generator().manual_seed(42)
    data = {"mobility": torch.zeros(10).long(), "amr": torch.zeros(10).float(),
            "expansion": torch.zeros(10).float()}
    idx = _epoch_indices(10, data, {"balanced_sampling": True}, gen,
                         cached_weights=torch.ones(10))
    assert idx.shape == (10,)


def test_focal_bce_gamma_zero():
    loss = _focal_bce(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0]), None, 0.0)
    assert torch.isfinite(loss)


def test_focal_bce_gamma_positive():
    loss = _focal_bce(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0]), None, 2.0)
    assert torch.isfinite(loss)


def test_focal_bce_with_weight():
    pw = torch.tensor([0.5])
    loss = _focal_bce(torch.tensor([0.5, -0.5]), torch.tensor([1.0, 0.0]), pw, 2.0)
    assert torch.isfinite(loss)


# ── evaluate with empty data ──────────────────────────────────────

def test_evaluate_empty_split():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    data = {
        "features": torch.randn(2, 56, 2728),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        "mobility": torch.tensor([0, 1]),
        "amr": torch.tensor([0.0, 1.0]),
        "expansion": torch.tensor([0.0, 1.0]),
    }
    metrics = evaluate(model, data, "cpu")
    assert "mobility_balanced_accuracy" in metrics


# ── CLI config validation ─────────────────────────────────────────

from dna_sentinel.cli import _load_config, _validate_config


def test_config_no_data_section():
    cfg = {"training": {"epochs": 1}, "model": {"max_windows": 56},
           "features": {"max_windows": [32, 16, 8], "window_sizes": [512, 2048, 8192], "strides": [256, 1024, 4096]}}
    _validate_config(cfg)


# ── API fasta latency test ────────────────────────────────────────

def test_benchmark_latency_no_model(tmp_path):
    """Test that benchmark fails gracefully with bad checkpoint."""
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    runner = CliRunner()
    r = runner.invoke(cli, ["benchmark", "--checkpoint", "/nonexistent.pt",
                            "--data-dir", str(tmp_path), "--out", str(tmp_path / "out.json")])
    assert r.exit_code != 0


# ── Predict with no checkpoint ────────────────────────────────────

def test_predict_fails_gracefully(tmp_path):
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    fa = tmp_path / "test.fa"
    fa.write_text(">test\nATGCGT" * 100)
    runner = CliRunner()
    r = runner.invoke(cli, ["predict", "--checkpoint", str(tmp_path / "nonexistent.pt"),
                            "--fasta", str(fa), "--json", "--interpret"])
    assert r.exit_code != 0


# ── Prepare with missing files ────────────────────────────────────

def test_prepare_missing_files(tmp_path):
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("data:\n  fasta_path: /nonexistent\n  out_dir: /tmp\n  limit: 100\n")
    runner = CliRunner()
    r = runner.invoke(cli, ["prepare", "--config", str(cfg)])
    assert r.exit_code != 0


# ── Features extraction with bad data ─────────────────────────────

def test_prepare_features_bad_config(tmp_path):
    from click.testing import CliRunner
    from dna_sentinel.cli import cli
    cfg = tmp_path / "cfg.yaml"
    # make invalid config (max_windows mismatch)
    cfg.write_text("model:\n  max_windows: 50\nfeatures:\n  max_windows: [32, 16, 8]\n  window_sizes: [512, 2048, 8192]\n  strides: [256, 1024, 4096]\ndata:\n  out_dir: /tmp\n")
    runner = CliRunner()
    r = runner.invoke(cli, ["prepare-features", "--config", str(cfg)])
    assert r.exit_code != 0


# ── model Calibrated evaluation ─────────────────────────────────

def test_calibrated_model_params():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    params = dict(model.named_buffers())
    assert "mob_t" in params
    assert "amr_t" in params
    assert "exp_t" in params


def test_model_evaluate_with_return():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    from dna_sentinel.train import evaluate
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    data = {
        "features": torch.randn(3, 56, 2728),
        "masks": torch.ones(3, 56, dtype=torch.bool),
        "mobility": torch.tensor([0, 1, 2]),
        "amr": torch.tensor([0.0, 1.0, 0.0]),
        "expansion": torch.tensor([0.0, 1.0, 0.0]),
    }
    metrics, mob_p, amr_p, exp_p = evaluate(model, data, "cpu", return_probs=True)
    assert mob_p.shape == (3, 3)
    assert amr_p.shape == (3,)
    assert exp_p.shape == (3,)


# ── prepare.py: build_labels edge cases ─────────────────────────

def test_build_labels(tmp_path):
    from dna_sentinel.prepare import build_labels
    # Create mock TSV files
    bb = tmp_path / "backbones.tsv"
    bb.write_text("sequence_accession\tpredicted_mobility\tbackbone_id\tcountry\tresolved_year\n"
                  "seq1\tconjugative\tbb1\tTR\t2020\n"
                  "seq2\tnon-mobilizable\tbb2\tUS\t2019\n")
    amr = tmp_path / "amr.tsv"
    amr.write_text("sequence_accession\tamr_any\nseq1\t1\nseq2\t0\n")
    labels = build_labels(str(bb), str(amr), expansion_country_threshold=15)
    assert "seq1" in labels
    assert labels["seq1"]["mobility"] == 2  # conjugative
    assert labels["seq1"]["amr"] == 1


# ── evaluate_records ──────────────────────────────────────────────

def test_evaluate_records_with_model():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    from dna_sentinel.utils import LabeledSequence, evaluate_records
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    records = [
        LabeledSequence("a", "ATGCGT" * 50, 0, 0, 0),
        LabeledSequence("b", "CATGCA" * 50, 1, 1, 1),
    ]
    metrics = evaluate_records(model, records, "cpu")
    assert "mobility_balanced_accuracy" in metrics
    assert "amr_auroc" in metrics


# ── api.py lifespan coverage ──────────────────────────────────────

def test_api_import():
    """Just importing the API module exercises module-level code."""
    import importlib
    import dna_sentinel.api
    importlib.reload(dna_sentinel.api)
    # Check that version is set
    assert hasattr(dna_sentinel.api, "_VERSION")


# ── Config from yaml ──────────────────────────────────────────────

def test_config_from_yaml():
    from dna_sentinel.model import CassiopeiaConfig
    cfg = CassiopeiaConfig.from_yaml({"model": {"hidden_dim": 64, "frp_out_dim": 128, "max_windows": 56}})
    assert cfg.hidden_dim == 64
    assert cfg.frp_out_dim == 128


def test_config_to_dict():
    from dna_sentinel.model import CassiopeiaConfig
    cfg = CassiopeiaConfig()
    d = cfg.to_dict()
    assert "hidden_dim" in d
    assert "frp_out_dim" in d


# ── circular position encoding ────────────────────────────────────

def test_cppe_forward():
    from dna_sentinel.model import CircularPositionEncoding
    cppe = CircularPositionEncoding(32)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    out = cppe(x, mask)
    assert out.shape == x.shape


def test_cppe_with_scale():
    from dna_sentinel.model import CircularPositionEncoding
    cppe = CircularPositionEncoding(32)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    scale = torch.zeros(2, 10, dtype=torch.long)
    out = cppe(x, mask, scale_ids=scale)
    assert out.shape == x.shape


# ── ScaleGate ──────────────────────────────────────────────────────

def test_scalegate_forward():
    from dna_sentinel.model import ScaleGate
    gate = ScaleGate(32, n=3)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    scale = torch.zeros(2, 10, dtype=torch.long)
    out = gate(x, scale, mask)
    assert out.shape == x.shape
