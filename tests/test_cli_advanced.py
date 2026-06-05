"""Advanced CLI tests for coverage."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from dna_sentinel.cli import (
    ConfigError,
    _load_config,
    _validate_config,
    cli,
)


def test_cli_all_commands_help():
    """Test that all cli commands have --help."""
    runner = CliRunner()
    commands = ["prepare", "prepare-features", "train", "cross-validate",
                 "benchmark", "predict", "serve", "experiment-list", "interpret"]
    for cmd in commands:
        r = runner.invoke(cli, [cmd, "--help"])
        assert r.exit_code == 0, f"Command '{cmd}' failed: {r.output[:200]}"


def test_dna_all_commands_help():
    """Test that all dna commands have --help."""
    from dna_sentinel.cli import dna
    runner = CliRunner()
    commands = ["prep", "features", "train", "cv", "bench",
                 "predict", "serve", "list", "interpret"]
    for cmd in commands:
        r = runner.invoke(dna, [cmd, "--help"])
        assert r.exit_code == 0, f"dna command '{cmd}' failed: {r.output[:200]}"


def test_load_config_file(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("data:\n  out_dir: /tmp/test\nmodel:\n  hidden_dim: 64\n")
    result = _load_config(str(cfg))
    assert result["data"]["out_dir"] == "/tmp/test"
    assert result["model"]["hidden_dim"] == 64


def test_validate_config_ok():
    cfg = {
        "model": {"max_windows": 56},
        "features": {
            "max_windows": [32, 16, 8],
            "window_sizes": [512, 2048, 8192],
            "strides": [256, 1024, 4096],
        },
    }
    _validate_config(cfg)


def test_validate_config_max_windows_mismatch():
    cfg = {
        "model": {"max_windows": 44},
        "features": {
            "max_windows": [32, 16, 8],
            "window_sizes": [512, 2048, 8192],
            "strides": [256, 1024, 4096],
        },
    }
    import pytest
    with pytest.raises(ConfigError):
        _validate_config(cfg)


def test_validate_config_strides_mismatch():
    cfg = {
        "model": {"max_windows": 56},
        "features": {
            "max_windows": [32, 16, 8],
            "window_sizes": [512, 2048],
            "strides": [256],
        },
    }
    import pytest
    with pytest.raises(ConfigError):
        _validate_config(cfg)


def test_predict_helps():
    runner = CliRunner()
    # Test the predict help with interpret flag
    r = runner.invoke(cli, ["predict", "--help"])
    assert "--interpret" in r.output


def test_benchmark_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["benchmark", "--help"])
    assert "--checkpoint" in r.output
    assert "-d" in r.output
    assert "-o" in r.output
    assert "--rc-average" in r.output


def _make_bench_fixture(tmp_path: Path):
    """Build a minimal data directory with feature cache + labels for benchmark tests."""
    import torch
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    from dna_sentinel.features import FEATURE_SCHEMA_VERSION
    feats = {
        "features": torch.zeros(1, 56, 2728),
        "struct_features": torch.zeros(1, 56, 49),
        "masks": torch.zeros(1, 56, dtype=torch.bool),
        "scale_ids": torch.zeros(1, 56, dtype=torch.long),
        "_schema_version": FEATURE_SCHEMA_VERSION,
        "_n_structural_features": 49,
    }
    labels = {
        "mobility": torch.tensor([0]),
        "amr": torch.tensor([0]),
        "expansion": torch.tensor([0]),
    }
    for split in ("val", "test", "heldout_test"):
        torch.save(feats, data_dir / f"{split}_features.pt")
        torch.save(labels, data_dir / f"{split}_labels.pt")
        with open(data_dir / f"{split}.jsonl", "w") as f:
            f.write(json.dumps({"sequence_id": "a", "dna": "ACGT" * 1500, "mobility": 0, "amr": 0, "expansion": 0}) + "\n")
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    m = Cassiopeia(CassiopeiaConfig(
        n_structural_features=49, hidden_dim=32, max_windows=56,
        n_canonical_features=2728, frp_out_dim=32, dropout=0.0,
        risk_weights=(0.4, 0.3, 0.3),
    ))
    ckpt = str(data_dir / "ckpt.pt")
    m.save(ckpt)
    return data_dir, ckpt


def test_benchmark_inference_mode_default():
    """Default benchmark path produces a report with inference_mode=cached_features."""
    from dna_sentinel.cli import cli
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, ckpt = _make_bench_fixture(Path(tmp))
        out = data_dir / "report.json"
        r = runner.invoke(cli, ["benchmark", "-m", ckpt, "-d", str(data_dir), "-o", str(out)])
        assert r.exit_code == 0, f"benchmark failed: {r.output[:500]}"
        report = json.loads(out.read_text())
        assert report["inference_mode"] == "cached_features"
        for split in ("validation", "test", "heldout"):
            assert split in report["splits"]


def test_benchmark_rc_average_flag():
    """--rc-average path produces a report with inference_mode=rc_averaged."""
    from dna_sentinel.cli import cli
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir, ckpt = _make_bench_fixture(Path(tmp))
        out = data_dir / "report_rc.json"
        r = runner.invoke(cli, ["benchmark", "-m", ckpt, "-d", str(data_dir), "-o", str(out), "--rc-average"])
        assert r.exit_code == 0, f"benchmark --rc-average failed: {r.output[:500]}"
        report = json.loads(out.read_text())
        assert report["inference_mode"] == "rc_averaged"


def test_train_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["train", "--help"])
    assert "--config" in r.output or "-c" in r.output
    assert "--experiment" in r.output or "-e" in r.output


def test_predict_with_text():
    """Test predict with --interpret flag."""
    runner = CliRunner()
    import os
    import tempfile
    # Create small FASTA
    tmp = tempfile.mkdtemp()
    fa_path = os.path.join(tmp, "test.fa")
    with open(fa_path, "w") as f:
        f.write(">test_seq\nATGCGT\n" * 100)
    ckpt = "nonexistent.pt"
    r = runner.invoke(cli, ["predict", "--checkpoint", ckpt, "--fasta", fa_path, "--interpret"])
    assert r.exit_code != 0  # will fail because no checkpoint


def test_validate_config_no_model_section():
    # Config without model section
    cfg = {"data": {"out_dir": "/tmp"}}
    _validate_config(cfg)


def test_dna_predict_flag():
    """Test dna predict help has short flags."""
    from dna_sentinel.cli import dna
    runner = CliRunner()
    r = runner.invoke(dna, ["predict", "--help"])
    assert r.exit_code == 0
    assert "-m" in r.output
    assert "-f" in r.output
    assert "-j" in r.output


def test_dna_bench_flag():
    """Test dna bench help has short flags."""
    from dna_sentinel.cli import dna
    runner = CliRunner()
    r = runner.invoke(dna, ["bench", "--help"])
    assert r.exit_code == 0
    assert "-m" in r.output
    assert "-d" in r.output
    assert "-o" in r.output


def test_dna_cv_flag():
    """Test dna cv help has short flags."""
    from dna_sentinel.cli import dna
    runner = CliRunner()
    r = runner.invoke(dna, ["cv", "--help"])
    assert r.exit_code == 0
    assert "-k" in r.output
