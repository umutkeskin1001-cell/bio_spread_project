"""Advanced CLI tests for coverage."""

import json
import tempfile
from pathlib import Path
from click.testing import CliRunner

from dna_sentinel.cli import (
    cli,
    _load_config,
    _validate_config,
    ConfigError,
)


def test_cli_all_commands_help():
    """Test that all cli commands have --help."""
    runner = CliRunner()
    commands = ["prepare", "prepare-features", "train", "cross-validate",
                 "benchmark", "predict", "serve", "experiment-list"]
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


def test_train_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["train", "--help"])
    assert "--config" in r.output or "-c" in r.output
    assert "--experiment" in r.output or "-e" in r.output


def test_predict_with_text():
    """Test predict with --interpret flag."""
    runner = CliRunner()
    import tempfile, os
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
