"""Additional CLI tests to boost coverage."""

import json
from pathlib import Path
from click.testing import CliRunner

from dna_sentinel.cli import cli, dna, _load_config, _validate_config, ConfigError


def test_cli_main_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0
    assert "Cassiopeia" in r.output


def test_dna_main_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["--help"])
    assert r.exit_code == 0
    assert "Cassiopeia" in r.output


def test_prepare_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["prepare", "--help"])
    assert r.exit_code == 0


def test_prepare_features_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["prepare-features", "--help"])
    assert r.exit_code == 0


def test_cross_validate_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["cross-validate", "--help"])
    assert r.exit_code == 0


def test_experiment_list_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["experiment-list", "--help"])
    assert r.exit_code == 0


def test_serve_help():
    runner = CliRunner()
    r = runner.invoke(cli, ["serve", "--help"])
    assert r.exit_code == 0
    assert "--port" in r.output


def test_dna_prep_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["prep", "--help"])
    assert r.exit_code == 0


def test_dna_features_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["features", "--help"])
    assert r.exit_code == 0


def test_dna_bench_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["bench", "--help"])
    assert r.exit_code == 0


def test_dna_cv_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["cv", "--help"])
    assert r.exit_code == 0


def test_dna_list_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["list", "--help"])
    assert r.exit_code == 0


def test_dna_interpret_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["interpret", "--help"])
    assert r.exit_code == 0


def test_dna_serve_help():
    runner = CliRunner()
    r = runner.invoke(dna, ["serve", "--help"])
    assert r.exit_code == 0


def test_config_validation_ok():
    cfg = {
        "model": {"max_windows": 56},
        "features": {"max_windows": [32, 16, 8], "window_sizes": [512, 2048, 8192], "strides": [256, 1024, 4096]},
    }
    _validate_config(cfg)


def test_config_validation_mismatch():
    cfg = {
        "model": {"max_windows": 50},
        "features": {"max_windows": [32, 16, 8], "window_sizes": [512, 2048, 8192], "strides": [256, 1024, 4096]},
    }
    import pytest
    with pytest.raises(ConfigError):
        _validate_config(cfg)


def test_config_validation_strides_mismatch():
    cfg = {
        "model": {"max_windows": 56},
        "features": {"max_windows": [32, 16, 8], "window_sizes": [512, 2048], "strides": [256]},
    }
    import pytest
    with pytest.raises(ConfigError):
        _validate_config(cfg)


def test_load_config(tmp_path):
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("data:\n  out_dir: /tmp")
    result = _load_config(str(cfg_path))
    assert result["data"]["out_dir"] == "/tmp"


def test_experiment_list_no_dir(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem():
        r = runner.invoke(cli, ["experiment-list"])
        assert r.exit_code == 0
        assert "No experiments directory" in r.output


def test_predict_with_no_model(tmp_path):
    """predict should fail gracefully with missing checkpoint."""
    fasta = tmp_path / "test.fa"
    fasta.write_text(">test\nATGCGT" * 100)
    runner = CliRunner()
    r = runner.invoke(cli, ["predict", "--checkpoint", str(tmp_path / "nonexistent.pt"),
                            "--fasta", str(fasta)])
    assert r.exit_code != 0


def test_benchmark_with_no_model(tmp_path):
    """benchmark should fail gracefully with missing checkpoint."""
    runner = CliRunner()
    r = runner.invoke(cli, ["benchmark", "--checkpoint", str(tmp_path / "nonexistent.pt"),
                            "--data-dir", str(tmp_path)])
    assert r.exit_code != 0
