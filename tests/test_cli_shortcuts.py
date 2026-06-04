"""Tests for CLI shortcuts and --help output."""

from click.testing import CliRunner

from dna_sentinel.cli import cli, dna


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Cassiopeia" in result.output


def test_dna_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["--help"])
    assert result.exit_code == 0
    assert "Cassiopeia" in result.output
    assert "Usage:" in result.output


def test_train_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["train", "--help"])
    assert result.exit_code == 0
    assert "--experiment" in result.output or "-e" in result.output


def test_predict_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--fasta" in result.output or "-f" in result.output


def test_benchmark_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "--help"])
    assert result.exit_code == 0
    assert "--checkpoint" in result.output or "-m" in result.output


def test_dna_predict_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["predict", "--help"])
    assert result.exit_code == 0
    assert "--fasta" in result.output or "-f" in result.output


def test_dna_bench_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--data-dir" in result.output or "-d" in result.output


def test_dna_cv_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["cv", "--help"])
    assert result.exit_code == 0
    assert "--folds" in result.output or "-k" in result.output


def test_dna_list_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["list", "--help"])
    assert result.exit_code == 0


def test_dna_interpret_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["interpret", "--help"])
    assert result.exit_code == 0
    assert "--fasta" in result.output or "-f" in result.output


def test_dna_prep_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["prep", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output or "-c" in result.output


def test_dna_features_help():
    runner = CliRunner()
    result = runner.invoke(dna, ["features", "--help"])
    assert result.exit_code == 0


def test_experiment_list_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["experiment-list", "--help"])
    assert result.exit_code == 0
