import subprocess
import sys
from pathlib import Path

import pytest

from bio_spread_project import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "data" / "sample_plasmid_records.csv"
RAW_BACKBONES = PROJECT_ROOT / "data" / "raw" / "plasmid_backbones.tsv"
RAW_AMR = PROJECT_ROOT / "data" / "raw" / "amr.tsv"


def test_cli_run_input_mode_emits_selection_reason(tmp_path, capsys):
    exit_code = cli.main(
        [
            "run",
            "--mode",
            "input",
            "--input",
            str(FIXTURE),
            "--output-dir",
            str(tmp_path / "cli_input"),
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Input mode: observation_records" in stdout
    assert "Selection reason: explicit_input_mode" in stdout


def test_cli_auto_mode_requires_explicit_surface_when_requested(tmp_path):
    with pytest.raises(ValueError, match="promote the run to geo mode"):
        cli.main(
            [
                "run",
                "--mode",
                "auto",
                "--records",
                str(RAW_BACKBONES),
                "--amr",
                str(RAW_AMR),
                "--output-dir",
                str(tmp_path / "cli_auto"),
                "--require-explicit-surface",
            ]
        )


def test_cli_defaults_follow_bio_spread_data_root(monkeypatch, tmp_path):
    custom_root = tmp_path / "custom_data"
    monkeypatch.setenv("BIO_SPREAD_DATA_ROOT", str(custom_root))
    parser = cli.build_parser()
    args = parser.parse_args(["run"])

    assert str(args.records).startswith(str(custom_root))
    assert str(args.amr).startswith(str(custom_root))
    assert str(args.geo_spread_features).startswith(str(custom_root))


def test_verify_project_has_release_mode():
    completed = subprocess.run(
        [sys.executable, "verify_project.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--release" in completed.stdout


def test_verify_project_ssl_warning_detects_non_openssl_backend():
    from verify_project import _ssl_backend_warning

    assert _ssl_backend_warning("LibreSSL 2.8.3")
    assert _ssl_backend_warning("OpenSSL 1.0.2")
    assert _ssl_backend_warning("OpenSSL 1.1.1") is None
    assert _ssl_backend_warning("OpenSSL 3.0.0") is None
