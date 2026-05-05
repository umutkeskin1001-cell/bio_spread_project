import os
import subprocess
import sys
from pathlib import Path

import pytest

from bio_spread_project import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
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
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) if not env.get("PYTHONPATH") else f"{SRC_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    completed = subprocess.run(
        [sys.executable, "-m", "bio_spread_project.cli", "verify", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--release" in completed.stdout


def test_release_verification_runs_real_checks():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) if not env.get("PYTHONPATH") else f"{SRC_ROOT}{os.pathsep}{env['PYTHONPATH']}"
    completed = subprocess.run(
        [sys.executable, "-m", "bio_spread_project.cli", "verify", "--release", "--skip-security"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "compile" in completed.stdout
    assert "pytest" in completed.stdout
    assert "release verification passed" in completed.stdout


def test_release_verification_bounds_nested_pytest_when_invoked_from_pytest(monkeypatch):
    from bio_spread_project import verification

    commands = []

    def capture_check(name, command, *, cwd, extra_env=None):
        commands.append((name, command, cwd))

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_cli.py::test_release_verification_runs_real_checks")
    monkeypatch.setattr(verification, "_run_check", capture_check)

    assert verification.run_verification(release=True, skip_security=True, project_root=PROJECT_ROOT) == 0

    pytest_command = next(command for name, command, _ in commands if name == "pytest")
    assert "tests/test_quality.py" in pytest_command
    assert "tests/test_validation_protocol_v2.py" in pytest_command
    assert "tests/test_evaluation_metrics.py" in pytest_command
    assert "-k" not in pytest_command


def test_verify_project_ssl_warning_detects_non_openssl_backend():
    from bio_spread_project.verification import ssl_backend_warning

    assert ssl_backend_warning("LibreSSL 2.8.3")
    assert ssl_backend_warning("OpenSSL 1.0.2")
    assert ssl_backend_warning("OpenSSL 1.1.1") is None
    assert ssl_backend_warning("OpenSSL 3.0.0") is None
