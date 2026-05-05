from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
from pathlib import Path


def ssl_backend_warning(backend: str) -> str | None:
    if not backend.startswith("OpenSSL "):
        return f"Unsupported SSL backend: {backend}"
    parts = backend.split()
    if len(parts) < 2:
        return f"Unparseable SSL backend version: {backend}"
    version = parts[1]
    major_minor = version.split(".")[:2]
    try:
        major = int(major_minor[0])
        minor = int(major_minor[1]) if len(major_minor) > 1 else 0
    except ValueError:
        return f"Unparseable SSL backend version: {backend}"
    if major < 1 or (major == 1 and minor == 0):
        return f"OpenSSL version too old: {backend}"
    return None


def _run_check(name: str, command: list[str], *, cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    print(f"{name}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    if extra_env:
        env.update(extra_env)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _release_pytest_command(python: str) -> list[str]:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Break release-verify recursion under pytest: exercise the slowest
        # scientific and metric contracts without spawning the entire suite from
        # inside itself.
        return [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_quality.py",
            "tests/test_validation_protocol_v2.py",
            "tests/test_evaluation_metrics.py",
        ]
    return [python, "-m", "pytest", "-q", "tests", "-k", "not test_release_verification_runs_real_checks"]


def run_verification(*, release: bool, skip_security: bool, project_root: Path | None = None) -> int:
    root = project_root or Path(__file__).resolve().parents[2]
    warning = ssl_backend_warning(ssl.OPENSSL_VERSION)
    if warning:
        print(warning)
    if not release:
        _run_check("compile", [sys.executable, "-m", "compileall", "-q", "src"], cwd=root)
        _run_check("pytest", [sys.executable, "-m", "pytest", "-q", "tests"], cwd=root)
        return 0

    python = sys.executable
    _run_check("compile", [python, "-m", "compileall", "-q", "src"], cwd=root)
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        _run_check("ruff", [python, "-m", "ruff", "check", "src", "tests"], cwd=root)
        _run_check("mypy", [python, "-m", "mypy", "src"], cwd=root)
    _run_check("pytest", _release_pytest_command(python), cwd=root)
    _run_check(
        "smoke-cli",
        [
            python,
            "-m",
            "bio_spread_project.cli",
            "run",
            "--mode",
            "input",
            "--input",
            "data/sample_plasmid_records.csv",
            "--output-dir",
            "reports/release_verify_smoke",
        ],
        cwd=root,
    )
    if not skip_security:
        _run_check("security", [python, "-m", "pip-audit", "-r", "requirements.txt"], cwd=root)
    print("release verification passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify BioSpread project health and release readiness")
    parser.add_argument("--release", action="store_true", help="Run release-grade verification checks")
    parser.add_argument("--skip-security", action="store_true", help="Skip dependency vulnerability audit")
    return parser
