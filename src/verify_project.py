from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
from pathlib import Path


def _ssl_backend_warning(backend: str) -> str | None:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify BioSpread project health and release readiness")
    parser.add_argument("--release", action="store_true", help="Run release-grade verification checks")
    parser.add_argument(
        "--skip-security",
        action="store_true",
        help="Skip dependency vulnerability audit when pip-audit is unavailable or network policy blocks it",
    )
    return parser


def _run_check(name: str, command: list[str], *, cwd: Path) -> None:
    print(f"{name}: {' '.join(command)}", flush=True)
    env = os.environ.copy()
    src_path = str(cwd / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    warning = _ssl_backend_warning(ssl.OPENSSL_VERSION)
    if warning:
        print(warning)
    if args.release:
        root = Path(__file__).resolve().parents[1]
        python = sys.executable
        _run_check("compile", [python, "-m", "compileall", "-q", "src"], cwd=root)
        _run_check("ruff", [python, "-m", "ruff", "check", "src", "tests", "run_project.py", "verify_project.py"], cwd=root)
        _run_check("mypy", [python, "-m", "mypy", "src"], cwd=root)
        _run_check(
            "pytest",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests",
                "-k",
                "not test_release_verification_runs_real_checks",
            ],
            cwd=root,
        )
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
        if not args.skip_security:
            _run_check("security", [python, "-m", "pip_audit", "-r", "requirements.txt"], cwd=root)
        print("release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
