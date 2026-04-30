#!/usr/bin/env python3
"""Run the full BioSpread verification suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RECORDS = PROJECT_ROOT / "data" / "raw" / "plasmid_backbones.tsv"
DEFAULT_AMR = PROJECT_ROOT / "data" / "raw" / "amr.tsv"
DEFAULT_GEO = PROJECT_ROOT / "data" / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"
DEFAULT_EXTERNAL_HOLDOUT = PROJECT_ROOT / "tests" / "fixtures" / "geo_holdout.tsv"


def _run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", str(PROJECT_ROOT / ".pycache"))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def _has_packaged_data() -> bool:
    return all(path.exists() for path in (DEFAULT_RECORDS, DEFAULT_AMR, DEFAULT_GEO))


def _ssl_backend_warning(version: str) -> str | None:
    if not version.startswith("OpenSSL "):
        return (
            f"Release environment warning: Python ssl backend is {version!r}. "
            "Use a Python build linked against OpenSSL 1.1.1+ for final competition packaging."
        )
    match = re.match(r"OpenSSL\s+(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return f"Release environment warning: could not parse ssl backend version {version!r}."
    major, minor, patch = (int(part) for part in match.groups())
    if (major, minor, patch) < (1, 1, 1):
        return (
            f"Release environment warning: Python ssl backend is {version!r}. "
            "Use OpenSSL 1.1.1+ for final competition packaging."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BioSpread verification checks")
    parser.add_argument(
        "--skip-run-if-data-missing",
        action="store_true",
        help="Skip full run/audit checks when packaged data files are unavailable",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run full release checks: ruff, mypy, pytest, production run, and artifact validation",
    )
    args = parser.parse_args(argv)

    if args.release:
        _run([sys.executable, "-m", "ruff", "check", "src", "tests"])
        _run([sys.executable, "-m", "mypy", "src/bio_spread_project"])
        _run([sys.executable, "-m", "pip_audit", "-r", "requirements.txt"])
        warning = _ssl_backend_warning(ssl.OPENSSL_VERSION)
        if warning:
            print(warning)
    _run([sys.executable, "-m", "compileall", "src"])
    _run([sys.executable, "-m", "pytest", "tests"])
    if not _has_packaged_data():
        if args.skip_run_if_data_missing:
            print("Packaged data not found; skipping run-level verification checks")
            print("BioSpread verification completed (partial)")
            return 0
        raise SystemExit(
            "Verification failed: packaged data files are missing. "
            "Use --skip-run-if-data-missing for CI/lightweight environments."
        )

    _run(
        [
            sys.executable,
            "run_project.py",
            "--mode",
            "geo",
            "--output-dir",
            "reports/competition_final" if args.release else "reports/run",
            "--external-holdout",
            str(DEFAULT_EXTERNAL_HOLDOUT),
            "--fail-on-quality-gates",
            "--fail-on-drift-fail",
            "--fail-on-trend-fail",
        ]
    )
    output_dir = PROJECT_ROOT / ("reports/competition_final" if args.release else "reports/run")
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    benchmark = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    drift_report = json.loads((output_dir / "drift_report.json").read_text(encoding="utf-8"))
    trend_report = json.loads((output_dir / "trend_report.json").read_text(encoding="utf-8"))
    release_gate = json.loads((output_dir / "release_gate.json").read_text(encoding="utf-8"))
    registry = json.loads((output_dir / "data_registry.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    model_registry_path = output_dir / "model_registry.jsonl"
    trend_report_path = output_dir / "trend_report.json"
    release_gate_path = output_dir / "release_gate.json"
    for artifact in (
        "report.md",
        "dashboard.html",
        "audit.json",
        "model_card.md",
        "manifest.json",
        "predictions.csv",
        "predictions.parquet",
        "features.parquet",
        "artifact_index.json",
    ):
        if not (output_dir / artifact).exists():
            raise SystemExit(f"Verification failed: {artifact} artifact is missing")
    if float(metrics["oof_roc_auc"]) < 0.82:
        raise SystemExit("Verification failed: OOF ROC AUC is below 0.82")
    if not audit["all_quality_gates_passed"]:
        raise SystemExit("Verification failed: audit quality gates did not pass")
    if not benchmark["all_quality_gates_passed"]:
        raise SystemExit("Verification failed: benchmark quality gates did not pass")
    tracks = audit.get("validation", {}).get("tracks", {})
    if tracks.get("temporal_holdout", {}).get("status") != "evaluated":
        raise SystemExit("Verification failed: temporal holdout was not evaluated")
    if tracks.get("external_holdout", {}).get("status") != "evaluated":
        raise SystemExit("Verification failed: external holdout was not evaluated")
    if not drift_report.get("all_passed", False):
        raise SystemExit("Verification failed: drift checks did not pass")
    if int(registry.get("input_count", 0)) < 1:
        raise SystemExit("Verification failed: data registry is empty")
    if not model_registry_path.exists():
        raise SystemExit("Verification failed: model registry artifact is missing")
    if not trend_report_path.exists():
        raise SystemExit("Verification failed: trend report artifact is missing")
    if not release_gate_path.exists():
        raise SystemExit("Verification failed: release gate artifact is missing")
    if trend_report.get("status") == "ok" and not trend_report.get("all_passed", False):
        raise SystemExit("Verification failed: trend checks did not pass")
    readiness = str(release_gate.get("readiness", "unknown"))
    trend_status = str(release_gate.get("trend_status", "unknown"))
    if readiness == "no_go":
        raise SystemExit("Verification failed: release gate readiness is no_go")
    if trend_status == "ok" and readiness != "go":
        raise SystemExit("Verification failed: release gate should be go when trend status is ok")
    if not manifest.get("run_id") or not manifest.get("created_at_utc"):
        raise SystemExit("Verification failed: run metadata is missing from manifest")
    if not manifest.get("environment", {}).get("ssl_backend"):
        raise SystemExit("Verification failed: ssl backend is missing from manifest")
    for key in ("semantic_input_hashes", "source_fingerprint", "config_fingerprint", "dependency_fingerprint"):
        if not manifest.get(key):
            raise SystemExit(f"Verification failed: manifest is missing {key}")
    print("BioSpread verification completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
