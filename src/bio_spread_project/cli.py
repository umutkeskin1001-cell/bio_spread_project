"""Command-line interface for BioSpread."""

from __future__ import annotations

import argparse
from pathlib import Path

from bio_spread_project.config_loader import ProjectPaths
from bio_spread_project.governance import (
    evaluate_model_registry_trend,
    load_model_registry,
    load_trend_thresholds,
    write_trend_report,
)
from bio_spread_project.orchestrator import run_pipeline
from bio_spread_project.runtime_policy import EnforcementPolicy, PipelineConfig
from bio_spread_project.verification import run_verification


def _default_trend_thresholds_path() -> Path:
    return Path(__file__).resolve().parents[2] / "project_config" / "trend_thresholds.json"


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_env()
    default_raw_backbones = paths.raw_backbones
    default_raw_amr = paths.raw_amr
    default_geo_spread_features = paths.geo_spread_features
    default_output = paths.default_output_dir

    parser = argparse.ArgumentParser(description="BioSpread geographic spread early-warning workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run BioSpread")
    run.add_argument("--input", type=Path)
    run.add_argument("--records", type=Path, default=default_raw_backbones)
    run.add_argument("--amr", type=Path, default=default_raw_amr)
    run.add_argument("--geo-spread-features", type=Path, default=default_geo_spread_features)
    run.add_argument(
        "--external-holdout",
        type=Path,
        default=None,
        help="Optional external holdout feature surface (TSV) for independent evaluation",
    )
    run.add_argument("--baseline-benchmark", type=Path, default=None)
    run.add_argument("--drift-thresholds", type=Path, default=None)
    run.add_argument("--trend-thresholds", type=Path, default=None)
    run.add_argument(
        "--quality-thresholds",
        type=Path,
        default=None,
        help="Optional JSON file overriding quality gate thresholds",
    )
    run.add_argument(
        "--mode",
        choices=("auto", "raw", "geo", "input"),
        default="auto",
        help="auto: prefer geo surface when present; raw: use records/amr; geo: force geo surface; input: force input CSV",
    )
    run.add_argument("--output-dir", type=Path, default=default_output)
    run.add_argument("--split-year", type=int, default=2020)
    run.add_argument("--horizon-years", type=int, default=3)
    run.add_argument(
        "--require-explicit-surface",
        action="store_true",
        help="Fail auto mode if it would implicitly promote raw inputs to the packaged GeoSpread feature surface",
    )
    run.add_argument("--fail-on-quality-gates", action="store_true")
    run.add_argument("--fail-on-drift-fail", action="store_true")
    run.add_argument("--fail-on-trend-fail", action="store_true")
    run.add_argument(
        "--require-trend-evidence",
        action="store_true",
        help="Treat insufficient model-registry history as a blocking release-gate failure",
    )

    trend = subparsers.add_parser("trend", help="Evaluate rolling trend from model registry history")
    trend.add_argument("--model-registry", type=Path, default=default_output / "model_registry.jsonl")
    trend.add_argument("--window-size", type=int, default=10)
    trend.add_argument("--trend-thresholds", type=Path, default=_default_trend_thresholds_path())
    trend.add_argument("--output", type=Path, default=default_output / "trend_report.json")
    trend.add_argument("--fail-on-fail", action="store_true")
    health = subparsers.add_parser("health", help="Check project health and scientific rigor status")
    health.add_argument("--check-types", action="store_true", help="Run mypy check")
    health.add_argument("--check-tests", action="store_true", help="Run pytest")
    verify = subparsers.add_parser("verify", help="Run verification checks")
    verify.add_argument("--release", action="store_true", help="Run release-grade verification checks")
    verify.add_argument("--skip-security", action="store_true", help="Skip dependency vulnerability audit")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "health":
        print("=== BioSpread Health & Scientific Rigor Check ===")
        exit_code = 0
        if args.check_types:
            import subprocess
            print("Running Mypy (Strict Mode)...")
            completed = subprocess.run(["mypy", "src/bio_spread_project"], check=False)
            if completed.returncode != 0:
                exit_code = completed.returncode
        if args.check_tests:
            import subprocess
            print("Running Pytest...")
            completed = subprocess.run(["pytest", "tests/"], check=False)
            if completed.returncode != 0:
                exit_code = completed.returncode
        print("Health check complete.")
        return exit_code
    if args.command == "verify":
        return run_verification(release=bool(args.release), skip_security=bool(args.skip_security))

    if args.command == "trend":
        thresholds = load_trend_thresholds(args.trend_thresholds)
        payload = evaluate_model_registry_trend(
            entries=load_model_registry(args.model_registry),
            window_size=args.window_size,
            thresholds=thresholds,
        )
        report_path = write_trend_report(args.output, payload)
        print(f"Model registry: {args.model_registry}")
        print(f"Trend report: {report_path}")
        print(f"Trend status: {payload.get('status', 'unknown')}")
        print(f"All checks passed: {payload.get('all_passed', False)}")
        if args.fail_on_fail and payload.get("status") == "ok" and not payload.get("all_passed", False):
            raise SystemExit(2)
        return 0

    policy = EnforcementPolicy(
        fail_on_quality_gates=args.fail_on_quality_gates,
        fail_on_drift_fail=args.fail_on_drift_fail,
        fail_on_trend_fail=args.fail_on_trend_fail,
        require_trend_evidence=args.require_trend_evidence,
        require_explicit_surface=args.require_explicit_surface,
    )
    config = PipelineConfig(
        input_path=args.input,
        backbone_records_path=args.records if args.input is None else None,
        amr_path=args.amr,
        geo_spread_features_path=args.geo_spread_features,
        external_holdout_path=args.external_holdout,
        baseline_benchmark_path=args.baseline_benchmark,
        drift_thresholds_path=args.drift_thresholds,
        trend_thresholds_path=args.trend_thresholds,
        quality_thresholds_path=args.quality_thresholds,
        output_dir=args.output_dir,
        run_mode=args.mode,
        split_year=args.split_year,
        horizon_years=args.horizon_years,
        policy=policy,
    )
    result = run_pipeline(config)
    print(f"Input mode: {result.input_mode}")
    print(f"Selection reason: {result.selection_reason}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Report: {result.report_path}")
    print(f"Audit: {result.audit_path}")
    print(f"Model card: {result.model_card_path}")
    print(f"Benchmark: {result.benchmark_path}")
    print(f"Drift report: {result.drift_report_path}")
    print(f"Data registry: {result.data_registry_path}")
    print(f"Model registry: {result.model_registry_path}")
    print(f"Trend report: {result.trend_report_path}")
    print(f"Release gate: {result.release_gate_path}")
    print(f"Artifact index: {result.artifact_index_path}")
    print(f"ROC AUC: {result.metrics['roc_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
