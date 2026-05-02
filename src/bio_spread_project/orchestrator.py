from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp
from typing import Any
from uuid import uuid4

import joblib

from bio_spread_project.artifact_transaction import ArtifactSet, commit_artifact_set
from bio_spread_project.audit import build_run_audit, render_model_card
from bio_spread_project.cache_keys import (
    config_fingerprint,
    dependency_fingerprint,
    semantic_table_hash,
    source_fingerprint,
)
from bio_spread_project.config_loader import ProjectPaths, load_project_config
from bio_spread_project.dashboard import generate_dashboard
from bio_spread_project.data import load_backbone_records_frame, load_records
from bio_spread_project.features import (
    FeatureConfig,
    build_backbone_features,
    build_backbone_features_lazy,
    feature_rows_to_frame,
)
from bio_spread_project.geo_reliability import (
    fit_geo_reliability_surface,
    load_geo_spread_features,
)
from bio_spread_project.governance import (
    build_governance_report,
    build_release_gate_report,
    evaluate_drift,
    evaluate_drift_checks,
    evaluate_quality_gates,
    evaluate_trend_from_registry,
    load_drift_thresholds,
    load_json,
    load_model_registry,
)
from bio_spread_project.input_selection import select_input_source
from bio_spread_project.io_utils import (
    sha256_file,
    write_dataclass_csv,
    write_dataclass_parquet,
    write_json,
    write_text,
)
from bio_spread_project.manifest import build_manifest, portable_path
from bio_spread_project.metrics import bootstrap_metric_intervals, evaluate_predictions
from bio_spread_project.model import fit_model_surface, select_primary_model
from bio_spread_project.model_metrics import validation_summary
from bio_spread_project.registry import append_model_registry_entry
from bio_spread_project.reporting import render_chic_report
from bio_spread_project.runtime_policy import PipelineConfig
from bio_spread_project.thresholds import load_thresholds
from bio_spread_project.visualization import plot_performance_summary, save_table_as_png


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    metrics: dict[str, Any]
    features_path: Path
    metrics_path: Path
    model_scorecard_path: Path
    artifact_index_path: Path
    predictions_path: Path
    report_path: Path
    audit_path: Path
    model_card_path: Path
    benchmark_path: Path
    drift_report_path: Path
    data_registry_path: Path
    model_registry_path: Path
    trend_report_path: Path
    release_gate_path: Path
    manifest_path: Path
    input_mode: str
    selection_reason: str


def _build_input_paths(selection: Any, config: PipelineConfig) -> dict[str, Path]:
    if selection.use_geo_reliability:
        paths = {"geo_spread_features": Path(selection.source_path)}
    elif selection.input_mode == "raw_backbone_records":
        paths = {"records": Path(selection.source_path)}
    else:
        paths = {"input": Path(selection.source_path)}
    if selection.resolved_amr_path is not None:
        paths["amr"] = Path(selection.resolved_amr_path)
    if config.external_holdout_path is not None:
        paths["external_holdout"] = Path(config.external_holdout_path)
    return paths


def _validate_external_holdout(training_source: Path, external_holdout: Path | None) -> None:
    if external_holdout is None:
        return
    if external_holdout.resolve() == training_source.resolve():
        raise ValueError("external holdout must be independent from the training feature surface")
    if external_holdout.exists() and training_source.exists():
        if sha256_file(external_holdout) == sha256_file(training_source):
            raise ValueError("external holdout must be independent from the training feature surface")

        # Genomic identity leakage check: ensure no backbone IDs are shared
        from bio_spread_project.data_io import read_table
        train_ids = set(read_table(training_source).select("backbone_id").to_series().to_list())
        holdout_ids = set(read_table(external_holdout).select("backbone_id").to_series().to_list())
        overlap = train_ids.intersection(holdout_ids)
        if overlap:
            count = len(overlap)
            sample = sorted(list(overlap))[:3]
            raise ValueError(f"Identity Leakage detected: {count} backbones shared between training and external holdout (e.g., {', '.join(sample)})")


def _build_config_from_kwargs(kwargs: dict[str, Any]) -> PipelineConfig:
    from bio_spread_project.runtime_policy import EnforcementPolicy

    policy = EnforcementPolicy(
        fail_on_quality_gates=kwargs.get("fail_on_quality_gates", False),
        fail_on_drift_fail=kwargs.get("fail_on_drift_fail", False),
        fail_on_trend_fail=kwargs.get("fail_on_trend_fail", False),
        require_trend_evidence=kwargs.get("require_trend_evidence", False),
        require_explicit_surface=kwargs.get("require_explicit_surface", False),
    )
    return PipelineConfig(
        input_path=kwargs.get("input_path"),
        backbone_records_path=kwargs.get("backbone_records_path"),
        amr_path=kwargs.get("amr_path"),
        geo_spread_features_path=kwargs.get("geo_spread_features_path"),
        external_holdout_path=kwargs.get("external_holdout_path"),
        baseline_benchmark_path=kwargs.get("baseline_benchmark_path"),
        drift_thresholds_path=kwargs.get("drift_thresholds_path"),
        trend_thresholds_path=kwargs.get("trend_thresholds_path"),
        quality_thresholds_path=kwargs.get("quality_thresholds_path"),
        output_dir=Path(kwargs.get("output_dir", "reports/run")),
        run_mode=kwargs.get("run_mode", "auto"),
        split_year=kwargs.get("split_year", 2020),
        horizon_years=kwargs.get("horizon_years", 3),
        policy=policy,
    )


def run_pipeline(config: PipelineConfig | None = None, **kwargs: Any) -> PipelineResult:
    if config is None:
        config = _build_config_from_kwargs(kwargs)
    elif kwargs:
        raise ValueError("Provide either PipelineConfig or keyword arguments, not both")

    paths = ProjectPaths.from_env()
    selection = select_input_source(config=config, paths=paths)

    if selection.use_geo_reliability:
        features = load_geo_spread_features(selection.source_path)
    else:
        if selection.input_mode == "raw_backbone_records":
            record_frame = load_backbone_records_frame(selection.source_path, amr_path=selection.resolved_amr_path)
            # Zero-copy Polars plan: raw production keeps observations columnar
            # through feature aggregation, avoiding dataclass/asdict cache churn.
            features = build_backbone_features_lazy(
                record_frame.lazy(),
                config=FeatureConfig(split_year=config.split_year, horizon_years=config.horizon_years),
            ).collect()
        else:
            record_rows = load_records(selection.source_path)
            features = feature_rows_to_frame(
                build_backbone_features(record_rows, split_year=config.split_year, horizon_years=config.horizon_years)
            )
    if features.is_empty():
        raise ValueError("No eligible feature rows produced")

    if selection.use_geo_reliability:
        run = fit_geo_reliability_surface(features)
    else:
        surface = fit_model_surface(features, load_project_config().models)
        primary_name, _ = select_primary_model(surface)
        run = surface[primary_name]

    _validate_external_holdout(Path(selection.source_path), config.external_holdout_path)
    metrics = {**run.metrics, **run.calibration, "primary_model": run.model_name}
    if selection.use_geo_reliability and config.external_holdout_path:
        holdout_features = load_geo_spread_features(config.external_holdout_path)
        ext_predictions = run.model.predict(holdout_features)
        ext_metrics = {**evaluate_predictions(ext_predictions), **bootstrap_metric_intervals(ext_predictions, n_resamples=200)}
        metrics.update({f"external_holdout_{k}": v for k, v in ext_metrics.items()})

    input_paths = _build_input_paths(selection, config)
    audit = build_run_audit(
        input_paths=input_paths,
        metrics=metrics,
        primary_model=run.model_name,
        input_mode=selection.input_mode,
        quality_thresholds_path=config.quality_thresholds_path,
    )

    stage = Path(mkdtemp(prefix=".staging_", dir=str(config.output_dir.parent)))
    try:
        write_dataclass_csv(stage / "features.csv", features)
        write_dataclass_parquet(stage / "features.parquet", features)
        write_dataclass_csv(stage / "predictions.csv", run.predictions)
        write_dataclass_parquet(stage / "predictions.parquet", run.predictions)
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "audit.json", audit)
        write_text(stage / "model_card.md", render_model_card(audit=audit, coefficient_summary=run.coefficient_summary))
        write_json(stage / "model_scorecard.json", {"primary_model": run.model_name, "scorecard": [{"model_name": run.model_name, **run.metrics}]})
        joblib.dump(run.model, stage / "model.joblib")

        benchmark = {
            "project": "BioSpread",
            "validation_summary": validation_summary(metrics),
            "quality_gates": audit["quality_gates"],
            "all_quality_gates_passed": audit["all_quality_gates_passed"],
        }
        write_json(stage / "benchmark.json", benchmark)
        if config.baseline_benchmark_path and Path(config.baseline_benchmark_path).exists():
            drift_payload = evaluate_drift(
                current=benchmark,
                baseline=load_json(config.baseline_benchmark_path),
                thresholds=load_drift_thresholds(config.drift_thresholds_path),
            )
        else:
            drift_payload = {"all_passed": True, "status": "not_evaluated", "reason": "baseline_not_provided"}
        write_json(stage / "drift_report.json", drift_payload)

        model_registry_path = append_model_registry_entry(
            stage / "model_registry.jsonl",
            {**validation_summary(metrics), "model_name": run.model_name, "input_mode": selection.input_mode, "all_quality_gates_passed": audit["all_quality_gates_passed"]},
        )
        registry_entries = load_model_registry(model_registry_path).to_dicts()
        thresholds = load_thresholds(
            quality_path=config.quality_thresholds_path,
            drift_path=config.drift_thresholds_path,
            trend_path=config.trend_thresholds_path,
        )
        quality_checks = evaluate_quality_gates(
            metrics=metrics,
            input_mode=selection.input_mode,
            leakage_audit_passed=audit.get("leakage_audit", {}).get("status") == "pass",
            thresholds=thresholds.quality,
        )
        drift_checks = evaluate_drift_checks(
            current=benchmark,
            baseline=load_json(config.baseline_benchmark_path) if config.baseline_benchmark_path and Path(config.baseline_benchmark_path).exists() else {"validation_summary": {}},
            thresholds=thresholds.drift,
        )
        trend_checks = evaluate_trend_from_registry(
            entries=registry_entries,
            thresholds=thresholds.trend,
            model_name=run.model_name,
            input_mode=selection.input_mode,
            window_size=10,
        )
        governance = build_governance_report(
            quality_checks=quality_checks,
            drift_checks=drift_checks,
            trend_checks=trend_checks,
            policy=config.policy,
        )
        write_json(
            stage / "trend_report.json",
            {
                "status": "ok" if not any(c.status == "not_evaluated" for c in trend_checks) else "insufficient_data",
                "all_passed": all(c.passed for c in trend_checks if c.status != "not_evaluated"),
                "trend_evidence_sufficient": not any(c.status == "not_evaluated" for c in trend_checks),
                "checks": {c.name: c.detail for c in trend_checks},
            },
        )
        release_gate = build_release_gate_report(
            audit=audit,
            drift_report=drift_payload,
            trend_report=load_json(stage / "trend_report.json"),
            allow_conditional_trend_release=config.policy.allow_conditional_release,
        )
        write_json(stage / "release_gate.json", release_gate)
        write_json(
            stage / "governance_report.json",
            {
                "readiness": governance.readiness,
                "blocked_by": governance.blocked_by,
                "quality_checks": [c.__dict__ for c in governance.quality_checks],
                "drift_checks": [c.__dict__ for c in governance.drift_checks],
                "trend_checks": [c.__dict__ for c in governance.trend_checks],
                "policy_flags": governance.policy_flags,
            },
        )

        generate_dashboard(audit=audit, output_path=stage / "dashboard.html")
        write_text(
            stage / "report.md",
            render_chic_report(
                predictions=run.predictions,
                metrics={**metrics, "all_quality_gates_passed": audit["all_quality_gates_passed"]},
                audit=audit,
                governance=governance,
                release_gate=release_gate,
                split_year=config.split_year,
                horizon_years=config.horizon_years,
            ),
        )
        plot_performance_summary(metrics, stage / "performance_summary.png")

        top_preds = sorted(run.predictions, key=lambda x: x.risk_probability, reverse=True)[:10]
        risk_headers = ["Rank", "Backbone ID", "Risk Prob", "Confidence", "Future Spread"]
        risk_rows = [[i+1, p.backbone_id, f"{p.risk_probability:.4f}", p.confidence_tier, p.n_new_countries_future] for i, p in enumerate(top_preds)]
        save_table_as_png(risk_headers, risk_rows, "HIGH-RISK CANDIDATE REGISTRY", stage / "risk_table.png")

        metric_headers = ["Metric", "Value", "Threshold", "Status"]
        metric_rows = [
            ["ROC AUC", f"{metrics.get('roc_auc', 0):.4f}", ">= 0.820", "PASS" if metrics.get("roc_auc", 0) >= 0.82 else "FAIL"],
            ["Avg Precision", f"{metrics.get('average_precision', 0):.4f}", "> Prev", "PASS" if metrics.get("average_precision", 0) > metrics.get("prevalence", 0) else "FAIL"],
            ["Spatial CV AUC", f"{metrics.get('group_oof_roc_auc', 0):.4f}", ">= 0.800", "PASS" if metrics.get("group_oof_roc_auc", 0) >= 0.80 else "FAIL"],
            ["Calibration ECE", f"{metrics.get('expected_calibration_error', 0):.4f}", "<= 0.100", "PASS" if metrics.get("expected_calibration_error", 1) <= 0.1 else "FAIL"],
        ]
        save_table_as_png(metric_headers, metric_rows, "CORE ANALYTIC METRICS", stage / "metrics_table.png")

        write_json(
            stage / "data_registry.json",
            {"project": "BioSpread", "input_count": len(input_paths), "inputs": {k: str(v) for k, v in input_paths.items()}},
        )

        root = paths.project_root
        manifest_artifacts = {
            "features": "features.csv",
            "features_parquet": "features.parquet",
            "predictions": "predictions.csv",
            "predictions_parquet": "predictions.parquet",
            "metrics": "metrics.json",
            "data_registry": "data_registry.json",
            "drift_report": "drift_report.json",
            "model_registry": "model_registry.jsonl",
            "trend_report": "trend_report.json",
            "audit": "audit.json",
            "model_card": "model_card.md",
            "model_scorecard": "model_scorecard.json",
            "release_gate": "release_gate.json",
            "governance_report": "governance_report.json",
            "report": "report.md",
            "manifest": "manifest.json",
            "artifact_index": "artifact_index.json",
        }
        write_json(
            stage / "manifest.json",
            build_manifest(
                selection=selection,
                run_mode=config.run_mode,
                policy=config.policy,
                input_hashes=audit["input_hashes"],
                semantic_input_hashes={name: semantic_table_hash(path) for name, path in input_paths.items() if path.exists()},
                split_year=config.split_year,
                horizon_years=config.horizon_years,
                run_metadata={"run_id": uuid4().hex, "created_at_utc": datetime.now(timezone.utc).isoformat()},
                primary_model=run.model_name,
                threshold_sources={
                    "quality": portable_path(config.quality_thresholds_path or (root / "project_config" / "quality_thresholds.json")),
                    "drift": portable_path(config.drift_thresholds_path or (root / "project_config" / "drift_thresholds.json")),
                    "trend": portable_path(config.trend_thresholds_path or (root / "project_config" / "trend_thresholds.json")),
                    "baseline": portable_path(config.baseline_benchmark_path or (root / "project_config" / "baseline_benchmark.json")),
                },
                quality_gates=audit["quality_gates"],
                artifacts=manifest_artifacts,
                environment=audit.get("environment", {}),
                source_fingerprint=source_fingerprint(root),
                config_fingerprint=config_fingerprint(root),
                dependency_fingerprint=dependency_fingerprint(root),
            ),
        )

        artifact_set = ArtifactSet(
            staging_dir=stage,
            files={name: stage / rel for name, rel in manifest_artifacts.items() if name != "artifact_index"},
        )
        commit_artifact_set(config.output_dir, artifact_set)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    if config.policy.fail_on_quality_gates and not audit["all_quality_gates_passed"]:
        raise RuntimeError("quality_gates failed")
    if config.policy.fail_on_drift_fail and not drift_payload.get("all_passed", False):
        raise RuntimeError("drift_checks failed")
    trend_payload = load_json(config.output_dir / "trend_report.json")
    if config.policy.fail_on_trend_fail and trend_payload.get("status") == "ok" and not trend_payload.get("all_passed", False):
        raise RuntimeError("trend_checks failed")
    if config.policy.require_trend_evidence and not trend_payload.get("trend_evidence_sufficient", False):
        raise RuntimeError("trend_evidence is insufficient")

    return PipelineResult(
        output_dir=config.output_dir,
        metrics=metrics,
        features_path=config.output_dir / "features.csv",
        metrics_path=config.output_dir / "metrics.json",
        model_scorecard_path=config.output_dir / "model_scorecard.json",
        artifact_index_path=config.output_dir / "artifact_index.json",
        predictions_path=config.output_dir / "predictions.csv",
        report_path=config.output_dir / "report.md",
        audit_path=config.output_dir / "audit.json",
        model_card_path=config.output_dir / "model_card.md",
        benchmark_path=config.output_dir / "benchmark.json",
        drift_report_path=config.output_dir / "drift_report.json",
        data_registry_path=config.output_dir / "data_registry.json",
        model_registry_path=config.output_dir / "model_registry.jsonl",
        trend_report_path=config.output_dir / "trend_report.json",
        release_gate_path=config.output_dir / "release_gate.json",
        manifest_path=config.output_dir / "manifest.json",
        input_mode=selection.input_mode,
        selection_reason=selection.selection_reason,
    )
