import concurrent.futures
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import polars as pl

from bio_spread_project.audit import build_run_audit, render_model_card
from bio_spread_project.cache_keys import (
    config_fingerprint,
    dependency_fingerprint,
    semantic_table_hash,
    source_fingerprint,
)
from bio_spread_project.config import load_project_config
from bio_spread_project.dashboard import generate_dashboard
from bio_spread_project.data import load_backbone_records, load_records
from bio_spread_project.drift import evaluate_drift, load_drift_thresholds, load_json
from bio_spread_project.features import build_backbone_features, feature_rows_to_frame
from bio_spread_project.geo_reliability import (
    fit_geo_reliability_surface,
    geo_rows_to_frame,
    load_geo_spread_feature_rows,
)
from bio_spread_project.input_selection import InputSelection, select_input_source
from bio_spread_project.io_utils import (
    sha256_file,
    write_dataclass_csv,
    write_dataclass_parquet,
    write_json,
    write_text,
)
from bio_spread_project.manifest import build_manifest, portable_path
from bio_spread_project.model import ModelRun, fit_model_surface, select_primary_model
from bio_spread_project.model_metrics import validation_summary
from bio_spread_project.registry import append_model_registry_entry
from bio_spread_project.release_gate import build_release_gate_report
from bio_spread_project.reporting import render_markdown_report
from bio_spread_project.runtime_policy import EnforcementPolicy, PipelineConfig
from bio_spread_project.trend import (
    evaluate_model_registry_trend,
    load_model_registry,
    load_trend_thresholds,
    write_trend_report,
)

logger = logging.getLogger(__name__)


@dataclass
class RunContext:
    selection: Optional[InputSelection] = None
    features: Optional[pl.DataFrame] = None
    run: Optional[ModelRun] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    input_paths: Dict[str, Path] = field(default_factory=dict)
    paths: Dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    features_path: Path
    predictions_path: Path
    metrics_path: Path
    model_scorecard_path: Path
    audit_path: Path
    model_card_path: Path
    benchmark_path: Path
    drift_report_path: Path
    data_registry_path: Path
    model_registry_path: Path
    trend_report_path: Path
    release_gate_path: Path
    manifest_path: Path
    report_path: Path
    metrics: dict[str, Any]
    input_mode: str
    selection_reason: str


class BioSpreadPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.state = RunContext()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> PipelineResult:
        logger.info("Starting BioSpread Pipeline Run")

        # 1. Selection
        self.state.selection = select_input_source(
            run_mode=self.config.run_mode,
            input_path=self.config.input_path,
            backbone_records_path=self.config.backbone_records_path,
            amr_path=self.config.amr_path,
            geo_spread_features_path=self.config.geo_spread_features_path,
            require_explicit_surface=self.config.policy.require_explicit_surface,
        )
        selection = self.state.selection

        # 2. Data & Features
        logger.info(f"Loading data in mode: {selection.input_mode}")
        if selection.use_geo_reliability:
            self.state.features = geo_rows_to_frame(load_geo_spread_feature_rows(selection.source_path))
        else:
            if selection.input_mode == "raw_backbone_records":
                records = load_backbone_records(selection.source_path, amr_path=selection.resolved_amr_path)
            else:
                records = load_records(selection.source_path)
            self.state.features = feature_rows_to_frame(
                build_backbone_features(
                    records,
                    split_year=self.config.split_year,
                    horizon_years=self.config.horizon_years,
                )
            )
        features = self.state.features

        if features.is_empty():
            raise ValueError("No eligible feature rows produced")

        # 3. Model Training/Selection
        logger.info("Fitting/Selecting model surface...")
        if selection.use_geo_reliability:
            # We pass the dataframe to fit_geo_reliability_surface (needs update)
            self.state.run = fit_geo_reliability_surface(features)
            import joblib
            joblib.dump(self.state.run.model, self.config.output_dir / "model.joblib")
        else:
            proj_config = load_project_config()
            surface = fit_model_surface(features, proj_config.models)
            primary_name, _ = select_primary_model(surface)
            self.state.run = surface[primary_name]
            self.state.run.model.save(self.config.output_dir / "model.joblib")
        run = self.state.run

        # 4. Evaluation & Audit
        self._validate_external_holdout()
        self.state.input_paths = self._collect_input_paths()
        self.state.metrics = {**run.metrics, **run.calibration, "primary_model": run.model_name}

        if selection.use_geo_reliability and self.config.external_holdout_path:
            holdout_features = geo_rows_to_frame(load_geo_spread_feature_rows(self.config.external_holdout_path))
            from bio_spread_project.evaluation import bootstrap_metric_intervals, evaluate_predictions
            ext_predictions = run.model.predict(holdout_features)
            ext_metrics = {
                **evaluate_predictions(ext_predictions),
                **bootstrap_metric_intervals(ext_predictions, n_resamples=200),
            }
            self.state.metrics.update({f"external_holdout_{k}": v for k, v in ext_metrics.items()})

        self.state.audit = build_run_audit(
            input_paths=self.state.input_paths,
            metrics=self.state.metrics,
            primary_model=run.model_name,
            input_mode=selection.input_mode,
            quality_thresholds_path=self.config.quality_thresholds_path,
        )

        if self.config.policy.fail_on_quality_gates and not self.state.audit["all_quality_gates_passed"]:
            raise RuntimeError("quality_gates failed")

        # 5. Reporting and Async Disk I/O
        self.state.paths = self._write_artifacts()

        logger.info("Pipeline run completed successfully.")
        return PipelineResult(
            **self.state.paths,
            metrics=self.state.metrics,
            input_mode=selection.input_mode,
            selection_reason=selection.selection_reason
        )

    def _collect_input_paths(self) -> dict[str, Path]:
        sel = self.state.selection
        if sel is None:
            raise RuntimeError("Pipeline input selection has not been initialized")
        if sel.use_geo_reliability:
            paths = {"geo_spread_features": sel.source_path}
        elif sel.input_mode == "raw_backbone_records":
            paths = {"records": sel.source_path}
        else:
            paths = {"input": sel.source_path}
        if sel.resolved_amr_path:
            paths["amr"] = sel.resolved_amr_path
        if self.config.external_holdout_path:
            paths["external_holdout"] = Path(self.config.external_holdout_path)
        return paths

    def _write_artifacts(self) -> dict[str, Path]:
        logger.info(f"Writing artifacts to {self.config.output_dir}")
        out = self.config.output_dir
        stage = out / f".staging_{uuid4().hex}"
        stage.mkdir(parents=True, exist_ok=False)
        run = self.state.run
        features = self.state.features
        if run is None:
            raise RuntimeError("Pipeline model run has not been initialized")
        if features is None:
            raise RuntimeError("Pipeline features have not been initialized")
        selection = self.state.selection
        if selection is None:
            raise RuntimeError("Pipeline input selection has not been initialized")
        audit = self.state.audit
        metrics = self.state.metrics

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Parallelize all disk writes
            f_csv = executor.submit(write_dataclass_csv, stage / "features.csv", features)
            f_parquet = executor.submit(write_dataclass_parquet, stage / "features.parquet", features)
            p_csv = executor.submit(write_dataclass_csv, stage / "predictions.csv", run.predictions)
            p_parquet = executor.submit(write_dataclass_parquet, stage / "predictions.parquet", run.predictions)
            m_json = executor.submit(write_json, stage / "metrics.json", metrics)
            a_json = executor.submit(write_json, stage / "audit.json", audit)
            mc_md = executor.submit(write_text, stage / "model_card.md", render_model_card(audit=audit, coefficient_summary=run.coefficient_summary))
            ms_json = executor.submit(write_json, stage / "model_scorecard.json", {"primary_model": run.model_name, "scorecard": [{"model_name": run.model_name, **run.metrics}]})

            # Wait for base paths
            features_path = f_csv.result()
            predictions_path = p_csv.result()
            metrics_path = m_json.result()
            audit_path = a_json.result()
            model_card_path = mc_md.result()
            model_scorecard_path = ms_json.result()
            f_parquet.result()
            p_parquet.result()

            # Sequential dependencies (logic that needs files written or compute)
            benchmark = {
                "project": "BioSpread",
                "validation_summary": validation_summary(metrics),
                "quality_gates": audit["quality_gates"],
                "all_quality_gates_passed": audit["all_quality_gates_passed"],
            }
            benchmark_path = write_json(stage / "benchmark.json", benchmark)

            if self.config.baseline_benchmark_path and self.config.baseline_benchmark_path.exists():
                drift_report = evaluate_drift(current=benchmark, baseline=load_json(self.config.baseline_benchmark_path), thresholds=load_drift_thresholds(self.config.drift_thresholds_path))
            else:
                drift_report = {"all_passed": True, "status": "not_evaluated", "reason": "baseline_not_provided"}
            drift_report_path = write_json(stage / "drift_report.json", drift_report)

            if self.config.policy.fail_on_drift_fail and not drift_report.get("all_passed", False):
                raise RuntimeError("drift_checks failed")

            model_registry_path = append_model_registry_entry(stage / "model_registry.jsonl", {**validation_summary(metrics), "model_name": run.model_name, "all_quality_gates_passed": audit["all_quality_gates_passed"]})

            trend_report = evaluate_model_registry_trend(entries=load_model_registry(model_registry_path), window_size=10, thresholds=load_trend_thresholds(self.config.trend_thresholds_path))
            trend_report_path = write_trend_report(stage / "trend_report.json", trend_report)

            if self.config.policy.fail_on_trend_fail and trend_report.get("status") == "ok" and not trend_report.get("all_passed", False):
                raise RuntimeError("trend_checks failed")
            if self.config.policy.require_trend_evidence and not trend_report.get("trend_evidence_sufficient", False):
                raise RuntimeError("trend_evidence is insufficient")

            release_gate = build_release_gate_report(audit=audit, drift_report=drift_report, trend_report=trend_report, allow_conditional_trend_release=self.config.policy.allow_conditional_release)
            release_gate_path = write_json(stage / "release_gate.json", release_gate)

            generate_dashboard(audit=audit, output_path=stage / "dashboard.html")

            report_path = write_text(stage / "report.md", render_markdown_report(predictions=run.predictions, metrics={**metrics, "all_quality_gates_passed": audit["all_quality_gates_passed"]}, calibration=run.calibration, split_year=self.config.split_year, horizon_years=self.config.horizon_years, coefficient_summary=run.coefficient_summary))

            data_registry_path = write_json(stage / "data_registry.json", {"project": "BioSpread", "input_count": len(self.state.input_paths), "inputs": {k: str(v) for k, v in self.state.input_paths.items()}})

            from bio_spread_project.paths import PROJECT_ROOT
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
                "report": "report.md",
                "manifest": "manifest.json",
                "artifact_index": "artifact_index.json",
            }
            if selection.use_geo_reliability:
                manifest_artifacts["drift_report"] = "benchmark.json"
                manifest_artifacts["model_registry"] = "benchmark.json"
            manifest_path = write_json(stage / "manifest.json", build_manifest(
                selection=selection,
                run_mode=self.config.run_mode,
                policy=self.config.policy,
                input_hashes=audit["input_hashes"],
                semantic_input_hashes={n: semantic_table_hash(p) for n, p in self.state.input_paths.items() if p.exists()},
                split_year=self.config.split_year,
                horizon_years=self.config.horizon_years,
                run_metadata={"run_id": uuid4().hex, "created_at_utc": datetime.now(timezone.utc).isoformat()},
                primary_model=run.model_name,
                threshold_sources={
                    "quality": portable_path(self.config.quality_thresholds_path or PROJECT_ROOT / "project_config" / "quality_thresholds.json"),
                    "drift": portable_path(self.config.drift_thresholds_path or PROJECT_ROOT / "project_config" / "drift_thresholds.json"),
                    "trend": portable_path(self.config.trend_thresholds_path or PROJECT_ROOT / "project_config" / "trend_thresholds.json"),
                    "baseline": portable_path(self.config.baseline_benchmark_path or PROJECT_ROOT / "project_config" / "baseline_benchmark.json"),
                },
                quality_gates=audit["quality_gates"],
                artifacts=manifest_artifacts,
                environment=audit.get("environment", {}),
                source_fingerprint=source_fingerprint(PROJECT_ROOT),
                config_fingerprint=config_fingerprint(PROJECT_ROOT),
                dependency_fingerprint=dependency_fingerprint(PROJECT_ROOT),
            ))
            artifact_stage_paths = {
                "features": features_path,
                "features_parquet": stage / "features.parquet",
                "predictions": predictions_path,
                "predictions_parquet": stage / "predictions.parquet",
                "metrics": metrics_path,
                "audit": audit_path,
                "model_card": model_card_path,
                "model_scorecard": model_scorecard_path,
                "benchmark": benchmark_path,
                "drift_report": drift_report_path,
                "data_registry": data_registry_path,
                "model_registry": model_registry_path,
                "trend_report": trend_report_path,
                "release_gate": release_gate_path,
                "manifest": manifest_path,
                "report": report_path,
            }
            for path in stage.iterdir():
                path.replace(out / path.name)
            stage.rmdir()

            artifact_paths = {
                name: out / path.name
                for name, path in artifact_stage_paths.items()
            }
            artifact_index = {
                "schema_version": "artifact_index_v1",
                "artifact_count": len(artifact_paths),
                "artifacts": {
                    name: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for name, path in artifact_paths.items()
                    if path.exists()
                },
            }
            write_json(out / "artifact_index.json", artifact_index)

        return {
            "features_path": out / "features.csv", "predictions_path": out / "predictions.csv", "metrics_path": out / "metrics.json",
            "model_scorecard_path": out / "model_scorecard.json", "audit_path": out / "audit.json", "model_card_path": out / "model_card.md",
            "benchmark_path": out / "benchmark.json", "drift_report_path": out / "drift_report.json", "data_registry_path": out / "data_registry.json",
            "model_registry_path": out / "model_registry.jsonl", "trend_report_path": out / "trend_report.json", "release_gate_path": out / "release_gate.json",
            "manifest_path": out / "manifest.json", "report_path": out / "report.md",
        }

    def _validate_external_holdout(self) -> None:
        selection = self.state.selection
        if selection is None:
            raise RuntimeError("Pipeline input selection has not been initialized")
        if not selection.use_geo_reliability or self.config.external_holdout_path is None:
            return
        holdout = Path(self.config.external_holdout_path)
        source = Path(selection.source_path)
        if holdout.resolve() == source.resolve() or (holdout.exists() and source.exists() and sha256_file(holdout) == sha256_file(source)):
            raise ValueError("external holdout must be independent from the training feature surface")


def run_pipeline(**kwargs: Any) -> PipelineResult:
    # This remains for backward compatibility but calls the new config-based entry point
    policy = EnforcementPolicy(
        fail_on_quality_gates=kwargs.get("fail_on_quality_gates", False),
        fail_on_drift_fail=kwargs.get("fail_on_drift_fail", False),
        fail_on_trend_fail=kwargs.get("fail_on_trend_fail", False),
        require_trend_evidence=kwargs.get("require_trend_evidence", False),
        require_explicit_surface=kwargs.get("require_explicit_surface", False),
    )
    config = PipelineConfig(
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
    return BioSpreadPipeline(config).run()
