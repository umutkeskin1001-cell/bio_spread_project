import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from bio_spread_project.config_loader import load_project_config
from bio_spread_project.data import load_backbone_records, load_records
from bio_spread_project.features import build_backbone_features
from bio_spread_project.geo_reliability import (
    FEATURE_COLUMNS,
    GeoSpreadFeatureRow,
    fit_geo_reliability_surface,
    leakage_audit,
    load_geo_spread_feature_rows,
    single_feature_leakage_scan,
)
from bio_spread_project.governance import (
    DriftThresholds,
    TrendThresholds,
    evaluate_drift,
    evaluate_model_registry_trend,
    load_drift_thresholds,
    load_quality_thresholds,
    load_trend_thresholds,
)
from bio_spread_project.metrics import (
    bootstrap_metric_intervals,
    calibration_summary,
    evaluate_predictions,
)
from bio_spread_project.model import BioSpreadRiskModel, fit_model_surface, select_primary_model
from bio_spread_project.orchestrator import run_pipeline
from bio_spread_project.reporting import render_markdown_report

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "sample_plasmid_records.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_BACKBONES = PROJECT_ROOT / "data" / "raw" / "plasmid_backbones.tsv"
RAW_AMR = PROJECT_ROOT / "data" / "raw" / "amr.tsv"
GEO_SPREAD_FEATURES = PROJECT_ROOT / "data" / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"
GEO_HOLDOUT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "geo_holdout.tsv"


def _require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"Packaged data files are missing: {', '.join(missing)}")


def test_build_backbone_features_uses_only_pre_split_history():
    records = load_records(FIXTURE)

    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    alpha = next(row for row in features if row.backbone_id == "bb_alpha")

    assert alpha.n_countries_pre == 2
    assert alpha.n_new_countries_future == 3
    assert alpha.label_geo_spread == 1
    assert 0.0 <= alpha.knownness_score <= 1.0


def test_model_outputs_ranked_probabilities_with_confidence_tiers():
    records = load_records(FIXTURE)
    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    model = BioSpreadRiskModel.train(features)

    predictions = model.predict(features)
    top = max(predictions, key=lambda row: row.risk_probability)

    assert top.backbone_id == "bb_alpha"
    assert top.risk_probability > 0.5
    assert top.confidence_tier in {"high", "medium", "review"}


def test_evaluation_and_calibration_summarize_reliability():
    records = load_records(FIXTURE)
    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    model = BioSpreadRiskModel.train(features)
    predictions = model.predict(features)

    metrics = evaluate_predictions(predictions)
    bootstrap = bootstrap_metric_intervals(predictions, n_resamples=80)
    calibration = calibration_summary(predictions, bins=3)

    assert metrics["n_backbones"] == 5
    assert metrics["roc_auc"] >= 0.5
    assert metrics["average_precision"] >= metrics["prevalence"]
    assert bootstrap["bootstrap_roc_auc_ci_low"] <= metrics["roc_auc"] <= bootstrap["bootstrap_roc_auc_ci_high"]
    assert calibration["expected_calibration_error"] >= 0.0 or calibration["brier_score"] >= 0.0


def test_report_marks_missing_holdout_tracks_as_not_evaluated():
    report = render_markdown_report(
        predictions=[],
        metrics={
            "roc_auc": 0.90,
            "average_precision": 0.80,
            "prevalence": 0.20,
            "top_k_precision": 0.70,
            "abstain_rate": 0.10,
            "all_quality_gates_passed": False,
        },
        calibration={"expected_calibration_error": 0.05, "brier_score": 0.10, "calibration_bins": []},
        split_year=2020,
        horizon_years=3,
    )

    assert "Group OOF ROC AUC: `not_evaluated`" in report
    assert "Temporal holdout ROC AUC: `not_evaluated`" in report


def test_pipeline_writes_predictions_and_report(tmp_path):
    output_dir = tmp_path / "run"

    result = run_pipeline(
        input_path=FIXTURE,
        output_dir=output_dir,
        split_year=2020,
        horizon_years=3,
    )

    assert result.predictions_path.exists()
    assert (output_dir / "features.parquet").exists()
    assert (output_dir / "predictions.parquet").exists()
    assert (output_dir / "artifact_index.json").exists()
    assert result.report_path.exists()
    assert result.model_scorecard_path.exists()
    assert result.benchmark_path.exists()
    assert result.drift_report_path.exists()
    assert result.model_registry_path.exists()
    assert result.trend_report_path.exists()
    assert result.release_gate_path.exists()
    assert result.manifest_path.exists()
    assert result.metrics["n_backbones"] == 5
    assert "BioSpread" in result.report_path.read_text(encoding="utf-8")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] is not None
    assert "T" in manifest["created_at_utc"]
    assert manifest["policy"]["fail_on_quality_gates"] is False
    assert "trend" in manifest["threshold_sources"]
    assert manifest["artifacts"]["features_parquet"] == "features.parquet"
    assert manifest["artifacts"]["artifact_index"] == "artifact_index.json"


def test_geo_pipeline_separates_final_predictions_from_validation_metrics(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_separation",
    )
    predictions = result.predictions_path.read_text(encoding="utf-8")
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert "risk_probability" in predictions
    assert metrics["validation_mode"] == "spatial_group_cv_stacked"
    assert "oof_roc_auc" in metrics
    assert metrics["roc_auc"] == metrics["oof_roc_auc"]


def test_geo_pipeline_exports_real_temporal_holdout_evidence(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_temporal",
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    temporal = audit["validation"]["tracks"]["temporal_holdout"]

    assert temporal["status"] == "evaluated"
    assert temporal["n_backbones"] > 0
    assert "average_precision" in temporal
    assert result.metrics["temporal_holdout_n_backbones"] < result.metrics["n_backbones"]


def test_geo_surface_requires_core_model_columns(tmp_path):
    broken = tmp_path / "broken.tsv"
    broken.write_text("backbone_id\tspread_label\tn_new_countries\nbb1\t1\t2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_geo_spread_feature_rows(broken)


def test_geo_surface_rejects_leakage_columns_in_actual_input(tmp_path):
    clean = GEO_SPREAD_FEATURES.read_text(encoding="utf-8").splitlines()
    header = clean[0] + "\tfuture_country_count"
    rows = [line + "\t0" for line in clean[1:4]]
    leaked = tmp_path / "leaked.tsv"
    leaked.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="leakage-prone columns"):
        load_geo_spread_feature_rows(leaked)


def test_geo_surface_derives_optional_geo_features_from_packaged_aliases():
    rows = load_geo_spread_feature_rows(GEO_SPREAD_FEATURES)
    first = rows[0]

    assert "geo_country_entropy_train" in first.features
    assert "geo_macro_region_entropy_train" in first.features
    assert "geo_dominant_region_share_train" in first.features
    assert "geo_country_record_count_train" in first.features


def test_geo_predictions_expose_honest_attribution_metadata(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_attribution",
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))

    assert "top_features" in audit["validation"]
    assert isinstance(audit["validation"]["top_features"], list)
    assert audit["validation"]["top_features"][0]["feature"]
    assert audit["validation"]["top_features"][0]["score"] >= 0.0


def test_calibration_summary_exports_bins_for_report():
    rows = load_geo_spread_feature_rows(GEO_SPREAD_FEATURES)
    run = fit_geo_reliability_surface(rows)

    assert "calibration_bins" in run.calibration
    assert len(run.calibration["calibration_bins"]) >= 5
    assert {"bin_start", "bin_end", "mean_prediction", "observed_rate", "count"} <= set(
        run.calibration["calibration_bins"][0]
    )


def test_load_backbone_records_joins_raw_amr_evidence():
    _require_files(RAW_BACKBONES, RAW_AMR)
    records = load_backbone_records(RAW_BACKBONES, amr_path=RAW_AMR, limit=500)

    assert records
    assert all(record.backbone_id for record in records)
    assert max(record.amr_gene_count for record in records) >= 0.0


def test_production_pipeline_generates_features_from_packaged_raw_tables(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR)

    result = run_pipeline(
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        output_dir=tmp_path / "production",
    )

    assert result.metrics["n_backbones"] > 100
    assert result.predictions_path.exists()
    assert result.model_scorecard_path.exists()
    assert result.manifest_path.exists()


def test_model_surface_selects_primary_model_from_scorecard():
    records = load_records(FIXTURE)
    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    config = load_project_config()

    surface = fit_model_surface(features, config.models)
    primary_name, scorecard = select_primary_model(surface)

    assert set(surface) == {"mobility", "amr_mobility", "clinical_hybrid"}
    assert primary_name in surface or primary_name == "geobio_reliability_ensemble"
    assert scorecard[0]["model_name"] == primary_name
    assert "selection_score" in scorecard[0]
    assert "validation_mode" in scorecard[0]
    assert "roc_auc" in scorecard[0] or "oof_roc_auc" in scorecard[0]
    assert "coefficient_summary" in scorecard[0]


def test_learned_model_surface_beats_prevalence_on_raw_tables(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR)
    result = run_pipeline(
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        output_dir=tmp_path / "production",
    )

    assert "validation_mode" in result.metrics
    assert result.metrics["roc_auc"] >= 0.60
    assert result.metrics["average_precision"] > result.metrics["prevalence"]
    assert result.metrics.get("expected_calibration_error", 0.0) <= 0.20 or result.metrics.get("brier_score", 0.0) <= 0.25


def test_geo_reliability_surface_reaches_competition_auc_target(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR, GEO_SPREAD_FEATURES)

    result = run_pipeline(
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        external_holdout_path=GEO_HOLDOUT_FIXTURE,
        output_dir=tmp_path / "production",
    )

    assert "validation_mode" in result.metrics
    assert result.metrics["primary_model"] == "geobio_reliability_ensemble"
    assert result.metrics["roc_auc"] >= 0.80
    assert result.metrics["average_precision"] >= 0.70
    assert result.metrics.get("expected_calibration_error", 0.0) <= 0.10 or result.metrics.get("brier_score", 0.0) <= 0.15


def test_geo_reliability_feature_contract_blocks_future_leakage():
    audit = leakage_audit(FEATURE_COLUMNS)

    assert audit["status"] == "pass"
    assert audit["blocked_columns"] == []
    assert all("label" not in column for column in FEATURE_COLUMNS)
    assert all("future" not in column for column in FEATURE_COLUMNS)
    assert all("test" not in column for column in FEATURE_COLUMNS)


def test_pipeline_writes_model_card_and_audit_artifacts(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR, GEO_SPREAD_FEATURES)
    result = run_pipeline(
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        external_holdout_path=GEO_HOLDOUT_FIXTURE,
        output_dir=tmp_path / "production",
    )

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    benchmark = json.loads(result.benchmark_path.read_text(encoding="utf-8"))
    model_card = result.model_card_path.read_text(encoding="utf-8")

    assert result.audit_path.exists()
    assert result.model_card_path.exists()
    assert audit["quality_gates"]["auc_at_least_target"] is True
    assert audit["quality_gates"]["adversarial_leakage_scan_passed"] is True
    assert audit["leakage_audit"]["status"] == "pass"
    assert float(audit["validation"]["max_single_feature_auc"]) < 0.95
    assert int(audit["validation"]["suspicious_feature_count"]) == 0
    assert "geo_spread_features" in audit["input_hashes"]
    assert manifest["artifacts"]["data_registry"] == "data_registry.json"
    assert manifest["artifacts"]["drift_report"] == "drift_report.json"
    assert manifest["artifacts"]["model_registry"] == "model_registry.jsonl"
    assert manifest["artifacts"]["trend_report"] == "trend_report.json"
    assert benchmark["all_quality_gates_passed"] is True
    assert "bootstrap_roc_auc_ci_low" in benchmark["validation_summary"]
    assert "geobio_reliability_ensemble" in model_card
    assert "validation" in model_card.lower()


def test_custom_quality_thresholds_can_fail_audit_gate(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR, GEO_SPREAD_FEATURES)
    strict_thresholds = tmp_path / "strict_thresholds.json"
    strict_thresholds.write_text(
        json.dumps(
            {
                "auc_min": 0.99,
                "average_precision_above_prevalence": True,
                "calibration_ece_max": 0.05,
                "bootstrap_auc_ci_low_min": 0.98,
                "bootstrap_average_precision_ci_low_above_prevalence": True,
                "group_auc_min": 0.98,
                "temporal_holdout_auc_min": 0.98,
                "max_single_feature_auc_max": 0.95,
                "suspicious_feature_count_max": 0,
            }
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        quality_thresholds_path=strict_thresholds,
        output_dir=tmp_path / "strict_run",
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["all_quality_gates_passed"] is False
    assert audit["quality_gates"]["auc_at_least_target"] is False
    assert float(audit["quality_thresholds"]["auc_min"]) == 0.99


def test_pipeline_can_enforce_quality_gate_failure_policy(tmp_path):
    _require_files(GEO_SPREAD_FEATURES)
    strict_thresholds = tmp_path / "strict_thresholds.json"
    strict_thresholds.write_text(
        json.dumps(
            {
                "auc_min": 0.99,
                "average_precision_above_prevalence": True,
                "calibration_ece_max": 0.05,
                "bootstrap_auc_ci_low_min": 0.98,
                "bootstrap_average_precision_ci_low_above_prevalence": True,
                "group_auc_min": 0.98,
                "temporal_holdout_auc_min": 0.98,
                "max_single_feature_auc_max": 0.95,
                "suspicious_feature_count_max": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="quality_gates"):
        run_pipeline(
            run_mode="geo",
            geo_spread_features_path=GEO_SPREAD_FEATURES,
            quality_thresholds_path=strict_thresholds,
            output_dir=tmp_path / "strict_policy_run",
            fail_on_quality_gates=True,
        )


def test_report_explains_biological_problem_and_reliability():
    records = load_records(FIXTURE)
    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    predictions = BioSpreadRiskModel.train(features).predict(features)
    metrics = evaluate_predictions(predictions)
    calibration = calibration_summary(predictions, bins=3)

    report = render_markdown_report(
        predictions=predictions,
        metrics=metrics,
        calibration=calibration,
        split_year=2020,
        horizon_years=3,
    )

    assert "coğrafi yayılım" in report.lower()
    assert "kalibrasyon" in report.lower()
    assert "bb_alpha" in report


def test_report_marks_metrics_as_cross_validated(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR)
    result = run_pipeline(
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        output_dir=tmp_path / "production",
    )
    report = result.report_path.read_text(encoding="utf-8")

    assert "validation" in report.lower()
    assert "Katsayı özeti" in report


def test_cli_run_module_executes_from_project_root(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR, GEO_SPREAD_FEATURES)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "run"

    env = os.environ.copy()
    src_root = project_root / "src"
    env["PYTHONPATH"] = str(src_root) if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bio_spread_project.cli",
            "run",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Input mode:" in completed.stdout
    assert (output_dir / "report.md").exists()
    assert (output_dir / "predictions.csv").exists()


def test_manifest_uses_portable_inputs(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR)
    result = run_pipeline(
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        output_dir=tmp_path / "production",
    )

    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert "data/raw/plasmid_backbones.tsv" in manifest
    assert "data/raw/amr.tsv" in manifest
    assert "/Users/" not in manifest


def test_pipeline_rejects_missing_records_path_in_auto_mode(tmp_path):
    _require_files(GEO_SPREAD_FEATURES)
    missing_records = tmp_path / "missing_records.tsv"
    assert not missing_records.exists()
    assert GEO_SPREAD_FEATURES.exists()

    try:
        run_pipeline(
            backbone_records_path=missing_records,
            geo_spread_features_path=GEO_SPREAD_FEATURES,
            output_dir=tmp_path / "run",
        )
    except ValueError as exc:
        message = str(exc)
        assert "does not exist" in message
        assert "records" in message
    else:
        raise AssertionError("Expected ValueError for a missing records path")


def test_pipeline_geo_mode_runs_without_records_path(tmp_path):
    _require_files(GEO_SPREAD_FEATURES)
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_only",
    )
    assert result.metrics["primary_model"] == "geobio_reliability_ensemble"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_mode"] == "geo"
    assert manifest["input_mode"] == "geo_reliability_feature_surface"
    assert "geo_spread_features" in manifest["inputs"]
    assert "records" not in manifest["inputs"]


def test_geo_mode_requires_feature_contract_columns(tmp_path):
    broken = tmp_path / "broken_geo.tsv"
    broken.write_text(
        "\t".join(["backbone_id", "spread_label", "n_new_countries", "T_eff_norm"]) + "\n"
        + "\t".join(["bb_x", "1", "2", "0.7"])
        + "\n",
        encoding="utf-8",
    )

    try:
        run_pipeline(
            run_mode="geo",
            geo_spread_features_path=broken,
            output_dir=tmp_path / "geo_broken",
        )
    except ValueError as exc:
        assert "missing required columns" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for a broken geo feature surface")


def test_cli_defaults_follow_bio_spread_data_root(monkeypatch, tmp_path):
    from bio_spread_project import cli

    custom_root = tmp_path / "custom_data"
    monkeypatch.setenv("BIO_SPREAD_DATA_ROOT", str(custom_root))
    parser = cli.build_parser()
    args = parser.parse_args(["run"])

    assert str(args.records).startswith(str(custom_root))
    assert str(args.amr).startswith(str(custom_root))
    assert str(args.geo_spread_features).startswith(str(custom_root))


def test_geo_reliability_reports_group_and_temporal_metrics(tmp_path):
    _require_files(GEO_SPREAD_FEATURES)
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_metrics",
    )
    assert "roc_auc" in result.metrics
    assert "bootstrap_roc_auc_ci_low" in result.metrics
    assert result.metrics["validation_mode"] == "spatial_group_cv_stacked"
    assert "max_single_feature_auc" in result.metrics
    assert result.metrics["max_single_feature_auc"] < 0.95
    registry = json.loads((tmp_path / "geo_metrics" / "data_registry.json").read_text(encoding="utf-8"))
    assert registry["project"] == "BioSpread"
    assert "geo_spread_features" in registry["inputs"]


def test_adversarial_leakage_scan_flags_near_deterministic_feature():
    rows: list[GeoSpreadFeatureRow] = []
    for index in range(24):
        label = 1 if index % 2 == 0 else 0
        rows.append(
            GeoSpreadFeatureRow(
                backbone_id=f"bb_{index}",
                label_geo_spread=label,
                n_new_countries_future=2 if label else 0,
                knownness_score=0.8,
                max_resolved_year_train=2010 + (index % 6),
                features={
                    "T_eff_norm": 0.3,
                    "H_obs_specialization_norm": 0.4,
                    "A_eff_norm": 0.5,
                    "coherence_score": float(label),
                    "backbone_purity_norm": 0.6,
                    "assignment_confidence_norm": 0.7,
                    "mash_neighbor_distance_train_norm": 0.2,
                    "orit_support": 0.4,
                    "H_external_host_range_norm": 0.3,
                    "geo_country_entropy_train": 0.1,
                    "geo_macro_region_entropy_train": 0.1,
                    "geo_dominant_region_share_train": 0.9,
                    "geo_country_record_count_train": 0.2,
                }
            )
        )
    scan = single_feature_leakage_scan(rows, auc_threshold=0.95)
    assert scan["max_single_feature_auc"] >= 0.99
    assert scan["suspicious_feature_count"] == 1


def test_pipeline_writes_data_registry_artifact(tmp_path):
    _require_files(RAW_BACKBONES, RAW_AMR)
    result = run_pipeline(
        run_mode="raw",
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        output_dir=tmp_path / "raw_registry",
    )
    registry_path = tmp_path / "raw_registry" / "data_registry.json"
    assert registry_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"]["data_registry"] == "data_registry.json"
    assert manifest["artifacts"]["drift_report"] == "drift_report.json"
    assert manifest["artifacts"]["model_registry"] == "model_registry.jsonl"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["input_count"] >= 2
    assert "records" in registry["inputs"]
    assert "amr" in registry["inputs"]


def test_drift_evaluation_flags_large_drop():
    baseline = {
        "validation_summary": {
            "roc_auc": 0.85,
            "average_precision": 0.8,
            "group_oof_roc_auc": 0.84,
            "temporal_holdout_roc_auc": 0.83,
            "external_holdout_roc_auc": 0.82,
            "max_single_feature_auc": 0.70,
            "suspicious_feature_count": 0,
        }
    }
    current = {
        "validation_summary": {
            "roc_auc": 0.75,
            "average_precision": 0.70,
            "group_oof_roc_auc": 0.80,
            "temporal_holdout_roc_auc": 0.81,
            "external_holdout_roc_auc": 0.80,
            "max_single_feature_auc": 0.71,
            "suspicious_feature_count": 0,
        }
    }
    report = evaluate_drift(current=current, baseline=baseline, thresholds=DriftThresholds())
    assert report["all_passed"] is False
    assert report["metric_checks"]["roc_auc"]["status"] == "fail"


def test_external_holdout_metrics_are_exported_for_geo_mode(tmp_path):
    _require_files(GEO_SPREAD_FEATURES, GEO_HOLDOUT_FIXTURE)
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        external_holdout_path=GEO_HOLDOUT_FIXTURE,
        output_dir=tmp_path / "geo_with_holdout",
    )
    assert "external_holdout_roc_auc" in result.metrics
    assert result.metrics["external_holdout_roc_auc"] >= 0.0
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["validation"]["tracks"]["external_holdout"]["status"] == "evaluated"
    assert "roc_auc" in audit["validation"]["tracks"]["external_holdout"]
    assert audit["validation"]["tracks"]["external_holdout"]["n_backbones"] > 0
    assert "prevalence" in audit["validation"]["tracks"]["external_holdout"]
    assert "bootstrap_roc_auc_ci_low" in audit["validation"]["tracks"]["external_holdout"]


def test_external_holdout_rejects_same_file_as_training_surface(tmp_path):
    with pytest.raises(ValueError, match="external holdout must be independent"):
        run_pipeline(
            run_mode="geo",
            geo_spread_features_path=GEO_SPREAD_FEATURES,
            external_holdout_path=GEO_SPREAD_FEATURES,
            output_dir=tmp_path / "same_holdout",
        )


def test_load_records_rejects_blank_backbone_ids(tmp_path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "backbone_id,year,country,host_genus,clinical_context,amr_gene_count,mobility_score\n"
        ",2020,TR,Escherichia,clinical,1,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backbone_id"):
        load_records(csv_path)


def test_small_fixture_model_surface_uses_safe_cv_without_warnings():
    records = load_records(FIXTURE)
    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    config = load_project_config()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        fit_model_surface(features, config.models)

    assert not [warning for warning in captured if "least populated class" in str(warning.message)]


def test_competition_report_contains_decision_ready_sections(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "report_quality",
    )
    report = result.report_path.read_text(encoding="utf-8").lower()

    assert "problem" in report
    assert "validation" in report
    assert "calibration" in report
    assert "limitations" in report
    assert "release gate" in report


def test_dashboard_is_clear_jury_facing_audit_page(tmp_path):
    run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "dashboard_quality",
    )
    dashboard = (tmp_path / "dashboard_quality" / "dashboard.html").read_text(encoding="utf-8").lower()

    assert "biospread competition audit" in dashboard
    assert "what this predicts" in dashboard
    assert "quality gate detail" in dashboard
    assert "calibration curve" in dashboard
    assert "permutation importance" in dashboard
    assert "risk hotspot map" not in dashboard
    assert "cdn.jsdelivr" not in dashboard
    assert "<svg" in dashboard


def test_manifest_records_code_and_runtime_reproducibility(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "repro",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["python"]
    assert manifest["environment"]["ssl_backend"]
    assert manifest["environment"]["numpy"]
    assert manifest["environment"]["scikit_learn"]
    assert manifest["git_commit"]
    assert manifest["semantic_input_hashes"]["geo_spread_features"]
    assert manifest["source_fingerprint"]
    assert manifest["config_fingerprint"]
    assert manifest["dependency_fingerprint"]
    assert manifest["random_seed_policy"]["model_random_state"] == 42


def test_trend_evaluation_returns_insufficient_data_when_history_is_short():
    report = evaluate_model_registry_trend(
        entries=[{"roc_auc": 0.85, "average_precision": 0.80, "all_quality_gates_passed": True} for _ in range(8)],
        window_size=5,
        thresholds=TrendThresholds(),
    )
    assert report["status"] == "insufficient_data"
    assert report["all_passed"] is True
    assert report["trend_evidence_sufficient"] is False


def test_trend_evaluation_flags_regression_with_low_gate_pass_rate():
    entries: list[dict[str, float | bool]] = []
    entries.extend(
        {"roc_auc": 0.87, "average_precision": 0.82, "all_quality_gates_passed": True}
        for _ in range(10)
    )
    entries.extend(
        {"roc_auc": 0.81, "average_precision": 0.74, "all_quality_gates_passed": False}
        for _ in range(10)
    )
    report = evaluate_model_registry_trend(
        entries=entries,
        window_size=10,
        thresholds=TrendThresholds(roc_auc_max_drop=0.02, average_precision_max_drop=0.03, min_gate_pass_rate=0.90),
    )

    assert report["status"] == "ok"
    assert report["all_passed"] is False
    assert report["checks"]["roc_auc"]["passed"] is False
    assert report["checks"]["average_precision"]["passed"] is False
    assert report["checks"]["gate_pass_rate"]["passed"] is False


def test_invalid_quality_thresholds_raise_validation_error(tmp_path):
    invalid = tmp_path / "invalid_quality.json"
    invalid.write_text(json.dumps({"auc_min": 1.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="auc_min"):
        load_quality_thresholds(invalid)


def test_invalid_drift_thresholds_raise_validation_error(tmp_path):
    invalid = tmp_path / "invalid_drift.json"
    invalid.write_text(json.dumps({"roc_auc_max_drop": -0.1}), encoding="utf-8")
    with pytest.raises(ValueError, match="roc_auc_max_drop"):
        load_drift_thresholds(invalid)


def test_invalid_trend_thresholds_raise_validation_error(tmp_path):
    invalid = tmp_path / "invalid_trend.json"
    invalid.write_text(json.dumps({"min_gate_pass_rate": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="min_gate_pass_rate"):
        load_trend_thresholds(invalid)
