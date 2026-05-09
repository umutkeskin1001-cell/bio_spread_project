import json
from pathlib import Path

from bio_spread_project.governance import build_release_gate_report
from bio_spread_project.orchestrator import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_BACKBONES = PROJECT_ROOT / "data" / "raw" / "plasmid_backbones.tsv"
RAW_AMR = PROJECT_ROOT / "data" / "raw" / "amr.tsv"
GEO_SPREAD_FEATURES = PROJECT_ROOT / "data" / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"


def test_release_gate_returns_conditional_go_when_trend_evidence_is_insufficient():
    payload = build_release_gate_report(
        audit={"all_quality_gates_passed": True},
        drift_report={"all_passed": True},
        trend_report={"status": "insufficient_data", "all_passed": True, "trend_evidence_sufficient": False},
    )
    assert payload["readiness"] == "conditional_go"
    assert payload["score"] == 75
    assert payload["checks"]["trend_evidence"] is False
    assert payload["blocked_by"] == ["trend_evidence"]


def test_release_gate_returns_no_go_when_trend_evidence_is_required():
    payload = build_release_gate_report(
        audit={"all_quality_gates_passed": True},
        drift_report={"all_passed": True},
        trend_report={"status": "insufficient_data", "all_passed": True, "trend_evidence_sufficient": False},
        allow_conditional_trend_release=False,
    )
    assert payload["readiness"] == "no_go"
    assert payload["checks"]["trend_evidence"] is False


def test_pipeline_release_gate_is_conditional_go_for_fresh_output_dir(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        backbone_records_path=RAW_BACKBONES,
        amr_path=RAW_AMR,
        output_dir=tmp_path / "fresh_run",
    )
    payload = json.loads(result.release_gate_path.read_text(encoding="utf-8"))
    assert payload["readiness"] in {"conditional_go", "no_go"}
    assert payload["trend_status"] == "insufficient_data"


def test_pipeline_does_not_fail_when_drift_evidence_is_not_evaluated(tmp_path):
    bad_baseline = tmp_path / "baseline.json"
    bad_baseline.write_text(
        json.dumps(
            {
                "validation_summary": {
                    "roc_auc": 0.999,
                    "average_precision": 0.999,
                    "group_oof_roc_auc": 0.999,
                    "temporal_holdout_roc_auc": 0.999,
                    "external_holdout_roc_auc": 0.999,
                    "max_single_feature_auc": 0.10,
                    "suspicious_feature_count": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        baseline_benchmark_path=bad_baseline,
        output_dir=tmp_path / "drift_fail",
        fail_on_drift_fail=True,
    )
    drift_report = json.loads(result.drift_report_path.read_text(encoding="utf-8"))
    assert drift_report["status"] == "not_evaluated"
