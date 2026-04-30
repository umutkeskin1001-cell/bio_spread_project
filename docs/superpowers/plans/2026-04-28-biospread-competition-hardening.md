# BioSpread Competition Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BioSpread a competition-ready, reproducible, scientifically defensible plasmid-backbone geographic spread early-warning project.

**Architecture:** Keep the current standalone Python package shape, but make the runtime contract stricter: inputs are validated before training, validation metrics are separated from final-fit predictions, artifacts are generated from real checks, and documentation mirrors measured behavior. Avoid broad rewrites; harden the existing pipeline, model surface, audit layer, reports, and tests.

**Tech Stack:** Python 3.9+, Polars, scikit-learn, NumPy, Numba, pytest, Hypothesis, ruff, strict mypy, joblib.

---

## Current Baseline

Fresh verification after the first hardening pass:

- `python3 -m pytest tests`: 41 passed, 9 warnings.
- `python3 -m ruff check src tests`: pass.
- `python3 -m mypy src/bio_spread_project`: pass.
- `python3 run_project.py --output-dir /tmp/biospread_final_verify`: pass, ROC AUC 0.823.

Remaining work is not about making the project merely run. It is about making the result hard to attack in a competition review.

---

## File Map

- `src/bio_spread_project/pipeline.py`: orchestration, policy enforcement, artifact generation.
- `src/bio_spread_project/geo_reliability.py`: GeoSpread feature contract, model training, OOF predictions, leakage checks.
- `src/bio_spread_project/model.py`: raw-record model surface.
- `src/bio_spread_project/data.py`: raw and CSV input loading.
- `src/bio_spread_project/quality.py`: quality gate logic.
- `src/bio_spread_project/drift.py`: drift checks against baseline.
- `src/bio_spread_project/trend.py`: model registry trend checks.
- `src/bio_spread_project/audit.py`: audit JSON and model card.
- `src/bio_spread_project/reporting.py`: markdown report.
- `src/bio_spread_project/dashboard.py`: HTML dashboard.
- `src/bio_spread_project/manifest.py`: reproducibility manifest.
- `tests/`: all behavior, regression, validation, and artifact tests.
- `docs/`: architecture, validation strategy, data contracts, release policy, operations runbook.
- `README.md`: public competition-facing project description.

---

## Task 1: Lock The Competition Definition

**Files:**
- Modify: `docs/problem_statement.md`
- Modify: `README.md`
- Modify: `docs/data_contracts.md`

- [ ] **Step 1: Write the public problem contract**

Add a concise statement that BioSpread predicts whether a plasmid backbone observed up to `split_year` will appear in at least two previously unseen countries within `horizon_years`.

- [ ] **Step 2: Define non-goals**

Document that the model is not for clinical diagnosis, patient decisions, or real-time outbreak declaration.

- [ ] **Step 3: Define judging claim boundaries**

State that the current packaged run is a retrospective early-warning benchmark over packaged GeoSpread features, not proof of field deployment performance.

- [ ] **Step 4: Verify docs**

Run:

```bash
rg "clinical diagnosis|field deployment|two previously unseen countries|split_year" README.md docs
```

Expected: all four concepts appear in public docs.

---

## Task 2: Split Final Predictions From Validation Predictions

**Files:**
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/model.py`
- Modify: `src/bio_spread_project/pipeline.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add failing test**

Add a test that asserts `predictions.csv` contains final-fit predictions while `metrics.json` uses OOF validation metrics.

```python
def test_geo_pipeline_separates_final_predictions_from_validation_metrics(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_separation",
    )
    predictions = (tmp_path / "geo_separation" / "predictions.csv").read_text(encoding="utf-8")
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))

    assert "risk_probability" in predictions
    assert metrics["validation_mode"] == "spatial_group_cv_stacked"
    assert "oof_roc_auc" in metrics
    assert metrics["roc_auc"] == metrics["oof_roc_auc"]
```

- [ ] **Step 2: Run test red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_geo_pipeline_separates_final_predictions_from_validation_metrics -q
```

Expected before implementation: fail if OOF fields are absent or conflated.

- [ ] **Step 3: Implement explicit OOF metadata**

Ensure `ModelRun.validation_predictions` stores OOF predictions and `ModelRun.predictions` stores final-fit predictions. Ensure `metrics.json` includes `oof_roc_auc`, `oof_average_precision`, `validation_mode`, and final prediction cohort counts.

- [ ] **Step 4: Run test green**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_geo_pipeline_separates_final_predictions_from_validation_metrics -q
```

Expected: pass.

---

## Task 3: Make Geo Feature Contract Explicit

**Files:**
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `docs/data_contracts.md`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add failing tests for required and derived columns**

Add tests:

```python
def test_geo_surface_requires_core_model_columns(tmp_path):
    broken = tmp_path / "broken.tsv"
    broken.write_text("backbone_id\tspread_label\tn_new_countries\nbb1\t1\t2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_geo_spread_feature_rows(broken)


def test_geo_surface_derives_optional_geo_features_from_packaged_aliases():
    rows = load_geo_spread_feature_rows(GEO_SPREAD_FEATURES)
    first = rows[0]

    assert "geo_country_entropy_train" in first.features
    assert "geo_macro_region_entropy_train" in first.features
    assert "geo_dominant_region_share_train" in first.features
    assert "geo_country_record_count_train" in first.features
```

- [ ] **Step 2: Run tests red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_geo_surface_requires_core_model_columns tests/test_bio_spread.py::test_geo_surface_derives_optional_geo_features_from_packaged_aliases -q
```

- [ ] **Step 3: Implement named contract constants**

Keep:

```python
FEATURE_COLUMNS = (...)
REQUIRED_FEATURE_COLUMNS = tuple(column for column in FEATURE_COLUMNS if not column.startswith("geo_"))
DERIVED_FEATURE_COLUMNS = {
    "geo_country_entropy_train": "log1p_n_countries_train",
    "geo_macro_region_entropy_train": "n_train_macro_regions",
    "geo_country_record_count_train": "log1p_member_count_train",
}
```

Compute `geo_dominant_region_share_train` as `1 / max(n_train_macro_regions, 1)`.

- [ ] **Step 4: Document the contract**

In `docs/data_contracts.md`, list required columns, optional derived columns, label columns, and blocked leakage columns.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py -q
```

Expected: pass.

---

## Task 4: Replace Pseudo Attribution With Honest Feature Importance

**Files:**
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `src/bio_spread_project/reporting.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add failing test**

```python
def test_geo_predictions_expose_honest_attribution_metadata(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "geo_attribution",
    )
    rows = json.loads(result.audit_path.read_text(encoding="utf-8"))

    assert "top_features" in rows["validation"]
    assert isinstance(rows["validation"]["top_features"], list)
    assert rows["validation"]["top_features"][0]["feature"]
    assert rows["validation"]["top_features"][0]["score"] >= 0.0
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_geo_predictions_expose_honest_attribution_metadata -q
```

- [ ] **Step 3: Move permutation importances into audit validation**

Ensure `build_run_audit()` preserves `metrics["top_features"]` under `audit["validation"]["top_features"]`.

- [ ] **Step 4: Remove misleading local attribution claim**

Replace “SHAP-lite” wording with “global permutation importance”. Do not claim SHAP unless SHAP is actually implemented.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_geo_predictions_expose_honest_attribution_metadata -q
```

Expected: pass.

---

## Task 5: Strengthen Calibration Evidence

**Files:**
- Modify: `src/bio_spread_project/calibration.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/reporting.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add calibration-bin test**

```python
def test_calibration_summary_exports_bins_for_report():
    rows = load_geo_spread_feature_rows(GEO_SPREAD_FEATURES)
    run = fit_geo_reliability_surface(rows)

    assert "calibration_bins" in run.calibration
    assert len(run.calibration["calibration_bins"]) >= 5
    assert {"bin_start", "bin_end", "mean_prediction", "observed_rate", "count"} <= set(run.calibration["calibration_bins"][0])
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_calibration_summary_exports_bins_for_report -q
```

- [ ] **Step 3: Implement bin payload**

Extend calibration output with stable bin dictionaries. Keep existing `expected_calibration_error` and `brier_score` keys unchanged.

- [ ] **Step 4: Render bins**

Add a small markdown table in `report.md` and a chart input in `dashboard.html`.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py -q
```

Expected: pass.

---

## Task 6: Make Drift And Trend Release Logic Non-Cosmetic

**Files:**
- Modify: `src/bio_spread_project/pipeline.py`
- Modify: `src/bio_spread_project/drift.py`
- Modify: `src/bio_spread_project/trend.py`
- Modify: `docs/release_policy.md`
- Test: `tests/test_release_gate.py`

- [ ] **Step 1: Add failing release-policy tests**

```python
def test_pipeline_fails_when_drift_policy_is_enabled_and_drift_fails(tmp_path):
    bad_baseline = tmp_path / "baseline.json"
    bad_baseline.write_text(json.dumps({"validation_summary": {"roc_auc": 1.0}}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="drift_checks"):
        run_pipeline(
            run_mode="geo",
            geo_spread_features_path=GEO_SPREAD_FEATURES,
            baseline_benchmark_path=bad_baseline,
            output_dir=tmp_path / "drift_fail",
            fail_on_drift_fail=True,
        )
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_release_gate.py::test_pipeline_fails_when_drift_policy_is_enabled_and_drift_fails -q
```

- [ ] **Step 3: Ensure drift report is always real**

If baseline exists, evaluate it. If baseline is missing, report `status=not_evaluated`, `all_passed=True`, and `reason=baseline_not_provided`.

- [ ] **Step 4: Enforce policy**

`fail_on_drift_fail=True` must raise if drift report has `all_passed=False`.

- [ ] **Step 5: Document release decisions**

In `docs/release_policy.md`, define `go`, `conditional_go`, and `no_go`.

- [ ] **Step 6: Verify**

Run:

```bash
python3 -m pytest tests/test_release_gate.py tests/test_trend.py -q
```

Expected: pass.

---

## Task 7: Add External Holdout Mode That Cannot Cheat

**Files:**
- Modify: `src/bio_spread_project/pipeline.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add failing test for holdout identity**

```python
def test_external_holdout_rejects_same_file_as_training_surface(tmp_path):
    with pytest.raises(ValueError, match="external holdout must be independent"):
        run_pipeline(
            run_mode="geo",
            geo_spread_features_path=GEO_SPREAD_FEATURES,
            external_holdout_path=GEO_SPREAD_FEATURES,
            output_dir=tmp_path / "same_holdout",
        )
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_external_holdout_rejects_same_file_as_training_surface -q
```

- [ ] **Step 3: Implement path/hash rejection**

Reject external holdout if resolved path matches training path or SHA-256 hash matches training hash.

- [ ] **Step 4: Add positive holdout fixture**

Create a small independent compatible holdout fixture under `tests/fixtures/geo_holdout.tsv` with required columns and mixed labels.

- [ ] **Step 5: Verify positive and negative paths**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_external_holdout_rejects_same_file_as_training_surface tests/test_bio_spread.py::test_external_holdout_metrics_are_exported_for_geo_mode -q
```

Expected: both pass after updating the positive test to use the fixture.

---

## Task 8: Improve Raw-Record Fallback Quality

**Files:**
- Modify: `src/bio_spread_project/data.py`
- Modify: `src/bio_spread_project/features.py`
- Modify: `src/bio_spread_project/model.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add input schema edge-case tests**

```python
def test_load_records_rejects_blank_backbone_ids(tmp_path):
    csv_path = tmp_path / "records.csv"
    csv_path.write_text(
        "backbone_id,year,country,host_genus,clinical_context,amr_gene_count,mobility_score\n"
        ",2020,TR,Escherichia,clinical,1,0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backbone_id"):
        load_records(csv_path)
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_load_records_rejects_blank_backbone_ids -q
```

- [ ] **Step 3: Implement strict blank-ID validation**

Filter truly empty trailing rows, but raise for rows where other fields exist and `backbone_id` is blank.

- [ ] **Step 4: Add temporal split tests**

Assert future records never affect pre-split feature values.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_build_backbone_features_uses_only_pre_split_history tests/test_bio_spread.py::test_load_records_rejects_blank_backbone_ids -q
```

Expected: pass.

---

## Task 9: Harden Small-Sample Cross Validation

**Files:**
- Modify: `src/bio_spread_project/model.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add no-warning test for small fixture**

```python
def test_small_fixture_model_surface_uses_safe_cv_without_warnings():
    records = load_records(FIXTURE)
    features = build_backbone_features(records, split_year=2020, horizon_years=3)
    config = load_project_config()

    with pytest.warns(None) as warnings:
        fit_model_surface(features, config.models)

    assert len(warnings) == 0
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_small_fixture_model_surface_uses_safe_cv_without_warnings -q
```

- [ ] **Step 3: Implement adaptive CV splits**

Use `min_class_count = min(np.bincount(y))`; set `n_splits = min(3, min_class_count)`. If `n_splits < 2`, use direct fit metrics and mark validation mode `direct_small_sample`.

- [ ] **Step 4: Verify**

Run:

```bash
python3 -m pytest tests -q
```

Expected: no sklearn “least populated class” warnings.

---

## Task 10: Make Reports Jury-Grade

**Files:**
- Modify: `src/bio_spread_project/reporting.py`
- Modify: `src/bio_spread_project/dashboard.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `README.md`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add report content test**

```python
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
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_competition_report_contains_decision_ready_sections -q
```

- [ ] **Step 3: Add report sections**

Add these sections to `report.md`:

- Problem.
- Input selected.
- Model and validation.
- Calibration.
- Leakage and audit.
- Release gate.
- Top candidates.
- Limitations.
- Reproducibility.

- [ ] **Step 4: Make README less inflated**

Replace unsupported phrases like “premium” with measured claims. Keep actual metrics.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_competition_report_contains_decision_ready_sections -q
```

Expected: pass.

---

## Task 11: Add Reproducibility Snapshot

**Files:**
- Modify: `src/bio_spread_project/manifest.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `src/bio_spread_project/pipeline.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add failing test**

```python
def test_manifest_records_code_and_runtime_reproducibility(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "repro",
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["environment"]["python"]
    assert manifest["environment"]["numpy"]
    assert manifest["environment"]["scikit_learn"]
    assert manifest["git_commit"]
```

- [ ] **Step 2: Run red**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_manifest_records_code_and_runtime_reproducibility -q
```

- [ ] **Step 3: Add environment payload to manifest**

Reuse audit environment fields and include them in manifest.

- [ ] **Step 4: Verify**

Run:

```bash
python3 -m pytest tests/test_bio_spread.py::test_manifest_records_code_and_runtime_reproducibility -q
```

Expected: pass.

---

## Task 12: Add Benchmark Regression Test

**Files:**
- Create: `tests/test_competition_regression.py`
- Modify: `project_config/quality_thresholds.json` only if justified by measured evidence.

- [ ] **Step 1: Add metric floor test**

```python
from pathlib import Path

from bio_spread_project.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_SPREAD_FEATURES = PROJECT_ROOT / "data" / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"


def test_packaged_competition_run_stays_above_metric_floor(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "competition_regression",
    )

    assert result.metrics["roc_auc"] >= 0.82
    assert result.metrics["average_precision"] >= 0.74
    assert result.metrics["max_single_feature_auc"] < 0.95
    assert result.metrics["suspicious_feature_count"] == 0.0
```

- [ ] **Step 2: Run test**

Run:

```bash
python3 -m pytest tests/test_competition_regression.py -q
```

Expected: pass with current packaged data.

- [ ] **Step 3: Add this to CI quality command**

Ensure `make quality` runs the new test.

---

## Task 13: Add CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`

- [ ] **Step 1: Add CI workflow**

Create:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install
        run: python -m pip install -r requirements-dev.txt
      - name: Ruff
        run: python -m ruff check src tests
      - name: Mypy
        run: python -m mypy src/bio_spread_project
      - name: Tests
        run: python -m pytest tests
```

- [ ] **Step 2: Verify locally**

Run:

```bash
python3 -m ruff check src tests
python3 -m mypy src/bio_spread_project
python3 -m pytest tests
```

Expected: all pass.

---

## Task 14: Add Final Release Checklist Command

**Files:**
- Modify: `verify_project.py`
- Modify: `Makefile`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Define release verification**

`verify_project.py` should run:

- ruff.
- mypy.
- pytest.
- production `run_project.py`.
- artifact existence checks.
- release gate readiness check.

- [ ] **Step 2: Add test for verify command**

```python
def test_verify_project_has_release_mode():
    completed = subprocess.run(
        [sys.executable, "verify_project.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--release" in completed.stdout
```

- [ ] **Step 3: Implement `--release`**

Make `python3 verify_project.py --release` run full quality checks and exit non-zero on failure.

- [ ] **Step 4: Wire Makefile**

Add:

```make
release-verify:
	python3 verify_project.py --release
```

- [ ] **Step 5: Verify**

Run:

```bash
make release-verify
```

Expected: exit 0.

---

## Task 15: Package A Clean Competition Artifact

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Create: `docs/competition_submission.md`

- [ ] **Step 1: Decide what ships**

Ship source, tests, docs, minimal packaged data needed for reproducibility, and final generated report. Do not ship temporary caches or duplicate old demo folders unless required.

- [ ] **Step 2: Document the exact reproduction command**

Add to `docs/competition_submission.md`:

```bash
python3 -m pip install -r requirements-dev.txt
python3 run_project.py --mode geo --output-dir reports/competition_final
python3 verify_project.py --release
```

- [ ] **Step 3: Add artifact manifest**

List expected final files:

- `reports/competition_final/report.md`
- `reports/competition_final/dashboard.html`
- `reports/competition_final/audit.json`
- `reports/competition_final/model_card.md`
- `reports/competition_final/manifest.json`
- `reports/competition_final/release_gate.json`
- `reports/competition_final/predictions.csv`

- [ ] **Step 4: Verify clean checkout behavior**

Run from a fresh clone or clean worktree:

```bash
python3 -m pip install -r requirements-dev.txt
python3 run_project.py --mode geo --output-dir reports/competition_final
python3 -m pytest tests
```

Expected: pass.

---

## Final Acceptance Criteria

The project is competition-ready only when all of these are true:

- `python3 -m pytest tests` passes with no failures.
- `python3 -m ruff check src tests` passes.
- `python3 -m mypy src/bio_spread_project` passes.
- `python3 run_project.py --mode geo --output-dir reports/competition_final` succeeds.
- `reports/competition_final/release_gate.json` is `go` or explicitly explained `conditional_go`.
- `README.md` claims exactly what the code verifies.
- `docs/data_contracts.md` fully describes required input columns and leakage-blocked columns.
- `docs/validation_strategy.md` explains OOF, group CV, calibration, bootstrap, leakage scan, and limitations.
- `docs/competition_submission.md` lets a judge reproduce the result without guessing.
- No report or model card uses inflated claims such as “clinical-ready”, “field-proven”, or “SHAP” unless implemented and verified.

---

## Execution Order

Recommended order:

1. Tasks 1-3: truthfulness and data contracts.
2. Tasks 4-7: scientific defensibility.
3. Tasks 8-9: fallback and small-sample robustness.
4. Tasks 10-11: jury-facing report and reproducibility.
5. Tasks 12-14: regression protection and release verification.
6. Task 15: final submission packaging.

Commit after each task group with focused messages:

```bash
git add <changed files>
git commit -m "test: lock biospread competition contract"
git commit -m "feat: harden geospread validation artifacts"
git commit -m "docs: align competition report with verified metrics"
git commit -m "ci: add release verification"
```

