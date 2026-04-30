# BioSpread Final Excellence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn BioSpread from a passing competition prototype into a rigorously defensible, reproducible, performant, and judge-ready scientific software artifact.

**Architecture:** Treat scientific validity as the core product. The implementation should first eliminate misleading validation and leakage risk, then make runtime fast, then split orchestration behind typed contracts, and finally polish release evidence.

**Tech Stack:** Python 3.9+, Polars, NumPy, scikit-learn, pytest, pytest-cov, mypy strict, ruff, pip-audit, GitHub Actions. Use Numba only after benchmark evidence, not as default complexity camouflage.

---

## Non-Negotiable Definition of Done

BioSpread is not “final” until all of these are true:

- `python3 -m ruff check src tests run_project.py verify_project.py` passes.
- `python3 -m mypy src` passes under strict mode.
- `python3 -m pytest --cov=src/bio_spread_project --cov-report=term-missing --cov-fail-under=92 -q tests` passes.
- `python3 -m pip_audit -r requirements.txt` passes with no ignored vulnerabilities.
- `python3 verify_project.py --release` passes from a clean checkout.
- `audit.json`, `manifest.json`, `model_card.md`, `dashboard.html`, `release_gate.json`, and `artifact_index.json` explain the run without needing source-code context.
- No production/release gate passes by substituting OOF metrics for missing temporal, external, drift, or trend evidence.
- Every stochastic training, split, bootstrap, and permutation path has an explicit seed or documented nondeterminism field in the manifest.

## Current Evidence From Final Review

- Ruff passes.
- Mypy strict passes.
- Full test suite passes: `55 passed`.
- Coverage passes but is thin: total `90.61%`, with `geo_reliability.py` at `77%`, `cli.py` at `75%`, `validation.py` at `76%`, `data.py` at `88%`, `drift.py` at `86%`.
- Full coverage run takes about `7:10`, so CI needs fast/full/release tiers.
- `pip-audit -r requirements.txt` reports no known vulnerabilities, so `Makefile` ignore flags should be removed.
- `geo_reliability.py` currently exports `temporal_holdout_roc_auc` as OOF AUC, not a real temporal holdout.
- `quality.py` and `reporting.py` contain fallback patterns that can make missing evidence look evaluated.
- `evaluation.py` still routes primary metrics through pure-Python `_auc()` and `_average_precision()`.
- `geo_reliability.py` has a Numba bootstrap AUC that remains pairwise O(N^2).
- `pipeline.py` remains a high-coupling orchestration file with `Any` at important domain boundaries.

## Phase 0: Freeze the Baseline Before Changing Behavior

**Files:**
- Create: `reports/baseline/final_review_metrics.json`
- Create: `reports/baseline/final_review_coverage.txt`
- Modify: `docs/competition_submission.md`

- [ ] **Step 1: Capture the current quality baseline**

Run:

```bash
python3 -m ruff check src tests run_project.py verify_project.py
python3 -m mypy src
python3 -m pytest --cov=src/bio_spread_project --cov-report=term-missing --cov-fail-under=90 -q tests
python3 -m pip_audit -r requirements.txt
```

Expected:

```text
ruff: pass
mypy: pass
pytest: 55 passed
coverage: >=90%
pip-audit: No known vulnerabilities found
```

- [ ] **Step 2: Generate one current production artifact set**

Run:

```bash
python3 verify_project.py --release
```

Expected: release run exits 0. Copy only summary metrics into `docs/competition_submission.md`; do not commit large generated reports unless the project already tracks them intentionally.

## Phase 1: Make Validation Scientifically Honest

**Files:**
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/quality.py`
- Modify: `src/bio_spread_project/model_metrics.py`
- Modify: `src/bio_spread_project/reporting.py`
- Modify: `src/bio_spread_project/audit.py`
- Test: `tests/test_quality.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add tests that forbid metric substitution**

Add or strengthen tests so these fail unless fixed:

```python
def test_geo_quality_does_not_backfill_temporal_from_oof():
    gates = evaluate_quality_gates(
        metrics={
            "roc_auc": 0.99,
            "average_precision": 0.90,
            "prevalence": 0.10,
            "expected_calibration_error": 0.01,
            "bootstrap_roc_auc_ci_low": 0.95,
            "bootstrap_average_precision_ci_low": 0.80,
            "group_oof_roc_auc": 0.95,
            "max_single_feature_auc": 0.50,
            "suspicious_feature_count": 0,
            "validation_mode": "spatial_group_cv_stacked",
        },
        input_mode="geo_reliability_feature_surface",
        leakage_audit_passed=True,
        thresholds=QualityThresholds(),
    )
    assert gates["temporal_holdout_auc_at_least_target"] is False
```

```python
def test_reporting_marks_missing_temporal_holdout_not_evaluated():
    markdown = render_markdown_report(
        predictions=[],
        metrics={
            "roc_auc": 0.90,
            "average_precision": 0.80,
            "prevalence": 0.20,
            "all_quality_gates_passed": False,
        },
        calibration={},
        split_year=2020,
        horizon_years=3,
        coefficient_summary="",
    )
    assert "Temporal holdout ROC AUC: `not_evaluated`" in markdown
```

- [ ] **Step 2: Remove fallback pass logic**

In `evaluate_quality_gates()`, require explicit keys:

```python
has_group = "group_oof_roc_auc" in metrics
has_temporal = "temporal_holdout_roc_auc" in metrics
has_external = "external_holdout_roc_auc" in metrics
```

Geo release gates should pass only when the metric exists and meets threshold, except when a separate explicit policy says the evidence is optional.

- [ ] **Step 3: Implement real temporal holdout**

In `fit_geo_reliability_surface()`:

1. Sort candidate rows by `max_resolved_year_train`.
2. Use the latest 20-30% as temporal validation.
3. Require both classes in train and validation.
4. Train a fresh estimator on temporal train only.
5. Export temporal ROC AUC, AP, ECE, Brier, n, positives, and prevalence.

If insufficient, export no temporal scalar metric and add:

```python
"temporal_holdout_status": "not_evaluated"
"temporal_holdout_reason": "insufficient_class_diversity"
```

- [ ] **Step 4: Make model card and dashboard evidence-aware**

Display `not_evaluated` as `not_evaluated`, never as copied OOF values. The reviewer should immediately see whether group, temporal, external, drift, and trend evidence is real.

- [ ] **Step 5: Verify**

Run:

```bash
python3 -m pytest -q tests/test_quality.py tests/test_bio_spread.py -k "quality or temporal or external or reporting"
python3 -m mypy src
```

## Phase 2: Eliminate Metric Performance Debt

**Files:**
- Modify: `src/bio_spread_project/evaluation.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Create: `tests/test_evaluation_metrics.py`
- Create: `scripts/benchmark_metrics.py`
- Modify: `project_config/config/performance_budgets.yaml`

- [ ] **Step 1: Test metric parity**

Add tests for:

- ties
- all-positive labels
- all-negative labels
- empty prediction input
- sklearn parity for mixed labels
- top-k precision unchanged

- [ ] **Step 2: Replace primary `_auc()` and `_average_precision()` use**

In `evaluate_predictions()`, convert labels/scores once:

```python
labels = np.asarray([row.label_geo_spread for row in predictions], dtype=int)
scores = np.asarray([row.risk_probability for row in predictions], dtype=float)
```

Then use `_fast_auc(labels, scores)` and `_fast_average_precision(labels, scores)`.

- [ ] **Step 3: Remove Numba pairwise bootstrap**

Delete `_numba_bootstrap_auc()` unless a benchmark proves it faster and scalable. Use one bootstrap implementation with deterministic `np.random.default_rng(seed)`.

- [ ] **Step 4: Add performance budget**

`benchmark_metrics.py` should generate 50k and 250k synthetic predictions and report:

```json
{
  "n_predictions": 50000,
  "evaluate_predictions_seconds": 0.0,
  "bootstrap_metric_intervals_seconds": 0.0
}
```

Set budgets in `performance_budgets.yaml` and check them in release verification.

## Phase 3: Rebuild Type Boundaries, Not Just Type Hints

**Files:**
- Create: `src/bio_spread_project/contracts.py`
- Modify: `src/bio_spread_project/pipeline.py`
- Modify: `src/bio_spread_project/model.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `src/bio_spread_project/manifest.py`
- Test: mypy and full tests

- [ ] **Step 1: Add domain payload types**

Create typed objects:

```python
@dataclass(frozen=True)
class PipelineOptions:
    output_dir: Path
    run_mode: str
    input_path: Path | None
    backbone_records_path: Path | None
    amr_path: Path | None
    geo_spread_features_path: Path | None
    external_holdout_path: Path | None
    split_year: int
    horizon_years: int
    policy: EnforcementPolicy
```

```python
@dataclass(frozen=True)
class LoadedDataset:
    selection: InputSelection
    rows: list[BackboneFeatures] | list[GeoSpreadFeatureRow]
    input_paths: dict[str, Path]
```

- [ ] **Step 2: Remove `Any` from pipeline method signatures**

`BioSpreadPipeline.run()` may keep a compatibility wrapper, but the internal path must use `PipelineOptions`, `LoadedDataset`, `ModelRun`, and typed artifact objects.

- [ ] **Step 3: Use `object` or specific unions for JSON payloads**

Replace broad `dict[str, Any]` in new code with:

```python
JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
```

Do not force this across the whole project in one patch; apply it to newly extracted modules first.

- [ ] **Step 4: Verify**

Run:

```bash
python3 -m mypy src
python3 -m pytest -q tests
```

## Phase 4: Split the Monolith Without Changing Behavior

**Files:**
- Modify: `src/bio_spread_project/pipeline.py`
- Create: `src/bio_spread_project/data_orchestrator.py`
- Create: `src/bio_spread_project/model_orchestrator.py`
- Create: `src/bio_spread_project/audit_orchestrator.py`
- Create: `src/bio_spread_project/artifact_writer.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Extract data loading**

`DataOrchestrator.load(options) -> LoadedDataset` owns:

- input selection
- raw/geo loading
- feature construction
- input path collection
- empty-row failure

- [ ] **Step 2: Extract model fitting**

`ModelOrchestrator.fit(dataset, options) -> ModelRun` owns:

- geo vs raw model choice
- final model persistence
- external holdout scoring

- [ ] **Step 3: Extract audit and gates**

`AuditOrchestrator.evaluate(dataset, model_run, options) -> AuditBundle` owns:

- audit building
- quality threshold loading
- policy enforcement
- drift/trend/release gate evaluation

- [ ] **Step 4: Extract artifact writing**

`ArtifactWriter.write(...) -> PipelineArtifacts` owns:

- features/predictions
- metrics/audit/model card/dashboard
- benchmark/drift/trend/release gate
- manifest/artifact index

- [ ] **Step 5: Keep old public API stable**

`run_pipeline(**kwargs)` and CLI behavior must remain backward-compatible.

## Phase 5: Parquet-First I/O With Schema Contracts

**Files:**
- Modify: `src/bio_spread_project/data.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/io_utils.py`
- Modify: `src/bio_spread_project/cli.py`
- Create: `tests/test_data_io.py`
- Modify: `docs/data_contracts.md`

- [ ] **Step 1: Centralize table loading**

Add:

```python
def read_table(path: str | Path, *, schema_overrides: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(source)
    if suffix in {".tsv", ".tab"}:
        return pl.read_csv(source, separator="\t", schema_overrides=schema_overrides)
    if suffix == ".csv":
        return pl.read_csv(source, separator=",", schema_overrides=schema_overrides)
    raise ValueError(f"Unsupported table format: {source}")
```

- [ ] **Step 2: Add explicit schema validation**

Each loader should validate required columns, cast types, and fail with column-specific errors.

- [ ] **Step 3: Dual-write large artifacts**

Keep human-readable CSV output, but add Parquet:

```text
features.csv
features.parquet
predictions.csv
predictions.parquet
```

Manifest must include both with byte hash and semantic hash.

- [ ] **Step 4: Add CLI conversion**

Add:

```bash
python3 -m bio_spread_project.cli convert-table --input data/raw/plasmid_backbones.tsv --output data/raw/plasmid_backbones.parquet
```

## Phase 6: Reproducibility, Provenance, and Cache Keys

**Files:**
- Create: `src/bio_spread_project/cache_keys.py`
- Modify: `src/bio_spread_project/io_utils.py`
- Modify: `src/bio_spread_project/manifest.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `verify_project.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Keep byte hashes**

Byte SHA-256 remains the source of exact provenance. Do not replace it with semantic hashes.

- [ ] **Step 2: Add semantic input hashes**

For tabular files, hash:

- column names and order
- dtypes
- row count
- null counts
- stable numeric summaries
- sorted categorical cardinalities

Store this under `semantic_input_hashes`.

- [ ] **Step 3: Add source/config fingerprints**

Manifest should include:

```text
source_fingerprint
config_fingerprint
dependency_fingerprint
random_seed_policy
```

- [ ] **Step 4: Add artifact index**

Create `artifact_index.json`:

```json
{
  "audit.json": {"purpose": "quality and leakage gates", "required_for_submission": true},
  "manifest.json": {"purpose": "reproducibility and provenance", "required_for_submission": true}
}
```

## Phase 7: Robustness and Edge-Case Testing

**Files:**
- Modify: `tests/test_fuzzing.py`
- Create: `tests/test_geo_reliability_edges.py`
- Create: `tests/test_cli_errors.py`
- Modify: `tests/test_release_gate.py`

- [ ] **Step 1: Add small-sample geo tests**

Cover:

- one class only
- fewer rows than CV splits
- missing required columns
- invalid numeric values
- duplicate backbone IDs
- external holdout with identical content under different path

- [ ] **Step 2: Add deterministic repeatability test**

Run the same fixture twice and assert core validation metrics and top predictions are identical within a strict tolerance.

- [ ] **Step 3: Raise coverage floor to 92%**

Focus on risk-heavy uncovered code first:

- `geo_reliability.py`
- `cli.py`
- `data.py`
- `drift.py`
- `validation.py`

Do not chase meaningless lines; test behavior and failure modes.

## Phase 8: Security and CI Release Discipline

**Files:**
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Modify: `constraints.txt`
- Create: `constraints-release.txt`
- Modify: `verify_project.py`

- [ ] **Step 1: Remove ignored vulnerabilities**

Makefile:

```make
security:
	$(PYTHON) -m pip_audit -r requirements.txt
```

- [ ] **Step 2: Split CI**

GitHub Actions should have:

- `fast`: ruff, mypy, focused tests
- `full`: all tests with coverage
- `release`: pip-audit, `verify_project.py --release`, artifact checks

- [ ] **Step 3: Add dependency lock discipline**

Keep `pyproject.toml` flexible. Use `constraints-release.txt` for known-good competition builds.

- [ ] **Step 4: Add Python version matrix**

Test at least Python 3.9 and 3.11 because project declares `>=3.9`.

## Phase 9: Judge-Facing Polish Without Marketing Noise

**Files:**
- Modify: `src/bio_spread_project/dashboard.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `docs/competition_submission.md`
- Modify: `docs/validation_strategy.md`
- Modify: `README.md`

- [ ] **Step 1: Make dashboard evidence-first**

First viewport should show:

- release readiness
- OOF AUC/AP
- temporal holdout status
- external holdout status
- leakage scan max single-feature AUC
- suspicious feature count

- [ ] **Step 2: Add model-card challenge section**

Include:

- what can invalidate the model
- known limitations
- data leakage controls
- calibration caveats
- minimum acceptable evidence before use

- [ ] **Step 3: Add competition submission map**

Map jury questions to artifacts:

```text
How do you prevent leakage? -> model_card.md, audit.json
How do you reproduce the run? -> manifest.json
How do you know it is stable? -> drift_report.json, trend_report.json
How do you inspect predictions? -> predictions.csv/parquet, dashboard.html
```

## Phase 10: Final Release Gate

**Files:**
- Modify: `verify_project.py`
- Modify: `docs/operations_runbook.md`

- [ ] **Step 1: Make release verification strict**

`verify_project.py --release` must fail when:

- quality gates fail
- temporal evidence is missing without explicit waiver
- external holdout is required but missing
- drift baseline is missing in release mode
- trend evidence is missing and conditional release is disabled
- security audit has any vulnerability
- coverage is below 92%
- artifact index is missing
- manifest lacks source/config/dependency fingerprints

- [ ] **Step 2: Add waiver mechanism**

If a competition dry-run needs to proceed with missing external evidence, require an explicit waiver file:

```json
{
  "waiver": "external_holdout_missing",
  "reason": "independent external dataset not released yet",
  "expires": "2026-05-15"
}
```

The dashboard and release gate must show waiver status as conditional, not pass.

- [ ] **Step 3: Final command**

Run:

```bash
python3 verify_project.py --release
```

Expected:

```text
BioSpread verification completed
```

## Execution Order

1. Phase 0: freeze evidence.
2. Phase 1: scientific validation honesty.
3. Phase 2: metric speed.
4. Phase 3: typed contracts.
5. Phase 4: orchestration split.
6. Phase 5: Parquet and schema contracts.
7. Phase 6: provenance and cache keys.
8. Phase 7: edge-case tests and coverage floor.
9. Phase 8: security and CI.
10. Phase 9: judge-facing polish.
11. Phase 10: strict release gate.

The plan is intentionally conservative: every phase leaves the project runnable, testable, and easier to defend than before.
