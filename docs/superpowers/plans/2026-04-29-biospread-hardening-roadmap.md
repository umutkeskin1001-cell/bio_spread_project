# BioSpread Competition Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BioSpread defensible for high-level competition review by tightening leakage guarantees, performance, type contracts, reproducibility, security, and jury-facing evidence.

**Architecture:** Keep the current standalone package, but split orchestration responsibilities behind typed boundaries before large behavioral changes. Prioritize scientific validity gates first, because fast or elegant code is not useful if validation evidence is inflated.

**Tech Stack:** Python 3.9+, Polars, NumPy, scikit-learn, Numba only where measured useful, pytest, mypy strict, ruff, pip-audit.

---

## Current Assessment

The submitted optimization plan is directionally useful, but it overstates or misplaces a few issues:

- `src/bio_spread_project/evaluation.py` has slow pure-Python `_auc` and `_average_precision`, and `evaluate_predictions()` still uses them. Bootstrap paths already use sklearn-based helpers for normal-sized datasets.
- A second O(N^2) hotspot exists in `src/bio_spread_project/geo_reliability.py` via `_numba_bootstrap_auc()`. Numba reduces overhead but does not change the pairwise positive-negative complexity.
- `data.py` uses Polars but lacks one loader abstraction that chooses Parquet/CSV/TSV from extension and enforces schemas consistently.
- `pipeline.py` is an orchestration hotspot with `Any` boundaries; mypy passes because the broad types hide domain contracts.
- Security audit currently reports no known vulnerabilities with `python3 -m pip_audit -r requirements.txt`, so the immediate issue is removing stale ignore flags from `Makefile`, not necessarily upgrading runtime dependencies.
- Tests, ruff, and mypy pass today, but full tests take about 5:41, so fast/release test layering should be added.

## Task 1: Lock Scientific Validation Semantics

**Files:**
- Modify: `src/bio_spread_project/quality.py`
- Modify: `src/bio_spread_project/model_metrics.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Test: `tests/test_bio_spread.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Add failing tests for missing temporal/external evidence**

Add tests proving geo-mode temporal and external holdout gates cannot silently pass by falling back to OOF AUC. Expected behavior:

```python
def test_geo_quality_requires_explicit_temporal_metric():
    gates = evaluate_quality_gates(
        metrics={
            "roc_auc": 0.99,
            "average_precision": 0.90,
            "prevalence": 0.20,
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

- [ ] **Step 2: Make validation tracks honest**

Update `evaluate_quality_gates()` so group, temporal, and external gates pass only when the corresponding metric key exists. If a competition run does not have external holdout evidence, leave that gate disabled only under an explicit policy, not by default fallback.

- [ ] **Step 3: Add true temporal holdout metrics**

In `fit_geo_reliability_surface()`, compute a temporal split from `max_resolved_year_train`: train on earlier backbones, validate on later backbones. Export:

```python
temporal_holdout_roc_auc
temporal_holdout_average_precision
temporal_holdout_n_backbones
temporal_holdout_prevalence
```

If insufficient classes exist, export status through `validation_tracks()` as `not_evaluated` and make release policy decide whether that is acceptable.

- [ ] **Step 4: Re-run validation**

Run:

```bash
python3 -m pytest -q tests/test_quality.py tests/test_bio_spread.py
python3 -m mypy src
```

Expected: tests pass, and audit JSON distinguishes `evaluated` from `not_evaluated`.

## Task 2: Replace O(N^2) Metric Paths With Vectorized Metrics

**Files:**
- Modify: `src/bio_spread_project/evaluation.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Test: `tests/test_bio_spread.py`
- Create: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: Add metric equivalence tests**

Create tests covering ties, all-positive, all-negative, empty predictions, and sklearn parity.

```python
def test_auc_ties_match_sklearn_behavior():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.8, 0.5, 0.8, 0.1])
    assert _fast_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
```

- [ ] **Step 2: Route `evaluate_predictions()` through `_fast_auc()` and `_fast_average_precision()`**

Keep the degenerate-class behavior currently used by BioSpread: AUC `0.5`, AP equal to prevalence for one-class non-empty data.

- [ ] **Step 3: Remove or quarantine pairwise AUC**

Delete `_auc()` if tests no longer require it, or rename it to `_slow_auc_reference()` and use it only in tests for tiny arrays.

- [ ] **Step 4: Replace `_numba_bootstrap_auc()`**

Use sklearn/vectorized rank-based AUC inside bootstrap or compute CIs through `bootstrap_metric_intervals()` consistently. Keep Numba only if a benchmark proves it faster on realistic data.

- [ ] **Step 5: Add a performance budget**

Add a test or script-level benchmark that evaluates 50k synthetic predictions under a fixed wall-time budget and document it in `project_config/config/performance_budgets.yaml`.

## Task 3: Add Typed Domain Contracts and Remove Pipeline `Any`

**Files:**
- Modify: `src/bio_spread_project/pipeline.py`
- Modify: `src/bio_spread_project/model.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Create: `src/bio_spread_project/types.py` or `src/bio_spread_project/contracts.py`
- Test: existing mypy and pytest suite

- [ ] **Step 1: Introduce aliases and protocols**

Define:

```python
FeatureRows = list[BackboneFeatures] | list[GeoSpreadFeatureRow]
MetricsPayload = dict[str, float | int | str | bool | list[dict[str, object]]]
AuditPayload = dict[str, object]

class PredictiveModel(Protocol):
    def predict(self, rows: FeatureRows) -> list[Prediction]: ...
```

- [ ] **Step 2: Type `BioSpreadPipeline` methods**

Replace:

```python
def load_data(self, selection: Any, split_year: int, horizon_years: int) -> Any
def process_model(self, features: Any, selection: Any) -> Any
```

with `InputSelection`, `FeatureRows`, and `ModelRun`.

- [ ] **Step 3: Type artifact writing**

Give `_write_artifacts()` explicit parameter and return types. Avoid `kwargs: Any` by creating a `PipelineOptions` dataclass parsed once in `run_pipeline()`.

- [ ] **Step 4: Re-run strict typing**

Run:

```bash
python3 -m mypy src
python3 -m pytest -q tests/test_cli.py tests/test_bio_spread.py
```

## Task 4: Build Parquet-First I/O Without Breaking CSV/TSV Compatibility

**Files:**
- Modify: `src/bio_spread_project/data.py`
- Modify: `src/bio_spread_project/geo_reliability.py`
- Modify: `src/bio_spread_project/io_utils.py`
- Test: `tests/test_bio_spread.py`
- Create: `tests/test_data_io.py`

- [ ] **Step 1: Add a central table reader**

Implement:

```python
def read_table(path: str | Path, *, schema: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pl.read_parquet(source)
    separator = "\t" if source.suffix.lower() in {".tsv", ".tab"} else ","
    return pl.read_csv(source, separator=separator, schema_overrides=schema)
```

- [ ] **Step 2: Replace direct `pl.read_csv()` calls**

Use `read_table()` in `load_records()`, `load_table()`, `load_backbone_records()`, and `load_geo_spread_feature_rows()`.

- [ ] **Step 3: Emit Parquet artifacts for large tables**

Keep existing CSV artifacts for human review, but also write:

```text
features.parquet
predictions.parquet
```

Record both in `manifest.json`.

- [ ] **Step 4: Add migration command**

Add CLI support for converting raw TSV/CSV inputs to Parquet with schema validation, without deleting source files.

## Task 5: Split Pipeline Orchestration by Responsibility

**Files:**
- Modify: `src/bio_spread_project/pipeline.py`
- Create: `src/bio_spread_project/orchestration.py`
- Create: `src/bio_spread_project/artifacts.py`
- Create: `src/bio_spread_project/validation_orchestrator.py`
- Test: `tests/test_bio_spread.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Extract data orchestration**

Move input selection plus feature loading to a `DataOrchestrator` that returns a typed `LoadedDataset`.

- [ ] **Step 2: Extract model orchestration**

Move model fitting, primary selection, model persistence, and external holdout scoring to `ModelOrchestrator`.

- [ ] **Step 3: Extract audit/artifact orchestration**

Move `audit.json`, model card, dashboard, registry, drift, trend, release gate, and manifest writing into `ArtifactOrchestrator`.

- [ ] **Step 4: Keep `BioSpreadPipeline.run()` small**

Target shape:

```python
dataset = self.data.load(options)
model_run = self.models.fit(dataset, options)
audit_bundle = self.audit.evaluate(dataset, model_run, options)
artifacts = self.artifacts.write(dataset, model_run, audit_bundle, options)
return PipelineResult.from_artifacts(...)
```

## Task 6: Strengthen Reproducibility and Semantic Caching

**Files:**
- Modify: `src/bio_spread_project/manifest.py`
- Modify: `src/bio_spread_project/audit.py`
- Modify: `src/bio_spread_project/io_utils.py`
- Create: `src/bio_spread_project/cache_keys.py`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Keep byte hashes for provenance**

Do not replace SHA-256 in audit; byte hashes are correct for reproducibility and legal provenance.

- [ ] **Step 2: Add semantic hashes separately**

For CSV/TSV/Parquet, hash schema, sorted column names, row count, null counts, and stable typed summaries. Store under `semantic_input_hashes`.

- [ ] **Step 3: Add code/config fingerprints**

Hash relevant Python source files, `pyproject.toml`, `requirements.txt`, and threshold JSON files. Store in manifest.

- [ ] **Step 4: Use semantic hashes for cache invalidation only**

Document that byte hash means exact artifact identity; semantic hash means “safe to reuse derived computation if schema/content semantics unchanged.”

## Task 7: Improve Jury-Facing Evidence

**Files:**
- Modify: `src/bio_spread_project/audit.py`
- Modify: `src/bio_spread_project/dashboard.py`
- Modify: `docs/validation_strategy.md`
- Test: `tests/test_bio_spread.py`

- [ ] **Step 1: Add a leakage evidence section**

Model card and dashboard should show:

```text
blocked feature tokens
max single-feature AUC
suspicious feature count
group split strategy
temporal holdout status
external holdout status
```

- [ ] **Step 2: Add limitations in plain language**

Keep it honest: retrospective benchmark, not clinical deployment, not patient-level decision support.

- [ ] **Step 3: Add downloadable audit bundle index**

Create `artifact_index.json` with artifact filenames, purpose, and whether each is required for competition submission.

## Task 8: Security, Dependency, and CI Hardening

**Files:**
- Modify: `Makefile`
- Modify: `requirements.txt`
- Modify: `constraints.txt`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/*` if present
- Test: local quality commands

- [ ] **Step 1: Remove stale pip-audit ignores**

Change `make security` to:

```make
security:
	$(PYTHON) -m pip_audit -r requirements.txt
```

- [ ] **Step 2: Pin a release constraints lock**

Keep broad package ranges in `pyproject.toml`, but create a locked `constraints-release.txt` generated from a known-good environment.

- [ ] **Step 3: Split CI tiers**

Add:

```text
fast: ruff, mypy, focused unit tests
full: all tests
release: full tests, pip-audit, production run, artifact validation
```

- [ ] **Step 4: Add runtime environment warning gate**

The current macOS environment emitted a urllib3 LibreSSL warning during audit. It did not fail, but release docs should recommend Python linked against OpenSSL 1.1.1+.

## Task 9: Performance Profiling and Budgets

**Files:**
- Create: `scripts/benchmark_metrics.py`
- Create: `scripts/profile_pipeline.py`
- Modify: `project_config/config/performance_budgets.yaml`
- Test: benchmark smoke command

- [ ] **Step 1: Benchmark before each optimization**

Measure:

```text
metric evaluation time
bootstrap CI time
Geo feature load time
model fit time
artifact write time
```

- [ ] **Step 2: Store results as JSON**

Write benchmark output to `reports/performance/benchmark.json` with machine metadata and dataset size.

- [ ] **Step 3: Gate regressions**

Fail release verification if metric evaluation or pipeline runtime exceeds configured budgets by more than a small tolerance.

## Task 10: Final Release Proof

**Files:**
- Modify: `verify_project.py`
- Modify: `docs/competition_submission.md`
- Test: full release command

- [ ] **Step 1: Extend `verify_project.py --release`**

Require:

```text
ruff clean
mypy strict clean
pytest clean
pip-audit clean without ignores
production geo run succeeds
audit/model_card/dashboard/manifest/artifact_index exist
all required quality gates pass
temporal evidence is evaluated or explicitly justified
```

- [ ] **Step 2: Document submission checklist**

Add a one-page checklist to `docs/competition_submission.md` that maps every jury concern to a concrete artifact.

- [ ] **Step 3: Run the final command**

```bash
python3 verify_project.py --release
```

Expected: exits 0 and prints `BioSpread verification completed`.

## Recommended Implementation Order

1. Scientific validation gates and honest track reporting.
2. Metric/vectorization performance fixes.
3. Typed pipeline contracts.
4. Parquet-first I/O and artifact dual-write.
5. Pipeline orchestration split.
6. Semantic cache keys and manifest fingerprints.
7. Dashboard/model-card evidence improvements.
8. Security/CI/release verification.

This order keeps the project continuously releasable while improving the highest-risk competition concerns first.
