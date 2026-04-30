# BioSpread Ruthless Architectural Audit And Implementation Brief

This document consolidates the full iterative audit of the BioSpread codebase. It is written for an implementation agent. Treat it as a concrete remediation backlog, not a discussion document.

## Executive Verdict

BioSpread has strong scientific intent, but the current implementation is not yet a defensible scientific workflow. It has broken public contracts, import-time failures, silent data coercion, stale tests, weak validation semantics, non-transactional artifact writing, contradictory documentation, and a 49 GB data footprint that is not repository-safe.

The target architecture must become:

```text
Validated Runtime Config
  -> Explicit Input Selection
  -> Strict Data Schema Firewall
  -> Leakage-Controlled Feature/Label Builder
  -> Model Training + OOF Validation
  -> Status-Rich Metrics + Calibration
  -> Quality/Drift/Trend Governance
  -> Transactional Artifact Writer
  -> Manifest/Registry/Report/Dashboard
```

The most urgent blockers are:

- `src/bio_spread_project/model.py` can fail at import because `ModelRun` references `BioSpreadRiskModel` before definition without postponed annotations.
- `run_project.py` imports path helper functions that are not present in the shown `paths.py`.
- Tests import `PlasmidRecord` and `GeoSpreadFeatureRow`, but implementation no longer defines those APIs.
- Docs/code say spread means at least 2 new countries, while `project_config/config.yaml` and `project_config/config/benchmarks.yaml` say `min_new_countries_for_spread: 3`.
- Tests and docs expect `artifact_index.json`, but the pipeline shown does not write it.
- Model card backfills missing temporal/external metrics with OOF metrics, which is scientifically misleading.
- The repository contains or expects a 49 GB `data/` tree without a sane external data strategy.

## Recommended Execution Order

1. Make import/test collection work: fix `model.py`, `run_project.py`, and stale test imports.
2. Create one canonical benchmark contract and resolve the 2-vs-3 new-country contradiction.
3. Harden path/config/input validation.
4. Harden data loading and Geo surface loading.
5. Rewrite feature construction and label construction around the benchmark contract.
6. Rewrite model training and selection to reject invalid matrices and weak validation.
7. Rewrite metrics/calibration to be status-rich and deterministic.
8. Rewrite quality/drift/trend/release governance to use structured gate payloads.
9. Refactor pipeline artifact writing into a transaction and emit `artifact_index.json`.
10. Fix reporting/model card/dashboard to display evidence without backfilling.
11. Collapse duplicate CLIs/scripts and fix verification.
12. Split tests, clean docs, and implement data manifest strategy.

## 1. Configuration, Runtime Policy, Pathing, And Input Selection

### Current Flaws

- `src/bio_spread_project/paths.py` captures `BIO_SPREAD_DATA_ROOT` at import time:
  - This makes CLI defaults stale if environment variables change after import.
  - Tests relying on `monkeypatch.setenv` become import-order dependent.
- `ModelSpec.weights` is an unrestricted `dict[str, float]`.
  - No required feature schema.
  - No duplicate model-name validation.
  - No finite-value validation.
  - No rejection of negative, NaN, or infinite weights.
- `ProjectConfig` does not prove objective weights sum to `1.0`.
- `PipelineConfig` accepts invalid states:
  - invalid `run_mode`
  - negative or zero `horizon_years`
  - absurd `split_year`
  - simultaneous `input_path` and `backbone_records_path`
- `input_selection.ensure_existing_file` checks only existence and regular file status.
  - It does not check readability, empty files, or normalized absolute paths.
- `select_input_source` mixes mode validation, policy, path resolution, file checks, and provenance.
- Auto mode can silently promote raw inputs to the packaged Geo surface unless explicitly blocked.
- `manifest.portable_path` depends on `Path.cwd()`, making manifests non-reproducible across launch directories.

### Required Fixes

Create lazy path resolution:

```python
from dataclasses import dataclass
from pathlib import Path
import os

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

@dataclass(frozen=True, slots=True)
class ProjectPaths:
    project_root: Path
    data_root: Path

    @classmethod
    def from_env(cls) -> "ProjectPaths":
        root = project_root()
        data = os.environ.get("BIO_SPREAD_DATA_ROOT")
        return cls(project_root=root, data_root=Path(data or root / "data").expanduser().resolve())

    @property
    def raw_backbones(self) -> Path:
        return self.data_root / "raw" / "plasmid_backbones.tsv"

    @property
    def raw_amr(self) -> Path:
        return self.data_root / "raw" / "amr.tsv"

    @property
    def geo_spread_features(self) -> Path:
        return self.data_root / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"

    @property
    def default_output_dir(self) -> Path:
        return self.project_root / "reports" / "run"
```

Make config dataclasses frozen, slotted, and self-validating:

- Validate model names are unique.
- Validate every model weight is finite and non-negative.
- Validate exact feature set or explicitly declared feature set.
- Validate objective weights sum to `1.0`.

Make `PipelineConfig.__post_init__` reject invalid runtime states.

Replace `ensure_existing_file` with `ensure_readable_file`:

- resolve path strictly
- require regular file
- require readable file
- reject empty files unless explicitly allowed

Make `portable_path(path, root=project_root)` root-anchored, not `cwd`-anchored.

## 2. Data Loading And Schema Validation

### Current Flaws

- Tests import `PlasmidRecord`, but `src/bio_spread_project/data.py` does not define it.
- `read_table` relies on extension and Polars inference.
  - Mixed column types can silently alter schema.
- `load_records` fills missing years with `0`.
  - Year `0` is then treated as pre-split evidence.
- `load_records` clips invalid AMR and mobility values instead of rejecting them.
  - Example: mobility `70` becomes `1.0`.
- `load_backbone_records` assumes required raw columns exist but does not assert them clearly.
- `pl.coalesce([])` can occur if identity columns are absent.
- `_derive_clinical_context` breaks if none of its source columns exist.
- `_calculate_weighted_amr` treats gene symbols as regex patterns.
- AMR aggregation is scientifically vague: `weight_expr.unique().sum()` is neither clean count nor calibrated burden.
- `write_dataclass_csv` and `write_dataclass_parquet` are misnamed now that the hot path is Polars.
- `sha256_file` uses mmap, which is fragile for very large biological files.

### Required Fixes

Restore or explicitly remove `PlasmidRecord`. If keeping compatibility, define:

```python
from dataclasses import dataclass
import math

@dataclass(frozen=True, slots=True)
class PlasmidRecord:
    backbone_id: str
    year: int | None
    country: str
    host_genus: str
    clinical_context: str
    amr_gene_count: float
    mobility_score: float

    def __post_init__(self) -> None:
        if not self.backbone_id.strip():
            raise ValueError("backbone_id must be non-empty")
        if self.year is not None and not 1900 <= self.year <= 2100:
            raise ValueError(f"year outside accepted range: {self.year}")
        if not math.isfinite(self.amr_gene_count) or self.amr_gene_count < 0.0:
            raise ValueError("amr_gene_count must be finite and non-negative")
        if not math.isfinite(self.mobility_score) or not 0.0 <= self.mobility_score <= 1.0:
            raise ValueError("mobility_score must be finite in [0, 1]")
```

Define explicit observation schema:

```python
OBSERVATION_SCHEMA = {
    "backbone_id": pl.String,
    "year": pl.Int64,
    "country": pl.String,
    "host_genus": pl.String,
    "clinical_context": pl.String,
    "amr_gene_count": pl.Float64,
    "mobility_score": pl.Float64,
}
```

Make `load_records`:

- require all columns
- strip IDs and strings
- reject null/blank IDs
- reject null/impossible years
- reject non-finite or out-of-range values
- never clip invalid values
- never invent year `0`

Make raw AMR loading:

- require `NUCCORE_ACC`
- coalesce gene column from `gene_symbol`/`gene_name`
- treat identifiers as strings, not regex programs
- deduplicate by `NUCCORE_ACC, gene_id`
- document whether score is weighted burden, gene count, or severity proxy

Replace mmap hashing with streaming hashing:

```python
def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

## 3. Feature Engineering And Label Construction

### Current Flaws

- `GENUS_TO_ORDER` is a hard-coded toy taxonomy.
- Unknown genus becomes `Unknown_<genus>`, inflating apparent diversity.
- `label_geo_spread` is hard-coded as `n_new_countries_future >= 2`.
- Config says `min_new_countries_for_spread: 3`.
- `knownness_score` is arbitrary:
  - `(record_count/8 + country_count/5) / 2`
  - no posterior meaning
  - no exposure correction
- No validation for split year, horizon, null years, or required columns.
- Future labels ignore surveillance bias.
- Feature aggregation uses means only for AMR and mobility.
  - It loses max, variance, trend, and distribution.
- Clinical detection is brittle and conflates human/clinical metadata.
- List aggregation for future/pre countries is more memory-heavy and fragile than anti-join set logic.

### Required Fixes

Create a `FeatureBuildConfig`:

```python
@dataclass(frozen=True, slots=True)
class FeatureBuildConfig:
    split_year: int
    horizon_years: int
    min_new_countries_for_spread: int
    knownness_record_weight: float = 0.55
    knownness_country_weight: float = 0.30
    knownness_host_weight: float = 0.15
```

Feature builder must:

- validate required observation columns
- reject invalid years and values
- normalize country/host/context strings
- use only `year <= split_year` for features
- use only `(split_year, split_year + horizon_years]` for labels
- build new-country counts with anti-join:

```python
new_country_counts = (
    future_countries
    .join(pre_countries, on=["backbone_id", "country"], how="anti")
    .group_by("backbone_id")
    .agg(pl.len().alias("n_new_countries_future"))
)
```

Knownness should be monotone and bounded, for example:

```python
knownness = 1.0 - exp(-(w_records * log1p(records) + w_countries * countries + w_hosts * hosts) / scale)
```

Unknown taxonomy should reduce support via a `taxonomy_support_pre` feature, not create synthetic diversity.

Add features:

- `max_amr_gene_count_pre`
- `sd_amr_gene_count_pre`
- `max_mobility_score_pre`
- `sd_mobility_score_pre`
- `taxonomy_support_pre`

## 4. Model Training, Prediction, And Model Selection

### Current Flaws

- `model.py` lacks `from __future__ import annotations`; `ModelRun` references `BioSpreadRiskModel` before definition.
- `ModelSpec.weights` values are ignored.
  - Only keys choose feature names.
- Unknown model features silently become constant zero columns.
- Missing feature values become zero through `.fill_null(0.0)`.
- Single-class training is not rejected before fitting.
- Small-sample validation falls back to in-sample predictions but is still treated as metrics.
- `Prediction` does not validate probability, tier, label, or knownness.
- Model selection uses arbitrary weights and does not penalize weak validation.
- Geo model subclasses `BioSpreadRiskModel` without honoring its initializer or fields.
- Pickle/joblib model persistence lacks metadata and schema hash.

### Required Fixes

Immediately add:

```python
from __future__ import annotations
```

Harden `Prediction`:

- `risk_probability` finite in `[0, 1]`
- `label_geo_spread` in `{0, 1}`
- `knownness_score` finite in `[0, 1]`
- tier in `{high, medium, review, not_evaluated}`
- non-empty `model_name` and `backbone_id`

Replace `_extract_matrix`:

- maintain a `FEATURE_EXPRESSIONS` map
- reject unknown feature names
- reject missing required columns
- reject non-finite matrix values
- use `float64` and contiguous arrays

Use `ModelSpec.weights` or rename them:

- If weights are intended as priors, multiply standardized features by non-negative prior weights before fitting.
- If not, rename field to `features`.

Training must:

- reject empty frames
- require `label_geo_spread`
- require at least two classes
- use `max_iter=1000`
- emit model metadata with feature names and schema

Validation must:

- use true OOF probabilities when feasible
- mark insufficient validation as `not_evaluated_*`
- never use apparent/in-sample scores as selection-grade metrics

Model selection must:

- require eligible validation mode
- penalize small `n`
- penalize high ECE
- prefer validated models over apparent metrics

## 5. Geo Reliability, Leakage Control, And External Holdout

### Current Flaws

- Tests expect `GeoSpreadFeatureRow`; implementation returns `pl.DataFrame`.
- Tests call `single_feature_leakage_scan(..., auc_threshold=...)`; implementation accepts `threshold`.
- `GeoBioReliabilityModel` fake-subclasses `BioSpreadRiskModel`.
- `load_geo_spread_feature_rows` silently filters invalid labels.
- Required columns omit columns actually used:
  - `new_macro_regions`
  - `metadata_support_depth_norm`
  - `metadata_missingness_burden`
- Runtime feature contract is hard-coded independently from `project_config/config/benchmarks.yaml`.
- `_feature_matrix` fills null model features with zero.
- `StratifiedGroupKFold(n_splits=5)` is not feasibility-checked.
- `StackingClassifier(cv=3)` inside group CV is not group-aware.
- `_make_calibrated_stack` is not actually calibrated.
- Temporal holdout uses arbitrary 70th percentile cutoff.
- Permutation importance is in-sample and uses global RNG.
- Leakage audit is lexical only.
- Single-feature leakage scan lacks single-class protection.
- Uncertainty is batch-relative z-score, not model/training-distribution uncertainty.
- Attribution is fake local attribution: row value times global importance.
- External holdout independence only checks path/hash, not overlapping `backbone_id`.

### Required Fixes

Define a `GeoSurfaceContract`:

- feature columns
- label column
- outcome column
- required base columns
- required support columns
- assignment mode
- `min_new_countries_for_spread`

Loading must:

- derive optional geo aliases before required-feature check
- reject invalid labels instead of filtering them
- reject wrong assignment mode
- reject null model features
- require temporal metadata
- produce `label_geo_spread`, `n_new_countries_future`, `knownness_score`

Feature matrix must reject missing/null/non-finite values.

Replace fake inheritance with composition or a shared protocol:

```python
@dataclass(slots=True)
class GeoBioReliabilityModel:
    estimator: Any
    feature_columns: tuple[str, ...]
    train_center: np.ndarray
    train_scale: np.ndarray
    importances: list[dict[str, Any]]
```

Validation must:

- compute feasible split count from class counts
- reject or mark not evaluated if insufficient classes
- use real calibration, e.g. `CalibratedClassifierCV`
- keep internal calibration group-aware if needed

Permutation importance:

- deterministic `np.random.default_rng(seed)`
- computed on validation/OOF or holdout data
- no global RNG

Leakage scan:

- accept both `threshold` and `auc_threshold` for compatibility
- handle single-class labels
- scan fixed feature columns and optionally broader numeric surface

External holdout:

- load holdout features
- reject same file/hash
- reject overlapping `backbone_id`

## 6. Evaluation Metrics, Calibration, And Bootstrapping

### Current Flaws

- Tests import `_fast_auc` and `_fast_average_precision`, but implementation defines `_fast_auc_kernel` and `_fast_ap_kernel`.
- Numba introduces compile latency and dependency risk without proven benefit.
- Average precision implementation may diverge from sklearn tie semantics.
- Metrics do not validate scores are finite probabilities.
- Single-class AUC returns `0.5` without status.
- Top-k precision uses undocumented `max(10, 1%)`.
- Bootstrap is naive row bootstrap, not grouped.
- Bootstrap silently includes degenerate samples by turning AUC into `0.5`.
- Calibration empty bins are encoded as zero mean/observed rate.
- Fixed-bin ECE lacks sample-support caveats.
- `validation_summary` omits key support/calibration metrics.

### Required Fixes

Expose:

- `_fast_auc`
- `_fast_average_precision`

Use exact vectorized AUC with tie handling.

Use sklearn `average_precision_score` unless a custom implementation is proven equivalent.

Validate arrays:

- labels 1D, binary
- scores 1D, same length
- scores finite in `[0, 1]`

`evaluate_predictions` should return:

- `n_backbones`
- `n_positive`
- `prevalence`
- `roc_auc`
- `average_precision`
- `top_k`
- `top_k_precision`
- `abstain_rate`
- `discrimination_status`

Bootstrap should:

- use deterministic `np.random.default_rng`
- support `groups`
- skip/count degenerate resamples
- emit `bootstrap_degenerate_resamples`
- emit `bootstrap_effective_resamples`

Calibration should:

- validate `bins >= 2`
- use `None` for empty-bin `mean_prediction` and `observed_rate`
- include maximum calibration error

`validation_summary` should include:

- sample sizes
- prevalence
- ECE
- Brier
- AP CI high/low
- degenerate bootstrap counts
- temporal/external AP

## 7. Quality Gates, Drift, Trend, And Release Governance

### Current Flaws

- `QualityThresholds` uses descriptors inside dataclass with `type: ignore`.
- Unknown threshold keys are silently ignored.
- Missing metrics become numeric defaults without reason metadata.
- Cross-validation detection uses substring matching.
- Drift passes when metrics are missing.
- Drift checks only drops, not suspicious large improvements.
- Trend compares all registry rows regardless of compatibility.
- Missing trend columns become zeros.
- Registry entries lack run/model/input/benchmark compatibility fields.
- Registry append is not lock-safe.
- Release gate emits arbitrary weighted score.
- Conditional release is not clearly separated from full release in all artifacts.

### Required Fixes

Make `QualityThresholds` frozen/slotted with `__post_init__`.

`load_quality_thresholds` must reject unknown keys.

`evaluate_quality_gates` should return structured gate objects:

```python
{
  "auc_at_least_target": {
    "passed": true,
    "status": "pass",
    "observed": 0.83,
    "threshold": 0.82,
    "reason": ""
  }
}
```

Validation modes must be enumerated, not substring-matched.

Drift:

- missing comparable required metric should fail or block
- include `status`
- include suspicious large-improvement review flags

Trend:

- filter registry history by compatibility:
  - `model_name`
  - `input_mode`
  - `validation_mode`
  - benchmark/version if present
- missing required trend columns should return `invalid_history`
- no fake zero means

Release gate:

- remove fake weighted score
- report `readiness`, `checks`, `blocked_by`, `policy_version`
- allow `conditional_go` only when explicitly permitted

## 8. Pipeline Orchestration, Artifact Writing, Manifest, Registry, Reproducibility

### Current Flaws

- Pipeline uses mutable `RunContext`.
- Optional state is dereferenced as required state.
- Geo loader double-wraps DataFrame.
- Model persistence differs between raw and geo paths.
- Artifact write futures for parquet are not named/explicitly awaited.
- Manifest can list artifacts that failed.
- `artifact_index.json` is expected but not emitted.
- Writes happen directly into final output directory.
- Failed runs can leave partial artifacts.
- `manifest.portable_path` is `cwd`-dependent.
- `config_fingerprint` ignores YAML files under `project_config/config`.
- `semantic_table_hash` scans TSV with CSV defaults.
- `semantic_table_hash` min/max/mean is collision-prone.
- `source_fingerprint` uses mtime cache and only top-level `*.py`.
- Model registry lacks run identity and compatibility metadata.
- Data registry lacks sizes/hashes/schema/row counts.

### Required Fixes

Generate `RunIdentity` once:

```python
@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    created_at_utc: str
```

Use local variables instead of mutable `self.state`.

Use artifact transaction:

- write to staging dir
- await all futures
- build artifact index from actual files
- write manifest after artifact index
- optionally promote staging atomically

Always write `artifact_index.json`:

```json
{
  "schema_version": "artifact_index_v1",
  "artifact_count": 0,
  "artifacts": {
    "metrics": {
      "path": "reports/run/metrics.json",
      "bytes": 123,
      "sha256": "..."
    }
  }
}
```

Make config fingerprint recursive over `.json`, `.yaml`, `.yml`, `.txt`.

Make source fingerprint recursive over `src/bio_spread_project/**/*.py`.

Make semantic table hash separator-aware for TSV.

Registry entry must include:

- schema version
- run id
- model name
- input mode
- validation mode
- source fingerprint
- config fingerprint
- dependency fingerprint
- quality gate result
- validation summary

## 9. Reporting, Dashboard, Audit Rendering, Scientific Communication

### Current Flaws

- `reporting.py` assumes headline metrics exist.
- Calibration rendering cannot handle `None` empty bins.
- Leakage status is read from metrics instead of audit.
- Quality gates collapse fail into soft `review`.
- Report lacks actual release-gate outcome and blockers.
- Model card backfills missing group/temporal/external metrics with OOF metrics.
- Dashboard assumes numeric values and can crash on missing metrics.
- Dashboard uses `innerHTML` with unsanitized values.
- Dashboard lacks release-gate/drift/trend context.
- “Feature Importance” label is vague; should say permutation importance if that is the method.
- Audit `all_quality_gates_passed` assumes gates are booleans.

### Required Fixes

Add safe formatter:

```python
def _fmt_value(value: Any) -> str:
    if value is None:
        return "not_evaluated"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
```

Reports should accept `audit` and `release_gate`.

Reports must display:

- validation mode
- evaluated/not-evaluated status per track
- release readiness
- blocked checks
- leakage audit status from audit
- calibration bins with missing values as `not_evaluated`

Model card must not backfill missing tracks:

```python
def _track_or_na(track: dict[str, Any], key: str) -> str:
    if track.get("status") != "evaluated":
        return str(track.get("status", "not_evaluated"))
    return _metric_or_na(track.get(key))
```

Dashboard:

- use `textContent`, not `innerHTML`
- use JS `fmt()` for missing values
- show release gate readiness and blockers
- show quality gate details with structured gates
- rename feature chart to `Permutation Importance` when applicable

Audit:

- support structured gates:

```python
def _gate_passed(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("passed", False))
    return bool(value)
```

## 10. CLI, Scripts, Verification, Packaging, Developer Operations

### Current Flaws

- `run_project.py` imports missing path helper functions.
- `cli.py` uses import-time path constants.
- `cli health` returns `0` even when mypy/pytest fail.
- CLI assumes `result.metrics["roc_auc"]` exists.
- `run_project.py` duplicates CLI parser and semantics.
- `run_project.py` does not support all modes supported by package CLI.
- `verify_project.py --release` can allow insufficient trend evidence.
- `verify_project.py` expects `artifact_index.json`, but pipeline does not write it.
- `verify_project.py` duplicates performance thresholds already in quality config.
- `Makefile clean-reports` is destructive over fixed directories.
- `requirements-dev.txt` mixes `-c constraints.txt`, `-e .[dev]`, and repeated dependencies.
- `constraints.txt` appears empty/useless in shown output.
- `numba` is runtime dependency because metrics are overengineered.
- Python `>=3.9` is claimed, but support must be verified or requirement raised.

### Required Fixes

Make `run_project.py` a shim:

```python
from bio_spread_project.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Make CLI defaults use `ProjectPaths.from_env()` inside `build_parser`.

Make health command return nonzero on failed checks.

Make CLI metric printing status-aware.

Verification should:

- use generated quality gates, not duplicate thresholds
- validate artifact index
- accept `--require-go`
- pass `--require-trend-evidence` when full release is required
- distinguish `conditional_go` local CI from `go` release

Requirements:

- Remove `numba` unless benchmark proves it necessary.
- Make `requirements-dev.txt` only:

```text
-c constraints.txt
-e .[dev]
```

- Either pin `constraints.txt` or remove it.

## 11. Documentation, Data Assets, Repository Hygiene, Test Architecture

### Current Flaws

- `data/` is 49 GB.
- Generated caches exist in workspace:
  - `.hypothesis`
  - `.mypy_cache`
  - `.coverage`
  - `.DS_Store`
  - `.pytest_cache`
  - `.ruff_cache`
- Root `.gitignore` should explicitly ignore `.hypothesis/`.
- `tests/test_bio_spread.py` is 780 lines and mixes unrelated concerns.
- Docs contradict config on spread threshold.
- README says security audit uses ignore file, but Makefile does not.
- Operations runbook gives invalid external holdout example using same file as training.
- Docs mention weighted release score, which should be removed.
- README hard-codes performance numbers without run id/artifact hash/date.
- CI release job likely cannot handle 49 GB data unless assets are actually committed or downloaded.
- Tests lack markers for data-required, slow, integration, property.

### Required Fixes

Data strategy:

- Keep tiny fixtures in Git.
- Move huge data to external asset bundle or Git LFS.
- Add `data/manifest.json` with path, bytes, SHA-256, source, license, and required use.

Add root `.gitignore` entries:

```gitignore
.hypothesis/
htmlcov/
reports/
output/
.data/
```

Create canonical benchmark contract:

```python
@dataclass(frozen=True, slots=True)
class GeoSpreadBenchmarkContract:
    split_year: int = 2015
    horizon_years: int = 5
    min_new_countries_for_spread: int = 3
    label_column: str = "spread_label"
    outcome_column: str = "n_new_countries"
    assignment_mode: str = "training_only"
```

Fix docs to match that contract.

Add docs consistency tests.

Split tests:

```text
tests/unit/
tests/integration/
tests/regression/
tests/property/
```

Add pytest markers:

```toml
markers = [
  "data_required: requires packaged project data",
  "slow: model training or release verification",
  "integration: end-to-end pipeline behavior",
  "property: property-based fuzz tests",
]
```

Fix runbook external holdout example:

```bash
--external-holdout tests/fixtures/geo_holdout.tsv
```

Add generated metrics block in README or remove hard-coded metrics.

## Final Acceptance Criteria

The implementation is not complete until all of these are true:

- `python -m compileall src` succeeds.
- `pytest` collection succeeds without import errors.
- Tests no longer import missing `PlasmidRecord` or `GeoSpreadFeatureRow`, unless those APIs are restored.
- One canonical spread threshold is used across config, code, docs, and tests.
- `bio-spread run --mode geo ...` writes:
  - `features.csv`
  - `features.parquet`
  - `predictions.csv`
  - `predictions.parquet`
  - `metrics.json`
  - `audit.json`
  - `model_card.md`
  - `benchmark.json`
  - `drift_report.json`
  - `data_registry.json`
  - `model_registry.jsonl`
  - `trend_report.json`
  - `release_gate.json`
  - `manifest.json`
  - `artifact_index.json`
  - `report.md`
  - `dashboard.html`
- `artifact_index.json` contains size and SHA-256 for every emitted artifact.
- Manifest path output is stable regardless of current working directory.
- Missing validation tracks render as `not_evaluated`, never as OOF fallback.
- Quality gates expose structured reasons.
- Drift fails or blocks when required comparable metrics are missing.
- Trend compares only compatible registry rows.
- Release report/dashboard/model card show `go`, `conditional_go`, or `no_go` plus blockers.
- External holdout rejects same file, same hash, and overlapping `backbone_id`.
- No huge external data is required for fast unit tests.
- CI has fast and full/release lanes with markers.

## Files Most Likely To Change

- `src/bio_spread_project/paths.py`
- `src/bio_spread_project/config.py`
- `src/bio_spread_project/runtime_policy.py`
- `src/bio_spread_project/input_selection.py`
- `src/bio_spread_project/data.py`
- `src/bio_spread_project/features.py`
- `src/bio_spread_project/model.py`
- `src/bio_spread_project/geo_reliability.py`
- `src/bio_spread_project/evaluation.py`
- `src/bio_spread_project/calibration.py`
- `src/bio_spread_project/model_metrics.py`
- `src/bio_spread_project/quality.py`
- `src/bio_spread_project/drift.py`
- `src/bio_spread_project/trend.py`
- `src/bio_spread_project/release_gate.py`
- `src/bio_spread_project/pipeline.py`
- `src/bio_spread_project/manifest.py`
- `src/bio_spread_project/cache_keys.py`
- `src/bio_spread_project/registry.py`
- `src/bio_spread_project/io_utils.py`
- `src/bio_spread_project/reporting.py`
- `src/bio_spread_project/dashboard.py`
- `src/bio_spread_project/audit.py`
- `src/bio_spread_project/cli.py`
- `run_project.py`
- `verify_project.py`
- `Makefile`
- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `constraints.txt`
- `README.md`
- `docs/*.md`
- `.gitignore`
- `.github/workflows/ci.yml`
- `tests/**`

## Non-Negotiable Scientific Rules

- Never silently convert missing evidence into zero evidence.
- Never silently drop malformed labels.
- Never backfill one validation track with another.
- Never report in-sample metrics as validation.
- Never let documentation define a different target than code/config.
- Never allow release governance to pass because metrics are absent.
- Never treat a same-file or overlapping holdout as independent.
- Never list an artifact in the manifest unless it was written and hashed.

