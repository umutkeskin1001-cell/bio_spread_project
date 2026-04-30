# BioSpread Validation Strategy

## Current Guarantees
- Cross-validated discrimination and calibration metrics are exported on every run.
- Group-aware cross-validation metrics are exported for robustness checks.
- Temporal holdout metrics are exported using recency-based split on train-year metadata.
- Adversarial single-feature leakage scan is exported and quality-gated.
- Leakage token audit is enforced for the fixed feature contract.
- Input hashes and runtime environment are written to `audit.json`.
- Consolidated gate decisions are exported to `benchmark.json`.
- Input file inventory (path, size, hash) is exported to `data_registry.json`.
- Validation tracks are labeled as `evaluated` or `not_evaluated` instead of being silently backfilled from another benchmark.
- Final-fit predictions and validation metrics are separated: `predictions.csv`
  contains final model scores, while OOF metrics in `metrics.json`, `audit.json`,
  and `benchmark.json` drive release decisions.
- Calibration bins are exported for report and dashboard inspection.

## Enforced Contracts
- Explicit run mode: `auto`, `raw`, `geo`, or `input`.
- Missing input files fail fast with clear errors.
- `auto` mode records the selected surface and selection reason in `manifest.json`.
- `--require-explicit-surface` blocks silent promotion from raw records to the packaged Geo surface.
- Geo feature surfaces must include mandatory base columns and minimum feature coverage.
- Geo runs must pass adversarial leakage gate:
  - `suspicious_feature_count == 0`
  - `max_single_feature_auc < 0.95`

## Test Layers
- Unit tests for feature engineering, model scoring, metrics, and reporting.
- Integration tests for end-to-end pipeline artifacts.
- Script-level test for `run_project.py`.

## Release-Gate Semantics
- `go`: quality, drift, and trend evidence all pass.
- `conditional_go`: quality and drift pass, but trend history is still insufficient.
- `no_go`: one or more blocking checks failed.

## Limitations
The packaged validation is retrospective. It evaluates whether train-side
features before the split can rank future geography expansion in held-out folds.
It does not establish clinical utility, patient-level safety, or field
deployment performance.
