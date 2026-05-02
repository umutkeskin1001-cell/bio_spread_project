# BioSpread

BioSpread is a small, standalone early-warning project extracted from the larger
plasmid-priority research codebase. It focuses on one question:

> Which plasmid backbones show signs of future geographic spread?

The project can run against local data and model inputs under `data/`, or
against an external data bundle pointed to by `BIO_SPREAD_DATA_ROOT`. The
runtime can score either raw backbone observations or a GeoSpread feature
surface, records which path it selected, and writes reproducible artifacts. It
does not import or execute the parent repository.

## Engineering And Scientific Rigor

BioSpread is engineered as a reproducible retrospective early-warning benchmark:

- **Vectorized Engine**: Polars-based loading and validation for packaged tables.
- **Advanced Ensemble**: A stacked ML architecture using **HistGradientBoosting** and **RandomForest** with a Meta-Learner blending layer.
- **Spatial Cross-Validation**: Backbone-aware GroupKFold validation is used for the packaged GeoSpread surface.
- **Reports**: Generates markdown and standalone HTML reports with AUC, calibration, release-gate, and feature-importance views.
- **Audit**: Exports input hashes, environment versions, quality gates, bootstrap intervals, and adversarial leakage scans.

BioSpread predicts whether a plasmid backbone observed up to `split_year` will
appear in at least two previously unseen countries within `horizon_years`. It is
not for clinical diagnosis, patient decisions, or real-time outbreak declaration.
The packaged metrics are a retrospective benchmark over packaged GeoSpread
features, not proof of field deployment performance.

## Current Verified Performance

The packaged high-reliability run is validated with out-of-fold
cross-validation, not an in-sample score:

- primary model: `geobio_reliability_ensemble`
- OOF ROC AUC: `0.824`
- OOF average precision: `0.752`
- positive prevalence: `0.366`
- expected calibration error: `0.041`
- quality gate: AUC >= `0.82`

Release-grade geo runs require independent external-holdout evidence in addition
to group OOF and temporal holdout metrics. A geo run without
`--external-holdout` still writes artifacts, but the quality gate is not fully
green.

## Quick Start

```bash
cd bio_spread_project
python3 -m bio_spread_project.cli run
```

This runs the packaged inputs and prints the selected input mode plus the reason
for that choice. For release-intended runs, use explicit geo mode with
`--external-holdout` and add `--require-explicit-surface` when using `auto` so
auto mode cannot silently promote raw inputs to the packaged Geo surface.

## Project Inputs

This working directory may include a full copied input bundle from the parent
project:

- `data/project_inputs/raw/`
- `data/project_inputs/external/`
- `data/project_inputs/geo_spread/`
- `data/project_inputs/silver/`
- `data/project_inputs/scores/`
- `project_config/`

Those files are large and are intentionally ignored by git by default. Keep them
locally under `data/` or point `BIO_SPREAD_DATA_ROOT` at an external bundle when
running the full high-reliability pipeline. The final competition model trains
from the GeoSpread feature surface because that is the strongest
leakage-controlled surface available to the standalone project. The simpler
raw-record path is still kept and tested as a fallback.

Outputs are written to `reports/run/`:

- `features.csv`
- `predictions.csv`
- `metrics.json`
- `model_scorecard.json`
- `audit.json`
- `model_card.md`
- `benchmark.json`
- `drift_report.json`
- `data_registry.json`
- `model_registry.jsonl`
- `manifest.json`
- `report.md`

## Run The Product Pipeline

```bash
cd bio_spread_project
PYTHONPATH=src python3 -m bio_spread_project.cli run \
  --output-dir reports/my_run \
  --mode auto \
  --split-year 2020 \
  --horizon-years 3
```

By default this uses `data/raw/plasmid_backbones.tsv` and `data/raw/amr.tsv`.
When `--mode auto` sees `data/project_inputs/geo_spread/inputs/backbone_scored.tsv`,
it records that promotion in the manifest and CLI output. To be explicit:

```bash
PYTHONPATH=src python3 -m bio_spread_project.cli run \
  --records data/raw/plasmid_backbones.tsv \
  --amr data/raw/amr.tsv \
  --geo-spread-features data/project_inputs/geo_spread/inputs/backbone_scored.tsv \
  --output-dir reports/high_reliability_run
```

Run modes:

- `--mode auto`: records path must exist; if geo surface exists, prefer it.
- `--mode raw`: force raw records + amr path.
- `--mode geo`: force geo feature surface.
- `--mode input`: force user-provided `--input` CSV.
- `--require-explicit-surface`: fail `auto` mode instead of silently promoting raw inputs to the packaged Geo surface.
- `--quality-thresholds path/to/thresholds.json`: optional custom quality-gate thresholds.
- `--external-holdout path/to/backbone_scored.tsv`: independent holdout surface required for geo-mode quality gates.
- `--baseline-benchmark path/to/baseline_benchmark.json`: baseline benchmark used for drift checks.
- `--drift-thresholds path/to/drift_thresholds.json`: drift alarm thresholds.
- `--trend-thresholds path/to/trend_thresholds.json`: rolling-regression trend thresholds.
- `--fail-on-quality-gates`: exit non-zero if any quality gate fails.
- `--fail-on-drift-fail`: exit non-zero if drift checks fail.
- `--fail-on-trend-fail`: exit non-zero when trend metrics regress after enough registry history exists.
- `--require-trend-evidence`: treat insufficient registry history as a blocking release-gate failure.

Input validation is strict: missing `--records`, `--amr`, or `--geo-spread-features`
paths now fail fast instead of silently falling back to another source.

You can keep large datasets outside the repository by setting:

```bash
export BIO_SPREAD_DATA_ROOT=/absolute/path/to/data
```

CLI/script defaults (`--records`, `--amr`, `--geo-spread-features`) automatically
resolve from `BIO_SPREAD_DATA_ROOT` when set.

You can point it at another compatible raw record table:

```bash
PYTHONPATH=src python3 -m bio_spread_project.cli run \
  --records path/to/plasmid_backbones.tsv \
  --amr path/to/amr.tsv \
  --output-dir reports/custom_run
```

For raw observation CSVs with the columns listed below, use:

```bash
PYTHONPATH=src python3 -m bio_spread_project.cli run \
  --input path/to/plasmid_records.csv \
  --output-dir reports/csv_run
```

Run tests:

```bash
cd bio_spread_project
python3 -m pytest tests
```

Run full verification:

```bash
cd bio_spread_project
python3 -m bio_spread_project.cli verify
```

Run release verification:

```bash
python3 -m bio_spread_project.cli verify --release
```

Release verification runs bytecode compilation, ruff, mypy, pytest, a CLI smoke
run, and dependency audit. Use `--skip-security` only when the dependency audit
tool is unavailable or blocked by local network policy.

Via Make targets:

```bash
make test
make quality
make verify
make release-verify
```

`make quality` runs lint (`ruff`), static type checks (`mypy`), and coverage-gated tests.
It also runs dependency audit (`pip-audit`) with temporary CVE ignores listed in
`project_config/security_audit_ignore.txt`.

Default quality thresholds are stored at:

- `project_config/quality_thresholds.json`
- `project_config/trend_thresholds.json`

Only `project_config/quality_thresholds.json`, `project_config/drift_thresholds.json`,
`project_config/trend_thresholds.json`, and `project_config/baseline_benchmark.json`
are loaded by the standalone runtime. Files under `project_config/config/` are
reference assets inherited from the parent research codebase.

Threshold files are validated strictly (numeric type/range checks) and fail fast on invalid values.

See:

- `docs/problem_statement.md`
- `docs/validation_strategy.md`
- `docs/operations_runbook.md`
- `docs/architecture.md`
- `docs/data_contracts.md`
- `docs/release_policy.md`

## Reliability And Leakage Controls

BioSpread writes two audit-focused outputs on every product run:

- `audit.json`: input SHA-256 hashes, environment versions, validation metrics,
  quality gates, and leakage-audit status.
- `model_card.md`: jury-readable summary of intended use, limitations,
  validation metrics, quality gates, and reproducibility details.
- `manifest.json`: includes `run_id`, `created_at_utc`, enforcement policy flags,
  selected input mode, selection reason, candidate input paths, and threshold source paths for traceability.
- `release_gate.json`: single decision artifact with `go`, `conditional_go`, or `no_go` readiness plus blocking-check detail.

The high-reliability model only uses train-side feature columns such as
mobility, host-range, AMR, support, and geography history signals. Columns with
future/outcome semantics such as labels, test outcomes, future spread, event
timing, and visibility expansion are blocked by the leakage audit.

Robustness checks are also exported on each run:

- group-aware OOF metrics (`group_oof_*`)
- temporal holdout metrics (`temporal_holdout_*`)
- bootstrap confidence intervals (`bootstrap_*`)
- adversarial single-feature leakage scan (`max_single_feature_auc`, `suspicious_feature_count`)
- rolling registry trend checks (`trend_report.json`)

You can run trend analysis directly from model registry history:

```bash
PYTHONPATH=src python3 -m bio_spread_project.cli trend \
  --model-registry reports/run/model_registry.jsonl \
  --window-size 10 \
  --trend-thresholds project_config/trend_thresholds.json \
  --output reports/run/trend_report.json
```

To make trend command fail CI when trend checks fail:

```bash
PYTHONPATH=src python3 -m bio_spread_project.cli trend \
  --model-registry reports/run/model_registry.jsonl \
  --fail-on-fail
```

## Input Format

CSV input must contain these columns:

- `backbone_id`
- `year`
- `country`
- `host_genus`
- `clinical_context`
- `amr_gene_count`
- `mobility_score`

The included `data/sample_plasmid_records.csv` is only a small test fixture. The
normal standalone product path uses `data/raw/plasmid_backbones.tsv` plus
`data/raw/amr.tsv`.

## Competition Reproduction

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m bio_spread_project.cli run --mode geo --output-dir reports/competition_final
python3 -m bio_spread_project.cli verify --release
```

See `docs/competition_submission.md` for the final artifact checklist.
