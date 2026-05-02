# BioSpread Architecture

## Runtime shape

BioSpread is a standalone package with one executable workflow:

- input selection
- feature loading or feature construction
- model fitting and validation
- artifact emission
- release-gate evaluation

The control-plane entry points are:

- `python -m bio_spread_project.cli run`
- `python -m bio_spread_project.cli verify`

## Data paths

There are two supported scoring surfaces:

- raw observation records:
  `data/raw/plasmid_backbones.tsv` plus `data/raw/amr.tsv`
- GeoSpread feature surface:
  `data/project_inputs/geo_spread/inputs/backbone_scored.tsv`

`auto` mode can still choose between them, but the selected path and the reason
are now recorded in `manifest.json` and emitted by the CLI.

## Artifact contract

Each run emits:

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
- `trend_report.json`
- `release_gate.json`
- `manifest.json`
- `report.md`

## Configuration policy

The standalone runtime treats these files as active runtime configuration:

- `project_config/quality_thresholds.json`
- `project_config/drift_thresholds.json`
- `project_config/trend_thresholds.json`
- `project_config/baseline_benchmark.json`

Files under `project_config/config/` are inherited reference assets from the
parent research codebase. They are not loaded by the standalone runtime unless
future work wires them in explicitly.
