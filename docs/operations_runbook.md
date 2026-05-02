# BioSpread Operations Runbook

## Standard Run
```bash
python3 run_project.py \
  --fail-on-quality-gates \
  --fail-on-drift-fail \
  --fail-on-trend-fail \
  --require-trend-evidence \
  --require-explicit-surface
```

## Quality Gate Run
```bash
make quality
python3 verify_project.py
```

## Release Python Runtime
Use a Python build linked against OpenSSL 1.1.1 or newer for final release packaging.
`verify_project.py --release` records the active SSL backend in `manifest.json` and prints a warning when the runtime is linked against LibreSSL or an older OpenSSL.

## Lightweight CI Verification
```bash
python3 verify_project.py --skip-run-if-data-missing
```

## External Holdout Evaluation
```bash
python3 run_project.py \
  --mode geo \
  --geo-spread-features data/project_inputs/geo_spread/inputs/backbone_scored.tsv \
  --external-holdout data/project_inputs/geo_spread/inputs/backbone_scored.tsv
```

## Standalone Trend Evaluation
```bash
PYTHONPATH=src python3 -m bio_spread_project.cli trend \
  --model-registry reports/run/model_registry.jsonl \
  --window-size 10 \
  --trend-thresholds project_config/trend_thresholds.json \
  --output reports/run/trend_report.json
```

## Artifacts to Inspect
- `reports/run/audit.json`
- `reports/run/benchmark.json`
- `reports/run/drift_report.json`
- `reports/run/data_registry.json`
- `reports/run/model_registry.jsonl`
- `reports/run/trend_report.json`
- `reports/run/manifest.json` (`run_id`, `created_at_utc`, policy flags, threshold sources)
- `reports/run/release_gate.json` (`readiness: go/conditional_go/no_go`, weighted score, blocked checks)

## Incident Checklist
1. Confirm `quality_gates` in `audit.json` and `benchmark.json`.
2. Check latest line in `model_registry.jsonl` for run metadata.
3. Check `trend_report.json` status (`ok` or `insufficient_data`), required history, and pass flags.
4. Confirm `release_gate.json` is `go` for release decisions. `conditional_go` is acceptable only for local or lightweight CI runs.
5. Validate input hashes from `data_registry.json`.
6. Re-run with explicit `--quality-thresholds` / `--drift-thresholds` / `--trend-thresholds` if temporary policy change is needed.
