# BioSpread Release Policy

## Readiness states

- `go`: quality, drift, and trend evidence all pass.
- `conditional_go`: quality and drift pass, but trend history is still
  insufficient for a fully evidenced release decision.
- `no_go`: one or more blocking checks failed.

Release gates are generated from real artifacts:

- quality gates from `audit.json`
- drift checks from `drift_report.json`
- trend checks from `trend_report.json`
- consolidated readiness from `release_gate.json`

## Production-intended flags

Use these flags when a run is intended to act like a release candidate:

```bash
python3 run_project.py \
  --fail-on-quality-gates \
  --fail-on-drift-fail \
  --fail-on-trend-fail \
  --require-trend-evidence \
  --require-explicit-surface
```

## Local and CI defaults

Local development and lightweight CI may allow `conditional_go` so long as:

- quality gates pass
- drift checks pass
- the run is not being published as a release decision

## Trend evidence policy

Trend evidence is considered sufficient only after the registry history reaches
the configured threshold in `project_config/trend_thresholds.json`.

Current default:

- `required_entries_for_go = 20`

Fresh output directories usually produce `conditional_go` because they do not
yet contain enough model registry history. Use `--require-trend-evidence` only
when a release candidate must be blocked until that history exists.
