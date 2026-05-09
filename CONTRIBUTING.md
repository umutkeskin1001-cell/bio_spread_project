# Contributing

## Setup

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -c constraints.txt -e .[dev]
```

## Standard checks

```bash
make lint
make typecheck
make test-cov
make smoke-cli
make verify
```

## Release-intended run

```bash
python3 -m bio_spread_project.cli run \
  --fail-on-quality-gates \
  --fail-on-drift-fail \
  --fail-on-trend-fail \
  --require-trend-evidence \
  --require-explicit-surface
```
