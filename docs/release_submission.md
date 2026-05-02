# BioSpread Release Submission

## Reproduce The Final Run

```bash
python3 -m pip install -r requirements-dev.txt
python3 run_project.py --mode geo --output-dir reports/release_final
python3 verify_project.py --release
```

For final packaging, run these commands with Python linked against OpenSSL 1.1.1
or newer. The release verifier records the SSL backend in `manifest.json` and
prints a warning for LibreSSL-based system Python builds.

## Expected Final Artifacts

- `reports/release_final/report.md`
- `reports/release_final/dashboard.html`
- `reports/release_final/audit.json`
- `reports/release_final/model_card.md`
- `reports/release_final/manifest.json`
- `reports/release_final/release_gate.json`
- `reports/release_final/predictions.csv`
- `reports/release_final/metrics.json`
- `reports/release_final/benchmark.json`
- `reports/release_final/drift_report.json`
- `reports/release_final/model_registry.jsonl`
- `reports/release_final/trend_report.json`

## Judging Notes

BioSpread predicts whether a plasmid backbone observed up to `split_year` will
appear in at least two previously unseen countries within `horizon_years`.

The packaged result is a retrospective early-warning benchmark over packaged
GeoSpread features. It is not clinical diagnosis, a patient-level decision
system, or proof of field deployment performance.

## Quality Gates

Before submission, these commands must pass:

```bash
python3 -m ruff check src tests
python3 -m mypy src/bio_spread_project
python3 -m pytest tests
python3 run_project.py --mode geo --output-dir reports/release_final
```
