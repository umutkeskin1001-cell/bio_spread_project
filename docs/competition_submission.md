# BioSpread Competition Submission

## Reproduce The Final Run

```bash
python3 -m pip install -r requirements-dev.txt
python3 run_project.py --mode geo --output-dir reports/competition_final
python3 verify_project.py --release
```

For final packaging, run these commands with Python linked against OpenSSL 1.1.1
or newer. The release verifier records the SSL backend in `manifest.json` and
prints a warning for LibreSSL-based system Python builds.

## Expected Final Artifacts

- `reports/competition_final/report.md`
- `reports/competition_final/dashboard.html`
- `reports/competition_final/audit.json`
- `reports/competition_final/model_card.md`
- `reports/competition_final/manifest.json`
- `reports/competition_final/release_gate.json`
- `reports/competition_final/predictions.csv`
- `reports/competition_final/metrics.json`
- `reports/competition_final/benchmark.json`
- `reports/competition_final/drift_report.json`
- `reports/competition_final/model_registry.jsonl`
- `reports/competition_final/trend_report.json`

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
python3 run_project.py --mode geo --output-dir reports/competition_final
```
