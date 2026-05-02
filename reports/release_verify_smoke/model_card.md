# BioSpread Model Card

## Model
- Name: `mobility`
- Input mode: `observation_records`
- Validation mode: `cross_validated`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.500`
- Minimum AUC target: `0.82`
- OOF average precision: `0.500`
- Positive prevalence: `0.400`
- Expected calibration error: `0.437`
- Brier score: `0.290`
- Group OOF ROC AUC: `0.500`
- Temporal holdout ROC AUC: `0.500`
- External holdout ROC AUC: `0.500`
- Bootstrap ROC AUC CI: `[0.500, 0.500]`
- Bootstrap AP CI: `[0.500, 0.500]`
- Max single-feature AUC: `0.500`
- Suspicious feature count: `0`
- Evaluation cohort: `5` backbones, `2` positives

## Quality Gates
- cross_validated: `pass`
- auc_at_least_target: `fail`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `fail`
- bootstrap_auc_ci_low_at_least_target: `fail`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `pass`
- temporal_holdout_auc_at_least_target: `pass`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `fail`
- adversarial_leakage_scan_passed: `pass`

## Leakage Guard
- Status: `fail`
- Feature count: `0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `clinical=0.356; country=0.329; mobility=0.307; amr=0.221; low_knownness=0.086; host=0.000`

## Reproducibility
- amr: `dea80560f0b6305b425f870437f2e8cd5adda577952500159196b9593a19a841`
- input: `87fcaf5148a0601d39d0a6566fbeb13474b321097fb0cc5c11aa7d6341357762`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `9cfd575b993e3d892e2d0b791ab21375fcdfc710`
