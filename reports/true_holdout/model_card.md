# BioSpread Model Card

## Model
- Name: `geobio_reliability_ensemble`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `direct`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.910`
- Minimum AUC target: `0.82`
- OOF average precision: `0.876`
- Positive prevalence: `0.359`
- Expected calibration error: `0.155`
- Brier score: `0.159`
- Group OOF ROC AUC: `0.910`
- Temporal holdout ROC AUC: `0.910`
- External holdout ROC AUC: `0.713`
- Bootstrap ROC AUC CI: `[0.910, 0.910]`
- Bootstrap AP CI: `[0.876, 0.876]`
- Max single-feature AUC: `0.500`
- Suspicious feature count: `0`
- Evaluation cohort: `887` backbones, `318` positives

## Quality Gates
- cross_validated: `fail`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `fail`
- bootstrap_auc_ci_low_at_least_target: `pass`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `fail`
- temporal_holdout_auc_at_least_target: `fail`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `fail`
- adversarial_leakage_scan_passed: `pass`

## Leakage Guard
- Status: `fail`
- Feature count: `0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `Meta-ensemble coefficients`

## Reproducibility
- amr: `dea80560f0b6305b425f870437f2e8cd5adda577952500159196b9593a19a841`
- external_holdout: `c10603fbfb6fce0f92650fe46e2bd2da41b6665c8b4f0dc6cf5b511136984e5d`
- geo_spread_features: `2b7fcfe8c5448d3a83d98183997e5d57eb78f328a3530ca02933155d71d6b334`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `f6daf02d33aeafeeeabb2213d88162d67ba77770`
