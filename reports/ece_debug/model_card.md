# BioSpread Model Card

## Model
- Name: `geobio_reliability_ensemble`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `spatial_group_cv_stacked`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.806`
- Minimum AUC target: `0.80`
- OOF average precision: `0.718`
- Positive prevalence: `0.359`
- Expected calibration error: `0.023`
- Brier score: `0.163`
- Group OOF ROC AUC: `0.806`
- Temporal holdout ROC AUC: `0.810`
- External holdout ROC AUC: `0.851`
- Bootstrap ROC AUC CI: `[0.778, 0.842]`
- Bootstrap AP CI: `[0.670, 0.774]`
- Max single-feature AUC: `0.738`
- Suspicious feature count: `0`
- Evaluation cohort: `887` backbones, `318` positives

## Quality Gates
- cross_validated: `pass`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `pass`
- bootstrap_auc_ci_low_at_least_target: `fail`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `pass`
- temporal_holdout_auc_at_least_target: `pass`
- temporal_holdout_ece_at_most_target: `pass`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `pass`
- adversarial_leakage_scan_passed: `pass`

## Leakage Guard
- Status: `pass`
- Feature count: `38.0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `Ensemble coefficients`

## Reproducibility
- external_holdout: `c10603fbfb6fce0f92650fe46e2bd2da41b6665c8b4f0dc6cf5b511136984e5d`
- geo_spread_features: `2b7fcfe8c5448d3a83d98183997e5d57eb78f328a3530ca02933155d71d6b334`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `0ad13c1fc2424f20a6f223fa92b043efb2acf5db`
