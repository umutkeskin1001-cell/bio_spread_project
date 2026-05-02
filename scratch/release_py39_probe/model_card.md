# BioSpread Model Card

## Model
- Name: `geobio_reliability_ensemble`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `spatial_group_cv_stacked`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.982`
- Minimum AUC target: `0.82`
- OOF average precision: `0.705`
- Positive prevalence: `0.053`
- Expected calibration error: `0.007`
- Brier score: `0.024`
- Group OOF ROC AUC: `0.982`
- Temporal holdout ROC AUC: `0.992`
- External holdout ROC AUC: `0.982`
- Bootstrap ROC AUC CI: `[0.978, 0.984]`
- Bootstrap AP CI: `[0.659, 0.739]`
- Max single-feature AUC: `0.945`
- Suspicious feature count: `0`
- Evaluation cohort: `6841` backbones, `362` positives

## Quality Gates
- cross_validated: `pass`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `pass`
- bootstrap_auc_ci_low_at_least_target: `pass`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `pass`
- temporal_holdout_auc_at_least_target: `pass`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `pass`
- adversarial_leakage_scan_passed: `pass`

## Leakage Guard
- Status: `pass`
- Feature count: `0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `Ensemble coefficients`

## Reproducibility
- geo_spread_features: `80015d8336d4183d76574c497120e0f7945f4ad356debabbb29052aa55fa9376`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `9cfd575b993e3d892e2d0b791ab21375fcdfc710`
