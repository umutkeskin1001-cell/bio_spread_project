# BioSpread Model Card

## Model
- Name: `geobio_reliability_ensemble`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `spatial_group_cv_stacked`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.923`
- Minimum AUC target: `0.82`
- OOF average precision: `0.894`
- Positive prevalence: `0.366`
- Expected calibration error: `0.059`
- Brier score: `0.107`
- Group OOF ROC AUC: `0.923`
- Temporal holdout ROC AUC: `0.939`
- External holdout ROC AUC: `0.923`
- Bootstrap ROC AUC CI: `[0.906, 0.939]`
- Bootstrap AP CI: `[0.864, 0.913]`
- Max single-feature AUC: `0.914`
- Suspicious feature count: `0`
- Evaluation cohort: `989` backbones, `362` positives

## Quality Gates
- cross_validated: `pass`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `fail`
- calibration_bin_gap_at_most_target: `pass`
- bootstrap_auc_ci_low_at_least_target: `pass`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `pass`
- temporal_holdout_auc_at_least_target: `pass`
- temporal_consistency_passed: `pass`
- feature_lineage_passed: `pass`
- disabled_feature_leak_passed: `pass`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `pass`
- adversarial_leakage_scan_passed: `pass`

## Leakage Guard
- Status: `pass`
- Feature count: `0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `not_available`

## Reproducibility
- geo_spread_features: `5d0b631336fd67cba06d6b3cecf2deec433dcbe22152d126f6d7334f55b2e6b4`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `f6daf02d33aeafeeeabb2213d88162d67ba77770`
