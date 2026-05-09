# BioSpread Model Card

## Model
- Name: `geobio_reliability_ensemble`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `spatial_group_cv_stacked`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.879`
- Minimum AUC target: `0.82`
- OOF average precision: `0.799`
- Positive prevalence: `0.359`
- Expected calibration error: `0.025`
- Brier score: `0.129`
- Group OOF ROC AUC: `0.879`
- Temporal holdout ROC AUC: `0.942`
- External holdout ROC AUC: `0.674`
- Bootstrap ROC AUC CI: `[0.859, 0.900]`
- Bootstrap AP CI: `[0.748, 0.834]`
- Max single-feature AUC: `0.915`
- Suspicious feature count: `0`
- Evaluation cohort: `887` backbones, `318` positives

## Quality Gates
- cross_validated: `pass`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `pass`
- calibration_bin_gap_at_most_target: `pass`
- bootstrap_auc_ci_low_at_least_target: `pass`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `pass`
- temporal_holdout_auc_at_least_target: `pass`
- temporal_consistency_passed: `fail`
- feature_lineage_passed: `pass`
- disabled_feature_leak_passed: `pass`
- external_holdout_auc_at_least_target: `fail`
- leakage_audit_passed: `pass`
- adversarial_leakage_scan_passed: `pass`
- low_knownness_slice_auc_at_least_target: `pass`
- low_knownness_slice_ap_at_least_target: `fail`
- low_knownness_slice_ece_at_most_target: `pass`

## Leakage Guard
- Status: `pass`
- Feature count: `0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `not_available`

## Reproducibility
- amr: `dea80560f0b6305b425f870437f2e8cd5adda577952500159196b9593a19a841`
- external_holdout: `14b61f57ae74dd7da95cbbef3ca3bedbc9299219b8a41dc4d29c0936839f7bce`
- geo_spread_features: `2b7fcfe8c5448d3a83d98183997e5d57eb78f328a3530ca02933155d71d6b334`
- records: `2f77791c9c8793890fae55ee0c8fa0c8e350641a69d83efc67c093da086d00c7`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `2047c473e30119ec239e8344eec3035e8b87d5bd`
