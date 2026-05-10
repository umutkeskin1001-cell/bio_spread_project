# BioSpread Model Card

## Model
- Name: `biospread_omega_v15`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `direct`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.819`
- Minimum AUC target: `0.82`
- OOF average precision: `0.728`
- Positive prevalence: `0.359`
- Expected calibration error: `0.047`
- Brier score: `0.168`
- Group OOF ROC AUC: `0.819`
- Temporal holdout ROC AUC: `0.819`
- External holdout ROC AUC: `0.781`
- Bootstrap ROC AUC CI: `[0.819, 0.819]`
- Bootstrap AP CI: `[0.728, 0.728]`
- Max single-feature AUC: `0.500`
- Suspicious feature count: `0`
- Evaluation cohort: `887` backbones, `318` positives

## Quality Gates
- cross_validated: `fail`
- auc_at_least_target: `fail`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `pass`
- calibration_bin_gap_at_most_target: `pass`
- bootstrap_auc_ci_low_at_least_target: `fail`
- bootstrap_average_precision_ci_low_above_prevalence: `fail`
- group_auc_at_least_target: `fail`
- temporal_holdout_auc_at_least_target: `fail`
- temporal_consistency_passed: `pass`
- feature_lineage_passed: `pass`
- disabled_feature_leak_passed: `pass`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `pass`
- adversarial_leakage_scan_passed: `pass`
- low_knownness_slice_auc_at_least_target: `pass`
- low_knownness_slice_ap_at_least_target: `pass`
- low_knownness_slice_ece_at_most_target: `pass`
- adversarial_robustness: `pass`

## Leakage Guard
- Status: `pass`
- Feature count: `0`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `genomic_oracle_active`

## Reproducibility
- amr: `dea80560f0b6305b425f870437f2e8cd5adda577952500159196b9593a19a841`
- external_holdout: `14b61f57ae74dd7da95cbbef3ca3bedbc9299219b8a41dc4d29c0936839f7bce`
- geo_spread_features: `2b7fcfe8c5448d3a83d98183997e5d57eb78f328a3530ca02933155d71d6b334`
- records: `2f77791c9c8793890fae55ee0c8fa0c8e350641a69d83efc67c093da086d00c7`
- Python: `3.13.12`
- NumPy: `2.4.4`
- scikit-learn: `1.8.0`
- Git commit: `2b92fb791d2c016da6bd1832b9e960cceed76c2e`
