# BioSpread Model Card

## Model
- Name: `BioSpread`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `direct`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.931`
- Minimum AUC target: `0.82`
- OOF average precision: `0.424`
- Positive prevalence: `0.052`
- Expected calibration error: `0.311`
- Brier score: `0.167`
- Group OOF ROC AUC: `0.931`
- Temporal holdout ROC AUC: `0.931`
- External holdout ROC AUC: `0.763`
- Bootstrap ROC AUC CI: `[0.931, 0.931]`
- Bootstrap AP CI: `[0.424, 0.424]`
- Max single-feature AUC: `0.500`
- Suspicious feature count: `0`
- Evaluation cohort: `6156` backbones, `318` positives

## Quality Gates
- cross_validated: `fail`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `fail`
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
- primary: `2b7fcfe8c5448d3a83d98183997e5d57eb78f328a3530ca02933155d71d6b334`
- Python: `3.13.12`
- NumPy: `2.4.4`
- scikit-learn: `1.8.0`
- Git commit: `2b92fb791d2c016da6bd1832b9e960cceed76c2e`
