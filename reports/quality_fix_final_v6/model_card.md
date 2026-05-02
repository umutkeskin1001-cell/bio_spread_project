# BioSpread Model Card

## Model
- Name: `geobio_reliability_ensemble`
- Input mode: `geo_reliability_feature_surface`
- Validation mode: `spatial_group_cv_stacked`
- Intended use: prioritize plasmid backbones for geographic-spread monitoring.
- Not intended use: clinical diagnosis or direct public-health intervention without expert review.

## Reliability
- OOF ROC AUC: `0.921`
- Minimum AUC target: `0.82`
- OOF average precision: `0.896`
- Positive prevalence: `0.366`
- Expected calibration error: `0.053`
- Brier score: `0.107`
- Group OOF ROC AUC: `0.921`
- Temporal holdout ROC AUC: `0.911`
- External holdout ROC AUC: `0.921`
- Bootstrap ROC AUC CI: `[0.903, 0.938]`
- Bootstrap AP CI: `[0.871, 0.918]`
- Max single-feature AUC: `0.912`
- Suspicious feature count: `0`
- Evaluation cohort: `989` backbones, `362` positives

## Quality Gates
- cross_validated: `pass`
- auc_at_least_target: `pass`
- average_precision_above_prevalence: `pass`
- calibration_ece_at_most_target: `fail`
- bootstrap_auc_ci_low_at_least_target: `pass`
- bootstrap_average_precision_ci_low_above_prevalence: `pass`
- group_auc_at_least_target: `pass`
- temporal_holdout_auc_at_least_target: `pass`
- external_holdout_auc_at_least_target: `pass`
- leakage_audit_passed: `pass`
- adversarial_leakage_scan_passed: `pass`

## Leakage Guard
- Status: `pass`
- Feature count: `29`
- Future/outcome columns are excluded from model features.

## Explanation Surface
- Top feature signal summary: `surv_intensity=0.023; gnn_embed_6=0.022; gnn_embed_2=0.011; geo_country_record_count_train=0.009; host_sampling_shannon=0.009`

## Reproducibility
- amr: `dea80560f0b6305b425f870437f2e8cd5adda577952500159196b9593a19a841`
- geo_spread_features: `80015d8336d4183d76574c497120e0f7945f4ad356debabbb29052aa55fa9376`
- Python: `3.9.6`
- NumPy: `1.26.4`
- scikit-learn: `1.6.1`
- Git commit: `a534279191cf1eebbdc23420644192df0ae4f3b8`
