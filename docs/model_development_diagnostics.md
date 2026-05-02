# Model Development Diagnostics

Generated: 2026-05-02T17:40:04.882560Z

This document summarizes ablations, leakage/quality signals, and top feature effects to help contributors quickly find weak spots.

## Reproduction

Run: `PYTHONPATH=src python3 -m bio_spread_project.cli run --mode geo --output-dir <dir>` after setting toggles in `project_config/config/enriched_features.yaml`.

## Ablation Summary

| Run | ROC AUC | OOF AUC | AP | ECE | Temporal AUC | Quality Gates | Leakage |
|---|---:|---:|---:|---:|---:|---|---|
| temporal_off | 0.844229 | 0.844229 | 0.786016 | 0.020004 | 0.8435705117400718 | PASS | pass |
| baseline | 0.940414 | 0.940414 | 0.922974 | 0.020141 | 0.9398799413968806 | PASS | pass |
| psge_only | 0.942438 | 0.942438 | 0.925810 | 0.021469 | 0.9406578589116934 | PASS | pass |
| rank_only | 0.873723 | 0.873723 | 0.798382 | 0.615489 | 0.8667881082342569 | FAIL | pass |
| country_debias_only | 0.933708 | 0.933708 | 0.905938 | 0.040734 | 0.9308560982250516 | PASS | pass |
| all_modeling | 0.893404 | 0.893404 | 0.828134 | 0.617300 | 0.8850367566025749 | FAIL | pass |

## Delta vs Temporal-Off Baseline

- `temporal_off`: ΔOOF AUC = `+0.000000`
- `baseline`: ΔOOF AUC = `+0.096185`
- `psge_only`: ΔOOF AUC = `+0.098209`
- `rank_only`: ΔOOF AUC = `+0.029495`
- `country_debias_only`: ΔOOF AUC = `+0.089479`
- `all_modeling`: ΔOOF AUC = `+0.049175`

## Top Feature Effects (from `audit.validation.top_features`)

### temporal_off
- `host_range_saturation_norm`: 1.394709
- `eco_clinical_context_saturation_norm`: 1.126899
- `H_external_host_range_norm`: 1.006859
- `geo_country_entropy_train`: 0.847461
- `log1p_n_countries_train`: 0.847461
- `A_eff_norm`: 0.816579
- `orit_support`: 0.814672
- `coherence_score`: 0.690606
- `amr_burden_saturation_norm`: 0.661633
- `replicon_architecture_norm`: 0.655515

### baseline
- `country_slope_train`: 3.478347
- `recent_expansion_flag`: 0.938574
- `geo_country_entropy_train`: 0.809186
- `log1p_n_countries_train`: 0.809186
- `coherence_score`: 0.671304
- `host_range_saturation_norm`: 0.654820
- `orit_support`: 0.629206
- `host_breadth_slope_train`: 0.480312
- `backbone_purity_norm`: 0.445132
- `H_obs_specialization_norm`: 0.430258

### psge_only
- `country_slope_train`: 3.211967
- `psge_7`: 0.952149
- `recent_expansion_flag`: 0.778263
- `coherence_score`: 0.722131
- `geo_country_entropy_train`: 0.674379
- `log1p_n_countries_train`: 0.674379
- `host_breadth_slope_train`: 0.650735
- `orit_support`: 0.623957
- `host_range_saturation_norm`: 0.613783
- `H_obs_specialization_norm`: 0.496286

### rank_only
- `geo_country_record_count_train`: 0.003692
- `recent_expansion_flag`: 0.002306
- `T_eff_norm`: 0.002066
- `backbone_purity_norm`: 0.001638
- `log1p_member_count_train`: 0.001632
- `replicon_architecture_norm`: 0.001299
- `H_external_host_range_norm`: 0.000990
- `amr_clinical_threat_norm`: 0.000901
- `amr_burden_saturation_norm`: 0.000888
- `metadata_missingness_burden`: 0.000855

### country_debias_only
- `recent_expansion_flag`: 0.057997
- `country_slope_train`: 0.044431
- `geo_country_record_count_train`: 0.032810
- `T_eff_norm`: 0.017997
- `orit_support`: 0.014645
- `H_obs_specialization_norm`: 0.014480
- `host_breadth_slope_train`: 0.011163
- `amr_burden_saturation_norm`: 0.010268
- `metadata_missingness_burden`: 0.009659
- `log1p_member_count_train`: 0.009540

### all_modeling
- `geo_country_record_count_train`: 0.002864
- `psge_2`: 0.002011
- `recent_expansion_flag`: 0.001904
- `T_eff_norm`: 0.001761
- `log1p_member_count_train`: 0.001197
- `psge_0`: 0.001133
- `backbone_purity_norm`: 0.001130
- `replicon_architecture_norm`: 0.001060
- `psge_7`: 0.000956
- `psge_1`: 0.000711

## Leakage Scan Snapshot

- `temporal_off`: max_single_feature_auc=0.7383268568206052, suspicious_feature_count=0, leakage_status=pass
- `baseline`: max_single_feature_auc=0.9144593653898685, suspicious_feature_count=0, leakage_status=pass
- `psge_only`: max_single_feature_auc=0.9144593653898685, suspicious_feature_count=0, leakage_status=pass
- `rank_only`: max_single_feature_auc=0.9144593653898685, suspicious_feature_count=0, leakage_status=pass
- `country_debias_only`: max_single_feature_auc=0.9144593653898685, suspicious_feature_count=0, leakage_status=pass
- `all_modeling`: max_single_feature_auc=0.9144593653898685, suspicious_feature_count=0, leakage_status=pass

## Current Recommended Default

- Keep `enable_temporal_trends: true`.
- Keep `enable_phylo_spatial_embedding: false` unless explicitly benchmarking for ROC-only gain.
- Keep `enable_rank_focal_loss: false` and `enable_soft_country_debiasing: false` (quality regressions observed).

## Absolute Optimization Strategy (May 2026)

The current state-of-the-art configuration incorporates:
1. **Synergy Interactions**: capturing multiplicative effects between biological and contextual features.
2. **Phylo-Propagation**: leveraging genomic distances to spread risk labels through the phylogenetic graph.
3. **Evidential Meta-Learner**: replaces standard stacking with an uncertainty-aware neural estimator and LightGBM ranker.

These components are designed to break the 0.84 AUC ceiling while maintaining zero-leakage and sub-5s execution.